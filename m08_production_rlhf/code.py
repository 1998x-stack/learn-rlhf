"""m08 Production RLHF — 分布式/生产式 RLHF 架构 (v0.7).

Run:  python m08_production_rlhf/code.py

在 m04 (v0.3) 的离散回答级 PPO 核心之上，模拟"生产级/分布式 RLHF"的关键工程
骨架（versions.md §9 v0.6 的生产架构，教学上收敛为单 GPU 仿真）：

    Prompt Dataset
         ↓
    Distributed Rollout Workers ───────── production: 多 GPU / vLLM 推理
         ↓
    Reward / Safety / Rule Evaluators
         ↓
    Replay Buffer ◄────────── rollout 与训练解耦（生产范式）
         ↓
    PPO Training Workers
         ↓
    Periodic Evaluation

本模块聚焦四块"系统件"（而非新 RL 算法）：

1. ReplayBuffer（回放缓冲）：rollout 阶段把 (prompt, action, log_prob,
   raw_reward, kl, value) 压进缓冲，trainer 阶段再从缓冲里 sample 一个
   minibatch——把"数据采集"和"策略更新"解耦，这正是生产分布式 RLHF 的
   rollout⇄trainer 流水线的简化镜像。
2. AdaptiveKLController（§11.3）：跟踪近期平均 KL，KL > 目标 → 增大 β，
   KL < 目标 → 减小 β，让 KL 稳定在 target 附近，既不 reward-hacking（太小）
   也不失效（太大）。
3. Checkpoint save/load：torch.save dict（policy/optimizer state、step、
   config），torch.load + load_state_dict 恢复，支持断点续训。
4. 周期评估 + 断点恢复：每 K 次迭代保存一次 checkpoint，最后 load 回来验证
   round-trip 一致（logits allclose + 恢复的 optimizer 可再走一步）。
"""

# --- v0.7: 生产级 RLHF 架构 ---
from __future__ import annotations

import copy
import os
import sys

# 本文件名为 code.py，运行时会以其所在目录作为 sys.path[0]；torch 内部会
# `import code`（标准库的 code 模块），若本目录留在 sys.path 中会被本文件错误遮蔽
# 导致导入崩溃。import 任何 torch 之前先剔除本目录。
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(os.path.expanduser(p)) != _here]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 配置与数据：离散回答级多候选任务（复用 m04 任务形状）
# ============================================================

prompts = [
    "1+1等于多少？",
    "如何安全地过马路？",
    "天空为什么是蓝色的？",
    "什么是过拟合？",
]

responses = [
    ["2。", "3。", "你猜呢？"],
    ["看信号灯并走斑马线。", "闭眼快速冲过去。", "跟着别人走就行。"],
    ["大气对短波散射更强。", "因为海洋把天空染蓝。", "蓝色比较好看。"],
    ["模型记住训练数据而泛化较差。", "模型训练太慢。", "模型参数太多。"],
]

num_prompts = len(prompts)
num_actions = len(responses[0])
prompt_ids = torch.arange(num_prompts, device=device)

# 带噪声的 SFT 标签（prompt 0/2 指向正确候选，prompt 1/3 指向错误候选）。
sft_labels = torch.tensor([0, 2, 0, 1], device=device)

# 人类偏好：每个 prompt 的第 0 个回答最好（Reward Model 学习目标）。
preference_pairs = [
    (pid, 0, 1) for pid in range(num_prompts)
] + [(pid, 0, 2) for pid in range(num_prompts)]
preference_pairs_t = torch.tensor(preference_pairs, device=device)


# ============================================================
# 1. 离散 RL 核心：Policy / Reward / Value（复刻 m04 形状）
# ============================================================

class PolicyModel(nn.Module):
    """prompt -> 每个候选回答的 logits。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, pids: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.prompt_embedding(pids))


class RewardModel(nn.Module):
    """(prompt, action) -> 标量奖励。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.action_embedding = nn.Embedding(num_actions, hidden_size)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pids: torch.Tensor, aids: torch.Tensor) -> torch.Tensor:
        h = torch.cat(
            [self.prompt_embedding(pids), self.action_embedding(aids)], dim=-1
        )
        return self.reward_head(h).squeeze(-1)


class ValueModel(nn.Module):
    """prompt -> 期望奖励标量（critic）。"""

    def __init__(self, num_prompts: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, pids: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.prompt_embedding(pids)).squeeze(-1)


# ============================================================
# 2. SFT（v0.0）：先教基础行为，冻结成 Reference（KL 锚点）
# ============================================================

policy = PolicyModel(num_prompts, num_actions).to(device)
sft_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
for _ in range(80):
    sft_optimizer.zero_grad()
    loss = F.cross_entropy(policy(prompt_ids), sft_labels)
    loss.backward()
    sft_optimizer.step()


def chosen_prob_0(model: nn.Module) -> float:
    """policy 给出受奖候选 0 的平均概率（真实质量代理，诊断用）。"""
    with torch.no_grad():
        return F.softmax(model(prompt_ids), dim=-1)[:, 0].mean().item()


sft_chosen = chosen_prob_0(policy)
print(f"[SFT] 候选0平均概率 = {sft_chosen:.4f}")

# Reference Policy：冻结的 SFT 策略 = KL 惩罚锚点。
reference_policy = copy.deepcopy(policy)
reference_policy.eval()
for p in reference_policy.parameters():
    p.requires_grad_(False)


# ============================================================
# 3. 训练 Reward Model（v0.1，BT loss；随后冻结为生产 evaluator）
# ============================================================

reward_model = RewardModel(num_prompts, num_actions).to(device)
rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)
for _ in range(80):
    pr = preference_pairs_t[:, 0]
    chosen = preference_pairs_t[:, 1]
    rejected = preference_pairs_t[:, 2]
    rm_loss = -F.logsigmoid(
        reward_model(pr, chosen) - reward_model(pr, rejected)
    ).mean()
    rm_optimizer.zero_grad()
    rm_loss.backward()
    rm_optimizer.step()

reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad_(False)
print("[RM] Bradley-Terry 训练完成；RM 已冻结为生产 evaluator。")


# ============================================================
# 4. 生产系统件 A：ReplayBuffer（回放缓冲，解耦采集与训练）
# ============================================================

class Rollout:
    """单条 rollout 样本（生产 rollout worker 产出的经验）。

    字段即 (prompt, action, log_prob, kl, reward, value)。
    `sample_batch` 会把这些条目按字段堆叠成"批量"张量。
    """

    __slots__ = ("prompt_id", "action", "log_prob", "kl", "reward", "value")

    def __init__(
        self,
        prompt_id: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        kl: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
    ):
        self.prompt_id = prompt_id
        self.action = action
        self.log_prob = log_prob
        self.kl = kl
        self.reward = reward
        self.value = value


class ReplayBuffer:
    """bounded FIFO 回放缓冲。

    push(rollout) 压入一条经验；超过 cap 时丢弃最旧的。
    sample_batch(size) 均匀采样 size 条，返回字段堆叠的 Rollout。

    这就是生产分布式 RLHF"rollout⇄训练解耦"的零拷贝直译：真实系统里
    rollout worker 与 PPO trainer 是不同进程/设备，靠 buffer 传递经验；
    采集方只管 push、训练方只管 sample，互不阻塞。
    """

    def __init__(self, cap: int = 256):
        self.data: list[Rollout] = []
        self.cap = cap

    def push(self, entry: Rollout) -> None:
        self.data.append(entry)
        if len(self.data) > self.cap:
            self.data = self.data[-self.cap:]

    def __len__(self) -> int:
        return len(self.data)

    def sample_batch(self, size: int) -> Rollout:
        """均匀采样 size 条，返回字段堆叠成 (size,) 的批量 Rollout。"""
        idx = torch.randint(0, len(self.data), (size,))
        return Rollout(
            prompt_id=torch.stack([self.data[i].prompt_id for i in idx]),
            action=torch.stack([self.data[i].action for i in idx]),
            log_prob=torch.stack([self.data[i].log_prob for i in idx]),
            kl=torch.stack([self.data[i].kl for i in idx]),
            reward=torch.stack([self.data[i].reward for i in idx]),
            value=torch.stack([self.data[i].value for i in idx]),
        )


# ============================================================
# 5. 生产系统件 B：Adaptive KL Controller（versions.md §11.3）
# ============================================================

class AdaptiveKLController:
    """自适应 KL 控制器：让 KL 稳定在 target_kl 附近。

        实际 KL > 目标 KL -> 增大 β（多惩罚 -> 拉回向 ref -> KL 下降）
        实际 KL < 目标 KL -> 减小 β（少惩罚 -> 允许发散 -> KL 上升）

    β 夹在 [beta_min, beta_max]；记录 (kl, beta) 轨迹供分析（教学即刻度）。
    """

    def __init__(
        self,
        target_kl: float,
        init_beta: float = 1e-3,
        beta_min: float = 1e-4,
        beta_max: float = 3.0,
        kp: float = 0.1,
        ema_decay: float = 0.9,
    ):
        self.target = target_kl
        self.beta = init_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        # 比例增益：β 按"近期平均 KL"相对目标的偏差做乘法式连续调节（§11.3）。
        self.kp = kp
        # 平滑：瞬时 batch KL 噪声大，用 EMA 追踪"近期平均 KL"再驱动 β，
        # 否则控制器会跟噪声震荡。
        self.ema = target_kl
        self.ema_decay = ema_decay
        self.trace: list[tuple[float, float]] = []

    def update(self, mean_kl: float) -> float:
        """输入本批 mean_kl，平滑后续调 β，返回最新 β。"""
        self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * float(mean_kl)
        err = (self.ema - self.target) / self.target
        err = max(-1.0, min(1.0, err))
        self.beta = self.beta * (1.0 + self.kp * err)
        self.beta = min(self.beta_max, max(self.beta_min, self.beta))
        self.trace.append((float(self.ema), float(self.beta)))
        return self.beta


# ============================================================
# 6. 生产系统件 C：Checkpoint save/load（断点续跑）
# ============================================================

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "m08_ckpt.pt")


def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    path: str = CKPT_PATH,
) -> None:
    """policy 权重 + optimizer 状态 + 元数据打成 dict，torch.save。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str = CKPT_PATH,
) -> dict:
    """加载 checkpoint 并恢复 policy 权重与 optimizer 状态，返回元数据 dict。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ============================================================
# 7. 生产主循环：rollout → RM → buffer → PPO(minibatch) → 评估 → checkpoint
# ============================================================

ROLLOUT_SIZE = 64    # 每轮采集的 rollout 样本数
MINIBATCH = 32       # 每次 PPO 更新从 buffer 采样批大小
BUFFER_CAP = 256     # 回放缓冲容量
PPO_UPDATES = 900    # 生产"迭代"数
PPO_EPOCHS = 2       # 每个 minibatch 的 PPO epoch
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
TARGET_KL = 1.0
BETA_INIT = 1e-3
GRAD_CLIP_NORM = 1.0   # 梯度裁剪：RL 数值稳定（生产 PPO）

value_model = ValueModel(num_prompts).to(device)
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
value_optimizer = torch.optim.Adam(value_model.parameters(), lr=3e-4)

buffer = ReplayBuffer(BUFFER_CAP)
kl_controller = AdaptiveKLController(target_kl=TARGET_KL, init_beta=BETA_INIT)


def rollout_step(replay: ReplayBuffer, n: int) -> None:
    """production" rollout worker "：当前策略采样 n 条回答 -> RM 打分 -> 压入 buffer。"""
    pids = torch.randint(0, num_prompts, (n,), device=device)
    with torch.no_grad():
        logits = policy(pids)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        ref_dist = Categorical(logits=reference_policy(pids))
        kl = log_probs - ref_dist.log_prob(actions)
        rewards = reward_model(pids, actions)
        values = value_model(pids)

    for p, a, lp, k, r, v in zip(
        pids.tolist(), actions.tolist(),
        log_probs.tolist(), kl.tolist(),
        rewards.tolist(), values.tolist(),
    ):
        replay.push(Rollout(
            prompt_id=torch.tensor(p, device=device),
            action=torch.tensor(a, device=device),
            log_prob=torch.tensor(lp, device=device),
            kl=torch.tensor(k, device=device),
            reward=torch.tensor(r, device=device),
            value=torch.tensor(v, device=device),
        ))


def evaluate_policy(model: nn.Module) -> float:
    """production 周期评估：受奖候选 0 平均概率（真实质量代理）。"""
    with torch.no_grad():
        return F.softmax(model(prompt_ids), dim=-1)[:, 0].mean().item()


def main() -> None:
    beta = BETA_INIT

    for update in range(PPO_UPDATES):
        # ---- 7.1 Rollout worker：经验压进 buffer（与训练解耦）----
        rollout_step(buffer, ROLLOUT_SIZE)
        batch = buffer.sample_batch(MINIBATCH)

        # ---- 7.2 取出经验，用当前 β 算 kl-penalized reward ----
        rewards = batch.reward - batch.kl * beta
        advantages = rewards - batch.value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards

        prompts_b = batch.prompt_id
        actions_b = batch.action
        old_log_probs = batch.log_prob

        # ---- 7.3 PPO minibatch 更新 ----
        for _ in range(PPO_EPOCHS):
            logits = policy(prompts_b)
            curr_dist = Categorical(logits=logits)
            curr_lps = curr_dist.log_prob(actions_b)
            ratio = (curr_lps - old_log_probs).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            entropy = curr_dist.entropy().mean()
            policy_loss = -torch.min(unclipped, clipped).mean() - ENTROPY_COEF * entropy
            policy_optimizer.zero_grad()
            policy_loss.backward()
            # 梯度裁剪：RL 更新不稳定 -> 在一次更新前把策略梯度的 total-norm
            # 截到 GRAD_CLIP_NORM，防止个别梯度尖峰扭曲整步更新。
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=GRAD_CLIP_NORM)
            policy_optimizer.step()

            vpred = value_model(batch.prompt_id)
            value_loss = F.mse_loss(vpred, returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=GRAD_CLIP_NORM)
            value_optimizer.step()

        # ---- 7.4 自适应 KL：本批 KL 均值驱动 β ----
        batch_kl_mean = batch.kl.mean().item()
        beta = kl_controller.update(batch_kl_mean)

        # ---- 7.5 周期性评估 + 打印刻度 ----
        if update % 25 == 0:
            ev = evaluate_policy(policy)
            print(
                f"[iter {update:03d}] beta={beta:.3e} | "
                f"kl={batch_kl_mean:.4f} | chosen={ev:.4f}"
            )

    # ---- 8. 落盘 + round-trip 验证（生产环境断点续跑）----
    save_checkpoint(policy, policy_optimizer, PPO_UPDATES, {"target_kl": TARGET_KL})

    # --- 8.1 用"全新"模型实例 load，验证权重与优化器都可恢复 ---
    loaded_policy = PolicyModel(num_prompts, num_actions).to(device)
    loaded_optimizer = torch.optim.Adam(loaded_policy.parameters(), lr=3e-3)
    ckpt_meta = load_checkpoint(loaded_policy, loaded_optimizer)

    # --- 8.2 round-trip 断言（1）：同一输入 logits 全等 ---
    with torch.no_grad():
        logits_orig = policy(prompt_ids)
        logits_load = loaded_policy(prompt_ids)
    max_diff = (logits_orig - logits_load).abs().max().item()
    assert max_diff < 1e-6, f"round-trip logits 不一致 max|x|={max_diff}"

    # --- 8.3 round-trip 断言（2）：恢复的 optimizer 能再走一步（可续训）---
    batch_r = buffer.sample_batch(MINIBATCH)
    lg = loaded_policy(batch_r.prompt_id)
    cont_loss = -Categorical(logits=lg).log_prob(batch_r.action).mean()
    loaded_optimizer.zero_grad()
    cont_loss.backward()
    loaded_optimizer.step()  # 恢复 state 后能正常 step => 续训可恢复。

    # ---- 9. Buffer 形状断言 ----
    sb = buffer.sample_batch(17)
    assert sb.prompt_id.shape == (17,), "prompt_id 形状错误"
    assert sb.action.shape == (17,), "action 形状错误"
    assert sb.kl.shape == (17,), "kl 形状错误"
    assert sb.reward.shape == (17,), "reward 形状错误"
    assert torch.isfinite(sb.kl).all(), "buffer KL 含 NaN/inf"

    # ---- 10. 诊断 & 断言（honest）----
    # recent_kls 取控制器 EMA 轨迹最后 10 步的"近期平均 KL"（§11.3 说追踪近期均值）。
    recent_kls = [k for k, _ in kl_controller.trace[-10:]]
    final_beta = kl_controller.beta
    beta_adapted = final_beta != BETA_INIT

    print("\n===== 自适应 KL 控制器轨迹（尾端 8 步）=====")
    for k, b in kl_controller.trace[-8:]:
        print(f"  kl(EMA)={k:.4f} -> beta={b:.3e}")
    print(f"最终 beta = {final_beta:.3e}（init={BETA_INIT:.3e}，是否自适应变化: {beta_adapted}）")

    final_ev = evaluate_policy(policy)
    mean_kl = sum(recent_kls) / len(recent_kls)

    print(f"\n候选0平均概率：SFT={sft_chosen:.4f} -> RL={final_ev:.4f}")
    print(f"近期平均KL={mean_kl:.4f}，目标={TARGET_KL}，β={final_beta:.3e}")
    print(f"checkpoint round-trip logits 最大差={max_diff:.3e} (<1e-6)，续训可继续 step")

    # (honest) 灾害断言：自适应 β 让 KL 不爆炸（近期 max 有界）
    assert max(recent_kls) < 3.0, f"KL 失控 max={max(recent_kls)}"
    # (honest) 且 RUN 都有真实推进：KL 稳定在目标邻域（平均值落在目标 ×[0.5,2]）
    assert TARGET_KL * 0.5 <= mean_kl <= TARGET_KL * 2.0, (
        f"近期平均KL={mean_kl:.4f} 应落在目标 {TARGET_KL} 的 [0.5,2.0] 邻域"
    )
    # 有效训练：受奖候选概率提升（SFT 的噪声偏好被 RL 修正）。
    assert final_ev > sft_chosen + 0.05, (
        f"RL 应提升候选0概率: SFT={sft_chosen:.3f} -> {final_ev:.3f}"
    )
    # β 确实被控制器自适应调节过。
    assert beta_adapted, "自适应 KL 控制器应改变 β"
    # checkpoint 元数据可读。
    assert ckpt_meta["step"] == PPO_UPDATES, "checkpoint 元数据 step 不正确"

    print(
        f"[PASS] m08 production rlhf (v0.7): rollout⇄buffer⇄trainer 解耦闭环，"
        f"自适应 KL（β:{BETA_INIT:.0e}→{final_beta:.1e}）把近期KL保持在目标 "
        f"{TARGET_KL} 邻域（均值 {mean_kl:.2f}），checkpoint 断点续跑 round-trip 一致"
    )


if __name__ == "__main__":
    main()