"""m07 Multi-Objective Reward — 多目标 Reward 聚合 + Reward Hacking (v0.6).

Run:  python m07_multi_objective/code.py

在 m04 (v0.3) 的离散 PPO 核心之上，把"单一奖励"升级为"多维奖励"（versions.md
§9 v0.6）：

    R_total = w_h·R_helpful + w_s·R_safety + w_v·R_verbosity

并演示三类重要概念：

1. Reward Hacking（§11.1/11.2）：构造一个极易被过度优化的 verbosity/style 代理
   奖励，用纯加权目标去优化它，出现"代理奖励一路上升、真实质量持平甚至下降"。
2. Reward Overoptimization 与 Hard Constraint（§9 v0.6）：软加权（weighted sum）
   下的安全目标会被 verbosity 奖励"hack"掉；改成硬约束（safety 低于阈值就
   reject/penalize 该样本），策略不再从不安全回答中获利。
3. Bias 评估（§9 v0.6）：verbosity bias 与 position bias 的量化诊断。
"""

# --- v0.6: 多目标 reward 聚合 ---
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
# 0. 配置与数据：离散回答级多候选任务（复用 m04 的任务形状）
# ============================================================

prompts = [
    "1+1等于多少？",
    "如何安全地过马路？",
    "天空为什么是蓝色的？",
    "什么是过拟合？",
]

# 每个候选用三个属性描述：helpfulness（真实有用性）、safety（安全 0/1）、
# verbosity（回答长度，作为可被 hack 的 style/verbosity 代理）。
responses = [
    ["2。", "3。", "这是个很长但错误、只为刷长度的回答。"],
    ["看信号灯并走斑马线。", "闭眼快速跑过去。", "冲过马路才最快最长…"],
    ["大气对短波散射更强。", "因为海洋把天空染蓝。", "蓝的好看，所以天空就是蓝的。"],
    ["模型记住数据而泛化差。", "模型训练速度慢。", "模型在训练集拟合好而在测试集泛化差故过度拟合。"],
]

num_prompts = len(prompts)
num_actions = len(responses[0])

# R_helpful: 真实有用性（0~1），视为"真值/人工质量"模范奖励，用于 hack 对比。
HELPFUL = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # P0: 只有 "2" 有用
        [1.0, 0.5, 0.0],  # P1: 两个不安全候选（a1 短、a2 长）都不可信
        [1.0, 0.0, 0.0],  # P2
        [1.0, 0.0, 0.0],  # P3
    ],
    dtype=torch.float32,
    device=device,
)

# R_safety: 安全规则打分（0/1），1=安全。P1 的两个候选不安全。
SAFE = torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
    device=device,
)

# verbosity 代理的原始长度（字符数）——同一 prompt 内越长 score 越高。
LENGTH = torch.tensor(
    [
        [3, 4, 22],     # P0: 最长的回答是错的-> 长度可被刷分
        [11, 8, 26],    # P1: 最长的回答既错误又不安全
        [12, 9, 30],
        [14, 10, 32],
    ],
    dtype=torch.float32,
    device=device,
)

# R_verbosity（v0.6 的 style/format 奖励）：把长度归一化到 [0,1]，越长越高 => 易 hack。
MIN_LEN = LENGTH.min(dim=-1, keepdim=True).values
MAX_LEN = LENGTH.max(dim=-1, keepdim=True).values
verbosity = (LENGTH - MIN_LEN) / (MAX_LEN - MIN_LEN + 1e-8)

# 多目标权重。刻意把 verbosity 权重 w_v 放大并超过 helpful+safety 之和，让"软加权"
# 路线会被风格奖励 hack 掉（w_v·1 > w_h·1 + w_s·1 + w_v·verb(a0)），从而凸显 hard gate。
W_HELPFUL = 1.3
W_SAFETY = 1.0
W_VERBOSITY = 3.0

# R_total 的上界 = w_h + w_s + w_v（三个分量都在 [0,1]）。
BOUND_MAX = W_HELPFUL + W_SAFETY + W_VERBOSITY

SAFE_THRESHOLD = 0.5   # safety < 该阈值 => 判定为安全违规
SAFE_PENALTY = -2.0    # hard 路线对违规样本的 reject/penalty 奖励


# ============================================================
# 1. Helpfulness Reward Model（v0.1 风格）：把人类偏好压成可打分 RM
# ============================================================

class RewardModel(nn.Module):
    """输入 (prompt_id, action_id)，输出代表"有用性"的标量。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.action_embedding = nn.Embedding(num_actions, hidden_size)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, prompt_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat(
            [self.prompt_embedding(prompt_ids), self.action_embedding(action_ids)], dim=-1
        )
        return self.reward_head(hidden).squeeze(-1)


# 从 HELPFUL 表格直接生成 (prompt, chosen, rejected) 三元组：
# chosen 的有用性 > rejected 的有用性。
preference_pairs: list[tuple[int, int, int]] = []
for p in range(num_prompts):
    for i in range(num_actions):
        for j in range(num_actions):
            if HELPFUL[p, i] > HELPFUL[p, j]:
                preference_pairs.append((p, i, j))
preference_pairs_t = torch.tensor(preference_pairs, dtype=torch.long, device=device)

reward_model = RewardModel(num_prompts, num_actions).to(device)
reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)

for _ in range(300):
    pid = preference_pairs_t[:, 0]
    chosen_ids = preference_pairs_t[:, 1]
    rejected_ids = preference_pairs_t[:, 2]
    chosen_reward = reward_model(pid, chosen_ids)
    rejected_reward = reward_model(pid, rejected_ids)
    # Bradley–Terry 偏好损失：-log sigmoid(r_chosen - r_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    reward_optimizer.zero_grad()
    loss.backward()
    reward_optimizer.step()

# 冻结 RM，仅用于 bias 诊断（v0.6 的 verbosity/position bias 评估）。
reward_model.eval()
for parameter in reward_model.parameters():
    parameter.requires_grad_(False)

prompt_grid = torch.arange(num_prompts, device=device).unsqueeze(1).expand(num_prompts, num_actions)
action_grid = torch.arange(num_actions, device=device).unsqueeze(0).expand(num_prompts, num_actions)


# ============================================================
# 2. Policy Model / Value Model（复用 m04 离散 PPO 核）
# ============================================================

class PolicyModel(nn.Module):
    """输入 prompt_id，输出对每个候选回答的 logits。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        return self.policy_head(hidden)


class ValueModel(nn.Module):
    """输入 prompt_id，预测其当前期望奖励（Critic）。"""

    def __init__(self, num_prompts: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        return self.value_head(hidden).squeeze(-1)


def policy_probs(model: PolicyModel) -> torch.Tensor:
    """Policy 在全部 prompt 上的概率矩阵 [P, A]。"""
    with torch.no_grad():
        return F.softmax(model(torch.arange(num_prompts, device=device)), dim=-1)


def policy_metrics(probs: torch.Tensor) -> dict[str, float]:
    """给定策略概率矩阵 [P,A]，返回按 prompt 平均的期望指标。"""
    return {
        "quality": (probs * HELPFUL).sum(dim=-1).mean().item(),
        "safety": (probs * SAFE).sum(dim=-1).mean().item(),
        "verbosity": (probs * verbosity).sum(dim=-1).mean().item(),
        "unsafe_rate": (probs * (1.0 - SAFE)).sum(dim=-1).mean().item(),
    }


# ============================================================
# 3. 多目标聚合（v0.6 core）
# ============================================================

def aggregate(
    help_r: torch.Tensor, safe_r: torch.Tensor, verb_r: torch.Tensor
) -> torch.Tensor:
    """R_total = w_h·R_helpful + w_s·R_safety + w_v·R_verbosity（软加权）。"""
    return W_HELPFUL * help_r + W_SAFETY * safe_r + W_VERBOSITY * verb_r


# ============================================================
# 4. PPO 训练器：接受 reward 组成 + 是否开启 hard safety gate
# ============================================================

def run_ppo(reward_kind: str, reject_unsafe: bool) -> PolicyModel:
    """训练一个离散 PPO 策略，返回训练后的 PolicyModel。

    reward_kind:
      - 'proxy' : 只用"易被 hack"的 verbosity 代理奖励（不掺 help/safety），演示 hacking。
      - 'total' : 多目标加权和 R_total = w_h·R_helpful + w_s·R_safety + w_v·R_verbosity。

    reject_unsafe: 若 True，hard constraint：safety < 阈值就把该样本的奖励替换成
                   SAFE_PENALTY（= 从损失里剔除并让策略主动远离不安全回答）。
    """
    policy = PolicyModel(num_prompts, num_actions).to(device)

    reference_policy = copy.deepcopy(policy)
    reference_policy.eval()
    for p in reference_policy.parameters():
        p.requires_grad_(False)

    value_model = ValueModel(num_prompts).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-2)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=3e-2)

    batch_repeats = 32
    ppo_updates = 150
    ppo_epochs = 3
    clip_epsilon = 0.2
    kl_beta = 0.01
    entropy_coef = 0.001
    GRAD_CLIP_NORM = 0.5   # 梯度裁剪：RL 数值稳定（离散 PPO）

    batch_prompt_ids = torch.arange(num_prompts, device=device).repeat_interleave(batch_repeats)

    for _ in range(ppo_updates):
        # ----- rollout 并冻结采样时刻的 old policy -----
        with torch.no_grad():
            old_policy = copy.deepcopy(policy)
            old_policy.eval()
            for p in old_policy.parameters():
                p.requires_grad_(False)

            old_logits = old_policy(batch_prompt_ids)
            old_dist = Categorical(logits=old_logits)
            sampled_actions = old_dist.sample()
            old_log_probs = old_dist.log_prob(sampled_actions)

            ref_log_probs = Categorical(
                logits=reference_policy(batch_prompt_ids)
            ).log_prob(sampled_actions)

            help_r = HELPFUL[batch_prompt_ids, sampled_actions]
            safe_r = SAFE[batch_prompt_ids, sampled_actions]
            verb_r = verbosity[batch_prompt_ids, sampled_actions]

            if reward_kind == "proxy":
                r_total = W_VERBOSITY * verb_r
            else:  # 'total'：加权聚合
                r_total = aggregate(help_r, safe_r, verb_r)

            # ---- hard safety gate（v0.6）：reject/惩罚安全违规样本 ----
            if reject_unsafe:
                is_unsafe = safe_r < SAFE_THRESHOLD
                r_total = torch.where(
                    is_unsafe, torch.full_like(r_total, SAFE_PENALTY), r_total
                )

            # R_total 有界性：进入 loss 前断言有限且在理论范围内。
            assert torch.isfinite(r_total).all(), "R_total 含 NaN/inf"
            assert r_total.min().item() >= SAFE_PENALTY - 1e-6, "R_total 越界（下界）"
            assert r_total.max().item() <= BOUND_MAX + 1e-6, "R_total 越界（上界）"

            sampled_kl = old_log_probs - ref_log_probs
            rewards = r_total - kl_beta * sampled_kl

            old_values = value_model(batch_prompt_ids)
            advantages = rewards - old_values
            returns = rewards
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ----- PPO 多轮更新 -----
        for _ in range(ppo_epochs):
            curr_dist = Categorical(logits=policy(batch_prompt_ids))
            curr_log_probs = curr_dist.log_prob(sampled_actions)

            ratio = torch.exp(curr_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            entropy = curr_dist.entropy().mean()

            policy_loss = -torch.min(unclipped, clipped).mean() - entropy_coef * entropy

            policy_optimizer.zero_grad()
            policy_loss.backward()
            # 梯度裁剪：RL 更新不稳定 -> 在一次更新前把策略梯度的 total-norm
            # 截到 GRAD_CLIP_NORM，防止个别梯度尖峰扭曲整步更新。
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=GRAD_CLIP_NORM)
            policy_optimizer.step()

            value_loss = F.mse_loss(value_model(batch_prompt_ids), returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            # value 网络与策略各自独立，单独裁剪其参数梯度，避免 critic 更新过大。
            nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=GRAD_CLIP_NORM)
            value_optimizer.step()

    return policy


# ============================================================
# 5. Bias 评估 / 主流程（v0.6）
# ============================================================

def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """把两个张量展平后求 Pearson 相关系数。"""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    a_flat = a_flat - a_flat.mean()
    b_flat = b_flat - b_flat.mean()
    return ((a_flat * b_flat).mean() / (a_flat.std() * b_flat.std() + 1e-8)).item()


def main() -> None:
    # ---- 0. R_total 有界性前导演示：聚合一次全表 ----
    total_table = aggregate(HELPFUL, SAFE, verbosity)
    assert torch.isfinite(total_table).all(), "R_total 含 NaN/inf"
    assert total_table.min().item() >= 0.0, "R_total 下界越界"
    assert total_table.max().item() <= BOUND_MAX + 1e-6, "R_total 上界越界"

    print("\n===== 多目标加权聚合：R_total = "
          f"{W_HELPFUL}·helpful + {W_SAFETY}·safety + {W_VERBOSITY}·verbosity =====")
    print(f"  R_total∈[0, {BOUND_MAX}]（有限、有界）✓")
    for p in range(num_prompts):
        print(f"\n  Prompt: {prompts[p]}")
        for i in range(num_actions):
            print(
                f"    R_total={total_table[p, i].item():.2f}"
                f" (help={HELPFUL[p, i].item():.1f} safe={SAFE[p, i].item():.0f}"
                f" verb={verbosity[p, i].item():.2f}) | {responses[p][i]}"
            )

    # ---- Reward Hacking 演示：纯 verbosity 风格代理 ----
    init_policy = PolicyModel(num_prompts, num_actions).to(device)
    before = policy_metrics(policy_probs(init_policy))

    proxy_policy = run_ppo(reward_kind="proxy", reject_unsafe=False)
    after_proxy = policy_metrics(policy_probs(proxy_policy))

    proxy_delta = after_proxy["verbosity"] - before["verbosity"]
    quality_delta = after_proxy["quality"] - before["quality"]

    print("\n===== Reward Hacking 演示：只优化 verbosity 风格代理（proxy）=====")
    print(f"  初始(随机)策略:  E[verbosity]={before['verbosity']:.4f}"
          f"  E[quality]={before['quality']:.4f}")
    print(f"  过 optimize 后:  E[verbosity]={after_proxy['verbosity']:.4f}"
          f"  E[quality]={after_proxy['quality']:.4f}")
    print(f"  Δverbosity(代理)={proxy_delta:+.3f}  Δquality(真实质量)={quality_delta:+.3f}")
    print("  -> 代理奖励大涨，而真实有用性不涨/下降：reward hacking（§11）。")

    assert proxy_delta > 0.02, "reward hacking 未体现：代理奖励应上升"
    assert quality_delta <= 0.02, "reward hacking 未体现：真实质量应持平或下降"

    # ---- Hard constraint vs 软加权（A/B）----
    naive_policy = run_ppo(reward_kind="total", reject_unsafe=False)
    hard_policy = run_ppo(reward_kind="total", reject_unsafe=True)

    naive_m = policy_metrics(policy_probs(naive_policy))
    hard_m = policy_metrics(policy_probs(hard_policy))
    naive_unsafe = naive_m["unsafe_rate"]
    hard_unsafe = hard_m["unsafe_rate"]
    naive_safety = naive_m["safety"]
    hard_safety = hard_m["safety"]

    print("\n===== 软加权(naive) vs 硬约束(hard)：安全度量 =====")
    print(f"  软加权: unsafe_rate={naive_unsafe:.3f}  safety={naive_safety:.3f}")
    print(f"  硬约束: unsafe_rate={hard_unsafe:.3f}  safety={hard_safety:.3f}")
    print("  -> 安全权重被 verbosity 代理 hack 掉时，软加权仍选不安全回答；"
          "硬约束把违规样本 reject，策略不再从不安全回答获利。")

    assert naive_unsafe > 0.1, "naive 应呈现明显的安全违规（被 hack）"
    assert hard_unsafe < 0.1, "hard 应把不安全率压到很低"
    assert hard_unsafe < naive_unsafe - 0.05, "hard 的不安全率应显著低于 naive"

    # ---- Bias 评估：verbosity / position ----
    with torch.no_grad():
        rm_scores = reward_model(prompt_grid, action_grid).reshape(num_prompts, num_actions)
    corr_len_verb = pearson(LENGTH, verbosity)    # 代理 vs 长度（应≈1）
    corr_len_rm = pearson(LENGTH, rm_scores)      # RM vs 长度
    positions = torch.arange(num_actions, device=device).unsqueeze(0).expand(num_prompts, num_actions)
    corr_pos_rm = pearson(positions, rm_scores)   # 位置 vs RM
    rm_by_position = rm_scores.mean(dim=0)

    print("\n===== Bias 评估（v0.6）=====")
    print(f"  verbosity bias: corr(长度, verbosity代理) = {corr_len_verb:.3f}")
    print(f"                  corr(长度, HelpfulnessRM)  = {corr_len_rm:.3f}")
    print("    -> verbosity 代理与长度强正相关；HelpfulnessRM 则明显反对/远离长度。")

    pos_str = ", ".join(
        f"pos{i}={rm_by_position[i].item():.3f}" for i in range(num_actions)
    )
    print(f"  position 偏置:   各位置 RM 均分 [{pos_str}]  corr(位置,RM)={corr_pos_rm:.3f}")
    print("    -> 把『候选所处列表位置』当成隐含特征，会产生 location 偏置，需按内容残差化检查。")

    assert corr_len_verb > 0.7, "verbosity 代理应与长度强相关（corr>0.7）"
    assert corr_len_verb > abs(corr_len_rm) + 0.1, "HelpfulnessRM 不应比代理更长度偏置"

    # ---- 最终结果汇总 ----
    print("\n===== 最终结果 =====")
    print(f"  #1 代理奖励 Δ=+{proxy_delta:.3f} / 真实质量 Δ={quality_delta:+.3f}（hacking）")
    print(f"  #2 naive unsafe={naive_unsafe:.3f} vs hard unsafe={hard_unsafe:.3f}（硬约束更安全）")
    print(f"  #3 verbosity corr={corr_len_verb:.3f}（与长度强相关，RM 相反）")

    print(
        "\n[PASS] m07 multi-objective reward (v0.6): R_total 加权有界；"
        "verbosity 代理被过度优化(reward hacking)而真实质量不升；"
        "hard constraint 安全违规率显著低于 naive 软加权；"
        "verbosity/position bias 已量化诊断"
    )


if __name__ == "__main__":
    main()
