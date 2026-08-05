"""m03 REINFORCE + KL — 用奖励信号 + KL 约束推动离散策略 (v0.2).

Run:  python m03_reinforce/code.py
"""

# --- v0.2: REINFORCE + KL ---
from __future__ import annotations

import os
import sys
import time

# 本文件名为 code.py，运行时会以其所在目录作为 sys.path[0]；
# torch 内部会 `import code`（此处 code 是 Python 标准库的 code 模块），
# 若本目录留在 sys.path 中，标准库的 code 会被本文件错误遮蔽导致导入崩溃。
# 因此在 import 任何 torch 之前，先把本文件所在目录从 sys.path 中剔除。
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(os.path.expanduser(p)) != _here]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

torch.manual_seed(42)

# ── 配置与数据（与 versions.md §6 / §5 一致，自包含）──
# 人类偏好：每个 prompt 下 candidate 0（正确回答）胜出。
num_prompts, num_actions = 4, 3
prompts = [
    "1+1等于多少？",
    "天空为什么是蓝色？",
    "如何安全地过马路？",
    "什么是过拟合？",
]
responses = [
    ["2。", "3。", "这个问题没有答案。"],
    ["因为大气对短波长可见光的散射更强。", "因为海洋把天空染蓝。", "因为蓝色比较好看。"],
    ["看信号灯、左右观察并走斑马线。", "闭眼快速跑过去。", "只要没有喇叭声就可以走。"],
    ["模型记住训练数据而泛化较差。", "模型训练速度太慢。", "模型参数太少。"],
]
prompt_ids = torch.arange(num_prompts)
# 含少量噪声的 SFT 演示：Prompt 1 与 Prompt 3 的示范不是 RM 眼中的最佳答案。
sft_labels = torch.tensor([0, 1, 0, 1])


class PolicyModel(nn.Module):
    """离散策略 π_θ(a|x)：prompt_id → 候选答案 logits。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        return self.policy_head(hidden)


class RewardModel(nn.Module):
    """r_φ(x, y)：偏好信号，m02 Bradley–Terry 训练而来（此处重新实现）。"""

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
        h = torch.cat(
            [self.prompt_embedding(prompt_ids), self.action_embedding(action_ids)],
            dim=-1,
        )
        return self.reward_head(h).squeeze(-1)


def expected_objective(
    model: nn.Module, reward_model: nn.Module, reference: nn.Module, kl_beta: float
) -> float:
    """REINFORCE 目标的解析期望 E[r_φ - β·KL]，供前后对比。

    D_KL(π_θ ‖ π_ref) 在离散策略上可精确求和（不靠采样），
    从而能稳定度量"策略向 RM 偏好移动是否真的提升了目标"。"""

    with torch.no_grad():
        probs = F.softmax(model(prompt_ids), dim=-1)
        reward = reward_model(
            prompt_ids.repeat_interleave(num_actions),
            torch.arange(num_actions).repeat(num_prompts),
        ).reshape(num_prompts, num_actions)
        ref_probs = F.softmax(reference(prompt_ids), dim=-1)
        kl_terms = (probs * (probs.log() - ref_probs.log())).sum(dim=-1, keepdim=True)
        objective = (reward - kl_beta * kl_terms) * probs
    return objective.sum(dim=-1).mean().item()


def print_policy(title: str, model: nn.Module) -> None:
    print(f"\n===== {title} =====")
    with torch.no_grad():
        probabilities = F.softmax(model(prompt_ids), dim=-1)
    for prompt_id, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt}")
        for action_id, response in enumerate(responses[prompt_id]):
            p = probabilities[prompt_id, action_id].item()
            print(f"  P={p:.4f} | {response}")


def main() -> None:
    # ── 1. SFT 得到参考策略（复用 v0.0）──
    policy = PolicyModel(num_prompts, num_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(600):
        loss = F.cross_entropy(policy(prompt_ids), sft_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 冻结 SFT 模型作为参考策略 π_ref，防止奖励驱动的策略遗忘基本能力（reward hacking）。
    reference_policy = PolicyModel(num_prompts, num_actions)
    reference_policy.load_state_dict(policy.state_dict())
    reference_policy.eval()
    for param in reference_policy.parameters():
        param.requires_grad_(False)

    # ── 2. 训练 Reward Model r_φ（复用 v0.1：偏好对 chosen=0 > rejected）──
    reward_model = RewardModel(num_prompts, num_actions)
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)
    preference_pairs = torch.tensor(
        [(pid, 0, r) for pid in range(num_prompts) for r in (1, 2)]
    )
    for _ in range(600):
        chosen = reward_model(preference_pairs[:, 0], preference_pairs[:, 1])
        rejected = reward_model(preference_pairs[:, 0], preference_pairs[:, 2])
        reward_loss = -F.logsigmoid(chosen - rejected).mean()
        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()

    kl_beta = 0.02
    obj_before = expected_objective(policy, reward_model, reference_policy, kl_beta)
    print_policy("SFT 之后（REINFORCE 之前）", policy)
    print(f"\n期望目标 E[r_φ - β·KL] before = {obj_before:.4f}")

    # ── 3. REINFORCE + KL（v0.2 核心）──
    # L = - E[ R(a) · logπ_θ(a) ]，其中 R(a) = r_φ(a) - β·log(π_θ/π_ref)。
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
    GRAD_CLIP_NORM = 1.0   # 梯度裁剪：REINFORCE 单样本估计更噪，用稍大阈值仍可挡尖峰
    for update in range(400):
        logits = policy(prompt_ids)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        with torch.no_grad():
            # sample-level KL 估计：logπ_θ(a) - logπ_ref(a)，见 versions.md §6。
            reference_dist = Categorical(logits=reference_policy(prompt_ids))
            sampled_kl = log_probs - reference_dist.log_prob(actions)
            reward = reward_model(prompt_ids, actions) - kl_beta * sampled_kl

        reinforce_loss = -(reward * log_probs).mean()
        policy_optimizer.zero_grad()
        reinforce_loss.backward()
        # 梯度裁剪：RL 更新不稳定 -> 在一次更新前把梯度的 total-norm 截到
        # GRAD_CLIP_NORM，防止单次采样碰到的奖励尖峰把整步更新拽飞。
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=GRAD_CLIP_NORM)
        policy_optimizer.step()

    obj_after = expected_objective(policy, reward_model, reference_policy, kl_beta)
    print_policy("REINFORCE+KL 之后", policy)

    # ── 4. 自验证 ──
    with torch.no_grad():
        final_probs = F.softmax(policy(prompt_ids), dim=-1)

    # Assert 1（核心且稳健）：期望目标显著提升 —— REINFORCE 确把概率推向 RM 偏好动作。
    assert obj_after > obj_before, (
        f"期望目标未提升: before={obj_before:.4f} after={obj_after:.4f}"
    )
    # Assert 2（稳健）：SFT 出错（示范选了 candidate 1）的 prompt 1/3 上，
    # RM 偏好的 candidate 0 概率显著上升 —— 奖励信号独立于 SFT 标签推动了策略。
    assert final_probs[1, 0].item() > 0.5, f"prompt1 概率未推向候选0: {final_probs[1,0].item():.4f}"
    assert final_probs[3, 0].item() > 0.5, f"prompt3 概率未推向候选0: {final_probs[3,0].item():.4f}"

    chosen = "  ".join(f"p{i}:{final_probs[i,0]:.3f}" for i in range(num_prompts))
    print(
        f"\n期望目标 E[r_φ - β·KL] before = {obj_before:.4f}  after = {obj_after:.4f}  "
        f"↑{obj_after - obj_before:+.4f}"
    )
    print(f"REINFORCE 之后 candidate0 概率 → {chosen}  （RM 偏好动作，越高越好）")
    print(
        "[PASS] m03 reinforce+kl: 期望目标 E[r_φ-β·KL] 显著提升 ↑，且 SFT 出错的 prompt 上 "
        "概率被推向 RM 偏好的候选 0（诚实声明：单样本 REINFORCE 亦见噪声）"
    )


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"(CPU 秒级，耗时 {time.time() - start:.2f}s)")