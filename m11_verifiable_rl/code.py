"""m11 可验证奖励 / GRPO — Reinforcement Learning from Verifiable Rewards (v1.0).

前十个模块（m01–m10）的奖励都来自"人类偏好"（RM / 偏好对 / AI Judge）。
本模块第一次改用【可验证奖励】：正确性由【精确验证器】计算得出，不依赖任何
人类标注，也不训练任何 Reward Model / Value Model。

关键差异（GRPO, DeepSeekMath 2024）：

    PPO  :  Advantage = r - V(s)       ← 需要一个 Value / Critic 网络
    GRPO :  A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
                                      ← 只用同一 prompt 的 N 个采样做组内相对优势，
                                        完全【不需要 Value 模型】

流程（versions.md §9 v1.0）：

    math prompt → policy 采样 N 个候选解 → 精确验证器判定对/错 (R=1/0)
        ↓
    组内相对优势 A_i = (R_i - mean)/(std+eps)
        ↓
    GRPO 更新：提高 A>0 的答案概率、降低 A<0 的（带 ratio/clip 稳定）
    + best-of-N / rejection sampling：N 越大，采样命中率越高

Run:  python m11_verifiable_rl/code.py
"""

# --- v1.0: 可验证奖励 / 推理 RL ---
from __future__ import annotations

import os
import re
import sys

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

# --- v1.0: 设备——GPU 可用则用 GPU，否则 CPU（同 m05–m10）---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 可验证算术数据（正确性"可计算" → 可由验证器判定，无需人工标注）
# ============================================================
# 每条 = (表达式, 由 eval 算出的正确整数答案)。
# "能否验证"是本模块的根基：ground-truth 由机器算出，
# 因此 reward 是 0/1 的精确真值，而非另一个模型估计的偏好分。
ARITH: list[tuple[str, int]] = [
    ("3+5", 8),
    ("12-7", 5),
    ("6*4", 24),
    ("7+8", 15),
    ("20+14", 34),
    ("9*9", 81),
    ("28-13", 15),
    ("16+17", 33),
]


def build_candidates(correct: int) -> list[str]:
    """每个 prompt 生成 3 个候选答案：[正确, +1 错误, -1 错误]。"""
    return [str(correct), str(correct + 1), str(correct - 1)]


_prompts: list[str] = []
_candidates: list[list[str]] = []
for expr, correct in ARITH:
    _prompts.append(f"{expr}=？")
    _candidates.append(build_candidates(correct))

num_prompts = len(ARITH)
num_candidates = 3
CORRECT_IDX = 0  # 候选 0 恒为正确答案（由 build_candidates 固定）

prompt_ids = torch.arange(num_prompts, device=device)

# 含噪声 SFT 起点：每个 prompt 给 1 条正确示范 + 2 条错误示范（候选 1）。
# → SFT 后每个 prompt 的 P(正确) ≈ 1/3（"半信半疑"），
#   组内既采得到对、也采得到错，才有内部方差可供 GRPO 计算相对优势。
sft_demos: list[int] = []
for _p in range(num_prompts):
    sft_demos.append(CORRECT_IDX)   # 正确示范 ×1
    sft_demos.append(1)             # 错误示范 ×2
    sft_demos.append(1)
sft_prompt_ids = torch.tensor(
    [i for i in range(num_prompts) for _ in range(3)], device=device,
)
sft_labels = torch.tensor(sft_demos, device=device)


# ============================================================
# 1. 精确验证器（outcome reward: R=1 若验证通过，否则 0）——无 RM、无人工标注
# ============================================================

def compute_answer(expr: str) -> int:
    """'执行器'：把表达式真正 eval 一遍得到 ground-truth。"""
    return int(eval(expr))


def verify_answer(expr: str, candidate: str, tol: float = 0.0) -> float:
    """精确验证器：解析候选数字，与 eval 算出的真值比对。

    返回 outcome reward：
            R = 1  验证通过（tol=0 即精确匹配）
            R = 0  验证失败
    没有任何可学习的奖励模型 —— 这正是"可验证奖励"的核心。
    """
    matched = re.search(r"-?\d+", candidate)
    if matched is None:
        return 0.0
    expected = compute_answer(expr)
    return 1.0 if abs(int(matched.group()) - expected) <= tol else 0.0


def verifier_rewards(prompt_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
    """对一批 (prompt, 候选下标) 返回 outcome reward [B]。"""
    rewards = torch.zeros(prompt_batch.size(0), device=device)
    for i in range(prompt_batch.size(0)):
        p = int(prompt_batch[i])
        a = int(action_batch[i])
        rewards[i] = verify_answer(ARITH[p][0], _candidates[p][a])
    return rewards


# ============================================================
# 2. Policy（离散候选级）—— 不再需要 Value / Critic
# ============================================================

class PolicyModel(nn.Module):
    """离散答级策略：prompt_id → logits[num_candidates]。

    GRPO 只更新这一个网络；没有 value head（vs m04–m06 的 Critic）。
    """

    def __init__(self, num_prompts: int, num_candidates: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_candidates)

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompts)
        return self.policy_head(hidden)


# ============================================================
# 3. SFT 起点（含噪声："半信半疑"，P(正确) ≈ 1/3）
# ============================================================

def sft_pretrain() -> tuple[PolicyModel, dict]:
    policy = PolicyModel(num_prompts=num_prompts, num_candidates=num_candidates).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    loss: torch.Tensor | None = None
    for _ in range(80):
        logits = policy(sft_prompt_ids)
        loss = F.cross_entropy(logits, sft_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss is not None and loss.item() < 1.0, "SFT loss 未下降"
    state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    return policy, state


def sample_acc(policy: nn.Module) -> float:
    """采样级可验证正确率：按当前策略采样，验证器判定为正确的比例。"""
    with torch.no_grad():
        dist = Categorical(logits=policy(prompt_ids))
        acts = dist.sample()
    return float(verifier_rewards(prompt_ids, acts).mean().item())


def correct_prob_of(policy: nn.Module) -> float:
    """每个 prompt 的 P(正确候选) 的均值 —— 机制检查：GRPO 是否提升正确答案概率。"""
    with torch.no_grad():
        probs = F.softmax(policy(prompt_ids), dim=-1)
    return float(probs[:, CORRECT_IDX].mean().item())


# ============================================================
# 4. GRPO：组内相对优势（无 Value）＋ 策略更新
# ============================================================

def group_advantage(rewards_group: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """组内相对优势 A_i = (R_i - mean(R)) / (std(R)+eps)，按 prompt 分组。

    与 PPO 的分水岭：不需要 Value 模型。
    若组内全部相同（全 0 或全 1），std=0 → advantage≈0 → 不更新（正确行为）。
    """
    mean = rewards_group.mean(dim=-1, keepdim=True)
    std = rewards_group.std(dim=-1, keepdim=True)
    return (rewards_group - mean) / (std + eps)


def train_grpo(
    policy: nn.Module,
    n_samples: int = 8,
    grpo_updates: int = 150,
    epochs: int = 3,
    clip_epsilon: float = 0.2,
    lr: float = 5e-3,
) -> tuple[float, float]:
    """在线 GRPO：每个 prompt 采样 N 个候选，验证器给 0/1，做组内相对优势更新。

    返回 (首轮平均 outcome reward, 末轮平均 outcome reward)。
    """
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    first_reward = 0.0
    last_reward = 0.0

    for update in range(grpo_updates):
        # 冻结采样时刻的策略作为 ratio 分母（old_logp）。
        old_policy = PolicyModel(num_prompts=num_prompts, num_candidates=num_candidates).to(device)
        old_policy.load_state_dict(policy.state_dict())
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad_(False)

        # ---- 4.1 rollout：从当前策略采样 N 个候选/每 prompt 得到 outcome reward ----
        batch_prompts = prompt_ids.repeat_interleave(n_samples)
        with torch.no_grad():
            old_dist = Categorical(logits=old_policy(batch_prompts))
            sampled = old_dist.sample()             # [B] 候选下标
            old_logp = old_dist.log_prob(sampled)   # [B]
            rewards = verifier_rewards(batch_prompts, sampled)   # [B] 0/1

        # ---- 4.2 组内相对优势（reshape 到 [num_prompts, n_samples]，按行分组）----
        rewards_matrix = rewards.reshape(num_prompts, n_samples)
        advantages_matrix = group_advantage(rewards_matrix)      # [P, N]
        advantages = advantages_matrix.reshape(-1).detach()      # [B]

        # ---- 4.3 GRPO 策略更新：A>0 的答案概率↑，A<0 的↓，带 clamp 稳定 ----
        for _ in range(epochs):
            cur_dist = Categorical(logits=policy(batch_prompts))
            cur_logp = cur_dist.log_prob(sampled)
            ratio = torch.exp(cur_logp - old_logp)
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            loss = -torch.min(unclipped, clipped).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        mean_rew = float(rewards.mean().item())
        if update == 0:
            first_reward = mean_rew
        last_reward = mean_rew

        if update % 50 == 0:
            print(
                f"grpo update={update:03d} | mean_reward={mean_rew:.3f}"
                f" | group_adv_std={advantages_matrix.std().item():.4f}"
            )

    return first_reward, last_reward


# ============================================================
# 5. best-of-N / rejection sampling 评估
# ============================================================

def best_of_n_accuracy(policy: nn.Module, n: int, trials: int = 300) -> float:
    """best-of-N：对每个 prompt 采样 n 个候选，只要有 1 个正确就计成功。

    best-of-N accuracy = 每个 prompt 经验上"至少采到一个正确解"的概率。
    N=1 时即 per-sample accuracy。
    """
    acc = 0.0
    with torch.no_grad():
        dist = Categorical(logits=policy(prompt_ids))
        for _ in range(trials):
            acts = dist.sample((n,))                 # [n, P]
            any_correct = 0.0
            for p in range(num_prompts):
                ok = any(
                    verify_answer(ARITH[p][0], _candidates[p][int(acts[j, p])]) == 1.0
                    for j in range(n)
                )
                any_correct += 1.0 if ok else 0.0
            acc += any_correct / num_prompts
    return acc / trials


# ============================================================
# 6. 主流程：SFT 基线 → GRPO → 提升 → best-of-N 演示
# ============================================================

def main() -> None:
    print(f"m11 verifiable RL · 可验证奖励 + GRPO (v1.0)   device={device}")

    # ---- v1.0 Step 1: 验证器自检 —— 顶层真值确实可精确计算 ----
    verifier_ok = sum(
        1 for p in range(num_prompts)
        if verify_answer(ARITH[p][0], _candidates[p][CORRECT_IDX]) == 1.0
    )
    print(f"[验证器自检] 顶层可验证正确候选 {verifier_ok}/{num_prompts}"
          f"（ground-truth 由 eval 精确算出，无 RM/人工标注）")

    # ---- v1.0 Step 2: 含噪声 SFT 基线 ----
    sft_policy, sft_state = sft_pretrain()
    base_acc = sample_acc(sft_policy)
    base_correct_prob = correct_prob_of(sft_policy)
    print(f"\n[SFT 基线] 采样级可验证正确率 = {base_acc:.3f} ；P(正确候选) = {base_correct_prob:.3f}")

    # ---- v1.0 Step 3: GRPO 用可验证 outcome reward（无 Value 模型）训练 ----
    grpo_policy = PolicyModel(num_prompts=num_prompts, num_candidates=num_candidates).to(device)
    grpo_policy.load_state_dict(sft_state)
    first_reward, last_reward = train_grpo(grpo_policy)

    trained_acc = sample_acc(grpo_policy)
    trained_correct_prob = correct_prob_of(grpo_policy)
    print(f"\n[GRPO 后] 采样级可验证正确率 = {trained_acc:.3f} ；P(正确候选) = {trained_correct_prob:.3f}")
    print(f"[GRPO] 平均 outcome reward（采样正确比例）：{first_reward:.3f} → {last_reward:.3f}")

    # ---- v1.0 Step 4: 机制级断言 ----
    import math
    assert math.isfinite(first_reward) and math.isfinite(last_reward), "GRPO reward 非有限"
    adv_check = group_advantage(torch.tensor([[1.0, 0.0, 0.0]], device=device))
    assert adv_check.numel() == 3 and bool(torch.isfinite(adv_check).all()), "group advantage 非有限"
    assert trained_acc > base_acc + 0.15, (
        f"可验证正确率应显著上升：基线={base_acc:.3f} -> GRPO后={trained_acc:.3f}"
    )
    assert trained_correct_prob > base_correct_prob + 0.1, (
        "机制：正确候选的概率应显著上升（GRPO 确实推高了 P(正确)）"
    )

    # ---- v1.0 Step 5: best-of-N / rejection sampling ----
    # best-of-N 本质上是"采样 N 个解、经验证器挑一个已验证正确的"的"在线拒绝采样 +
    # 验证器把关"，是 v1.0 比"单样本采样"更高的一种用法。
    # 演示对象用 SFT 基线（非确定性、有真实散布）最能体现 N 越大成功概率越高；
    # 对近乎满分的 GRPO policy 也打印对照（此时各 N 已饱和，提升幅度自然变小）。
    sft_bon1 = best_of_n_accuracy(sft_policy, n=1)
    sft_bon4 = best_of_n_accuracy(sft_policy, n=4)
    sft_bon16 = best_of_n_accuracy(sft_policy, n=16)
    print(f"\nbest-of-N（SFT 弱策略，可验证筛选）: N=1 -> {sft_bon1:.3f}  "
          f"N=4 -> {sft_bon4:.3f}  N=16 -> {sft_bon16:.3f}")
    assert sft_bon16 > sft_bon4 > sft_bon1, "best-of-N 成功率应随 N 递增"
    assert sft_bon4 > base_acc, "best-of-N 应优于 单样本(per-sample) 采样级正确率"

    grpo_bon1 = best_of_n_accuracy(grpo_policy, n=1)
    grpo_bon8 = best_of_n_accuracy(grpo_policy, n=8)
    print(f"best-of-N（GRPO 训练后，对照）: N=1 -> {grpo_bon1:.3f}  N=8 -> {grpo_bon8:.3f}")

    print(f"\n[断言] GRPO 后验证器判定正确的采样比例 高于 含噪声 SFT 基线"
          f"（{base_acc:.3f} -> {trained_acc:.3f}）")
    print("[断言] GRPO 机制有效：正确候选的概率被组内相对优势推高"
          f"（{base_correct_prob:.3f} -> {trained_correct_prob:.3f}），优势为有限值")
    print("[断言] best-of-N / rejection sampling 提升：N 越大 至少命中一次正确的成功率越高"
          f"（SFT弱策略 N=1:{sft_bon1:.3f} < N=4:{sft_bon4:.3f} < N=16:{sft_bon16:.3f}）")
    print("[PASS] m11 verifiable_rl: 以精确验证器为奖励源（无 RM、无 Value 模型），"
          "GRPO 用组内相对优势把采样正确率从 SFT 基线拉起，best-of-N 进一步放大")


if __name__ == "__main__":
    main()