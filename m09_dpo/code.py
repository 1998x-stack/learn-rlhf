"""m09 DPO — 直接偏好优化 (v0.8).

与 PPO 类模块的最大不同：DPO 没有奖励模型、没有 rollout、没有 value/critic，
而是直接从离线的 chosen/rejected 偏好对优化 Policy（Rafailov et al., 2023）。

Run:  python m09_dpo/code.py
"""

# --- v0.8: 离线偏好优化 DPO ---
from __future__ import annotations

import os
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

torch.manual_seed(42)

# --- v0.8: 与 m01 相同的离散数据集基准 ---
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
num_prompts = len(prompts)
num_actions = len(responses[0])
prompt_ids = torch.arange(num_prompts)
# m01 设立的含噪声 SFT 标签：Prompt 1 与 Prompt 3 的示范并非最佳答案。
sft_labels = torch.tensor([0, 1, 0, 1])

# --- v0.8: 离线偏好对（chosen=0 才是正确答案）---
# 每条记录 = (prompt_idx, chosen_idx, rejected_idx)：candidate 0 是"好回答"，
# 1 和 2 是"差回答"。这正是 m01 里"答案 0 才是正确"的机制：
# SFT 把 prompt 1/3 推向了错误答案 1，DPO 用它学到隐式奖励把每个 prompt 拉向 chosen=0。
preference_pairs = [
    (p, 0, 1) for p in range(num_prompts)
] + [
    (p, 0, 2) for p in range(num_prompts)
]


class PolicyModel(nn.Module):
    """离散回答级策略：prompt_id → logits[num_actions]。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        logits = self.policy_head(hidden)
        return logits


def sft_pretrain(policy: nn.Module, steps: int = 30, lr: float = 1e-2) -> float:
    """用 SFT 训练一个初始 Policy，返回最终 loss（作为 DPO 的起点策略）。

    刻意只做部分收敛（而非训练到 99% 置信），给 DPO 留出重排的空间：
    因为 SFT 标签含噪声（prompt 1/3 被导向错误答案 1），
    若 SFT 把策略彻底压死在错误答案上，DPO 将难以通过偏好对把它拉回 chosen=0。
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        logits = policy(prompt_ids)
        loss = F.cross_entropy(logits, sft_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss is not None and loss.item() < 1.0, "SFT loss 未下降"
    return float(loss.item())


def per_action_log_prob(model: nn.Module) -> torch.Tensor:
    """返回每个 prompt 的每个候选回答 log 概率 [num_prompts, num_actions]。"""
    return F.log_softmax(model(prompt_ids), dim=-1)


@torch.no_grad()
def per_action_prob(model: nn.Module) -> torch.Tensor:
    """返回每个 prompt 的每个候选回答概率 [num_prompts, num_actions]。"""
    return F.softmax(model(prompt_ids), dim=-1)


def dpo_loss(log_pi: torch.Tensor, logref: torch.Tensor, beta: float) -> torch.Tensor:
    """数值稳定版 DPO 损失（Rafailov et al., 2023，v0.8）。

    L = -log σ[ β( log(πθ(y_w|x)/πref(y_w|x)) - log(πθ(y_l|x)/πref(y_l|x)) ) ]

    用 `-logsigmoid(...)` 计算，而非 `log(1 + exp(...))`，避免大值时上溢出。

    - log_pi: 当前策略 πθ 的 log 概率 [num_prompts, num_actions]
    - logref: 冻结参考策略 πref 的 log 概率 [num_prompts, num_actions]
    """
    total = torch.tensor(0.0, dtype=torch.float32)
    for prompt_idx, chosen, rejected in preference_pairs:
        logpi_w = log_pi[prompt_idx, chosen] - logref[prompt_idx, chosen]
        logpi_l = log_pi[prompt_idx, rejected] - logref[prompt_idx, rejected]
        total = total + -F.logsigmoid(beta * (logpi_w - logpi_l))
    return total / len(preference_pairs)


def print_pair_prob(title: str, prob: torch.Tensor) -> None:
    print(f"\n===== {title} =====")
    for prompt_id, prompt in enumerate(prompts):
        p0 = prob[prompt_id, 0].item()
        p1 = prob[prompt_id, 1].item()
        p2 = prob[prompt_id, 2].item()
        print(f"  Prompt[{prompt_id}] {prompt}:  chosen0={p0:.4f}  rejected1={p1:.4f}  rejected2={p2:.4f}")


def main() -> None:
    # v0.8 Step 1: SFT 训练初始 Policy，并冻结一份作为参考策略 π_ref。
    policy = PolicyModel(num_prompts=num_prompts, num_actions=num_actions)
    sft_loss = sft_pretrain(policy)
    ref_policy = PolicyModel(num_prompts=num_prompts, num_actions=num_actions)
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad_(False)          # π_ref 全程冻结
    print(f"[SFT] 起始策略 loss={sft_loss:.4f}（SFT 已把 prompt 1/3 偏到错误答案 1）")

    sft_prob = per_action_prob(policy)

    # ---- v0.8 Step 2: 直接用 DPO 优化 Policy（无 RM / 无 rollout / 无 value）。----
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    beta = 1.0
    loss = None
    logref = per_action_log_prob(ref_policy)
    for step in range(300):
        log_p = per_action_log_prob(policy)
        loss = dpo_loss(log_p, logref, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss is not None and loss.item() < 1.0, "DPO loss 未下降"

    dpo_prob = per_action_prob(policy)

    print_pair_prob("SFT 基线（DPO 之前）", sft_prob)
    print_pair_prob("DPO 之后", dpo_prob)

    # ---- v0.8 Step 3: 机制级断言。----
    # chosen=0 概率应上升（尤其 SFT 曾指向答案 1 的 prompt 1/3），
    # rejected=1、2 概率应下降。
    chosen_up = bool((dpo_prob[:, 0] > sft_prob[:, 0]).all().item())
    rejected_down = bool((dpo_prob[:, 1] < sft_prob[:, 1]).all().item()
                         and (dpo_prob[:, 2] < sft_prob[:, 2]).all().item())
    assert chosen_up, "chosen(candidate 0) 概率未相对 SFT 上升"
    assert rejected_down, "rejected(candidate 1/2) 概率未相对 SFT 下降"

    print(f"\n[断言] chosen(candidate 0) 每个 prompt 概率均上升：{chosen_up}")
    print(f"[断言] rejected(candidate 1 & 2) 每个 prompt 概率均下降：{rejected_down}")
    print("[PASS] m09 dpo: 无 RM/rollout/value，直接在线下偏好对上优化 Policy，"
          "chosen 概率上升、rejected 概率下降")


if __name__ == "__main__":
    main()