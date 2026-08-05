"""m02 Reward Model — Bradley–Terry 偏好建模 (v0.1).

Run:  python m02_reward_model/code.py
"""

# --- v0.1: Bradley–Terry Reward Model ---
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

# ── 配置与数据（与 versions.md §6 / §5 一致，自包含）──
# 人类偏好：每个 prompt 下 candidate 0（正确回答）胜出，
#           candidate 0 > candidate 1 且 candidate 0 > candidate 2。
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

# 每个 prompt 生成两个偏好对：(prompt_id, chosen=0, rejected=1) 与
# (prompt_id, chosen=0, rejected=2)。chosen=0 是标注的"好答案"。
# 每个 prompt 生成两个偏好对：(prompt_id, chosen=0, rejected=1) 与
# (prompt_id, chosen=0, rejected=2)。chosen=0 是标注的"好答案"。
_preference_tuples = []
for prompt_id in range(num_prompts):
    _preference_tuples.append((prompt_id, 0, 1))
    _preference_tuples.append((prompt_id, 0, 2))
preference_pairs: torch.Tensor = torch.tensor(_preference_tuples)


class RewardModel(nn.Module):
    """r_phi(x, y) → 标量奖励。用 prompt_embedding ⊕ action_embedding
    拼接后过 tanh MLP，输出每个 (prompt, action) 的奖励分数。"""

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


def pair_accuracy(reward_model: nn.Module, pairs: torch.Tensor) -> torch.Tensor:
    """偏好正确率：chosen(胜者) 分数 > rejected(败者) 分数的比例。"""
    pair_prompt_ids = pairs[:, 0]
    chosen_action_ids = pairs[:, 1]
    rejected_action_ids = pairs[:, 2]
    with torch.no_grad():
        chosen_rewards = reward_model(pair_prompt_ids, chosen_action_ids)
        rejected_rewards = reward_model(pair_prompt_ids, rejected_action_ids)
    return (chosen_rewards > rejected_rewards).float().mean()


def print_score_table(model: RewardModel) -> None:
    print("\n===== Reward Model 分数 =====")
    with torch.no_grad():
        for prompt_id, prompt in enumerate(prompts):
            repeated_prompt_ids = torch.full((num_actions,), prompt_id, dtype=torch.long)
            action_ids = torch.arange(num_actions)
            scores = model(repeated_prompt_ids, action_ids)
            print(f"\nPrompt: {prompt}")
            for action_id, response in enumerate(responses[prompt_id]):
                print(f"  Reward={scores[action_id].item():.4f} | {response}")


def main() -> None:
    reward_model = RewardModel(num_prompts=num_prompts, num_actions=num_actions)
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)

    reward_loss = None
    for step in range(300):
        pair_prompt_ids = preference_pairs[:, 0]
        chosen_action_ids = preference_pairs[:, 1]
        rejected_action_ids = preference_pairs[:, 2]

        chosen_rewards = reward_model(pair_prompt_ids, chosen_action_ids)
        rejected_rewards = reward_model(pair_prompt_ids, rejected_action_ids)

        # Bradley–Terry 偏好损失：-log sigmoid(r_chosen - r_rejected)
        reward_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()

    # 训练后评估正确率：chosen 分数 > rejected 分数的偏好对占比。
    accuracy = pair_accuracy(reward_model, preference_pairs).item()

    # 自验证：输出 shape 正确、BT loss 已下降、学会"候选 0 胜出"的偏好排序。
    assert reward_model(prompt_ids, torch.zeros(num_prompts, dtype=torch.long)).shape == (num_prompts,)
    assert (
        reward_loss is not None and reward_loss.item() < 0.1
    ), "BT loss 未充分下降"
    assert accuracy >= 0.95, f"RM 偏好正确率过低: {accuracy:.4f} < 0.95"

    print(f"\n最终 BT loss = {reward_loss.item():.4f}")
    print(f"偏好对正确率 accuracy = {accuracy:.4f} (chosen>rejected)")
    print_score_table(reward_model)
    print(f"\n[PASS] m02 reward_model: RM 学会偏好排序 candidate 0>1、0>2（accuracy={accuracy:.4f}）")


if __name__ == "__main__":
    main()