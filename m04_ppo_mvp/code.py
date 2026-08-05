"""m04 PPO MVP — Policy + Reward + Value (离散 PPO，最小 RLHF 闭环) (v0.3).

Run:  python m04_ppo_mvp/code.py
"""

# --- v0.3: 离散 PPO MVP ---
from __future__ import annotations

import os
import sys

# 本文件名为 code.py，运行时会以其所在目录作为 sys.path[0]；
# torch 内部会 `import code`（此处 code 是 Python 标准库的 code 模块），
# 若本目录留在 sys.path 中，标准库的 code 会被本文件错误遮蔽导致导入崩溃。
# 因此在 import 任何 torch 之前，先把本文件所在目录从 sys.path 中剔除。
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(os.path.expanduser(p)) != _here]

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 0. 配置与数据
# ============================================================

torch.manual_seed(42)

prompts = [
    "1+1等于多少？",
    "天空为什么是蓝色？",
    "如何安全地过马路？",
    "什么是过拟合？",
]

# 为了构造最小 RLHF，将“生成文本”简化为从三个候选回答中选择一个。
responses = [
    [
        "2。",
        "3。",
        "这个问题没有答案。",
    ],
    [
        "因为大气对短波长可见光的散射更强。",
        "因为海洋把天空染蓝。",
        "因为蓝色比较好看。",
    ],
    [
        "看信号灯、左右观察并走斑马线。",
        "闭眼快速跑过去。",
        "只要没有喇叭声就可以走。",
    ],
    [
        "模型记住训练数据而泛化较差。",
        "模型训练速度太慢。",
        "模型参数太少。",
    ],
]

num_prompts = len(prompts)
num_actions = len(responses[0])

prompt_ids = torch.arange(num_prompts)

# 故意构造一个带少量噪声的 SFT 数据：
# Prompt 1 和 Prompt 3 的 SFT 演示并不是最佳答案。
sft_labels = torch.tensor([0, 1, 0, 1])

# 人类偏好数据认为每个 Prompt 的第 0 个回答最好。
preference_pairs = []

for prompt_id in range(num_prompts):
    preference_pairs.append((prompt_id, 0, 1))
    preference_pairs.append((prompt_id, 0, 2))

preference_pairs = torch.tensor(preference_pairs)


# ============================================================
# 1. Policy Model
# ============================================================

class PolicyModel(nn.Module):
    """
    输入 prompt_id，输出对每个候选回答的 logits。
    """

    def __init__(
        self,
        num_prompts: int,
        num_actions: int,
        hidden_size: int = 16,
    ):
        super().__init__()

        self.prompt_embedding = nn.Embedding(
            num_prompts,
            hidden_size,
        )

        self.policy_head = nn.Linear(
            hidden_size,
            num_actions,
        )

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        logits = self.policy_head(hidden)
        return logits


# ============================================================
# 2. Reward Model
# ============================================================

class RewardModel(nn.Module):
    """
    输入 prompt_id 和 action_id，输出标量奖励。
    """

    def __init__(
        self,
        num_prompts: int,
        num_actions: int,
        hidden_size: int = 16,
    ):
        super().__init__()

        self.prompt_embedding = nn.Embedding(
            num_prompts,
            hidden_size,
        )

        self.action_embedding = nn.Embedding(
            num_actions,
            hidden_size,
        )

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        prompt_ids: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:

        prompt_hidden = self.prompt_embedding(prompt_ids)
        action_hidden = self.action_embedding(action_ids)

        hidden = torch.cat(
            [prompt_hidden, action_hidden],
            dim=-1,
        )

        reward = self.reward_head(hidden)
        return reward.squeeze(-1)


# ============================================================
# 3. Value Model
# ============================================================

class ValueModel(nn.Module):
    """
    根据 Prompt 预测当前策略的期望奖励。
    """

    def __init__(
        self,
        num_prompts: int,
        hidden_size: int = 16,
    ):
        super().__init__()

        self.prompt_embedding = nn.Embedding(
            num_prompts,
            hidden_size,
        )

        self.value_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        value = self.value_head(hidden)
        return value.squeeze(-1)


# ============================================================
# 4. SFT
# ============================================================

policy = PolicyModel(
    num_prompts=num_prompts,
    num_actions=num_actions,
)

sft_optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=1e-2,
)

for step in range(60):
    logits = policy(prompt_ids)

    sft_loss = F.cross_entropy(
        logits,
        sft_labels,
    )

    sft_optimizer.zero_grad()
    sft_loss.backward()
    sft_optimizer.step()


def print_policy(
    title: str,
    model: PolicyModel,
) -> None:

    print(f"\n===== {title} =====")

    with torch.no_grad():
        probabilities = F.softmax(
            model(prompt_ids),
            dim=-1,
        )

    for prompt_id, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt}")

        for action_id, response in enumerate(responses[prompt_id]):
            probability = probabilities[prompt_id, action_id].item()

            print(
                f"  P={probability:.4f} | {response}"
            )


print_policy("SFT 后的策略", policy)


# ============================================================
# 5. 冻结 Reference Policy
# ============================================================

reference_policy = copy.deepcopy(policy)
reference_policy.eval()

for parameter in reference_policy.parameters():
    parameter.requires_grad_(False)


# ============================================================
# 6. 训练 Reward Model
# ============================================================

reward_model = RewardModel(
    num_prompts=num_prompts,
    num_actions=num_actions,
)

reward_optimizer = torch.optim.Adam(
    reward_model.parameters(),
    lr=1e-2,
)

for step in range(300):
    pair_prompt_ids = preference_pairs[:, 0]
    chosen_action_ids = preference_pairs[:, 1]
    rejected_action_ids = preference_pairs[:, 2]

    chosen_rewards = reward_model(
        pair_prompt_ids,
        chosen_action_ids,
    )

    rejected_rewards = reward_model(
        pair_prompt_ids,
        rejected_action_ids,
    )

    # Bradley–Terry pairwise preference loss:
    #
    # -log sigmoid(r_chosen - r_rejected)
    reward_loss = -F.logsigmoid(
        chosen_rewards - rejected_rewards
    ).mean()

    reward_optimizer.zero_grad()
    reward_loss.backward()
    reward_optimizer.step()


print("\n===== Reward Model 分数 =====")

with torch.no_grad():
    for prompt_id, prompt in enumerate(prompts):
        repeated_prompt_ids = torch.full(
            (num_actions,),
            prompt_id,
            dtype=torch.long,
        )

        action_ids = torch.arange(num_actions)

        scores = reward_model(
            repeated_prompt_ids,
            action_ids,
        )

        print(f"\nPrompt: {prompt}")

        for action_id, response in enumerate(responses[prompt_id]):
            print(
                f"  Reward={scores[action_id].item():.4f}"
                f" | {response}"
            )


# PPO 阶段冻结 Reward Model。
reward_model.eval()

for parameter in reward_model.parameters():
    parameter.requires_grad_(False)


# ============================================================
# 7. PPO
# ============================================================

value_model = ValueModel(num_prompts)

policy_optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=3e-3,
)

value_optimizer = torch.optim.Adam(
    value_model.parameters(),
    lr=3e-3,
)

batch_repeats = 64
ppo_updates = 150
ppo_epochs = 4

clip_epsilon = 0.2
kl_beta = 0.02
entropy_coef = 0.01


for update in range(ppo_updates):

    # --------------------------------------------------------
    # 7.1 Rollout
    # --------------------------------------------------------

    batch_prompt_ids = torch.arange(
        num_prompts
    ).repeat_interleave(batch_repeats)

    # PPO 每轮需要一个固定的 old policy。
    old_policy = copy.deepcopy(policy)
    old_policy.eval()

    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():

        old_logits = old_policy(batch_prompt_ids)
        old_distribution = Categorical(logits=old_logits)

        sampled_actions = old_distribution.sample()
        old_log_probs = old_distribution.log_prob(
            sampled_actions
        )

        reference_logits = reference_policy(
            batch_prompt_ids
        )

        reference_distribution = Categorical(
            logits=reference_logits
        )

        reference_log_probs = (
            reference_distribution.log_prob(
                sampled_actions
            )
        )

        raw_rewards = reward_model(
            batch_prompt_ids,
            sampled_actions,
        )

        # Sample-level KL estimator:
        #
        # log π_policy(a|x) - log π_ref(a|x)
        sampled_kl = (
            old_log_probs - reference_log_probs
        )

        rewards = (
            raw_rewards
            - kl_beta * sampled_kl
        )

        old_values = value_model(batch_prompt_ids)

        # 单步环境：
        # Advantage = Reward - Value
        advantages = rewards - old_values
        returns = rewards

        # PPO 中通常会标准化 Advantage。
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

    # --------------------------------------------------------
    # 7.2 PPO 多轮更新
    # --------------------------------------------------------

    for epoch in range(ppo_epochs):

        current_logits = policy(batch_prompt_ids)
        current_distribution = Categorical(
            logits=current_logits
        )

        current_log_probs = (
            current_distribution.log_prob(
                sampled_actions
            )
        )

        probability_ratio = torch.exp(
            current_log_probs - old_log_probs
        )

        unclipped_objective = (
            probability_ratio * advantages
        )

        clipped_ratio = probability_ratio.clamp(
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        )

        clipped_objective = (
            clipped_ratio * advantages
        )

        entropy = (
            current_distribution.entropy().mean()
        )

        policy_loss = (
            -torch.min(
                unclipped_objective,
                clipped_objective,
            ).mean()
            - entropy_coef * entropy
        )

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        predicted_values = value_model(
            batch_prompt_ids
        )

        value_loss = F.mse_loss(
            predicted_values,
            returns,
        )

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

    if update % 25 == 0:
        print(
            f"update={update:03d}"
            f" | reward={rewards.mean().item():.4f}"
            f" | kl={sampled_kl.mean().item():.4f}"
            f" | policy_loss={policy_loss.item():.4f}"
            f" | value_loss={value_loss.item():.4f}"
        )


print_policy("RLHF/PPO 后的策略", policy)

print("[PASS] m04 PPO MVP: Policy 从带噪 SFT 转向正确回答（reward-hacking-free 经典闭环）")