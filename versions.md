



# 1. RLHF 是什么

**RLHF：Reinforcement Learning from Human Feedback，基于人类反馈的强化学习。**

它解决的问题不是“让模型继续预测下一个 token”，而是：

> 当多个回答都语言通顺时，怎样让模型更倾向于人类认为正确、有用、安全、清晰的回答？

经典 RLHF 将模糊的人类偏好转化为可优化的数值奖励：

```text
人类比较回答
    ↓
训练奖励模型 Reward Model
    ↓
奖励模型为新回答打分
    ↓
使用强化学习优化语言模型
```

RLHF 的本质可以概括为：

\[
\text{Human Preference}
\rightarrow
\text{Reward Function}
\rightarrow
\text{Policy Optimization}
\]

经典路线由人类偏好强化学习、语言模型偏好微调、摘要 RLHF 和 InstructGPT 等工作逐步形成（Christiano et al., 2017；Ziegler et al., 2019；Stiennon et al., 2020；Ouyang et al., 2022）。

---

# 2. RLHF 完整流程

```text
                  ┌───────────────────┐
                  │ 预训练语言模型 LM │
                  └─────────┬─────────┘
                            ↓
                  ┌───────────────────┐
                  │ 1. SFT 监督微调   │
                  │ Prompt → 好回答   │
                  └─────────┬─────────┘
                            ↓
                      SFT Policy π_SFT
                            │
          ┌─────────────────┴──────────────────┐
          ↓                                    ↓
生成多个候选回答                    人类比较候选回答
y₁, y₂, y₃ ...                     y_chosen > y_rejected
          │                                    │
          └─────────────────┬──────────────────┘
                            ↓
                  ┌───────────────────┐
                  │ 2. 奖励模型 RM    │
                  │ rθ(x, y) → 标量   │
                  └─────────┬─────────┘
                            ↓
                  ┌───────────────────┐
                  │ 3. PPO 强化学习   │
                  │ 最大化奖励并限制  │
                  │ 偏离 SFT 模型程度 │
                  └─────────┬─────────┘
                            ↓
                     RLHF Policy π_RLHF
```

---

# 3. 三个核心阶段

## 3.1 SFT：先教模型基本行为

训练数据：

```text
Prompt x
    ↓
人工编写或高质量回答 y*
```

损失函数仍然是标准语言模型交叉熵：

\[
\mathcal{L}_{SFT}
=
-\sum_{t=1}^{T}
\log \pi_\theta(y_t^* \mid x,y_{<t}^*)
\]

SFT 的作用：

- 教模型理解指令格式；
- 建立基本回答能力；
- 将策略限制在合理回答区域；
- 为后续采样提供较高质量候选；
- 作为 RLHF 中的参考模型。

如果直接对未经 SFT 的基座模型做强化学习，搜索空间太大，奖励模型很容易被利用。

---

## 3.2 Reward Model：学习人类偏好

对于同一个 Prompt：

```text
Prompt: 解释什么是过拟合

回答 A：正确、清楚、完整
回答 B：错误或含糊

人工标注：
A > B
```

奖励模型输入：

\[
(x,y)
\]

输出一个标量：

\[
r_\phi(x,y)\in\mathbb{R}
\]

通常使用 Bradley–Terry 偏好模型：

\[
P(y_w \succ y_l\mid x)
=
\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\]

其中：

- \(y_w\)：chosen / winner；
- \(y_l\)：rejected / loser；
- \(\sigma\)：Sigmoid。

奖励模型损失：

\[
\mathcal{L}_{RM}
=
-\log
\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\]

奖励模型并不需要预测“绝对正确分数”，只需要满足：

\[
r_\phi(x,y_w)>r_\phi(x,y_l)
\]

---

## 3.3 PPO：优化模型行为

策略模型生成回答：

\[
y\sim\pi_\theta(y\mid x)
\]

奖励模型给出奖励：

\[
r_\phi(x,y)
\]

但不能只优化奖励：

\[
\max_\theta
\mathbb{E}[r_\phi(x,y)]
\]

否则模型可能找到奖励模型的漏洞，即 **Reward Hacking**。

经典 RLHF 使用 KL 惩罚：

\[
R(x,y)
=
r_\phi(x,y)
-
\beta
\log
\frac{\pi_\theta(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
\]

等价地，可以理解为：

\[
\max_\theta
\mathbb{E}_{y\sim\pi_\theta}
[r_\phi(x,y)]
-
\beta
D_{KL}
\left(
\pi_\theta\Vert\pi_{\text{ref}}
\right)
\]

其中：

- \(\pi_\theta\)：当前策略；
- \(\pi_{\text{ref}}\)：冻结的 SFT 模型；
- \(\beta\)：KL 约束强度。

PPO 使用概率比：

\[
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
\]

裁剪目标：

\[
\mathcal{L}_{PPO}
=
-
\mathbb{E}
\left[
\min
\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

裁剪用于防止一次更新改变策略过多。

---

# 4. RLHF 中的四个模型

| 模型 | 是否训练 | 作用 |
|---|---:|---|
| Policy Model | 是 | 生成回答 |
| Reference Model | 否 | 限制 Policy 偏离 SFT 模型 |
| Reward Model | 通常冻结 | 为完整回答打分 |
| Value Model / Critic | 是 | 预测期望奖励，计算 Advantage |

实际部署时，经常将 Policy 和 Value Head 放在同一个语言模型中：

```text
Transformer Hidden States
       ├── LM Head → token logits
       └── Value Head → 每个 token 的 value
```

---

# 5. RLHF MVP 的设计

真正的 token-level RLHF 需要：

- tokenizer；
- causal language model；
- 自回归采样；
- attention mask；
- response mask；
- token-level log probability；
- reward broadcast；
- value head；
- GAE；
- PPO minibatch；
- 多模型显存管理。

为了先理解核心闭环，下面构建一个 **离散回答级 RLHF MVP**：

```text
Prompt
    ↓
Policy 从三个候选回答中选择一个
    ↓
Reward Model 为回答打分
    ↓
PPO 更新回答选择概率
```

它不是完整语言模型，但完整包含：

- SFT；
- chosen/rejected 偏好数据；
- Bradley–Terry Reward Model；
- Reference Policy；
- KL penalty；
- Value Model；
- PPO clipping；
- Advantage normalization；
- entropy bonus。

---

# 6. 可运行的 PyTorch RLHF MVP

```python
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
```

---

# 7. MVP 的预期结果

在带噪声的 SFT 数据中：

```text
天空为什么是蓝色？
SFT 倾向候选 1：因为海洋把天空染蓝

什么是过拟合？
SFT 倾向候选 1：模型训练速度太慢
```

Reward Model 从偏好对中学习到：

```text
候选 0 > 候选 1
候选 0 > 候选 2
```

经过 PPO 后，策略概率会转向候选 0：

```text
天空为什么是蓝色？
P(正确回答) ≈ 1

什么是过拟合？
P(正确回答) ≈ 1
```

这演示了经典闭环：

```text
不完全正确的 SFT Policy
        ↓
人类偏好比较
        ↓
Reward Model
        ↓
PPO 修正 Policy
```

---

# 8. MVP 中各部分对应真实 RLHF 的什么

| MVP | 真实语言模型 RLHF |
|---|---|
| `prompt_id` | tokenized prompt |
| 三个候选 action | 自回归生成的完整 response |
| Policy 输出三个 logits | Transformer 输出词表 logits |
| 一个 action 的 log-prob | response 所有 token 的 log-prob 之和 |
| Reward Model 输入 ID | Reward Transformer 输入 prompt + response |
| 单步 Value | 每个 response token 的 Value |
| `reward - value` | 使用 GAE 计算 token-level advantage |
| 离散 PPO | token-level PPO |
| 固定三个回答 | 动态 rollout 生成回答 |

---

# 9. 版本迭代路线

## v0.0：SFT Baseline

```text
Prompt → Demonstration
        ↓
Cross-Entropy
```

组件：

- Policy Model；
- SFT 数据；
- response-only loss。

目标：

- 先保证模型能够正常回答；
- 建立 RLHF 参考策略。

缺失：

- 没有偏好学习；
- 所有示范回答权重相同；
- 无法利用 chosen/rejected 排序。

---

## v0.1：离散回答级 Reward Model

```text
Prompt
  ├─ Answer A
  └─ Answer B
        ↓
Human Preference
        ↓
Reward Model
```

新增：

- pairwise preference 数据；
- Bradley–Terry loss；
- chosen/rejected accuracy；
- reward margin 分析。

关键指标：

\[
\text{Preference Accuracy}
=
P(r(y_w)>r(y_l))
\]

这一版本可以单独检查奖励模型，而不急着做强化学习。

---

## v0.2：REINFORCE + KL

```text
Policy 采样回答
     ↓
Reward Model
     ↓
Reward × log probability
```

目标函数：

\[
\mathcal{L}
=
-
R(x,y)\log\pi_\theta(y\mid x)
\]

并加入：

\[
R
=
r_\phi-\beta KL
\]

优点：

- 实现简单；
- 能验证奖励是否真的可以推动策略。

问题：

- 梯度方差大；
- 更新不稳定；
- 缺少 Value baseline；
- 容易一次改变策略过多。

---

## v0.3：PPO + Value Model

即上面的 MVP。

新增：

- old policy；
- probability ratio；
- PPO clipping；
- value model；
- advantage normalization；
- entropy bonus。

这是经典 RLHF 最小闭环。

---

## v0.4：Token-level Tiny LM RLHF

将离散回答策略替换为小型语言模型：

```text
Tokenizer
   ↓
Tiny Transformer / GPT
   ↓
Auto-regressive Rollout
   ↓
Reward Model
   ↓
Token-level PPO
```

需要新增：

### 数据 Mask

```text
[Prompt Tokens] [Response Tokens]
      mask=0          mask=1
```

只对 response token 优化：

\[
\log\pi(y\mid x)
=
\sum_t
m_t\log\pi(y_t\mid x,y_{<t})
\]

### Token-level KL

\[
KL_t
\approx
\log\pi_\theta(y_t\mid s_t)
-
\log\pi_{\text{ref}}(y_t\mid s_t)
\]

### 序列奖励分配

最终奖励通常加在最后一个 response token：

```text
token 1: -β KL₁
token 2: -β KL₂
token 3: -β KL₃
最后 token: RM Reward - β KL_T
```

---

## v0.5：GAE 与完整 PPO

引入 Temporal Difference error：

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
\]

广义优势估计：

\[
A_t^{GAE}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l}
\]

新增：

- GAE；
- value clipping；
- minibatch shuffle；
- gradient accumulation；
- gradient clipping；
- reward whitening；
- advantage whitening；
- response length mask；
- EOS 处理；
- invalid response penalty。

---

## v0.6：多维偏好与安全约束

单一 Reward Model 很容易把不同目标混在一个分数中。

```text
Helpfulness Reward
Correctness Reward
Safety Reward
Style Reward
Format Reward
       ↓
Reward Aggregator
```

可采用：

\[
R_{\text{total}}
=
\sum_i w_iR_i
\]

但安全目标有时不适合简单线性加权，而应作为硬约束：

```text
if safety_violation:
    reject_sample
else:
    optimize_helpfulness
```

新增评估：

- 不同语言；
- 长短回答；
- 边界安全样本；
- 奖励模型群体偏差；
- verbosity bias；
- position bias；
- reward disagreement。

---

## v0.7：生产级 RLHF

```text
Prompt Dataset
     ↓
Distributed Rollout Workers
     ↓
Reward / Safety / Rule Evaluators
     ↓
Replay Buffer
     ↓
PPO Training Workers
     ↓
Periodic Evaluation
```

新增：

- 多 GPU rollout；
- vLLM/SGLang 推理服务；
- ZeRO/FSDP 训练；
- Policy、Reference、Reward、Value 的模型并行；
- rollout 与训练解耦；
- checkpoint 和恢复；
- adaptive KL controller；
- reward normalization；
- 多奖励加权；
- 在线监控。

典型总奖励：

\[
R
=
w_hR_{\text{helpful}}
+
w_cR_{\text{correct}}
+
w_sR_{\text{safety}}
+
w_fR_{\text{format}}
-
\beta KL
\]

---

## v0.8：DPO 类离线偏好优化

DPO 不再显式训练 Reward Model，也不进行在线 PPO rollout，而是直接从 chosen/rejected 优化策略（Rafailov et al., 2023）。

DPO 损失：

\[
\mathcal{L}_{DPO}
=
-
\log\sigma
\left[
\beta
\left(
\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right)
\right]
\]

流程：

```text
Prompt + chosen + rejected
           ↓
直接优化 Policy
```

优点：

- 不需要单独部署 Reward Model；
- 不需要 rollout；
- 不需要 Value Model；
- 训练过程接近普通 SFT；
- 工程复杂度低。

局限：

- 主要利用固定离线偏好数据；
- 不能自然地在线探索新回答；
- 数据覆盖不足时难以发现新策略；
- 仍然依赖偏好数据质量。

严格来说，DPO 属于偏好优化，而不是经典的 PPO-RLHF，但经常作为 RLHF 工程替代方案。

---

## v0.9：RLAIF 与规则反馈

将部分人类偏好替换为 AI Feedback：

```text
Policy 生成多个回答
        ↓
Judge Model 比较
        ↓
Preference Dataset
        ↓
DPO / Reward Model / PPO
```

优势：

- 标注速度快；
- 成本低；
- 可以大规模覆盖；
- 可以使用明确原则进行审核。

风险：

- Judge 和 Policy 共享错误；
- 偏好长度更长、格式更漂亮的回答；
- 自我偏好；
- 奖励模型和生成模型共谋式偏差；
- 人类目标被 Judge 风格替代。

合理结构是：

```text
规则验证器
    >
可执行验证器
    >
异源 Judge
    >
同源 Judge
```

---

## v1.0：可验证奖励与推理强化学习

对于数学、代码、工具调用等任务，可以不用模糊的人类偏好作为主要奖励：

```text
数学答案 → 精确答案检查
代码      → 单元测试
SQL       → 数据库执行
工具调用  → Schema + 环境结果
证明      → Proof Checker
```

奖励：

\[
R=
\begin{cases}
1,&\text{验证通过}\\
0,&\text{验证失败}
\end{cases}
\]

或者提供更密集的过程奖励：

\[
R_t=\text{step verifier}(s_t,a_t)
\]

这一方向通常会结合：

- outcome reward；
- process reward；
- rejection sampling；
- group-relative advantage；
- best-of-N；
- curriculum；
- difficulty filtering。

它与传统“完全依赖人工偏好”的 RLHF 已经不同，更接近：

> Reinforcement Learning from Verifiable Rewards。

---

# 10. 版本总表

| 版本 | 核心能力 | 训练信号 | 主要问题 |
|---|---|---|---|
| v0.0 | SFT | 标准答案 | 不会利用偏好排序 |
| v0.1 | Reward Model | chosen/rejected | 尚未改变策略 |
| v0.2 | REINFORCE | RM reward + KL | 方差大、不稳定 |
| v0.3 | PPO MVP | RM + Value + PPO | 仍是回答级简化 |
| v0.4 | Token-level PPO | token log-prob | Mask 和长度处理复杂 |
| v0.5 | GAE PPO | token reward + GAE | 超参数敏感 |
| v0.6 | 多目标 RLHF | 帮助性、安全性等 | 奖励冲突 |
| v0.7 | 分布式 RLHF | 多奖励、在线 rollout | 成本和系统复杂度高 |
| v0.8 | DPO 家族 | 离线偏好对 | 在线探索较弱 |
| v0.9 | RLAIF | AI Judge 偏好 | Judge 偏差 |
| v1.0 | 可验证 RL | 执行器/验证器 | 只适用于可验证领域 |

---

# 11. RLHF 最容易踩的坑

## 11.1 Reward Hacking

模型优化的是：

\[
\text{Reward Model 的分数}
\]

而不是真实的人类满意度。

例如 Reward Model 偏好长回答，Policy 可能不断增加：

- 标题；
- 列表；
- 总结；
- 重复解释；
- 不必要的免责声明。

Reward 上升不代表真实质量上升。

---

## 11.2 Reward Overoptimization

RL 训练早期：

```text
Reward 上升
Human Evaluation 上升
```

继续训练后可能变成：

```text
Reward 继续上升
Human Evaluation 下降
```

这说明 Policy 已经开始利用 Reward Model 的分布外漏洞。

因此必须监控：

- Reward score；
- KL；
- response length；
- entropy；
- 人工胜率；
- 独立 Judge 胜率；
- 安全指标；
- 通用能力回归。

---

## 11.3 KL 太小或太大

KL 太小：

```text
Policy 快速偏离 SFT
→ Reward hacking
→ 语言退化
→ 模式坍塌
```

KL 太大：

```text
Policy 几乎不改变
→ RLHF 没有效果
```

生产系统通常使用自适应 KL：

```text
实际 KL > 目标 KL
→ 增大 β

实际 KL < 目标 KL
→ 减小 β
```

---

## 11.4 Reward Model 只会做域内比较

如果 Reward Model 训练数据主要是：

```text
简短问答
```

却让 PPO 生成：

```text
长篇代码、复杂数学证明、工具调用
```

Reward Model 可能在分布外给出任意高分。

所以 RM 数据分布必须覆盖 Policy 的实际 rollout 分布。

---

## 11.5 Preference 数据没有差异

如果 chosen 和 rejected 都很好，或者只是措辞略有不同：

\[
r(y_w)-r(y_l)
\]

很难学。

如果 rejected 全是明显垃圾，Reward Model 又只会学习表面特征。

好的偏好对应包含：

- 都语言通顺；
- 存在真实质量差异；
- 差异覆盖正确性、帮助性、安全性、简洁性；
- 难度逐渐增加；
- 有足够的边界样本。

---

## 11.6 PPO 的实现错误通常藏在 Mask 中

真实语言模型 RLHF 最常见的问题包括：

- Prompt token 被加入 PPO loss；
- padding token 被计算奖励；
- EOS 后 token 未 Mask；
- chosen response 长度不同但未归一化；
- Reference log-prob 使用错误 tokenizer；
- old log-prob 在 PPO epoch 中被重新计算；
- KL 符号写反；
- reward 加到了所有 token；
- value bootstrap 越过 EOS；
- advantage 没有 detach。

这些错误不会总是报错，但会让训练方向完全改变。

---

# 12. 建议的学习实现顺序

```text
v0.0 SFT Baseline
        ↓
v0.1 离散回答级 Reward Model
        ↓
v0.2 REINFORCE + KL
        ↓
v0.3 PPO + Value Model
        ↓
v0.4 Token-level Tiny LM RLHF
        ↓
v0.5 GAE 与完整 PPO
        ↓
v0.6 多维偏好与安全约束
        ↓
v0.7 生产级 / 分布式 RLHF
        ↓
v0.8 DPO 类离线偏好优化
        ↓
v0.9 RLAIF 与规则反馈
        ↓
v1.0 可验证奖励与推理强化学习
```

核心认识是：

\[
\boxed{
\text{RLHF}
=
\text{SFT 行为先验}
+
\text{偏好建模}
+
\text{受约束的策略优化}
}
\]

Reward Model 负责回答“人类更喜欢什么”，PPO 负责回答“怎样提高这种偏好的概率”，Reference Model 和 KL 则负责防止模型为了奖励而偏离正常语言分布。
