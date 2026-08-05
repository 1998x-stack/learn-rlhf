# m00: RLHF 全景 — 从人类偏好到策略优化的学习路线

[返回根目录](../README.md)

---

## The Problem

预训练语言模型（LM）本质上只做一件事：**预测下一个 token**。它能接续任何语言通顺的文本，但完全不知道"人类想要什么"——不知道哪种回答更正确、更有用、更安全、更清晰。

当多个回答都语言通顺时，靠交叉熵无法区分优劣：SFT 只是让模型模仿示范，所有示范回答权重相同，无法利用 **chosen/rejected** 的排序信息。

RLHF（Reinforcement Learning from Human Feedback）要解决的正是这个问题。它最根本的动机（见 `learn-rlhf/versions.md` §1）是把**模糊的人类偏好转化为可优化的数值奖励**：

```text
Human Preference
    ↓
Reward Function
    ↓
Policy Optimization
```

换句话说，用户的偏好往往是模糊的"这个更好 / 那个不好"的排序，而优化器需要明确的数值信号。RLHF 就在这两者之间建起一座桥。

## The Solution

不一次性实现一整套分布式 RLHF。**从最简的离散回答级闭环入手**，沿着 `versions.md` §2 的完整流程逐级推进——先跑通"偏好 → 奖励 → 策略"的最小圈，再逐步升级到 token-level、GAE、生产分布式：

```text
预训练 LM ⟶ SFT ⟶ 采样候答 + 人工比较 ⟶ Reward Model⟶ PPO（带 KL 约束）⟶ RLHF Policy
```

这套流程对应 `versions.md` 的完整版本树（v0.0–v1.0）。本系列 m00–m12 从 **SFT（v0.0）** 一路推进到 **可验证奖励 / 推理 RL（v1.0）**，每条前置目标都是"看得懂、跑得通、能扩展"。

## How It Works

整个 RLHF 是一条"数据流 + 四个模型"组成的闭环：

```
Prompt
  → Policy 采样候选回答
  → 人工/SFT 偏好规则
  → Reward Model 打分
  → PPO（带 KL 约束）更新 Policy
```

数据流（SFT → 偏好比较 → RM → PPO）：

1. **SFT 微调**：先用人类示范（prompt → 好回答）训练 Policy 基础行为
2. **人类偏好比较**：对同一 prompt 生成多个候选回答，人工标注 chosen > rejected
3. **RM 打分**：Bradley–Terry Reward Model 学习"人类更喜欢谁"，为完整回答打分
4. **PPO 优化**：最大化期望奖励，同时用 KL 约束限制策略不过度偏离 SFT 模型

其中整个流程共有**四个模型**（见 `versions.md` §4），职责与是否训练各异：

| 模型 | 是否训练 | 作用 |
|---|---|---|
| Policy Model | 是 | 生成（采样）回答 |
| Reference Model | 否 | 限制 Policy 偏离 SFT 模型（冻结） |
| Reward Model | 通常冻结 | 为完整回答打分 |
| Value Model / Critic | 是 | 预测期望奖励，计算 Advantage |

实际部署时，为了显存，Policy 与 Value head 常共用同一个语言模型：Transformer 隐藏状态分出 `LM Head → token logits` 与 `Value Head → value`。

## Module Map

本系列共 13 个模块（m00–m12），覆盖 `versions.md` 的 **v0.x（核心迭代）/ v1.0（可验证奖励）**：

| 模块 | 里程碑 | 覆盖版本 | 内容 |
|------|--------|---------|------|
| **m00** | RLHF 全景 | — | 学习路线 + 版本树导览（本文档） |
| **m01** | 离散回答级 Policy + SFT | v0.0 | prompt→候选回答，CE loss，建立参考策略 |
| **m02** | Bradley–Terry Reward Model | v0.1 | 偏好对、BT loss、accuracy、margin |
| **m03** | REINFORCE + KL | v0.2 | 采样、RM 奖励、`-R·logπ`、KL penalty |
| **m04** | PPO + Value（离散 MVP） | v0.3 | old policy、ratio、clip、advantage、entropy |
| **m05** | 字符级 Tiny LM + Token-level PPO | v0.4 | response mask、token KL、序列奖励分配 |
| **m06** | GAE + 完整 token-level PPO | v0.5 | TD error、GAE、value clip、whitening |
| **m07** | 多目标 Reward 聚合 | v0.6 | 多奖励加权/硬约束、bias 评估 |
| **m08** | 分布式/生产 RLHF | v0.7 | rollout 解耦、buffer、checkpoint、adaptive KL |
| **m09** | DPO 离线偏好优化 | v0.8 | 移除 RM/rollout，直接优化 policy |
| **m10** | RLAIF + 规则反馈 | v0.9 | Judge 偏好、多级验证器 |
| **m11** | 可验证奖励 / 推理 RL | v1.0 | outcome/process reward、GRPO、best-of-N |
| **m12** | 端到端收束 + 深坑清单 | 收束 | SFT→RM→RL→verifier 全链路 |

## 核心推进原则（Key Design Principles）

`versions.md`（§2–§3）把 RLHF 拆成**三个步骤，每一步都建立在上一步之上，任何一步都不建议跳级**：

① 先 **SFT 建立行为先验**（教模型基本回答能力、理解指令格式、限制探索空间）→ ② 再 **偏好建模**（Reward Model 学会"人类更喜欢什么"）→ ③ 最后 **受约束的策略优化**（PPO 把偏好概率最大化，同时用 Reference Model + KL 防止偏离正常语言分布）。

> Reward Model 负责回答"人类更喜欢什么"，PPO 负责回答"怎样提高这种偏好的概率"，Reference Model 和 KL 负责防止模型为了奖励而偏离正常语言。

但这个流程有很多能**踩到的坑**（见 `versions.md` §11）：

- **Reward Hacking**：模型优化的是 RM 分数而非真实人类满意度，Reward 上升不代表质量上升
- **Reward Overoptimization**：训练后期 Reward 继续上升但人工评估下降，说明 Policy 在利用 RM 的分布外漏洞，必须监控 KL、长度、人工胜率
- **KL 太小或太大**：KL 太小 → 快速偏离 SFT → reward hacking / 语言退化；KL 太大 → RLHF 几乎没有效果。生产系统用自适应 KL 动态调整 β

## Going Deeper

本系列（m00–m12）覆盖 `versions.md` 中从 **v0.0（SFT）到 v1.0（可验证奖励）** 的完整版本树。其中：

- **v0.x（m01–m10）** 走的是"人类偏好"主干，从离散 MVP 到生产分布式与离线偏好优化（DPO）。
- **v1.0（m11）** 转向"Verifiable Reward"，对数学 / 代码等可验证任务不再用模糊偏好，改用执行器 / 验证器。
- **m12_integration** 把 v0.x 与 v1.0 全链路收束，集中汇总 `versions.md` 中各类坑（Reward Hacking、Overoptimization、Mask 坑……）。

更远一步的生产与分布式细节（多 GPU rollout、vLLM/SGLang 推理、ZeRO/FSDP 训练、模型并行、adaptive KL controller）在 **m08** Going Deeper 展开。

如果你学完 m12，想接续更前沿方向，请回到 `versions.md` 的版本树，按同一套"先偏好、再约束、再扩展"的节奏继续。

## 模块定位

m00 是整个学习路线在第 **0** 号位置的**全景导览**：它不写任何代码，只负责在两件事之间建立连接——

1. **学习路线**（本页）：先解答"为什么 RLHF 要从人类偏好到策略优化"，用《The Problem → The Solution → How It Works》建立系统级心智模型；
2. **版本树**（`versions.md`）：把抽象的心智模型对齐到具体的 v0.x / v1.0 版本序列与四个模型的职责。

读完本模块，你应该已经"带着一张清晰地图进入 m01 的第一个 SFT 组件"。后续模块的 `[返回根目录](../README.md)` 链接，最终都会回到这张全景图上。