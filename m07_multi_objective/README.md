[返回根目录](../README.md)

# m07: 多目标 Reward 聚合 + Reward Hacking（v0.6）


---

## The Problem

m04/m05/m06 里都只有**一个标量奖励**（Reward Model 的分数）。但真实 RLHF 的"人类满意度"是**多维的**——一个回答要同时满足有用、正确、安全、风格合适、格式规范，还常常彼此冲突（`versions.md` §9 v0.6）：

```text
Helpfulness Reward
Correctness Reward
Safety Reward
Style Reward
Format Reward
       ↓
Reward Aggregator
```

经典做法是把它们线性加权成一个总奖励：

$$ R_{\text{total}} = \sum_i w_i R_i = w_h R_{\text{helpful}} + w_s R_{\text{safety}} + w_f R_{\text{format}} - \beta\,\text{KL} $$

但这里藏着两个大坑：

1. **Reward Hacking / Overoptimization（`versions.md` §11.1/11.2）**：模型优化的是 Reward Model 的分数，不是真实人类满意度。如果某个分量（比如"风格/长度"）很容易被刷高，策略就会拼命去刷它——**代理奖励一路涨，真实质量却持平甚至下降**。
2. **安全目标不适合单纯线性加权（§9 v0.6）**：安全往往应该是**硬约束**，而不是可以被别的目标"买通"的软权重:

```text
if safety_violation:
    reject_sample                # 硬约束：违规直接拒掉
else:
    optimize_helpfulness        # 只有安全的才优化
```

本模块用 m04 那套**离散回答级 PPO** 核心，把单一奖励升级成**多目标 Reward**，并亲手演给你看：加权会怎样被 hack，硬约束又是怎样把安全守住。

## The Solution

在 m04 的 `Policy + Reward + Value + PPO` 上，把"单一 RM 分数"替换为**三个分量**：

```
R_total = 1.3·R_helpful + 1.0·R_safety + 3.0·R_verbosity
                            │
              ┌─────────────┴──────────────┐
              │ R_helpful  真值/人工质量(0~1) │
              │ R_safety   规则打分(0/1)     │
              │ R_verbosity 长度代理(0~1)    │ ← 极易被 hack 的风格奖励
              └─────────────────────────────┘
```

- **R_helpful**：由 Helpfulness Reward Model（m02 的 Bradley–Terry 写法）训出，代表"真实有用性"。
- **R_safety**：安全规则打分（0/1）。P1「如何安全地过马路」里有两个候选是不安全的。
- **R_verbosity**：把回答长度在同一个 prompt 内归一化到 [0,1]。它只是"长度"，与真实质量无关——是一个**天然可被过度优化的代理**。

然后做三件事：

1. **Reward Hacking 演示**：单独用一个 PPO 去最大化 `R_verbosity`（纯代理），对比一开始的随机策略，证明"代理奖励大涨而真实质量不涨/下降"。
2. **Hard constraint vs 软加权 的 A/B**：同样跑两个 PPO——
   - **naive**：奖励就是 `R_total`（加权和，硬要让安全部分也参与加权）；
   - **hard**：同样的 `R_total`，但**只要 `safety < 阈值` 就把该样本的奖励替换成一个大负数**（= 把违规样本从损失里剔除、并推着策略远离它）。
   比较两者的 `unsafe_rate`。
3. **Bias 评估（§9 v0.6 列表）**：算 `corr(长度, verbosity代理)`、`corr(长度, HelpfulnessRM)`，以及 `corr(位置, RM)`——量化 verbosity bias 与 position bias。

## How It Works

**为什么 w_v 要放大到 3.0 超过 helpful+safety 之和？** 这是本模块最重要的"剧本"设定。拿高危的 P1（过马路）看：

| P1 候选 | R_helpful | R_safety | R_verbosity | R_total |
|---|---|---|---|---|
| a0 看信号灯走斑马线 | 1.0 | 1 | 0.17 | **2.8（安全）** |
| a2 冲过马路最快最长 | 0.0 | 0 | 1.0 | **3.0（不安全）** |

因为 `w_v` 足够大（3.0），**纯加权和会把最高分给到那个"既错误又不安全"的长回答**——安全权重（1.0）在 verbosity 这条 hack 通道面前根本拦不住。这模拟了生产中"Safety RM 权重偏低被风格奖励反超"的真实事故。

- **Reward Hacking 的本质（§11）**：PPO 只知道 `R_total`，它不知道 `R_verbosity` 是"风格"而非"质量"。当 `w_v` 足够大，策略的最优解就是把所有概率压到 `verbosity=1.0` 的长回答上。于是 `E[verbosity]` 从 `0.38` 冲上 `1.0`（+0.62），但 `E[真实质量]` 从 `0.44` 跌到 `0.0`（-0.44）。**Reward 涨了，Human Eval 崩了——这就是 Overoptimization**。这个现象不是用 print 糊弄，而是用断言锁死的。

- **Hard gate 的实现**：在 rollout 得到 `r_total` 后：

```python
is_unsafe = safe_r < SAFE_THRESHOLD
r_total = torch.where(is_unsafe, 大负数惩罚, r_total)
```

被替换成 `SAFE_PENALTY=-2.0` 的样本，其 advantage 变成长深负值，PPO 会把这块概率**主动让出来**，让给同 prompt 里安全的最优答案（a0）。这就是 `versions.md` §v0.6 的 `reject_sample` 路线：**不是修改权重，而是让策略永远无法从不安全回答中获利**。

- **Bias 诊断的意义**：`corr(长度, verbosity代理)≈0.87`（代理本质上就是"长度"），而 `corr(长度, HelpfulnessRM)≈-0.50`——说明一个合格的 Helpfulness RM 不会偏爱长文本，偏长文本的奖励其实是 bias。`corr(位置, RM)≈-0.85` 也警告：**奖励与候选所在列表位置强相关**时，就是 location bias，需要按内容做残差化才能判定真实偏好。

## Code Walkthrough（版本锚点 `# v0.6`，同 `versions.md` §9 v0.6）

**Step 1｜数据** — 复用 m04 的 4 个 prompt × 3 候选形状，但每个候选显式给出 `HELPFUL / SAFE / LENGTH` 三张表。`responses` 只是展示文本；训练用的是这三张表的属性，避免依赖真实的字符长度统计。

**Step 2｜Helpfulness RM**（`# v0.1` 风格）— 从 `HELPFUL` 表生成 `(prompt, chosen, rejected)` 偏好三元组，用 Bradley–Terry 损失训一个 `RewardModel`，冻结后**只用来做 bias 诊断**（不参与 PPO，保证加权演示的确定性）。

**Step 3｜Policy / Value / 聚合** — 复用 m04 的 `PolicyModel` / `ValueModel` 与离散 PPO 核。新增 `aggregate()` 实现 `R_total = w_h·R_helpful + w_s·R_safety + w_v·R_verbosity`。

**Step 4｜`run_ppo(reward_kind, reject_unsafe)`** — 一个训练器跑出三种策略：

- `kind='proxy', reject=False` → reward hacking 演示；
- `kind='total', reject=False` → naive 软加权；
- `kind='total', reject=True` → hard 硬约束。

Rollout 里对 `R_total` 做**有界性断言**（有限、下界 ≥ 惩罚、上界 ≤ `BOUND_MAX`），再扣 KL、算 advantage、PPO clip。

**Step 5｜评估 → [PASS]** — 依次断言：

1. `R_total` 全表 `isfinite` & 在 `[0, BOUND_MAX=5.3]`；
2. reward hack：`Δverbosity > 0.02` **且** `Δquality ≤ 0.02`（代理涨、质量不升）；
3. hard 胜：`naive_unsafe > 0.1`、`hard_unsafe < 0.1`、`hard_unsafe < naive_unsafe - 0.05`；
4. bias：`corr(长度, 代理) > 0.7`、`corr(长度, 代理) > |corr(长度, RM)| + 0.1`。

运行：

```bash
python m07_multi_objective/code.py
```

## Key Design Decisions

- **`w_v = 3.0` 制造 hack，而不是靠巧合**：把 verbosity 权重调到超过 `w_h + w_s`，才保证"不安全长回答"在加权和里拿到最高分——这样演示失败也不会依赖网络初始随机而跑偏。
- **R_helpful 用确定性 oracle 表，R_verbosity 是 length 代理**：代理故意做成一个与长度挂钩的东西，才能让"过优化代理 → 质量崩"被断言锁死；帮助分 RM 只做 bias 诊断，保证核心对标可用真值。
- **hard gate 用 `torch.where` 替换 reward 而非删行**：`reject_sample` 的"reject"既去掉该样本对策略的增益，又用一个负奖励**主动把概率分量让出来**，避免"只是不优化违规，但违规概率留在原位"——这是让 hard 严格优于 naive 的关键。
- **指标 `unsafe_rate` = 各 prompt 上有门 prob 选择不安全动作的平均**：只数 P1 的两个不安全候选，确定性、无采样抖动，断言稳定。
- **`R_total` 有界断言放在 loss 之前**：证明"加权和有限、在理论范围"是每条 rollout 都保证的，而不是事后检查。

## Going Deeper

- **Reward Overoptimization 的完整曲线（§11.2）**：本模块只演示了"代理涨、质量跌"的终点。完整故事是——早期 `reward ↑ / human eval ↑`，继续训练才`reward ↑ / human eval ↓`；真实生产要在过程中监控 reward、KL、response length、entropy、人工胜率、独立 Judge 胜率、通用能力回归。
- **拒绝采样与 `best-of-N`**：这里的"hard reject"和 v1.0 的 rejection sampling 一脉相承——只让"通过校验"的样本进入训练。真实部署常配合 `best-of-N`：采样 N 个回答，选 safety 校验通过且 reward 最高的一个喂给用户。
- **多目标加权里的 reward 冲突**：多目标会经常出现 reward 冲突（帮助性和安全性打架）。除了加权和、硬约束，还有 `min`（取最小分量）、`product`、`lexicographic`（先保安全再加权）等聚合策略，都是后续可选方向。
- **bias 量化后的残差化**：看到 reward 与长度/位置强相关时，不能直接说"模型喜欢长/位置 0"——要先把真实质量当作协变量做残差（partial correlation）。`versions.md` §v0.6 的 "verbosity bias / position bias" 正是这么要求。
- **这是 m08 的桥**：m07 的单机多维奖励 + hard gate，走进 m08 的分布式/生产 RLHF（rollout 解耦、adaptive KL、在线监控）就是工程放大版。

## 模块定位

这是 `learn-rlhf` 里把"RLHF 奖励从一个标量变成一组目标"的拐点（`v0.6`）：在 m04 的离散 PPO 完整闭环上，把单一 Reward 换成 `w_h R_helpful + w_s R_safety + w_v R_verbosity`，并亲自演示两件生产必学的课——**Reward Hacking（代理涨、质量降，§11）** 与 **安全硬约束胜过软权重（§9 v0.6）**，最后量化评价 verbosity/position bias。`R_total` 每条 rollout 都断言**有界**；代理过度优化与硬约束胜出均用可测断言锁死；`[PASS]` 字样与断言一致性。

版本：**v0.6** · 运行：`python m07_multi_objective/code.py`（CPU 秒级）