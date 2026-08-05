# m11: 可验证奖励与 GRPO —— Reinforcement Learning from Verifiable Rewards（v1.0）

[返回根目录](../README.md)

---

## The Problem

m01–m10 的奖励最终都来自**人类偏好**：人标注 (chosen, rejected)、训练 Reward Model、让 AI Judge 打分。这条路的隐含成本是把"什么才算好"模糊地编码进一个可被利用的奖励信号里。

但对一大类任务根本不需要这么绕：**数学、代码、SQL、工具调用、定理证明** —— 它们的正确性是**机器可计算的**。`3+5` 的答案是多少、一段代码是否通过单元测试、一个 SQL 是否能跑出对的结果，都不需要人去打分，更不需要训练一个可能被 hack 的 Reward Model。

`versions.md §9（v1.0）` 把这条路线总结为：

```
数学答案 → 精确答案检查
代码     → 单元测试
SQL      → 数据库执行
工具调用 → Schema + 环境结果
证明     → Proof Checker
```

而它的奖励也不再是"人类偏好分数"，而是**可验证奖励**（outcome reward：验证通过=1、否则=0）。这就是 **RL from Verifiable Rewards（RLVR，可验证奖励强化学习）** —— 与"完全依赖人工偏好的传统 RLHF"已经有本质区别。

可验证奖励 + 推理 RL 带来两个直接问题：

1. **reward 是极稀疏、非光滑的 0/1**，而且每个 prompt 是"单步"任务，没有中间 step —— PPO 那一套 Value / Critic / GAE 在这里既昂贵又不自然。有没有更简单、不需要 Critic 的目标？
2. **给定一个不确定的策略，如何用验证器进一步榨出正确率** —— 除了改策略，还有没有别的"推理时"手段？

本模块分别回答：**GRPO（组内相对优势）** 和 **best-of-N / rejection sampling**。

## The Solution

选**可验证算术题**（正确性机器可算），把"奖励从哪里来 + 怎么优化"都换掉：

```
math prompt  a+b=？
      │
      ▼
Policy 对每个 prompt 采样 N 个候选解（rollout N samples）
      │
      ▼
精确验证器 Verifier：对每个候选解判定 对(1) / 错(0)   ← 无 RM、无人工标注
      │
      ▼
GRPO 组内相对优势  A_i = (R_i - mean(R_group)) / (std(R_group)+eps)
      │（不需要任何 Value / Critic 模型 —— 这就是 GRPO 的关键洞察）
      ▼
更新策略：A>0 的答案概率↑、A<0 的↓（带 ratio / clip 稳定）
      ▼
per-sample/best-of-N N=1 正确率 0.32 → 1.00  + best-of-N 把成功率进一步滚高
```

最小闭环是：**可验证任务 → 采样 N 个解 → 验证器给 0/1 → 组内相对优势 → 更新策略**。之后再用 **best-of-N / rejection sampling** 展示另一种"不训练也能提正确率"的手段。

## How It Works

### outcome reward 与 process reward（versions.md §9 原文）

v1.0 给了两种可验证奖励：

| 奖励类型 | 定义 | 粒度 |
|---|---|---|
| **outcome reward** | $R=1$ 若整段回答验证通过，否则 $0$ | 稀疏，只看最终答案 |
| **process reward** | $R_t=\text{step verifier}(s_t,a_t)$ | 稠密，给每个中间推理步骤打分 |

本模块用 **outcome reward**（每个候选解整体 0/1），因为算术题的正确答案就是那个稀疏的最终结果。真实推理场景（DeepSeek-R1 训练）则往往给过程也打分 —— 但对教学来说，outcome 已经把"验证器当奖励源"的核心讲透了。

### GRPO vs PPO —— 组内相对优势消掉 Value 模型

PPO 的 Advantage 是:

$$A_t = R_t - V(s_t)$$

它需要学一个 **Value / Critic 网络** 去预测期望奖励作为 baseline。可验证奖励场景下，这个 Critic 显得笨重而多余。

GRPO（DeepSeekMath，2024）的做法是：对同一个 prompt，**采样一组 N 个候选解**，然后用这组的**均值/标准差**做基准 baseline，完全不需要价值网络：

$$A_i = \frac{R_i - \mathrm{mean}(R_{\text{group}})}{\mathrm{std}(R_{\text{group}})+\epsilon}$$

直觉：

- 同组内 **正确解（R=1）得到正优势** → 提高它们出现的概率；
- **错误解（R=0）得到负优势** → 压低它们；
- 若同组 **全对或全错**（std=0）→ advantage≈0 → 不更新（正确地停止）；
- 它和 PPO 一样可以用 **ratio + clip** 稳定（`min(ρA, clip(ρ)A)`），但不再需要第二套 Value 网络的参数与训练。

这就是 GRPO 的关键：**用"一组样本的组内相对位置"取代"价值模型对期望奖励的估计"** —— 在可验证（或群体同类）reward 下反差更小、更省钱。

### best-of-N / rejection sampling

即使策略不是完美的，只要它**有概率产出正确解**，我们就能在采样时多试几次、用验证器把关：

$$\text{best-of-N success} = 1 - (1 - p)^N$$

- $N=1$ 就是 per-sample 正确率 $p$；
- $N$ 越大，"至少一次 $p$ 命中"的成功率越高（$p=0.33,N=16$ 可达 ~0.998）；
- 配合"经验证器挑一个正确解"就是 **rejection sampling / 自举** —— 这种"生成的候选解提升"是 v1.0 里与"改策略"正交的提升手段，也是 **best-of-N**。

## Code Walkthrough（版本锚点 `# v1.0`）

**Step 0｜可验证算术数据** — `ARITH` 存 8 个 `(表达式, 正确整数)`；`build_candidates` 为每题生成 3 个候选（正确 / +1 错 / -1 错），候选下标 0 恒为正确。`compute_answer` 用 `eval` 真算一遍 ground-truth。

**Step 1｜精确验证器** — `verify_answer(expr, candidate)` 解析候选里的整数，与 `eval` 真值**精确比对**，返回 outcome reward 1/0。没有任何可学习参数 —— 这就是"可验证奖励"。

**Step 2｜含噪声 SFT 起点** — 每条 prompt 给 1 条正确 + 2 条错误示范，得到 `P(正确)≈1/3` 的不确定策略。这一步很重要：让组内既采得到对、也采得到错，GRPO 才有内部方差去计算相对优势。

**Step 3｜GRPO 循环**（`train_grpo`）：

- **Rollout**：对每个 prompt 采样 $N=8$ 个候选，冻结采样时刻策略算 `old_logp`，验证器返回 0/1。
- **组内相对优势**：`reshape` 到 `[num_prompts, N]`，`group_advantage` 按行算 `A=(R-mean)/(std+eps)`；
- **策略更新**：对每个样本算 `ratio=exp(logπθ - logπold)`，用 `min(ratio·A, clip(ratio)·A)` 的 PPO-clip surrogate 更新。反复多 epoch。

**Step 4｜机制断言**：

- 验证器自检：顶层真值 8/8 可精确算出；
- per-sample / best-of-N N=1 可验证正确率 **0.315 → 1.000**（严格上升；基线用与 Step 5 同款 300-trial 估计器，避免 0.500/0.315 两个偏置不同数值并存的误导）；
- 正确候选的概率从 **0.333 → 0.999**（机制上确实推高了 P(正确)）；
- 优势全程有限（`isfinite` 断言）；
- 退化组锁定：组内全对/全错（std=0）时优势应≈0 → 冻结更新（断言）；
- 退出码 0、打印 `[PASS]`。

**Step 5｜best-of-N / rejection sampling** — 在 SFT 弱策略上 `best_of_n_accuracy` 显示 `N=1 0.315 < N=4 0.795 < N=16 0.998`（N 越大成功率越高；`N=1` 即 Step 2 打印的 SFT 基线，二者用同一 300-trial 估计器，数字一致不重复），并对近乎满分的 GRPO 策略打印对照（已饱和、提升幅度自然变小）。

Run（CPU 秒级）：

```bash
python m11_verifiable_rl/code.py
python -m m11_verifiable_rl.code
```

## Key Design Decisions

- **选可验证算术，而非开放式问答**：让"谁来判对"有一个**客观、可复现**的答案 —— `eval` 就是验证器，无人介入、不可被 hack。这是与 m01–m10（人类偏好）的分水岭。
- **outcome reward**：用最干净的 0/1 展示"验证器当奖励源"；真实推理才需 process reward（README 里点明区别）。
- **含噪声 SFT 起点（P(正确)≈1/3）**：组内 A/B 混合，GRPO 才能学到"把正确抬起来"。若 SFT 已经确定地错，同组全 0，std=0 → 优势≈0 → 训不动（这正是 v1.0 里"困难过滤/惩罚"要处理的 cold-start，见 Going Deeper）。
- **GRPO 不引入 Value 网络**：只用一个 `PolicyModel`，直接对比 m04–m06 的 PPO 需要额外 Value + GAE，突出"组内相对优势消掉 v-Critic"的核心。
- **best-of-N 演示放在弱策略上**：GRPO 训练后策略已近乎满分，best-of-N 处处饱和看不出"随 N 上升"的单调性，故把"随 N 递增"的演示放在有真实散布的 SFT 弱策略上（更诚实地展示这一机制），再用已接近满分的 GRPO 策略打对照。

## Going Deeper

- **RLVR 与 DeepSeek-R1**：当模型必须靠**求解**得到可由执行器验证的答案（代码、带格式的数学推导）时，把**格式奖励**加上**问题级可验证奖励**喂给 RL，模型会长出"在可验证任务上内部探索"的行为 —— 这就是 R1 里"推理 RL"（RLVR）的雏形。本模块的 outcome reward + GRPO 正是其最小单元。
- **从 outcome → process**：单步算术用不上 process reward；真实的长链推理往往两者叠加：`总奖励 = w_out·outcome + w_step·process_step`。过程验证器更难写（要判"这一步推导对不对"），是需要另找一台批注/验证机的工程挑战。
- **rejection sampling / best-of-N 的耦合度**：把 `best_of_n_accuracy` 里"采样 N 个、经验证器挑一个已验证正确的"的那一套，其实就是在做 **rejection sampling**（拒绝验证不过的解、接受验证通过的解）→ 可在线**收集 SFT 数据**（对应 RFT / R1 的思想）。它与"改策略"互为正交：采样技巧 + 训练可以叠加，本模块分别演示。
- **curriculum / difficulty filtering**：当策略对"难题"已确定不会做对，同组全是 0，GRPO 更新不了 → 需要先是数据工程：按难度梯度排序、过滤掉太难的 prompt（v1.0 里"curriculum、difficulty filtering"）。本模块用"含噪声 SFT"跨过这一坑。

## 模块定位

这是 **v1.0 —— "可验证奖励与推理强化学习"** 的教学实现：把 m01–m10 的"人类偏好/大模型 Judge" 换成**精确执行器/验证器**作为奖励源，并用 **GRPO（组内相对优势）** 省掉 Value 模型、用 **best-of-N / rejection sampling** 再榨正确率。它和"纯人工偏好 RLHF" 已经是不同的范式（RLVR），从 m10（用验证器合成离线偏好 + DPO）到本模块（用验证器直接给在线 0/1 + GRPO）是一脉：把"验证器可用性"推向可训练、可推理、可在线提升。

版本：**v1.0** · 运行：`python m11_verifiable_rl/code.py`（CPU 秒级）