[返回根目录](../README.md)

# m10: RLAIF 与规则衡器合成偏好（v0.9）


---

## The Problem

m02–m08 的 RLHF 到头来都要**依赖人类标注的 (chosen, rejected) 偏好对**，m09 的 DPO 也只是把"预标注好的离线偏好"直接吃进去。但人类标注慢、贵、有噪声、难以规模化。**有没有可能把"偏好"这一环也自动化**，让机器自己（或者用一条可验证规则）替代人去"比较哪个回答更好"？

`versions.md §9（v0.9）` 说得很清楚：把部分人类偏好替换为 **AI Feedback（AI 反馈）**，由一台 **Judge Model 比较** 生成偏好数据集，再喂给 DPO / RM / PPO。而这又暴露了一个更深的问题：

> 不是所有 Judge 都一样可靠。一个只看"回答写得长不长、漂不漂亮"的 Judge，跟一个真正检查"答案对不对"的验证器，会给出截然不同的偏好 —— 后者能让 policy 学到真实正确，前者只会把 policy 带沟里。

`versions.md` 因此给了一个**验证器层级**：

```
规则验证器 > 可执行验证器 > 异源 Judge > 同源 Judge
```

本模块就用一个"答案可计算"的算术任务，把所有层级都当场演示出来。

## The Solution

选**可验证算术题**正是因为它的正确性是**机器可算的**：`3+5` 的结果是多少，`eval` 一下就知道了 —— 不需要人，也不需要另一个高斯 LM 来"猜测"哪个回答更对。于是我们可以干净地做 RLAIF：

```
可验证算术题  3+5=？
      │
      ▼
Policy 生成多个候选回答（正确/错误/长但漂亮的错误）
      │
      ├──► 规则衡器(rule)   = 硬判数字与 eval 一致 → 自动合成偏好
      ├──► 可执行衡器(exec) = 真正 eval 一遍 → 自动合成偏好
      ├──► 长度偏置 Judge   = 只留"最长最漂亮"的回答  (verbosity bias)
      └──► 同源 Judge       = 取 policy 自己最偏好的答案 (self-preference)
      │
      ▼
三路偏好数据集 (chosen, rejected) —— 全部自动合成，无人工标注
      │
      ▼
分别 DPO 训练三份 policy，打印各自的【真实可验证正确率】
```

最终对比真实正确率（验证器在测试时如何通过）：

- 训练自**规则/可执行衡器**偏好的 policy → 真实正确率最高；
- 训练自**长度偏置**偏好的 policy → 真实正确率最低（它真的被"又长又漂亮"的错误回答骗进去了）；
- 训练自**同源**（自我偏好）偏好的 policy → 和 SFT 基线差不多，毫无提升（固化自己的错误）。

这就是"哪个信号源更可信"的**可量化证据**。

## How It Works

### RLAIF 与传统 RLHF 的分水岭

| 传统 RLHF（m01–m08） | RLAIF（本模块） |
|---|---|
| 人标注 (chosen, rejected) | **机器/规则**合成 (chosen, rejected) |
| 标注慢、贵、有主观噪声 | 快、低成本、可规模化、可审计 |
| 人类目标 = 源目标 | 目标来自验证器，可能与人类目标偏 |

RLAIF 的流程（`v0.9` § 原文）：

```
Policy 生成多个回答
      │
      ▼
Judge Model 比较           ←── 本模块里 Judge 就是"验证器/启发式"
      │
      ▼
Preference Dataset          ←── 自动标注，零人工
      │
      ▼
DPO / Reward Model / PPO    ←── 本模块用 DPO（复用 m09 损失）
```

**为什么选算术？** 为了让第一步"生成回答后谁来判对错"有**客观基准**。算术答案可以由 `eval` 算出来，所以`rule_reward / executable_reward` 能无条件给出正误，这正是"规则 >可执行"这一层的力量 —— 而人写的数值正是从真实里算出来的，而非"另一个模型猜的"。

### 验证器层级如何对应本代码

```
rule_reward        # 规则验证器：候选必须是紧凑整数且==真解（最强先验）
executable_reward  # 可执行验证器：eval 表达式 → 与候选里的数字比对
length_judge       # 异源 Judge（启发）：只看 len(candidate)，被"漂亮长文"骗
synthesise_same_prefs  # 同源 Judge：取 π 自己 argmax（自我偏好，固化自身错误）
```

- **规则 ≈ 可执行**：在纯算术题上两者重合（都是"数字对不对"），所以我们让它们各自对 8 个 prompt 显式自检，证明 `rule_ok = 8/8`、`exec_reward_ok = 8/8` —— 保证真的"算了"，而不是糊弄。
- **长度偏置 Judge**：`length_judge` 只回 `len(candidate)`。候选 1 是故意写长、但数字错误的"漂亮回答"，于是该 Judge 会把**错误的长回答**标为 chosen。
- **同源 Judge**：`synthesise_same_prefs(policy)` 读的就是 policy 自己 `argmax` 的偏好 —— Judge 和 Policy 共享同一套错误，所以偏好只会加固 SFT 已有的噪声，无法纠错。

### DPO 复用

训练直接复用 m09 的数值稳定 DPO 损失：

$$
\mathcal{L}_{DPO} = - \texttt{logsigmoid}\big(\beta(\log\pi_w - \log\pi_l)\big)
$$

只不过 `(chosen, rejected)` 现在是**自动合成**的。`ref` 冻结为 SFT 起点。

## Code Walkthrough（版本锚点 `# v0.9`）

**Step 0｜可验证算术数据** — `ARITH` 存 8 个 `(表达式, 真确整数)`；`build_candidates` 为每题生成 3 个候选（正确+简短 / 错误+冗长漂亮 / 错误+简短），`_candidates` 记录每个是否携带正确答案，供后续"真实正确率"打分。

**Step 1｜衡器自检 + SFT 起点** — 先让 `rule_reward` 与 `executable_reward` 在 8 个 prompt 上确认"顶层衡器确实认得出正确候选"，打印 `rule_ok / exec_ok = 8/8`。随后 `sft_pretrain` 训一个**含噪声**起点 policy（`sft_labels` 仅一半指向正确），保证真实可验证正确率在 `0.50` 左右。

**Step 2｜自动合成三路偏好** — `synthesise_rule_prefs`（规则衡器）、`synthesise_biased_prefs`（长度 Judge，用 `length_judge` 打分）、`synthesise_same_prefs(sft_policy)`（同源自我偏好）。**没有一行人力工标注**。

**Step 3｜三份 DPO** — 从同一 `sft_state` 出发，分别在规则 / 偏置 / 同源偏好集上 `train_dpo`，得到 `rule_pol / biased_pol / same_pol`。

**Step 4｜真实正确率对比 + 断言** — `policy_accuracy` 用 `.argmax` 候选是否携带正确数字。断言：

- `rule_acc > base_acc`（机制级：AI/规则偏好把真实正确率从 0.50 抬到 1.00）；
- `rule_acc > biased_acc`（量化 Judge 偏差：规则路径胜过偏置路径 1.00 vs 0.00）；
- `rule_acc > same_acc`（消除自我偏好：1.00 vs 0.50）；
- `base_acc >= same_acc`（同源不背超基线）。

运行（CPU 秒级）：

```bash
python m10_rlaf/code.py
python -m m10_rlaf.code
```

## Key Design Decisions

- **选算术，而非开放式问答**：为的是"正确性可计算"从而能**自动合成偏好**。这是 RLAIF 与 m01–m09 的本质区别 —— 把"谁来人工判对"替换为"机器可算的真值"。
- **含噪声的 SFT 起点（0.50 真实正确率）**：给它留一个真实正确率低于 100% 的起点，DPO 才能展示"把它拉到规则衡器筛出的正确"。偏置/同源路径被留下来做对照。
- **length_judge 独立于 policy（异源但偏置）**：它是"另一个信号源"，而非 policy 派生，用来演示"即使是异源 Judge，若信号是长度而非正确性，也照样被骗"，从而凸显"规则/可执行 > 挑剔的异源启发式"。
- **同源 Judge 用 policy 的 argmax 自己**：精确对应 `v0.9` 风险"自我偏好 / Judge 与 policy 共享错误"，度量结果（0.50 = 不改善）一目了然。
- **三个模型只在各自偏好集上 DPO、共享同一 SFT 起点**：保证只有"偏好来源"这个变量在变，其它全部一致 —— 公平对照。

## Going Deeper

- **RLAIF 的优点**：标注快、成本低、可大规模、可用明确原则（rule/可以审计）复核 —— `v0.9` 优势段。
- **RLAIF 的风险**（本模块都演示到）：
  - 📐 **长度/格式偏差**：Judge 偏好更长、更漂亮的回答 → `length_judge` 路径真实正确率 0.00（被 100% 骗）；-- 对应 `v0.9` "偏好长度更长、格式更漂亮的回答"。
  - 🔄 **自我偏好（self-preference）**：同源 Judge 与自己 policy 共享错误 → `same_prefs` 路径 0.50，无改善。
  - ⚠️ **Reward/生成模型共谋偏差、人类目标被 Judge 风格替代**：v0.9 另两类风险，本模块主要在 README 收列。
- **验证器层级（the hierarchy）**：当正确性可计算时，永远优先 `规则  →  可执行  →  异源Judge  →  同源Judge`。在算术/代码/数学/工具调用这类可验证领域，规则和可执行衡器是**无标注真值**；而在开放式对话领域没有这种真值，只能用异源 AI 模型当 Judge（要额外防 moral 偏差）。
- **演进**：m11（verifiable reward / GRPO / best-of-N / rejection-sampling）会把"验证器可用性"推向 outcome/process reward 与组内相对优势 —— 都是本模块验证器层级的天然延伸。

## 模块定位

这是 **v0.9 ——"RLAIF 与规则反馈"** 的教学实现：在 m09 DPO 基础上把"偏好从哪里来"从**隐式人类**换成**机器/规则/启发式 Judge**，并量化验证器层级那几个可靠性差异，证明在可计算领域**规则/可执行衡器合成偏好证大于一切启发式**。它是"人工标注 DPO"（m09）到"可验证奖励 / 推理 RL"（m11, v1.0）之间的关键一档。

版本：**v0.9** · 运行：`python m10_rlaf/code.py`（CPU 秒级）