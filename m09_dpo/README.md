[返回根目录](../README.md)

# m09: 离散回答级 DPO 离线偏好优化（v0.8）


---

## The Problem

m03–m08 的经典 RLHF 路线都要**在线展开**一整条强化学习流水线：

- m02 先训练一个 **Reward Model**（BT 偏好建模）；
- m03/m04/m05/m06 用 RM 给**临时采样的回答（rollout）**打分，算 advantage；
- 还要一个 **Value / Critic** 网络估计 baseline，并加 PPO clip、GAE 等一堆机制。

在线组件多、工程复杂、且推理阶段还得单独部署一套 RM。有没有可能**只用已经标注好的离线 (chosen, rejected) 偏好数据，就直接把 Policy 优化好**，而**完全不需要 RM、不需要 rollout、不需要 value**？

`versions.md §9（v0.8）` 说得很清楚 —— 有，这就是 **DPO（Direct Preference Optimization，Rafailov et al., 2023）**。

## The Solution

核心洞察：**偏好对比所需的"奖励差"其实可以只用策略本身（当前策略 + 参考策略）的此刻概率直接表达**，不需要单独训练一个 RM 去近似它。DPO 把"偏好→奖励→策略"三步合并成一步：直接把策略推向"更偏好 chosen、更讨厌 rejected"。

```
SFT 先训练一个初始 Policy
        │
        ▼
  π_ref (冻结参考策略, 不再更新)
        │
        ▼
离线偏好对  (prompt, y_chosen=0, y_rejected=1)
            (prompt, y_chosen=0, y_rejected=2)
        │
        ▼
  直接优化 Policy π_θ   ←── 无 RM / 无 rollout / 无 Value
        │
        ▼
  chosen 概率上升、rejected 概率下降
```

DPO 损失（`versions.md §9` 原文）：

$$
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
$$

其中 `y_w` = chosen，`y_l` = rejected，`π_ref` 是冻结的参考策略（通常取 SFT 模型），`β` 控制离开参考的力度。

## How It Works

**DPO 的推导直觉**（一句话）：RLHF 里"奖励最大化"的最优解有一个**闭式**形式 —— 最优策略与参考策略之比正比于奖励的指数：

$$
\pi^*(y\mid x)
\propto
\pi_{\text{ref}}(y\mid x)
\exp\big(\tfrac{r(x,y)}{\beta}\big)
\;\Rightarrow\;
r(x,y)=
\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\ +\ \text{const}
$$

把这个 "`r` 的表达式" 代回到 Bradley–Terry 偏好模型：

$$
P(y_w \succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)
$$

奖励模型那一步就被**消掉了**：直接用 `log(π_θ/π_ref)` 的差代替 `r` 差，得到的正是上面的 DPO 损失。所以：

- **DPO 不需要 RM**：等式两边同时消掉 RM，偏好信号直接落在策略上。
- **DPO 不需要 rollout**：只用离线数据里**已存在的** chosen/rejected，不必为了拿 reward 而命令策略在线采样新回答。
- **DPO 不需要 Value**：没有优势估计，没有 TD/GAE，损失就是普普通通的一个 log-sigmoid，反向传播跟 SFT 一样简单。

**数值实现**：直接按公式展开会出现 `log(1 + exp(z))`，当 `z` 很大时可能溢出。本模块改用恒等等价且稳定的：

$$
\mathcal{L}_{DPO}
=
- \texttt{logsigmoid}\big(\beta \cdot (\text{log}\pi_w - \text{log}\pi_l)\big)
$$

其中
`logπ_w = logπ_θ(y_w|x) − logπ_ref(y_w|x)`，
`logπ_l = logπ_θ(y_l|x) − logπ_ref(y_l|x)`（严格来说取的是**对数比值**，代码里直接用各自 log 概率相减）。

## Code Walkthrough（版本锚点 `# v0.8`）

**Step 1｜数据与 SFT 初始化** — 复用 m01 的离散数据集：`num_prompts=4`、`num_actions=3`、`sft_labels=[0,1,0,1]`（含噪声地把 prompt 1/3 导向了错误的答案 1）。用 `sft_pretrain()` 训一个起点 `policy`，再 `load_state_dict` 复制成 `reference_policy` 并 `requires_grad_(False)` 冻结 —— 这就是 `π_ref`。

**Step 2｜离线偏好对** — `preference_pairs` 是 `(prompt_idx, chosen, rejected)`，每个 prompt 两条：`chosen=0` vs `rejected=1` 和 `chosen=0` vs `rejected=2`（candidate 0 才是正确答案）。

**Step 3｜DPO 训练循环** — 对每一步：用当前策略的 `log_softmax` 减参考的 `log_softmax` 得到 `logπ_w / logπ_l`，累计 `-logsigmoid(β·(logπ_w − logπ_l))` 作为 `dpo_loss`，`loss.backward()` + Adam 更新。**只有 `π_θ` 的梯度**：`π_ref` 全程冻结、无 RM 参数、无 Value、无采样。

**Step 4｜自验证 + 打印** — 断言 shape 与最终的 loss<1.0；打印每个 prompt 的 chosen/rejected 概率 `SFT 基线` vs `DPO 之后`；断言：

- `chosen(candidate 0)` 每个 prompt 概率均**上升**（含 SFT 曾推错到答案 1 的 prompt 1/3）；
- `rejected(candidate 1 & 2)` 每个 prompt 概率均**下降**。

运行（CPU 秒级）：

```bash
python m09_dpo/code.py
python -m m09_dpo.code     # 从仓库根目录运行也可以
```

## Key Design Decisions

- **只用离线偏好对，在线组件全部清零**：这是 DPO 与 m03–m08 最本质的分水岭。RM/rollout/Value 全部不需要 —— 这正是 `versions.md §9.3` 强调的"工程复杂度低"。
- **SFT 只做部分收敛（30 步）而非训练到 99% 置信**：因为 SFT 标签含噪声（prompt 1/3 被导向错误答案 1）。若 SFT 把策略死死压到错误答案上，DPO 要从"参考已经极度偏向错误答案"的起跑线把它拉回，会慢很多。留一点柔韧性，DPO 的"纠偏"课程效果更干净。
- **`-F.logsigmoid(...)` 而非 `log(1+exp(...))`**：稳定性（avoid overflow），这是 RLHF 工程里非常常见的数值细节。
- **`sys.path` 剔除本目录 + `from __future__ import annotations`**：同 m01，因文件名是 `code.py` 会 shadow 标准库 `code`，需在 import torch 前剔除（见根 `AGENTS.md`）。

## Going Deeper

- **DPO 与 PPO-RLHF 的取舍**（`versions.md` 优/缺）：

| DPO（本模块） | PPO-RLHF（m03–m08） |
|---|---|
| 离线：只用固定偏好数据 | 在线：需要采样/RM/value |
| 无 RM/Value | 训练 RM + Value |
| 工程简单、训练近似 SFT | 机制复杂、超参数多 |
| 不能主动探索新回答 | 能通过采样探索新回答 |
| 覆盖不足时难发现新策略 | 可在线试错 |

- **DPO 的性质**：严格来说，DPO 属**偏好优化**而非**经典的 PPO-RLHF**（`versions.md §9.3` 明确指出），但常作为 RLHF 的工程替代方案，两者可互补。m10（RLAIF）与 m11（可验证奖励/GRPO）会继续在这条线上展开。
- **真实 DPO**：大规模下常用偏好数据经逐 token 掩码计算 log-prob、按 batch 掩码规约处理；`β`、SFT 质量、数据清洗是主要性能杠杆；更多进阶见 `versions.md §9.1–§9.3`。

## 模块定位

这是 **v0.8 ——"DPO 类离线偏好优化"** 的教学实现：在 m01（SFT）+ 偏好对基础上，证明**去掉 RM、rollout、value 组件仍然可以对齐偏好**。它是"在线经典 RLHF"（m02–m08）到"离线偏好优化 / 可验证奖励"（m10–m12）之间的关键转折点。

版本：**v0.8** · 运行：`python m09_dpo/code.py`（CPU 秒级）