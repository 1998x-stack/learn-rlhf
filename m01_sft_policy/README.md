# m01: 离散回答级 Policy + SFT（v0.0）

[返回根目录](../README.md)

---

## The Problem

当前的"模型"没有任何关于**回答质量**的度量。给定一个问题（prompt），模型可以吐出任意的 token 序列，但我们没有办法让它的输出倾向"好答案"。

针对一个固定 prompt，我们预先枚举出若干**候选回答**（candidate responses），把问题变成一个**离散动作选择**问题：`prompt → 选择哪个候选回答`。这样"质量"就有了意义——哪个答案更好，就该给它更高的概率。

但模型最初对所有候选回答一视同仁。它需要一个**先知的引导**：先学习"人类/高质量示范是什么样的"。这正是 `versions.md` §3.1 里 **SFT（监督微调）** 的作用：

- 教模型理解指令格式；
- 建立基本回答能力；
- 将策略限制在合理回答区域；
- 为后续采样提供较高质量候选；
- 作为 RLHF 中的参考模型（`versions.md` §6 里的 `π_ref`）。

一句话：SFT 是 RLHF 的"地基"。没有它，直接强化学习会因搜索空间太大而被奖励模型轻松 hack（`versions.md` §3.1）。

## The Solution

离散 Action 定义：每个 `prompt` 有若干候选回答，`sft_labels` 给定每个 prompt 应该选哪个。用一个轻量 `PolicyModel` 把 `prompt_id` 映射到候选回答的 `logits`，再用交叉熵在 SFT 示范标签上做监督微调。

```
prompt_id ──► Embedding(hidden=16) ──► policy_head(Linear) ──► logits[num_actions]
                                                                       │
                                        sft_labels = [0,1,0,1] ◄──────┤
                                                                       ▼
                                                    CE Loss = -Σ log π(y* | x)
                                                                       │
                                                    Optimizer (Adam) 迭代
                                                                       ▼
                                            SFT 后的策略（概率集中在示范答案）
```

## How It Works

**交叉熵（CE）损失**就是标准语言模型的分类损失，在离散 Action 场景下退化为一次 softmax 分类：

$$ \mathcal{L}_{SFT} = -\sum_i \log \pi_\theta(y_i^* \mid x_i) $$

- `x`：prompt（离散 prompt_id）；
- `y*`：SFT 示范答案（sft_label 下标）；
- `π_θ`：`PolicyModel` 输出的 softmax 概率。

我们故意构造了含少量噪声的 SFT 数据（`versions.md §5`：label `[0,1,0,1]`——第 1、3 个 prompt 的示范并非最佳答案），好让后续模块演示"SFT 学到偏好不完美"的现象。

**参考策略的意义**：训练完成后的 `PolicyModel` 就是 RLHF 里的参考模型 `π_ref`。后续 RL 阶段会限制 `Policy` 不要偏离它太多,否则模型会在奖励驱动的指导下遗忘基本能力、输出退化文本（reward hacking）。这一步把"回答质量"这个概念第一次**可训练化**——这是后面所有 RLHF 组件的前提。

## Code Walkthrough（版本锚点 `# v0.0`）

**Step 1｜数据块（config + data）**— 4 个 prompt、每个 3 个候选回答；导出 `num_prompts=4`、`num_actions=3`、`prompt_ids`、`sft_labels=[0,1,0,1]`。这是 `versions.md §5` 的配置与数据基准。

**Step 2｜`PolicyModel`** — `Embedding(num_prompts, 16)` 得到隐向量，再 `Linear(16, num_actions)` 得到每个候选答案的 `logits`。`forward(prompt_ids) → logits[B, num_actions]`。

**Step 3｜SFT 训练循环** — 对每步：前向 → `F.cross_entropy(logits, sft_labels)` → 反向 → Adam 更新（`lr=1e-2`），共 60 步。

**Step 4｜自验证 + `print_policy`** — 断言 `logits.shape == (4, 3)`；断言最终 SFT loss < 1.0（证明" loss 下降"不是空话）；打印 `最终 SFT loss`；把每个 prompt 的候选概率逐个打印出来（`P=0.9954 | 2。`……）。退出码 0 且打印 `[PASS]`。

运行：

```bash
python m01_sft_policy/code.py
```

## Key Design Decisions

- **离散 Action 而非 token 级**：把"回答"抽象成单个离散动作，用一个 `Linear` 头完成。这是教学上最简自包含的起点（极简版 `v0.0`）；后续 `m05` 才会升级到 token 级语言模型。
- **`prompt_embedding` 直接索引 `prompt_id`**：因为是离散类别，直接查表即可，不需要测字级编码器。
- **含噪声 SFT 标签**：故意在 prompt 0/2 选择"非最佳"答案作为示范，提前预习"SFT 数据也可能存在噪声"。
- **`sys.path` 剔除本目录 + `from __future__ import annotations`**：由于本文件叫 `code.py`，直接运行时 torch 内部的 `import code` 会撞上标准库 `code` 模块；先把本文件所在目录从 `sys.path` 移除，保证 `torch` 正常导入（参见根 `AGENTS.md` 说明）。

## Going Deeper

- **真实 RLHF 的 SFT 数据**：规模、质量与覆盖度都极高（人工标注 / 蒸馏示范），且要覆盖指令多样性。本模块只是 4 条的"教学玩具"。
- **SFT 是必要前序**：没有 SFT，强化学习搜索空间过大，奖励模型极易被利用（§3.1）。SFT 先用最小代价把策略推到合理区域，RL 再在局部求精。
- **参考策略 `π_ref`**：这里的模型正是后面约束的锚点。`m03` 的 KL 项、`m04` 的 PPO、`m09` 的 DPO 都会用到 `π_ref`。

## 模块定位

这是 `learn-rlhf` 序列的**第一个真正写码模块**（`v0.0`）：建立离散 Action 级 `Policy` 并完成监督微调，产出可直接观测概率的参考策略。它是 m02（Reward Model 学习偏好）、m03（REINFORCE + KL）、m04（PPO MVP）…… 全部后续模块的**概念地基**。从这里开始，"生成的概率"第一次与"质量"挂钩。

版本：**v0.0** · 运行：`python m01_sft_policy/code.py`（CPU 秒级）
