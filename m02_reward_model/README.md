[返回根目录](../README.md)

# m02: Bradley–Terry Reward Model（v0.1）


---

## The Problem

m01 训出了一个能输出候选回答概率的策略，但策略目前**不知道哪个答案更好**。要让模型"倾向好答案"，必须先有一个**质量度量**——给每个 (prompt, answer) 一个标量奖励 `r(x, y)`。

问题在于：人类不容易说"这个答案值 3.7 分"，但**很容易比较两个答案哪个更好**——"A 比 B 好"（A ≻ B）。所以奖励模型不需要学"绝对正确分数"，只需要学**相对偏好排序**（`versions.md §3.2`）：

> RM 并不需要预测"绝对正确分数"，只需要满足 `r_phi(x, y_w) > r_phi(x, y_l)`。

一句话：把"打分的难题"转成"比较的易题"，再用 **Bradley–Terry 模型**把它变成可微的损失来训练。

## The Solution

对每个 prompt，构造偏好对 `(prompt_id, chosen=0, rejected=1)` 和 `(prompt_id, chosen=0, rejected=2)`——其中 chosen=0 是好答案（`versions.md §5`：每个 prompt 的第 0 个候选是标注正确）。用一个 `RewardModel` 给任意 `(prompt, action)` 打分，并用 Bradley–Terry 偏好损失训练：

```
prompt_id ──► Embedding ─┐
                         ├── cat ──► tanh MLP ──► r(prompt, action)
action_id ──► Embedding ─┘                                    │
                                                              ▼
             偏好对 (chosen=0, rejected=1|2)
                                                              ▼
       BT loss = -log σ(r_chosen - r_rejected)　（让胜者分数更高）
                                                              ▼
                                                  accuracy: chosen 分数 > rejected 比例
```

训练后：偏好对正确率 `accuracy ≥ 0.95`，打印每个 prompt 的分数表，候选 0 的奖励最高。

## How It Works

**Bradley–Terry 偏好模型**把"A 优于 B"的概率建模为两者奖励之差过 sigmoid：

$$ P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) $$

让标注偏好发生的概率最大，等价于最小化负对数似然：

$$ \mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) $$

要点（`versions.md §3.2`）：

- **只需要相对排序**：`-log σ(r_w - r_l)` 只依赖**分数差**。把所有分数整体平移一个常数，损失不变。所以 RM 学到的是"谁比谁好"，而不是"绝对分数"。
- **Sigmoid 把分数差映射到 (0,1)**：差值越大，胜者概率越接近 1，损失趋近 0；差值越负，损失越大。
- **对每个 prompt 用全连接 Embedding**：`prompt_embedding(prompt_id)` 和 `action_embedding(action_id)` 拼起来，过一个 `tanh` 隐藏层再压到 `1` 个标量。

## Code Walkthrough（版本锚点 `# v0.1`）

**Step 1｜数据块**— 与 m01 一致的 4 prompt × 3 candidate；构造 `preference_pairs`（8 个偏好对，每个 prompt 两个：`(id, 0, 1)`、`(id, 0, 2)`），`chosen=0` 是好答案。

**Step 2｜`RewardModel`** — `prompt_embedding(num_prompts,16)` 与 `action_embedding(num_actions,16)` 拼接成 `[B,32]`，过 `Linear(32,16)→Tanh→Linear(16,1)`，`.squeeze(-1)` 得到 `reward[B]`。

**Step 3｜训练循环** — 拆出 `chosen_action_ids` / `rejected_action_ids`，分别打分；`reward_loss = -F.logsigmoid(chosen - rejected).mean()`；Adam（`lr=1e-2`）迭代 300 步。

**Step 4｜自验证 + 输出** — 断言输出 shape `(num_prompts,)`；断言 BT loss < 0.1（证明下降）；用 `pair_accuracy` 计算 `(chosen > rejected).float().mean()` 并断言 `accuracy ≥ 0.95`；打印每 prompt 分数表，**候选 0 的奖励明显最高**；退出码 0 并打印 `[PASS]`。

运行：

```bash
python m02_reward_model/code.py
```

## Key Design Decisions

- **只训练 RM，不动 Policy**：本模块是 `v0.1`，先把"如何学习偏好"单独验证清楚；策略优化（REINFORCE/PPO）放到 m03/m04。这样 RM 是否学会排序可以被独立观测（accuracy 给出定量的证明）。
- **`num_actions=3` 的固定候选**：把 token 级文本抽象成离散回答，`action_embedding` 直接查表，得到极简自包含的起点（后续 m05 才升级到 token 级）。
- **偏好对用 `chosen=0`**：数据本身告诉模型"第 0 个候选是胜者"，所以最终 accuracy 应接近 1；这验证了 BT 机制确实从成对比较中恢复了隐藏的偏好排序。
- **`sys.path` 剔除本目录 + `from __future__ import annotations`**：本文件叫 `code.py`，直接运行时 torch 内部 `import code` 会撞上标准库 `code`；先把自己的目录从 `sys.path` 移除（同 m01，见根 `AGENTS.md`）。

## Going Deeper

- **BT loss 只学相对排序**：`r_w - r_l` 平移不变，所以 RM 分数**没有绝对意义**——真实 RLHF 里会做数值稳定性/归一化，PPO 阶段只依赖相对高低。
- **真实 RM 的输入**：真实场景 RM 输入是 token 序列（续写已生成的部分），用语言模型编码，再在顶层加 reward head——与这里的 Embedding 拼接受训本质相同，只是编码器更复杂。
- **奖励被 exploit**：RM 若学得粗糙，策略会找到它的漏洞（**Reward Hacking**），这也是为什么 RL 阶段需要 KL 约束 `π_ref`（m03/m04 用到）。

## 模块定位

这是 `learn-rlhf` 序列里**"从人类偏好到数值奖励"**的模块：把 m01 的离散回答与**成对偏好**喂给 Bradley–Terry RM，训练出 `r_phi(x,y)`。它是 m03（REINFORCE + KL，用 RM 给采样回答打分）的**直接前序**——有了（用 accuracy ≥ 0.95 验证过的）reward 信号，后续强化学习才"有东西可最大化"。

版本：**v0.1** · 运行：`python m02_reward_model/code.py`（CPU 秒级）