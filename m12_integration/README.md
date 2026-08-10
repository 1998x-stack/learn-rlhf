# m12: 收束 —— 端到端 RLHF 全链路 + 深坑清单（SFT → RM → PPO/KL → DPO → Best-of-N）

[返回根目录](../README.md)

---

## The Problem

前面 m01–m11 每个模块都只讲 RLHF 流水线的一环：SFT 给了带噪起点，Reward Model 学偏好，REINFORCE / PPO 用奖励去更新策略，DPO 用偏好对直接改策略，RLAIF 换了个奖励来源，Verifiable RL 换成机器可验证的 0/1 奖励。

但**单独跑通每一环 ≠ 真正懂 RLHF**。真正的困难不在任何单一环节，而在把整条链串起来时，那些"看起来能跑、结果全错"的实现细节。`versions.md §11` 总结了这十条坑：

1. **Prompt token 被加入 PPO loss** —— 位置对奖励的累计方向直接错。
2. **padding token 被计算奖励** —— 无行为意义的占位符拿到 reward。
3. **EOS 后 token 未 Mask** —— value bootstrap 越过 response 边界，n-step return 失真。
4. **chosen response 长度不同但未归一化** —— DPO / BT 的 log 概率差没法公平累加。
5. **Reference log-prob 使用错误 tokenizer** —— 参照系的 logp 和策略根本不对齐。
6. **old log-prob 在 PPO epoch 中被重新计算** —— 老策略漂移，ratio 整体崩塌。
7. **KL 符号写反** —— 本该 `-β·KL` 惩罚越界，写反后反而鼓励模型偏离参考策略。
8. **reward 广播到了所有 token** —— 末位评分被加到整个 response 上。
9. **value bootstrap 越过 EOS** —— bootstrap 目标取到了响应之外的 value。
10. **advantage 没有 detach** —— 梯度经由 baseline 泄漏回去。

这些问题**通常不会报错，只会让训练方向完全改变**——比崩溃更难发现。

本模块的目的就是：在一个文件里端到端串起整条流水线，把 `versions.md §11（RLHF 最容易踩的坑）` 的清单**编码成活跃断言**。今后任何一步改动若悄悄把某个坑改回来，启动即会输出 `[深坑未过]` 并 `SystemExit`——而不是静默地训练到错误方向。

## The Solution

用与 m03–m11 相同的**字符级算术环境**（`a+b=?`，`V` 字符、`PROMPT_LEN` 长 prompt、`RESPONSE_LEN` 长答案、定长无 EOS），把整条 RLHF 流水线压缩成一个可独立运行的文件：

```
valid prompt（a+b=?）   +  pref (chosen, rejected) 偏好对（正确 / 错误答案）
   │
   ▼
Stage 1 — SFT (v0.0)  在 SFT 答案上 CE 训练，得到带噪的不完美起点 P(target)≈0.75
   │
   ▼  （冻结 Reference Policy —— PPO/KL 与 DPO 的共享锚点）
Stage 2 — Reward Model (v0.1) Bradley–Terry，在偏好对上学会 chosen>rejected，冻结
   │
   ▼
Stage 3 — PPO + KL (v0.4) :旧 logp 冻结 / response-mask / -β·KL / advantage detach / 序列奖励广播
   │         P(target) 约 0.74 → 0.87（RM reward 驱动，verifier 只评估）
   ▼
Stage 4 — DPO (v0.8) 离线偏好直接调 : -logsigmoid(β·(logw_chosen - logl_rejected))，P(target)→0.999
   │
   ▼   （per-stage 总结表：SFT / RL-PPO+KL / DPO 三列对比）
Stage 5 — Verifier / Best-of-N (v1.0) 真实采样 N 候选 + 验证器筛选：SFT 弱策略上 N=1 < N=4 < N=16 单调上升
   │
   ▼
深坑清单验收（versions.md §11）：10 条坑编码为 7 条活跃断言，全过 → [PASS]
```

最小闭环是：**SFT 起步 → (RM + Reference) 冻结 → PPO+KL 在线调 → DPO 离线备选项 → 验证器 Best-of-N 兜底**。五阶段各用一个目标函数，机制级 `[检查]` 把 `versions.md §11` 的关键坑逐一钉死。

## How It Works

一段典型的 RLHF 从单一「Q：`a+b=?`」环境出发，走五步：

| 阶段 | 版本 | 目标 | 输出 |
|---|---|---|---|
| **SFT** | v0.0 | 让策略先能产出 SFT 带噪答案 | `P(target)≈0.75`（不完美起点） |
| **RM (BT)** | v0.1 | 学偏好：chosen>rejected | `rm_acc≈1.0`，冻结 |
| **PPO+KL** | v0.4 | 冻结 RM score + ratio-clip + `-β·KL` | `P(target)` 从约 0.74 提升到约 0.87 |
| **DPO** | v0.8 | 无 RM，`-logsigmoid(β·(logw-logl))` | `P(target)→0.999` |
| **Verifier/Best-of-N** | v1.0 | 可验证正确性 + 推理时采样 | N=1 0.7x < N=4 0.9x < N=16 0.999（SFT 弱策略单调）；训练后已饱和 |

### SFT：带噪起点，而不是完美

`versions.md §9 迭代路线` 里 v0.0（SFT）建立了参考策略。这里特意只用 400 步 CE loss 训出 `P(target)≈0.75`——一个**不完美的起点**。因为 RLHF 的价值恰恰是把不完美的起点推向目标区域；起点若太确定（≈1.0），RL 就没东西可学了。[检查] 直接断言 `P(target)<0.9`。

### Reward Model：偏好对打分，而不是逐 token 广播

RM 是独立的 `RewardModel`（内部复用 TinyLM 的 value head），只在输入的**整段**上给一个 score。PPO rollout 真正调用冻结 RM；精确匹配 verifier 只记录训练是否提升真实正确率，绝不混入 policy loss。关键约束是：**RM 只负责打整段分数，逐 token 的 reward 分布由 PPO 的"序列奖励广播"单独决定**。

### PPO + KL：以 mask 为护栏，KL 符号不可反

PPO 的目标是：
```
response mask 只覆盖 response 位置（深坑①）
token_reward = 末位 RM评分广播到 response 末 token（深坑⑧）
            − β·KL(π ‖ π_ref)          （深坑⑦：-β 而不是 +）
old_logp 冻结，不重算                （深坑⑥）
同一个 response 上用 ratio & clip 稳定更新
advantage = target - old_value，恒 detached（深坑⑩）
entropy bonus 防collapse（=0.001）
```
定长 + 无 EOS/padding 的 `RESPONSE_LEN` 约定（`check("response 定长")`）、`chosen/rejected 长度一致` 断言把深坑②（padding 计数）与 ④（长度未归一化）并案按掉。三个 PPO epoch 内自取样 rollout，旧对数概率**冻结**，`[检查]` 断言 `old_logp.requires_grad == False`；KL 检查不依赖 sampled log-ratio 的正负，而是直接验证单独保留的 `kl_penalty == -β·(old_logp-ref_logp)`；advantage 恒 detached。Policy trunk、LM head 与 value head 共享一个 optimizer，避免 value head 被两套 Adam 重复 step。

### DPO：删掉 RM 与 rollout，直接用偏好

DPO 是最小化的偏好优化：不需要奖励模型，也不需要在训练中 rollout。用**同一份冻结的 SFT reference（共享锚点）**算 π_ref，目标函数只有 `-logsigmoid(β·(log πθ(chosen) - log π_ref(chosen) - (log πθ(rejected) - log π_ref(rejected))))`。它把 policy 拉向 chosen、离开 rejected——但要注意**前提是 chosen/rejected 长度一致（已归一化）**，否则 log 概率差没法可靠累加（深坑②）。

### 总结表：把三条线并置对比

收束前打印 per-stage 汇总表，列：

```
stage      P(target) mean     greedy_acc
SFT        0.7378             0.500
RL/PPO+KL  0.8741             1.000
DPO        0.9993             1.000
```

这列（SFT → RL/PPO+KL → DPO）就是整条链路的进度地图：`P(target)` 一路上涨，`greedy_acc` 保持高稳，最后用 v1.0 验证器兜底。

### Best-of-N：不训练也还能再榨

最后一段（v1.0）做**真实的推理时采样**：对同一 prompt 采 N 个候选，用精确验证器挑出"已验证正确"的那个；N 越大"至少采到一个正确解"的成功率越高（`1-(1-p)^N`）。为了让"随 N 单调上升"诚实可见，把演示放在有真实散布的 **SFT 弱策略**上（打印 `N=1 < N=4 < N=16`），再把已收敛（近乎满分、已饱和）的 DPO 策略打出来做对照。这就是 m11（可验证奖励）的核心手段，也是沿 v0.0→v1.0 的收尾。

## Code Walkthrough

文件 `m12_integration/code.py`（可执行到 [PASS]，按 `[Stage]` 分隔、每阶段一行统计）：

1. **`sys.path` 自护**（MANDATORY）——本文件名为 `code.py`，运行时会以所在目录作为 `sys.path[0]`；而 torch 内部会 `import code`（标准库 `code` 模块）。这段把本目录从 `sys.path` 剔除，# 避免标准库 `code` 被本文件**遮蔽**导致导入崩溃。必须在任何 `import torch` 之前执行。
2. **`TinyLM`/`RewardModel`**——字符级小模型；`build_context` 把 prompt 与 response 拼成一段；`target_prob`/`greedy_acc` 分别写对「正确回答概率」与「贪婪解码准确率」。

3. **Stage 1/5 SFT（v0.0）**——交叉熵带 SFT 答案，`P(target)` 落入 0.75。冻结 `reference_policy`。
4. **Stage 2/5 Reward Model（BT，v0.1）**——训练、冻结 RM；`rm_acc` 打印偏好准确率。
5. **Stage 3/5 PPO + KL（v0.4）**——`batch_size=128`、`ppo_epochs=3`、KL β=0.1、clip 0.2；rollout 同时记录 RM score 与 verifier 0/1，但只有 RM score 进入 reward；单 optimizer 更新共享 policy/value；两条长度检查 + 五条深坑断言锁住梯度与 mask 边界。
6. **Stage 4/5 DPO（v0.8）**——用同一冻结 reference 做 π_ref，DPO_loss 滚动；`[DPO-standalone]` 打出 SFT→DPO 的 P + greedy。
7. **per-stage 总结表**——打印 `stage / P(target) mean / greedy_acc` 三横三列。
8. **Stage 5/5 Verifier / Best-of-N（v1.0）**——确定性验证器 + 真实 best-of-N（每 prompt 采 N 个候选、任一命中即成功）；在 SFT 弱策略上演示 N=1 < N=4 < N=16 单调上升，并打印训练后（已饱和）策略作对照。
9. **深坑清单（versions.md §11）验收**——七条活跃断言逐条打印 `[检查] … : 通过`；任一失败则列出 `[深坑未过] …` 并以非 0 退出。全通过打印 `[PASS] … end-to-end 收束`。

运行：

```bash
python m12_integration/code.py        # 直接运行（已避开 code.py 遮蔽）
python -m m12_integration.code       # 从仓库根目录运行亦可
```

### 逐条断言 ↔ 坑的对应

| `[检查]` 名称（code 里） | 对应深坑 |
|---|---|
| `response 定长（无 padding 逃生）` | ② padding 计数 |
| `chosen/rejected 长度一致（DPO/BT 归一化前提）` | ④ 长度未归一化 |
| `SFT 做出带噪的不完美起点` | v0.0（起点保证，非坑） |
| `RM（可在偏好对区分 chosen/rejected）` | v0.1（BT 收敛，非坑） |
| `advantage 已 detach（无梯度泄漏）` | ⑩ |
| `old_logp 冻结且不重新计算` | ⑥ |
| `RM 奖励只落在最后 response token` | ⑧ 奖励广播 |
| `KL 符号正确（reward 的 KL 项严格等于 -β·log-ratio）` | ⑦ |
| `Prompt token 不进入 PPO loss（位置mask prompt区=0 且 response 全覆盖 + logp 宽度=RESPONSE_LEN）` | ① |

> 坑③（EOS 后未 Mask）、⑤（reference 用了错误 tokenizer）、⑨（value bootstrap 越过 EOS）由**定长 + 无 EOS/padding** 的结构约定在代码层面一并兜住，故未单列活跃断言；把 `RESPONSE_LEN` / reference 来源改坏时，前述长度与 logp 对齐断言会先失败。

代码级：每条坑都能退回、立刻被抓回来。

## Key Design Decisions

- **一个文件五阶段**：不拆子模块，是为了让"串起来"本身成为被练习的对象——每阶段以 `print` 打一行，末尾 `[PASS]` 作为整链通过的单一判定。
- **定长 response（无 EOS/padding）**：`RESPONSE_LEN` 全程固定，让 response mask / 长度归一化 / value bootstrap 这些坑在**结构上**就不可行（长度断言兜住）。教学设计价值高于极简实现。
- **RM 只给整段 score、分布由 PPO 广播**：m02（偏好打分）与 m04/m05（token 奖励分配）两个概念在**代码里物理分离**，避免"末位评分被所有 token 共享"这一最反直觉的坑在整链上复活。
- **PPO epoch 共用同一份旧 rollout**：old 对数概率在 epoch 前一次性算出并冻结，天然不再有"过 epoch 重算"的空间。
- **Best-of-N 在弱策略上演示单调性**：训练后的策略已近乎满分，best-of-N 处处饱和看不出"随 N 上升"（与 m11 相同）。故把"随 N 递增"的演示放在有散布的 SFT 弱策略上（更诚实地展示验证器筛选这一机制），再用已接近满分的最终策略打对照——展示"不更新策略也能提正确率"，与改进策略正交；沿 v0.0→v1.0 提供一条可验证奖励推理时的收尾。

## Going Deeper

- 试着把 `versions.md §11` 的每条坑单独**改回去**（如 `token_reward` 换成 `+ KL`），再运行 code.py——观察启动即输出 `[深坑未过]`、退出码非 0，而非 `[PASS]`。这是比读源码更深的「手把手验证」。
- 想继续沿 v1.0 走：在 Stage 5 把 single-pass 采样扩成 `best-of-8/16`，观察正确率 / 出货率曲线；再换成 process reward（每 step 可验证），对照 outcome vs process 的差别。

## 模块定位

- **前序**：`m01_sft_policy`(v0.0) → `m02_reward_model`(v0.1) → `m03_reinforce`/`m04_ppo_mvp`/`m05_tiny_lm`→`m06_gae`(v0.5) → `m07_multi_objective`(v0.6) →`m08_production_rlhf`(v0.7)→`m09_dpo`(v0.8)→`m10_rlaf`(v0.9)→`m11_verifiable_rl`(v1.0)。
- **本模块（m12）**：整链收束。不新增算法，把 v0.0→v1.0 的每一环在同一文件里串跑，并把 `versions.md §11（RLHF 最容易踩的坑）`的 10 条坑清单编码为即时断言。
- **学习路径**（`versions.md §12 建议的学习实现顺序`）：这条路上，m12 建议放在 m01–m11 之后跑——只有亲眼见过每一环，才谈得上把它一条线串起来再让「坑」自己暴露。

[返回根目录](../README.md)
