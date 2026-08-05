# m08: 分布式/生产式 RLHF 架构（v0.7）

[返回根目录](../README.md)

---

## The Problem

**如何进行 FLOP 数十亿参数级的大规模 RLHF，仍然让训练"稳、可续、可跨越机器"？**

m01–m07 已经把 RLHF 的**算法**打通：SFT → Reward Model → REINFORCE/PPO → token-level GAE → 多目标奖励。但坑在于：毫尖端实验室《`versions.md` §9 v0.6》“生产级 RLHF”列出的是一整套**系统架构**，而不是某一个算法：

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

这些"生产件"里，教学最容易忽略的坑是：

- **rollout 和训练没有解耦**：单进程里"边采样边更新"。真实系统里 rollout（推理：多 GPU / vLLM 服务）和训练（PPO：另一批 GPU）是**不同进程、不同设备**，二者之间必须有一个**缓冲**做异步数据管道。
- **β 写死在代码里**：`versions.md §11.3` 说过，KL 太小 → reward hacking / 模式坍塌；KL 太大 → RLHF 失去效果。生产上用一个**自适应 KL 控制器**动态调 β，而不是手焊常数。
- **没有 checkpoint**：几十万 GPU 时训练一旦掉线就全部重来。生产必须把 policy + optimizer + step 落盘成 checkpoint，支持断点续跑。

> 一句话：m08 不再发明一个新 RL 算法，而是把 m01–m07 的 RL 核心包一层**生产系统件**（回放缓冲、自适应 KL、checkpoint），并把 rollout 与训练解耦成"生产者/消费者"流水线。教学上在**单 GPU 上仿真**这套架构。

## The Solution

用 m04 的离散 PPO 核心，外面罩上四块生产组件，组成一个可运行的**单 GPU 生产闭环仿真**：

```
                ┌──────────────────────────────────────────────┐
                │            PRODUCTION RLHF LOOP               │
                └──────────────────────────────────────────────┘
                                                                
  Policy π_θ ──► [Rollout Worker]  采样 N 条 (prompt, action)
                      │
                      ▼
                [RM / Reward Evaluator]  r = reward(prompt, action)
                      │
                      ▼
                 [ Replay Buffer ]  push(prompt, action, log_prob, kl, reward, value)
                      ▲
                      │                      采样/解耦：trainer 只管 push、不必阻塞 rollout
   Policy ◄──── [ PPO Trainer ]  sample_batch(minibatch) → PPO 更新 → 更新 π_θ
       ▲             │
       │             ▼
       │      [ Adaptive KL Controller ]  实际KL > 目标 → 增大 β；< 目标 → 减小 β
       │             │
       └─────────────┴────► (periodic) [ Evaluation ] → [ Checkpoint save/load ]
```

数据流向是**生产者—消费者**：

```text
   rollout worker（生产者）         buffer（队列）             PPO trainer（消费者）
   policy 采样 N 条回答  ─────────►  push()   …  sample_batch()  ──────►  用 minibatch 更新 policy
   RM 打分, 记 log_prob/kl         (cap 限长, 丢最旧)                   并据此把新 policy 交回给 rollout
```

四个核心组件：

1. **`ReplayBuffer`**：bounded FIFO。rollout worker 只 `push`，PPO trainer 只 `sample_batch`。`cap` 用不到时丢弃最旧样本，防止缓冲无界增长。这就是生产里 rollout 与训练进程间 data channel 的"零拷贝直译"。
2. **`AdaptiveKLController`**（`versions.md §11.3`）：维护一个 `β`。
   ```
   实际 KL > 目标 KL  →  增大 β（多惩罚 → 拉回向 ref → KL 降）
   实际 KL < 目标 KL  →  减小 β（少惩罚 → 允许发散 → KL 升）
   ```
   β 用乘法式比例调节（`β ← β·(1 + kp·err)`，`err` 是近期平均 KL 相对目标的偏差加以 clamp），并夹在 `[β_min, β_max]`，防止一次失控。
3. **Checkpoint**：`save_checkpoint` 把 `{model_state_dict, optimizer_state_dict, step, config}` `torch.save` 到 `checkpoints/m08_ckpt.pt`；`load_checkpoint` 用 `load_state_dict` 恢复。round-trip 必须**字节级复现**（同一输入 logits 全等）。
4. **周期评估 + 断点续跑**：每若干迭代打印受奖候选 0 的概率（真实质量代理），最后落盘 → 用全新实例 load 回来 → 断言 logits 一致、恢复的 optimizer 能再走一步。

## How It Works

**为什么 rollout 与训练解耦能加速生产？** 采样（rollout）只吃推理，PPO 更新只吃训练。两者对设备、批量的需求完全不同（推理要吞吐、训练要显存/通信）。解耦后可以让**很多 rollout worker 同时产生经验**，慢慢灌进 buffer，训练 side 按自己的节奏消费——这正是分布式系统的本质：生产者不需要等消费者，反之亦然。单 GPU 版里，这演成同一进程内"先 rollout 一批 feed buffer，再从 buffer sample 一个 minibatch"，但解耦的**因果结构**完全保留。

**自适应 KL 平滑和收敛**：离散单步里，`kl = logπ_θ(a) - logπ_ref(a)` 对采样到的候选 a 估计。它瞬时噪声大（某个 prompt 卷到 action0，其它不卷），如果拿**瞬时 batch mean**去跳 β 会震荡甚至形成"极端双稳态"（β 大到 crush 策略 → KL 归零；消一消 → 策略又卷成一 → KL 爆炸）。所以控制器**追的是近期平均 KL**（EMA），再按它相对目标偏差驱动 β——这正是 §11.3"追踪近期平均 KL"的正解。结束时，近期 KL EMA 稳定在目标 1.0 的邻域（实测 ≈ 0.71），β 也从 1e-3 自适应爬到 ≈ 0.7；受奖候选 0 概率从 SFT 的 0.50 升到 0.67——训练"有效且受控"。

**checkpoint 为什么能"元级等价"**：`state_dict` 含 Adam 的动量和方差一阶/二阶矩。`torch.save` 是整棵 state 的 pickle，`load_state_dict` 复原后**权重与优化器状态都完全等价**，因此轻微一步 step 都能恢复，行为不漂移。断点续跑就是 `save(step) → load(step+1)`。

## Code Walkthrough（版本锚点 `# v0.7`，同 versions.md §9 v0.6）

**Step 1｜离散核心（复刻 m04）** — `PolicyModel / RewardModel / ValueModel`。SFT 用带噪标签（prompt 0/2 指向正确候选、1/3 指向错误），得到"半对半错"的初学策略，作为 Reference（KL 锚点）。

**Step 2｜`Rollout` + `ReplayBuffer`** — `Rollout` 是单条经验（`prompt_id, action, log_prob, kl, reward, value`）；`ReplayBuffer.push` 压入、超 `cap` 丢最旧，`sample_batch(size)` 均匀采样并把字段 `torch.stack` 成一个批量 `Rollout`——形状断言 `(17,)` 验证。

**Step 3｜`AdaptiveKLController`** — `update(mean_kl)` 内做 EMA + 比例律调整 `beta` 并 `clamp`。`trace` 记录 `(kl, beta)` 手感。

**Step 4｜`save_checkpoint / load_checkpoint`** — 落盘与恢复封装；`config` 里带 `target_kl` 元数据。

**Step 5｜生产主循环** —

```python
for update in range(PPO_UPDATES):
    rollout_step(buffer, ROLLOUT_SIZE)   # 生产者：采样→RM→push
    batch = buffer.sample_batch(MINIBATCH) # 消费者：取一个 minibatch
    rewards        = batch.reward - batch.kl * beta    # kl-penalized reward
    advantages     = normalize(rewards - batch.value)  # 单步 advantage
    for _ in range(PPO_EPOCHS):
        ratio   = exp(policy(a) - batch.log_prob)
        policy_loss = -mean(min(ratio·adv, clip(ratio)·adv)) - ε·entropy
        policy_optimizer.step(); value_optimizer.step()
    batch_kl_mean  = batch.kl.mean()
    beta = kl_controller.update(batch_kl_mean)         # 自适应 KL 驱动 β
    if update % 25 == 0: 评估 & 打印
# 落盘 + round-trip 验证：[PASS]
save_checkpoint(policy, policy_optimizer, step, {…})
loaded = new PolicyModel(); load_checkpoint(loaded, new_optimizer)
assert (policy(ids)-loaded(ids)).abs().max() < 1e-6     # logits 全等
loaded_optimizer.backward().step()                       # 恢复的优化器可续训
```

**Step 6｜[PASS] 断言** —

1. Buffer：`sample_batch(17)` 各字段形状 `(17,)`、KL 有限。
2. Checkpoint round-trip：同输入 logits 最大差 `<1e-6`（实测 `0.00e+00`）；恢复的 optimizer 能再 step（续训可行）。
3. CK 元数据：`ckpt_meta["step"] == PPO_UPDATES`。
4. KL：近期平均 KL 落在目标 ×[0.5,2.0] 邻域（实测 0.71）且 max < 3.0（不爆炸）；`β` 确实被自适应调节（1e-3 → 0.70）。
5. RL 有效：候选0概率 SFT 0.50 → RL 0.67。

运行：

```bash
python m08_production_rlhf/code.py
```

## Key Design Decisions

- **一条 `ReplayBuffer` 写死了"生产者-消费者"边界**：仿真真实分布式 rollout⇄训练的两个进程通过 buffer 传经验，且用 `cap` 限长丢最旧——这既是教学点睛（为什么生产系统要 buffer），也让单 GPU 版诚实反映缓冲语义。
- **自适应 KL 用 EMA + 比例律，而非 bang-bang 硬切**：离散 batch 的瞬时 KL 噪声大，直接用"KL>target 就 β×1.5、KL<target 就 β÷1.5"会在 β 上限/下限来回打钟，形成 β 双稳态极限环（策略一会儿崩回 ref、一会儿崩成一）。EMA 追踪"近期平均 KL"（正对应 §11.3）后，调节平滑、能落在目标邻域。这是本模块最容易翻车的角落。
- **β 的 `[β_min, β_max]` 动态范围要匹配 reward 强度**：若奖励梯度 >> β·KL 上限，策略会无视 β、直接崩到受奖动作（KL 爆炸）；反过来 β 上限太大又瞬间把策略钉死回 ref（chosen 掉回 0.5、KL≈0）。本模块把 Reward Model 训到"温和"水平（80 轮 BT），并设 `β_max≈3`，才在离散任务里获得一个真实、可平衡的稳态点。
- **checkpoint 保存 state_dict + optimizer_state**（权威），而不是只存原始权重：恢复后才能继续 step，才是真正的"续跑"，而不仅仅是"载权重推理"。
- **sys.path 剔除本目录**（同 m01–m07，规避 `code.py` 遮蔽标准库 `code`）。

## Going Deeper

到这里，本模块作为 `learn-rlhf` 在学习算法层面已经收束完毕（用单 GPU 仿真了生产架构）；真实生产版的"真·分布式"全都是上面简化原件的依原放大：

- **真正的 multi-GPU rollout**：用 vLLM / SGLang 启动**推理服务器**（不同进程），多路并发 `generate` 拆 prompt 投喂；replay buffer 变成跨进程的 message queue / Ray Data 等。
- **PPO 训练并行**：Policy + Reward + Value 塞进同一个 GPU 的**显存（memory）复用**，或分开 DDP；参数更新用 **ZeRO / FSDP**（`torch.distributed.fsdp`），把模型分片到 N 卡。
- **Reference / Reward / Value 的模型并行**：Reference 冻、Reward 冻，只有 Policy 和 Value 的 head 需要训练，推理时可与 rollout 分开卡。
- **`ReplayBuffer` 在生产里还要存 token 级经验**：语言模型 rollout 的 "action" 是整条 response 的所有 token，buffer 存 `(tokens, logprob, reward, mask)`；`m06` 的 GAE 在 train 端做个 batch 计算。
- **自适应 KL 的变体**：本模块用 EMA + 比例；生产里还常用 piecewise（命中率区间采样）或 AdaFactor/超参搜索。`versions.md §11.3` 的"增大/减小 β"是基本原理，本模块给了最通用的平滑实现。
- **更多生产件**：reward normalization、多奖励加权（`m07`）、在线监控、checkpoint 版本管理、分布式通信——都是 v0.9+ 的内容。

## 模块定位

`learn-rlhf` 的算法版图在 m07 已经完整（SFT→RM→PPO→GAE→多目标）。m08 把同一套 RL 核心**包进生产级系统壳**（v0.7）：**（缓冲、自适应 KL、checkpoint 断点续跑、周期评估）**，让"研究能跑"升级成"production 可靠"。它是浅出层：把分布式/多 GPU/vLLM/FSDP 都留在 Going Deeper，只把**架构理念**（解耦、自适应、可恢复）在单 GPU 上仿真出来。

m07（v0.6 多目标）是 m08 的前位概念——m07 引入多奖励与硬约束，m08 则示范如何让这套复杂引擎具备"暂停重来"（checkpoint）与"自动收稳"（adaptive KL）的能力；之后的 m09（v0.8 DPO）会对比"在线 rollout"与"离线偏好"两种工程权衡。

版本：**v0.7** · 运行：`python m08_production_rlhf/code.py`（CPU 秒级）