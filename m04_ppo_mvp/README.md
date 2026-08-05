# m04: PPO + Value Model（离散 PPO MVP）(v0.3)

[返回根目录](../README.md)

---

## The Problem

m03 用 REINFORCE + KL 已经能让策略往奖励高的方向走。但 REINFORCE 有个工业级硬伤：**它直接用采样到的奖励作为"该往哪个方向走"的标签**，而采样奖励方差极大，更新方向抖动剧烈；只要一次采样碰巧拿到高奖励，策略就会被粗暴地拽向那个动作，容易步子太大、甚至把前面 SFT + KL 约束一起毁掉。

同时，m03 里 **Policy 更新用的是同一份 rollout**（`policy 自己采样的 log_prob`）——每次更新后就变了，样本很快过期，样本利用率低。

PPO（Proximal Policy Optimization，`versions.md` §3.3）要解决两件事：

1. **更新要"稳"** ——每步更新不许把策略改太多（clipping + KL 限制）；
2. **样本要"复用"** ——引入一个冻结的 `old policy`，让同一次 rollout 可以反复优化多轮（epoch），旧样本还能用。

再加上一个 `Value Model`，用 `Advantage = Reward − Value` 代替裸 `Reward`，**减去方差来源**，让梯度信号更干净——这就是本期加入第 4 个模型的原因（`versions.md` §4 的"四模型"完整版）。

这是本教程的**最小 RLHF 闭环**：SFT → Reward Model → Reference → Value → PPO 一次跑通。

## The Solution

在 m02 已训好的 Reward Model 与 m01 已训好的 Policy 基础上，新增一个 **Value Model**，用单步环境的 `Advantage = Reward − Value` 作为优化信号，并实现完整的 PPO 训练循环：

```
Policy(动作选择)                    Value Model / Critic
      │ Sample action a ──► Reward Model ──► r                        │
      │                                                              │
      │  old_policy(冻结, 本轮采样用)         ──► V(s)                │
      │        │                                                        │
      │   ratio = exp( logπ_θ(a) − logπ_old(a) )                        │
      │   A = r − V(s)  （再标准化 (A−mean)/std）                        │
      │        │                                                        │
      │   L = −min( ρ·A, clip(ρ,1−ε,1+ε)·A ) − entropy_bonus            │
      └────────┴─────────────────────────────────► Adam 更新 Policy
```

四个模型分工（对照 `versions.md` §4）：

| 模型 | 是否训练 | 作用 |
|---|---:|---|
| Policy Model | ✅ | 生成回答，PPO 优化对象 |
| Reference Policy | ❌ 冻结 | `sft` 后的副本，KL 约束锚点 |
| Reward Model | ❌ 冻结 | 为候选回答打分 |
| Value Model | ✅ | 预测期望奖励，算 Advantage |

## How It Works

**为什么需要 `old policy`？** 策略是概率分布，`π(a)` 随参数一直变。要让"一个样本被用好几轮"（分母 `π_old(a)` 与分子 `π_θ(a)` 相对），必须把采样那一刻的策略固定成 `old_policy`，每轮 rollout 冻结一次（`copy.deepcopy` + 关闭 `requires_grad`），本轮的多次 epoch 都基于同一份 `old_policy` 的样本。

**概率比（probability ratio）**是 PPO 的核心：一个动作在旧策略下的概率是 `π_old(a|x)`，当前策略下是 `π_θ(a|x)`，二者比值：

$$ \rho_t(\theta) = \frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)} $$

- `ρ ≈ 1`：新旧策略对该动作概率近似，更新平滑；
- `ρ >> 1`：当前策略把这个动作概率抬高了很多，PPO 会**裁剪它**，防止步子太大。

**裁剪目标（clipping）**：

$$ \mathcal{L}_{PPO} = -\mathbb{E}\left[\min\left(\rho_t A_t,\ \operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\right)\right] $$

取 `min` 的作用：当 `A` 为正（好动作）时，就算 `ρ` 涨到 2，也只用 `1+ε=1.2` 对应的梯度，**放弃那部分收益**去换取稳定；当 `A` 为负（坏动作）时，`clip(ρ,1−ε,1+ε) = 0.8` 兜底，防止策略因一个坏样本退到不可逆错误。这就是 PPO 经典的"该多少给多少，超了不要"。

**Value baseline 与 Advantage**：单步环境下 `Advantage = Reward − Value`。`Value` 是"期望奖励"，`Reward − Value` 衡量这个动作**相对于期望高了多少**——它的均值接近 0，方差比裸 `Reward` 小得多。这正是 m04 相比 m03 REINFORCE 的关键改进：用价值估计当 baseline 做"减均值"，从根源上压低梯度噪声。

**Advantage 标准化**：`advantages = (advantages − mean) / (std + 1e−8)`。这一步把整批 advantage 缩放到 `~N(0,1)`，统一了不同 prompt 的奖励尺度，避免"某个 prompt 天然高分就把梯度带偏"；`+1e−8` 防除零。

**Entropy bonus**：`− entropy_coef·H(π)` 当作正则项加在目标里，鼓励动作保持一定多样性，防止过早坍缩到单一回答，缓解探索不足。

**KL 约束**：与 m03 一致，`reward = raw_reward − β·(logπ_θ−logπ_ref)`，对动作抽样级 KL 估计（`versions.md` §3.3）。没有它，策略会为了奖励直接跑到 reward-hacking 上去（输出语义乱码但高分）。

## Code Walkthrough（版本锚点 `# v0.3`，同 `versions.md` §6）

**Step 1｜数据** — 与 m01–m03 相同的 4 prompt × 3 候选；`sft_labels=[0,1,0,1]`（带噪）构造"不完全正确的 SFT Policy"；`preference_pairs` 作为人类偏好 `(0比1好, 0比2好)`。

**Step 2｜三个模型** — `PolicyModel`（prompt→action logits）、`RewardModel`（prompt+action→标量）、新增 `ValueModel`（prompt→期望奖励）。

**Step 3｜SFT + 冻结 Reference**（§5）— 60 步 CE 得到噪声 SFT policy，`copy.deepcopy` 冻结成 `reference_policy`。

**Step 4｜训练 Reward Model**（§6）— 300 步 Bradley–Terry：`−log sigmoid(r_chosen − r_rejected)`；随后冻结 RM。

**Step 5｜RLHF——PPO 循环**（§7）：

- **Rollout**：对 batch 采样动作，记录 `old_log_probs`、`reference_log_probs`、`raw_rewards`，算 `sampled_kl` 与 `rewards`，Value 给出 `old_values` → Advantage 并标准化。
- **PPO 多轮更新**：对 `old_log_probs` 冻结的样本，在每个 epoch 算当前 `log_probs`→`ratio`→`clip`→`policy_loss = −min(...)−entropy_coef·entropy`；同时 Value 用 `F.mse_loss(value, returns)` 更新。
- 每 25 轮打印 `reward / kl / policy_loss / value_loss`。

**Step 6｜[PASS]**— 运行结束时打印 `[PASS] m04 PPO MVP: Policy 从带噪 SFT 转向正确回答（reward-hacking-free 经典闭环）`，退出码 0。

运行：

```bash
python m04_ppo_mvp/code.py
```

## Key Design Decisions

- **新加 Value Model + Advantage**：从"裸奖励"升级到"`reward−value`"，这是 REINFORCE(m03) 到 PPO 的点睛之笔，也是后面 GAE(m06) 的地基。
- **clipping 而非信任域**：TRPO 需要解带约束的优化问题、实现复杂；PPO 一个 `clamp` + `min` 就近似实现了"不跑太远"的约束，简单实用。
- **Advantage 标准化**：让不同 scale 的奖励在一个张量里比较、稳定训练（尤其扣完 KL 后数量级被拉平）。
- **Entropy bonus 当探索剂**：防止策略在 PPO 强力拉扯下过早坍缩。
- **sys.path 剔本目录**：因本文件名为 `code.py`，直接运行时 torch 内部 `import code` 会撞上本文件；导入前把本目录从 `sys.path` 移除（见根 `AGENTS.md`）。

## Going Deeper

- **真实 PPO 是 token 级 + GAE**：这里每个"动作"就是候选回答、Value 每提示一个标量；真实 RLHF 每 token 一个动作、每个 token 一个 Value、用 GAE 累积折扣回报（`m06`）。**GAE 是本模块 Advantage 的完整版推广**。
- **minibatch / 多 epoch**：真实 PPO 会把一个 rollout buffer 切成 minibatch 多轮扫，用 `old_policy 冻结的 log_prob` 反复复用；本模块 `ppo_epochs=4` 已在重复利用。
- **Adaptive KL**：`kl_beta` 这里固定；生产环境会按 KL 是否超阈值自动调 `β`（`m08`）。
- **Reward Hacking**：本模块的 KL 惩罚就是防 reward hacking：策略不能只贪 `raw_reward`，还得离 `π_ref` 不远，否则 KL 项把奖励拉回来。

## 模块定位

这是 `learn-rlhf` 里**第一次完整画出"最小 RLHF 闭环"**的模块（`v0.3`）：把 m01 的 `Policy`、m02 的 `Reward Model`、m03 的 KL，加 **m04 新增的 Value Model + PPO（clipping、ratio、advantage 归一化、entropy bonus）** 串成一条 `SFT→RM→RL` 的可运行链路，后验策略概率收敛到正确回答（candidate 0 ≈ 1.0），演示了标准 RLHF 从"SFT 学到偏好不完美"到"人类偏好纠偏"的完整故事。它是 `m05/m06`（token 级 + GAE）的直接地基：把这里的"离散 PPO 单步 advantage"放大成 token 级自回归 PPO 就是后面的事。

版本：**v0.3** · 运行：`python m04_ppo_mvp/code.py`（CPU 秒级）