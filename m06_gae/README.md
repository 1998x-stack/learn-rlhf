# m06: GAE + 完整 Token-level PPO（v0.5）

[返回根目录](../README.md)

---

## The Problem

m05（v0.4）已经能跑通完整的**字符级 TinyGPT + token-level PPO**，但它的优势估计用的是 **cheap n-step return**：`returns_t = r_t + γ·returns_{t+1}`，然后 `Advantage = returns - value`。

这个简单折现方法有两个让真实 RLHF 吃不消的问题（`versions.md` §9 v0.5、§11.6）：

1. **只折现、不做 value 基线消偏**：n-step return 只把末 token 的奖励 `γ` 折现回溯，并没有用 value model 估计的"期望累计奖励"去当基线做 advantage 校正。前面的 token 拿到的是"折扣后的整体奖励"，而不是"这一步比期望好多少"。
2. **方差大、bias 不好控**：单靠折扣因子 `γ` 无法在"估计偏差（bias）"和"抽样方差（variance）"之间精细调节。真实 RLHF 里奖励稀疏、rollout 噪声大，一个糟糕的优势估计会让 PPO 一会儿 push 一下 pull 一下，训练发散。

另一个工程层面的缺失（v0.5 要补齐）：**value clipping**、**advantage whitening（标准化）**、**reward whitening**，以及 response 长度 mask / EOS 处理的正确性约束。这些是"真正能在生产里用"的 PPO 必备件。

一句话：m05 用 `n-step return` 做 `Advantage`，m06 换成**正式的理论优势估计 GAE（Generalized Advantage Estimation, Schulman et al., 2016）**，并补上 v0.5 的工程安全带。

## The Solution

用 GAE 取代 n-step return，同时保留 m05 的全部 token-level 机制：

```
TD error:   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
GAE:        A_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}
```

其中 `V` 来自（每次采样时刻冻结的）旧策略 value head 在**每个 response token 位置**的输出，`r_t` 是 m05 的序列奖励分配（`-β·KL_t`，末 token 叠加 RM score），`bootstrap` 在单回合终局取 0。

```
Token-level rollout
     ↓
逐 token 的 value V(s_t)（旧策略，冻结）
     ↓
δ_t = r_t + γV(s_{t+1}) - V(s_t)    ← TD error
     ↓
A_t = Σ_l (γλ)^l δ_{t+l}             ← GAE advantage（前向公式）
     ↓
advantage whitening → value clipping → token-level PPO
```

与 m05 相比，多出的核心区别就是：**用 TD-error 打底、用 λ 做 bias-variance 调节、在每一步用 value 校正**，而不是简单地把末 token 奖励折现回去。

## How It Works

**TD（Temporal Difference）误差** `δ_t = r_t + γV(s_{t+1}) - V(s_t)` 是"单步 TD 之惊喜"：它量的是"真实拿到这一步的奖励 + 估计的下一步价值"与"估计的当前价值"之差。`δ_t` 为正，说明这一步的表现比价值模型的预期好，应该把该步的概率抬高。

**GAE** 把所有后续步的 TD 误差用 `(γλ)` 的幂加权求和：

$$ A_t^{GAE}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta_{t+l} $$

两个系数各司其职：

- **γ（折扣）**：用于"价值延迟多远"。`γ<1` 把未来奖励贴现到一个有限的现值，避免无穷回报发散，也降低了长时间的累积方差。
- **λ（GAE 的平衡旋钮）**：控制**bias-variance 权衡**。当 `λ→0`，GAE 退化成只看当前步的 TD(0)：bias 低（不用估计很远的未来）、但单步噪声带来的方差高；当 `λ→1`，GAE 退化成 Monte-Carlo（等价于把整条轨迹的奖励全部加到当前步）：利用了更多真实奖励所以**bias 更低 / 逼近真实回报**，但奖励噪声被完整累积、**方差更高**。`λ` 越小越"短路"（牺牲一点信息换稳定），越大越"长视"（更准但更抖）。实际常取 `λ≈0.95, γ≈0.99`，这里取 `γ=0.9, λ=0.95` 便于 CPU 教学演示。

**为什么 token 级 value bootstrap 能压低方差（对比 m05 的 n-step/MC）**：m05 的 n-step return 把末 token 的奖励一路乘 `γ^l` 折现回去，只有奖励本身驱动；而 GAE 的每一步都"预减去 `V(s_t)`、再预加 `γV(s_{t+1})`"，即用价值函数把"可预期的部分"剔掉，只让“surprise（TD error）”进入梯度。奖励的很大一部分是天气性的、可被价值预测的部分，减去它之后剩下的优势方差自然更小。这就是 v0.4→v0.5 的核心动机。

**Value clipping（原版 OpenAI PPO）**：把"预测的 value"夹到 `[V_old - ε, V_old + ε]`，再取其与 unclipped 的平方误差中更大的一侧作为 value loss：

```
value_pred_clipped = V_old + (V_new - V_old).clamp(-ε, +ε)
value_loss = max(MSE(V_new, returns), MSE(value_pred_clipped, returns))
```

防止单次更新把 value 推得离旧值太远，避免 value 与优势互相喂出一个"炮筒状的"不可控目标。

**Advantage whitening**：对整个 response-token 集合把 GAE 减均值除方差，缩到 `N(0,1)`，统一不同 token / 不同 sample 的奖励尺度——和 m04/m05 用同一个技巧，但这里是对 GAE 结果做标准化。

**Reward whitening**：`token_reward` 先缩放到零均值、单位方差再喂进 GAE，避免奖励绝对值尺度（`+1` 的 RM reward vs 很小的 `-βKL`）差异干扰 TD error。

**EOS / terminal 处理**：响应序列是固定长度、单回合（episode）——最后一个 response token 是我们能看到的最后一个状态，其后无更多步，于是 `bootstrap = 0`（terminal episode）。这是 `versions.md` §11.6 里"value bootstrap 越过 EOS"这条最大的坑的显式处理。

## Code Walkthrough（版本锚点 `# v0.5`，同 versions.md §9）

**Step 1｜数据 + TinyLM（同 m05）** — 词表 `0-9`、4 个 prompt、带噪 SFT（`'12'/'23'` 各给对/错一条示范），SFT 后正确 target 概率 `≈0.75`。`TinyLM` 共享主体上分 `lm_head` + `value_head`，`response_log_probs` 与 `response_values` 用同一个因果偏移约定。

**Step 2｜`compute_gae_loop`** — 自后向前推进：

```python
for t in reversed(range(T)):
    v_next = values[:, t+1] if t+1 < T else bootstrap    # 末位置用 0
    delta = token_reward[:,t] + gamma*v_next - values[:,t]
    deltas[:,t] = delta                                  # δ_t
    acc     = delta + (gamma*lam)*acc                     # 反向累积
    advantages[:,t] = acc                                # A_t = δ_t + γλ·A_{t+1}
```

**Step 3｜`compute_gae_closed`** — 用**加和定义** `A_t = Σ_{l}(γλ)^l δ_{t+l}` 逐步展开成另一条独立实现（向量化张量），作为 sanity 重算路径。两条路径应 `atol<1e-4` 一致。

**Step 4｜训练循环** — rollout 采样 → 冻结 old policy → 算 `token_reward`（序列奖励广播 `-βKL + RM`）→ reward whitening → GAE → advantage whitening → 一致性断言 → token-level PPO（ratio/clip/min）→ value clipping 的 value loss → 合并 backward、分别 step。

**Step 5｜[PASS]** — 断言：

1. GAE advantage 与 TD error 均 `torch.isfinite(...).all()`（无 NaN/inf）；
2. `GAE==recompute`：loop 版与闭式重算最大绝对差 `< 1e-4`；
3. 正确 target 平均概率**上升**（`0.75→1.00`）；
4. greedy 解码准确率 `>0.75`（实测 `1.000`）；
5. `total_loss` 下降（实测 `3.448→0.000`）。

运行：

```bash
python m06_gae/code.py
```

## Key Design Decisions

- **GAE 用"未标准化的原始版"做一致性校验**：advantage whitening 会把 GAE 缩到 `N(0,1)`、改变幅值，所以 `loop==recompute` 断言必须在**标准化前**的 `gae_raw` 上做，否则 `atol=1e-4` 永远过不了。这是这个模块最容易弄错的角落。
- **value 的 returns 用原始 GAE 还原**（`returns = gae_raw + old_values`），而不是用标准化后的 advantage——否则 value 拟合目标就被"调幅"污染，价值函数失去与真实回报的对应。
- **value clipping 用原版 OpenAI 形式（对预测 clamp、取 max 加强）**：直接 clamp target 在 `[V_old±ε]` 过于激进，会让"较难的 prompt"（如 `'23'→'09'`）的价值更新被锁死而停在错误区（调试中发现会导致 `SFT=0.5→RL=0.0` 的跑偏）；`clamp(pred)` 的 max 形式更稳定，既限幅又不至于禁止 value 追上目标回报。
- **`bootstrap=0` 是 terminal episode 的显式选择**：固定长度、到末 token 即终局，后面既无新 reward 也无新 value，`V(s_{T+1})=0`。这直接对应 `versions.md §11.6`"value bootstrap 越过 EOS"这条坑的正面答案。
- **因果偏移沿用 m05 的约定**：`values` 与 `log_probs` 同取 `[PROMPT_LEN-1 : PROMPT_LEN-1+RESPONSE_LEN]`，保证 GAE 的 `V(s_t)` 和 `V(s_{t+1})` 在同一个"将要生成第 t 个 response token"的状态上定义。
- **sys.path 剔除本目录**（同 m01–m05，规避 `code.py` 遮蔽标准库 `code`）；一次 backward 合并共享主体的 policy+value 梯度后分别 step。

## Going Deeper

- **λ vs γ 的分工**：γ 决定"多长", λ 决定"多依赖价值 vs 多依赖真实奖励"。真实生产常用 `γ=0.95~1.0, λ=0.95`；把 λ 拉到 1 就是 MC 行为，拉到 0 就是 TD(0)，能直观观察 bias-variance 变化。
- **advantage whitening 是"工程必要"，不是"理论必要"**：GAE 的相对排序已足够指导 PPO，但不同 scale 会让 `clip_epsilon` 行为不一致；标准化后 clip 的 `±1±0.2` 才有统一含义。
- **value clipping 与 advantage 标准化一起用**：如果只用 clip 而不用标准化，梯度的绝对幅值会漂移；两者搭配是全套稳定 PPO 的常见握手件。
- **这是 m07 多目标的桥梁**：多 reward / 多 λ 的加权、以及 adaptive KL 都在后续模块登场；这里只处理**单一 return + GAE** 的干净拼图。
- **更本质的**：GAE 的价值在于"**用 bootstrapped value 把低频、可预测的不确定性从高维、高方差的 MC 回报中剥掉**"——这是 Reinforce（v0.2）→ PPO（v0.3→v0.4）→ GAE（v0.5）一脉相承的改进主线。

## 模块定位

这是 `learn-rlhf` 里把 PPO 从"能跑"推到"**可用/对齐**"的关键一步（`v0.5`）：m05 已经完成 token 级自回归 + token-level PPO，但 n-step return 方差大、无法调节 bias-variance；m06 用**正式的 GAE** 取代之——以 TD-error 在 token 级做 value bootstrap、以 λ 旋钮平滑，并补齐 **value clipping / advantage-reward whitening / terminal bootstrap** 三项工程件，最终以 `GAE==recompute(atol<1e-4)`、优势有限、正确概率上升、loss 下降四条 `[PASS]` 自证。

m05 是 m06 的直接地基：把 m05 的"廉价 n-step" 换成 GAE，就进入 v0.5。

版本：**v0.5** · 运行：`python m06_gae/code.py`（CPU 秒级）