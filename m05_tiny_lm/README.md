# m05: 字符级 Tiny LM + Token-level PPO（v0.4）

[返回根目录](../README.md)

---

## The Problem

m04 的 PPO 已经能跑通完整的 RLHF 闭环，但它是**离散回答级**的：每个"动作"就是选中一个候选回答，Value 每个 prompt 只给一个标量。真实的语言模型 RLHF 不是这样的——模型是**自回归地一个 token 一个 token 生成整个回答**，而"哪个 token 该往哪个方向改"才是 PPO 真正要优化颗粒度。

要在 token 级做强化学习，会冒出一堆 m04 没有的新问题（`versions.md` §9 v0.4）：

1. **mask**：输入是 `[prompt token][response token]`，PPO / KL / value 只能作用在 response token 上，绝不能把 prompt token 也优化（`versions.md` §11.6 第一条坑）；
2. **token-level log-prob**：`log π_θ(y|x) = Σ_t m_t·logπ(y_t|…)`，是**每个 response token 的 log-prob 之和**，而不是一个整体的 action log-prob；
3. **token-level KL**：每个 response token 都要跟 frozen reference 比 `logπ_θ - logπ_ref`；
4. **序列奖励分配**：序列级 reward function 给的是"整个回答"一个分数，怎么分到每个 token 上？这里把它**只广播给最后一个 response token**，其余 token 只承担 KL 惩罚；
5. **reward 广播 + value head**：PPO 需要一个**每个 token 都预测标量**的 Value/Critic。

一句话：m04 的"回答 = 一个 action"在 m05 变成了"回答 = 一串 token，每个 token 是一个 action（step）"，PPO 的坐标轴从"回答序号"变成了"自回归生成时的每一个 token"。

## The Solution

用一个**字符级 Tiny GPT**（一层因果自注意力 + LM head + Value head）完成从离散到 token 级的升级：

```
Tokenizer (char)
     ↓
TinyGPT (因果自注意力, 只能看到自己及之前的 token)
     ↓
Auto-regressive Rollout：逐 token 生成 response（固定长度）
     ↓
规则奖励函数：整个 response 和正确 target 比对，给 0/1
     ↓
Token-level PPO（response mask / token KL / 序列奖励广播 / n-step returns）
```

任务：给定两位数字组成的 prompt（如 `"12"`），模型必须自回归拼出对应的两位"目标码"（如 `"08"`）。SFT 对其中两个 prompt **故意给带噪示范**（正确和错误各一条），于是策略对它们"半对半错"；Token-level PPO 用"答对才给奖"把它们全部推向正确目标码，从而演示**token 级 RLHF 把 SFT 的不完美纠正过来**。

## How It Works

**因果自注意力 = 自回归语言模型的核心**。`TinyLM` 用一张下三角 mask 让第 `t` 个 token 只能看到 `≤t` 的自己及之前所有 token——这样 logits at position t 预测的是"下一个 token是哪"。这正是语言模型与"离散 action 分类器"的本质区别：**输出是一个变长、相互依赖的 token 序列**，而 m04 是给定 prompt 挑一个固定候选。

**数据 Mask（response mask）**。把 `[prompt tokens][response tokens]` 拼成一个序列喂进模型：

```text
[prompt token][prompt token][resp token][resp token]
           mask=0             mask=1         mask=1
```

只在 mask=1 的 response 位置计算 PPO loss、KL、value loss 和 advantage。prompt token 绝不参与优化。

**Token-level log-prob**。PPO 需要知道"当前策略给这条已采样 response 的每个 token 打多少分"。对采样到的 token，按位置取 `log π_θ(y_t | x, y_<t)`，response 部分加起来就是序列 log-prob：

$$ \log\pi_\theta(y\mid x)=\sum_{t} m_t\,\log\pi_\theta(y_t\mid x,y_{<t}) $$

**Token-level KL**。对每个 response token，比较当前(旧)策略与参考策略在**同一个已采样 token** 上的 log-prob：

$$ \text{KL}_t=\log\pi_\theta(y_t\mid s_t)-\log\pi_{\text{ref}}(y_t\mid s_t) $$

**序列奖励分配（reward broadcast）**——`versions.md` §9 v0.4 的关键机制：

```text
token 1 (resp 首位):   reward = -β·KL₁
token 2 (resp 末位):   reward = -β·KL₂ + Rule Reward ← 序列奖励广播到最后
```

这里的精确匹配规则为"整个回答"打 `+1`/`0`，我们把它**加到最后一个 response token** 上；其它 response token 只拿 KL 惩罚（`-β·KL_t`），防止策略偏离 SFT 的合理语言分布。它扮演序列级 reward function，不是一个训练出来的 Reward Model；真实 RM 的偏好学习见 m02/m04。

**Token-level PPO**。得到每个 response token 的 `returns`（这里用 `cheap n-step`：`returns_t = r_t + γ·returns_{t+1}`，把末 token 的序列奖励折现回溯给前面的 token，让前面的 token 也分享到"答对"信号；正式 GAE 见 m06）。`Advantage = returns - value`，做 token 级标准化，然后对 response 的每个 token 算 PPO ratio 和 clip。

## Code Walkthrough（版本锚点 `# v0.4`，同 `versions.md` §9）

**Step 1｜数据 + token 化** — 词表 `0-9`（10 个数字字符）；4 个互不重复的 prompt，每个对应一个唯一目标码（`prompts`/`TARGETS`）；`WRONG_RESPONSE="55"` 是与任何合法目标都不冲撞的错误示范；`sft_responses_of()` 为带噪的 2 个 prompt 各出（正确、错）两条示范，其余只出正确示范。

**Step 2｜`TinyLM`** — 字符 Embedding + 位置编码 + 一层因果自注意力 + MLP；共享主体之上分两个头：`lm_head`（→每个位置下一 token 的 logits）与 `value_head`（→每个位置的标量 value，Critic）。

**Step 3｜SFT（v0.0）** — 用 `prompt + 示范 response` 的上下文做教师强制，对 response 位置的 logits（**注意因果偏移**：`logits[p-1]` 预测位置 `p`）算交叉熵。带噪 prompt 学到正确和错误各 50% → 不确定；其余 prompt 学确定。打印 SFT 正确概率 `≈0.75`。

**Step 4｜冻结 Reference** — `copy.deepcopy(policy)` 关掉梯度，作为 KL 锚点。

**Step 5｜Token-level PPO 循环**：

- Rollout：在当前策略下自回归采样每个 prompt 的 response（`rollout()` 固定 `RESPONSE_LEN` 步）。
- 冻结 old policy：`copy.deepcopy` 采样时刻的策略，作为 ratio 分母。
- 计算 `old_logp / ref_logp`（response 位置）→ `tok_kl` → 序列奖励分配（末 token 加 `raw_reward`，其余只 `-βKL`）。
- `returns = n-step`；`advantage = returns - old_values`；token 级标准化。
- `ratio = exp(curr_logp - old_logp)`，`clip`，PPO loss；Value 用 `MSE(values, returns)`。
- policy loss + value loss 共享主体，合并一次 backward 后由同一个 optimizer step。

**Step 6｜[PASS]** — 断言：正确 target 平均概率从约 0.75 **显著上升**（固定种子下超过 0.95）、greedy 解码准确率为 1.0、total loss 下降；打印 `[PASS]`，退出码 0。

运行：

```bash
python m05_tiny_lm/code.py
```

## Key Design Decisions

- **因果偏移索要是最容易错的地方**。`logits[b,t]` 预测位置 t+1 的 token，所以 response 第 k 个 token（位于 p=r(A)+k）由 `logits[b,p-1]` 预测。SFT 交叉熵、`response_log_probs`、old/current log-prob、KL **全部**用这个约定。写错一个就会像 `versions.md` §11.6 说的"不报错但训练方向完全错"。
- **序列奖励只广播给最后一个 token**：忠实还原 v0.4 的机制；让前面的 token"承压"去与 ref 对齐（KL），最后的答案 token 承接精确匹配奖励。配合 `n-step return` 让前面的 token 也能折现分享奖励。
- **`target_demo` 与合法 target 不冲撞**：错误示范用 `"55"`，不和任何正确目标码重叠，避免"同一个 token 既是对的又是错的"这种自相矛盾的监督，保证 SFT 能被清晰记忆、RLHF 能干净纠偏。
- **带噪 SFT 制造"半对半错"的可修正起点**：对两个 prompt 各给正确/错误一条示范 → 模型对它们不确定而非"确定地错"，这样 reward=1 的样本自然出现，PPO 才能把概率抬起来。这更贴近"SFT 数据有噪声"的真实情况。
- **policy 与 value 共享主体，也共享一个 optimizer**：两项 loss 来自同一次前向，合并后只 backward/step 一次。`value_head` 本来就是 `policy.parameters()` 的子集；若再交给第二个 Adam，它会在一轮里被两套状态重复更新。
- **sys.path 剔除本目录**：本文件叫 `code.py` 会遮蔽标准库 `code`（同 m01–m04）。

## Going Deeper

- **这正是 m06（GAE）的铺垫**。这里的 `n-step return`（`returns_t = r_t + γ·returns_{t+1}`）只做了简单折现；m06 会用 TD-error 和 GAE 做更精确的 multi-step advantage、value clipping、return 对齐。
- **`response mask` 是与 padding 的坑**：真实 RLHF 里 response 长度不一，如果 padding token 也被算进 PPO/奖励，训练方向就会骗走（`versions.md` §11.6 第 2-4 条）。这里用固定 response 长度保证了 mask 一致。
- **sparse reward 的 cold-start**：若 SFT 对某个 prompt"自信地错"，正确答案可能采不到，reward=1 永远不出现 → 训不动。工程上靠 SFT 高质量 + noisy demo + 过程奖励/GAE 缓解。本模块用带噪 SFT 制造"不确定起点"绕开它。
- **Value / Advantage drift**：token 级 advantage 标准化（`(A-mean)/std`）与 m04 相同，用来统一不同 token/不同 prompt 的奖励尺度。
- **KL 只是一个 sample 级估计**（`old_log - ref_log` 在同一 token 上），真正的 KL 散度需对分布积分，这只是单样本 MC 估计，够教学；自适应 KL 等留待 m08。

## 模块定位

这是 `learn-rlhf` 里**第一次真正做到 token 级自回归语言模型 RLHF**（`v0.4`）：把 m01–m04 的"离散 action 选择"推广为"字符级 TinyGPT 在 Response 上逐 token 生成 + PPO"。它完整演示了 v0.4 的四大新机制——**response mask、token 级 log-prob/KL、序列奖励广播、token-level PPO**——并以 `[PASS]` 证明强化学习真的把"半对半错的 SFT 回答"纠正成了"受奖励的正确回答"。它是 m06（GAE/token 级完整 PPO）的**直接地基**：把这里的 `n-step return` 换成 GAE，就进入 v0.5。

版本：**v0.4** · 运行：`python m05_tiny_lm/code.py`（CPU 秒级）
