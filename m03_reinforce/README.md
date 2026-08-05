# m03: REINFORCE + KL（v0.2）

[返回根目录](../README.md)

---

## The Problem

m01 用 SFT 让策略输出了候选回答的概率，m02 用 Bradley–Terry 训练了一个能判断"哪个回答更好"的奖励模型 `r_φ(x,y)`。但**策略还没用上这个奖励**：它只知道照着 SFT 示范复现，而示范本身是有噪声的（prompt 1、3 的示范并非 RM 眼中的最佳答案）。

现在要把奖励当成**训练信号**去推动策略——这正是 RLHF 的"强化学习"一环。最直觉的做法就是经典的 **REINFORCE**：

$$ \mathcal{L} = -\mathbb{E}\big[\, R(x,y)\cdot \log\pi_\theta(y\mid x)\,\big] $$

即：更高奖励的回答，要**提高**它被选中的概率（梯度方向为 `-R·∇logπ`）；更低奖励的回答，降低它的概率。但单独最大化 `E[r_φ]` 会让策略去**钻奖励函数漏洞**（Reward Hacking）——所以还要用 KL 把策略约束在 SFT 参考策略附近（`versions.md §3.3`）：

$$ R(x,y) = r_\phi(x,y) - \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} $$

一句话：**REINFORCE 让"reward 信号"真正推动策略，KL 防止策略在追求奖励时走火入魔**。

## The Solution

复用 m01 的 `PolicyModel` 得到策略、m02 的 `RewardModel` 得到奖励，然后把 SFT 模型**冻结**成 `reference_policy`（`π_ref`）。此后循环：

```
policy 采样一个动作 a ~ π_θ(a|x)
        │
        ├─ logπ_θ(a|x)        （当前策略，需梯度）
        └─ logπ_ref(a|x)      （参考策略，冻结，无梯度）
        │
sample-level KL = logπ_θ(a) - logπ_ref(a)
        │
reward = r_φ(x,a) - β·sampled_KL
        │
REINFORCE loss = - (reward · logπ_θ(a)).mean()
        │
        反向 → 更新政策
```

训练 400 步后，观察策略概率：SFT 示范错了的 Prompt 1/3，其 `candidate 0`（RM 偏好的好答案）概率从近 0 跃升到 0.98 / 0.96，期望目标 `E[r - β·KL]` 从 0.08 涨到 4.95。

> **诚实声明**：这里的 reward 是一次**单样本**采样估计的，梯度噪声天然很大（这正是本模块的核心教学点）。我们断言的是**期望目标显著提升** + **两个 SFT 出错 prompt 的概率确实被推向 RM 偏好动作**——而不是脆弱的"全部 prompt 收敛"。想看到真正的稳定方案（Value baseline + 裁剪），见 m04 PPO。

## How It Works

**为什么 `-R·logπ_θ` 能推动策略**：令 `p = π_θ(a|x)`。梯度为 `-R·(∇log p) = -R·(∇p/p)`。直观上：
- 若 `R > 0`（这个回答更好），梯度让 `p` 增大；
- 若 `R < 0`（更差），梯度让 `p` 减小。

**为什么要 KL 项**（`versions.md §3.3`）：仅最大化 `E[r_φ]` 时，策略会钻 RM 的漏：生成 RM 认为"高分"但实际**组语言退化 / 重复 / 偏移 SFT 分布**的内容。`-β·log(π_θ/π_ref)` 对"偏离参考太多"给出惩罚，把策略钉在"合理语言区域"内。等价地它就是在最大化 `E[r_φ] - β·D_KL(π_θ‖π_ref)`。

**sample-level KL 估计**：因为只采样了一个 `a`，我们用 `logπ_θ(a) - logπ_ref(a)` 作为该样本处的 KL 替身；这是 `versions.md §6` 的标准做法，也是 PPO 模块 `old_policy` 想法的先声。

**方差（教学重点）**：REINFORCE 的估计器用"一次采样"代替期望，故梯度方差**大**。这直接暴露 REINFORCE 的两个缺陷：
1. 无 baseline —— reward 全体平移会改变梯度（理论可证明），但 variance 高；
2. 更新可以一次性把策略推过头 —— 没有大裁剪限制单步幅度。

这正是 m04 PPO 用 `probability ratio + clip` 和 `Value baseline→advantage` 想解决的。

## Code Walkthrough（版本锚点 `# v0.2`）

**Step 1｜SFT 生成参考策略** — 复用 m01：对 `sft_labels=[0,1,0,1]` 交叉熵 600 步；`load_state_dict` 复制并**冻结**（`requires_grad_(False)`、`eval()`）。

**Step 2｜训练 Reward Model** — 复用 m02：8 个 `(pid, chosen=0, rejected=1|2)` 偏好对，用 Bradley-Terry loss 训练 600 步（每步对 8 个偏好对）。这一步提供 `r_φ(x,a)`。

**Step 3｜REINFORCE + KL 主循环** — 400 步：`Categorical.sample()` 采动作 `a`，取 `log_prob`；无梯度下算 `reference` 的 `log_prob`，得 `sampled_kl`；`reward = r_φ - β·kl`；`loss = -(reward * log_probs).mean()` 反向更新策略。

**Step 4｜自验证 + 输出** — `expected_objective` 用解析积分（非采样）算 `E[r_φ - β·KL]`；Assert1：`obj_after > obj_before`（证明目标真被提升）；Assert2：SFT 出错的 prompt 1/3 上 `candidate0 概率 > 0.5`。打印前后概率表 + `[PASS]`。

运行：

```bash
python m03_reinforce/code.py
```

## Key Design Decisions

- **`reference_policy` 复制（等价 deepcopy）后冻结**：RL 阶段必须用 SFT 的原始分布做 KL 锚点；如果 `π_ref` 继续学到，KL 就失去意义。代码里用 `load_state_dict(policy.state_dict())` 复制权重，再 `requires_grad_(False)` + `eval()`。这是与 m01/m02 的根本区别。
- **奖励和 ref 都不带梯度**：`reward`、`KL`、`sampled_kl` 都包在 `torch.no_grad()` 里，只有 `log_probs` 需要梯度 —— 避免把梯度传到 RM/ref 里，这是正常 REINFORCE 的做法。
- **`expected_objective` 用解析 KL 而非采样**：因为离散动作 3 个，KL 可以精确求和。用 -它来进行**前后对比的公正度量**，而不是依赖一条采样路径。
- **测试诚实**：不硬断 "所有 prompt 都收敛到 0"（噪声下不稳），断言"期望目标提升 + 两个出错 prompt 被拉向 RM 偏好" —— 让学生看到 REINFORCE **有效果**，同时保留"方差大"的提示。
- **`sys.path` 剔除 + `from __future__`**：同 m01/m02（本文件叫 `code.py` 会遮蔽标准库，见 `AGENTS.md`）。

## Going Deeper

- **REINFORCE 方差从哪来**：单样本采样的无偏估计量方差正比于**奖励幅度**。当 RM 输出的"高分答案"其实分布很窄时，只有小机会采到，梯度普遍稀疏而大。真实的 RLHF 用 batch ∈ 正数十甚至上百来稀释方差。
- **KL 太大 / 太小的权衡**（`versions.md §11.3`）：
  - β **太大**（KL 太强）→ 策略几乎不变，RLHF 没效果；
  - β **太小**（KL 太弱）→ 策略快速偏离 SFT → Reward hacking → 语言退化 → 模式坍塌。
  - 生产里用**自适应 KL**：real_KL > 目标 → 增大 β；< → 减小 β。
- **PPO（m04）补什么**：Value baseline（减掉期望奖励，降方差）+ 概率裁剪（防止单步更新过多）+ 多轮遍历 old_policy。这就是 REINFORCE 的"方差问题 → 稳定方案"的升级。

## 模块定位

这是 `learn-rlhf` 序列里**第一次真正用强化学习信号（RM 奖励 + KL）驱动策略** 的模块（`v0.2`）：复用了 m01 的 SFT 策略与 m02 的 RM，把"如何最大化奖励"落成可跑的 REINFORCE。它是 m04（PPO + Value，修 REINFORCE 的方差）的**直接前置**：这里暴露的问题，正是下一章要解的。

版本：**v0.2** · 运行：`python3 m03_reinforce/code.py`（CPU 秒级）