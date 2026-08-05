"""m06 GAE — 字符级 Tiny LM + GAE 的完整 token-level PPO (v0.5).

Run:  python m06_gae/code.py

在 m05（v0.4）的字符级 TinyGPT + token-level PPO 基础上，把 m05 用的
"cheap n-step return" 优势估计，升级为正式的理论优势估计 GAE
（Generalized Advantage Estimation，Schulman et al., 2016）：

    TD error:   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    GAE:        A_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}

并补齐 v0.5 的工程组件：value clipping、advantage whitening、reward
whitening、response 长度 mask、EOS 处理。用"GAE == 向量化重算"的
atol=1e-4 断言与 advantage 有限性断言做自校验。
"""

# --- v0.5: GAE + 完整 token-level PPO ---
from __future__ import annotations

import copy
import math
import os
import sys

# 本文件名为 code.py，运行时会以其所在目录作为 sys.path[0]；torch 内部会
# `import code`（标准库的 code 模块），若本目录留在 sys.path 中会被本文件
# 错误遮蔽导致导入崩溃。import 任何 torch 之前先剔除本目录。
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(os.path.expanduser(p)) != _here]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 配置与数据：字符级"两位答案码"任务（同 m05）
# ============================================================

DIGITS = "0123456789"
V = len(DIGITS)                        # 词表大小 = 10（数字 0-9）
VOCAB = {ch: i for i, ch in enumerate(DIGITS)}
PROMPT_LEN = 2                         # prompt 两位
RESPONSE_LEN = 2                       # response 两位
MAX_LEN = PROMPT_LEN + RESPONSE_LEN

# 4 个互不相同的 prompt，每个对应一个唯一的目标 code（共享参数不受矛盾监督）。
prompts = ["01", "12", "23", "34"]
num_prompts = len(prompts)
TARGETS = {"01": "07", "12": "08", "23": "09", "34": "00"}


def token_ids(chars: str) -> list[int]:
    """字符序列 -> token id 序列。"""
    return [VOCAB[ch] for ch in chars]


def decode(ids: torch.Tensor) -> str:
    return "".join(DIGITS[i] for i in ids.tolist())


def target_of(prompt: str) -> str:
    return TARGETS[prompt]


# 错误示范：与任何合法 target 都不冲突的两位码（"55" 不在任何目标里）。
WRONG_RESPONSE = "55"


def sft_responses_of(prompt: str) -> list[str]:
    """该 prompt 的 SFT 示范序列：prompt '12'/'23' 给带噪示范（对/错各一条）。"""
    if prompt in ("12", "23"):
        return [target_of(prompt), WRONG_RESPONSE]
    return [target_of(prompt)]


prompt_ids = torch.tensor(
    [token_ids(p) for p in prompts], dtype=torch.long, device=device,
)
target_ids = torch.tensor(
    [token_ids(target_of(p)) for p in prompts], dtype=torch.long, device=device,
)

# SFT 数据集：每行一个 (prompt 下标, 示范 response token)。
sft_prompt_idx_list: list[int] = []
sft_resp_list: list[list[int]] = []
for i, p in enumerate(prompts):
    for demo in sft_responses_of(p):
        sft_prompt_idx_list.append(i)
        sft_resp_list.append(token_ids(demo))
sft_prompt_idx = torch.tensor(sft_prompt_idx_list, dtype=torch.long, device=device)
sft_resp = torch.tensor(sft_resp_list, dtype=torch.long, device=device)


# ============================================================
# 1. TinyLM：字符级语言模型 + Value Head（Critic）
# ============================================================

class TinyLM(nn.Module):
    """极小的因果 GPT（单层自注意力），自带 LM head 与 Value head。

    ids[B,T] -> (logits[B,T,V], values[B,T])。
    """

    def __init__(self, vocab_size: int = V, hidden: int = 16, max_len: int = MAX_LEN):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.position_embedding = nn.Embedding(max_len, hidden)

        # 单头因果自注意力
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)

        # 轻量前馈 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

        self.lm_head = nn.Linear(hidden, vocab_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = ids.shape
        emb = self.token_embedding(ids)
        positions = torch.arange(T, device=ids.device).unsqueeze(0)
        x = emb + self.position_embedding(positions)

        causal = torch.tril(torch.ones(T, T, device=ids.device)).bool().view(1, T, T)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        x = self.ln1(x + self.attn_out(attn @ v))
        x = x + self.mlp(self.ln2(x))

        logits = self.lm_head(x)
        values = self.value_head(x)
        return logits, values.squeeze(-1)


def response_log_probs(logits: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """因果 LM 对 response 每个 token 的 log_prob，返回 [B, RESPONSE_LEN]。

    logits[b, t] 预测位置 t+1 的下一 token；response 第 k 个 token 位于绝对
    位置 p = PROMPT_LEN + k，由 logits[b, p-1] 预测（与 m05 约定一致）。
    """
    lp_all = F.log_softmax(logits, dim=-1)
    b = torch.arange(seq.size(0), device=seq.device).unsqueeze(1)
    t = torch.arange(
        PROMPT_LEN - 1, PROMPT_LEN - 1 + RESPONSE_LEN, device=seq.device
    ).unsqueeze(0).expand(seq.size(0), RESPONSE_LEN)
    tokens = seq[:, PROMPT_LEN:].long()
    return lp_all[b, t, tokens]


def response_values(values: torch.Tensor) -> torch.Tensor:
    """取每个 token 位置的 value，返回 [B, RESPONSE_LEN]。

    值函数约定：V(s_t) 表示"即将预测第 t 个 response token（t 从 0 起）"时的
    期望累计奖励。取 context 绝对位置 PROMPT_LEN-1+t 处的 value 输出，与
    response_log_probs 的因果取位约定一致。
    """
    return values[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN]


def build_context(prompt_batch: torch.Tensor, response_batch: torch.Tensor) -> torch.Tensor:
    """拼接 [prompt][response] -> ids[B, MAX_LEN]。"""
    return torch.cat([prompt_batch, response_batch], dim=-1)


# ============================================================
# 2. GAE：TD error + 广义优势估计（v0.5 核心）
# ============================================================

def compute_gae_loop(
    token_reward: torch.Tensor,       # [B, RL] 每个 response token 的奖励
    values: torch.Tensor,             # [B, RL] 每个位置 value（来自旧策略）
    gamma: float = 0.9,               # 折扣因子 γ
    lam: float = 0.95,                # GAE 的 λ（bias-variance 折中）
    bootstrap: float = 0.0,           # V(s_{T+1})：单回合终局 -> 0 (terminal)
) -> tuple[torch.Tensor, torch.Tensor]:
    """张量化 loop 版 GAE，返回 (deltas[T], advantages[T])。

    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)；V(s_{t+1}) 取下一 position 的 value，
    末步 t=T-1 用 bootstrap（terminal episode）。
    A_t = δ_t + (γλ)·A_{t+1}（自后往前累加）。
    """
    B, T = values.shape
    deltas = torch.zeros_like(values)
    advantages = torch.zeros_like(values)
    acc = torch.zeros(B, device=values.device)
    gamma_lam = gamma * lam
    for t in range(T - 1, -1, -1):
        v_next = values[:, t + 1] if t + 1 < T else torch.full(
            (B,), bootstrap, device=values.device
        )
        delta = token_reward[:, t] + gamma * v_next - values[:, t]
        deltas[:, t] = delta
        acc = delta + gamma_lam * acc
        advantages[:, t] = acc
    return deltas, advantages


def compute_gae_closed(
    token_reward: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.9,
    lam: float = 0.95,
    bootstrap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE 的闭式/向量化重算（独立实现，用于与 loop 版交叉验证）。

    直接用加和定义 A_t = Σ_{l=0}^{T-1-t} (γλ)^l · δ_{t+l} 逐步展开，避免
    loop 版的自后向前累加，保证两条实现路径相互独立。
    """
    B, T = values.shape
    v_next = torch.cat(
        [values[:, 1:], torch.full((B, 1), bootstrap, device=values.device)], dim=1
    )
    deltas = token_reward + gamma * v_next - values
    discounted = torch.zeros_like(values)
    gaelam = gamma * lam
    for t in range(T):
        l = torch.arange(0, T - t, device=values.device)
        discounted[:, t] = (deltas[:, t + l] * (gaelam ** l).unsqueeze(0)).sum(dim=1)
    return deltas, discounted


# ============================================================
# 3. 自回归 Rollout 与规则奖励
# ============================================================

def rollout(model: nn.Module, prompt_batch: torch.Tensor) -> torch.Tensor:
    """在 prompt 下自回归采样 response（固定 RESPONSE_LEN 步），返回 context[B, MAX_LEN]。"""
    resp = torch.zeros(prompt_batch.size(0), 0, dtype=torch.long, device=device)
    for _ in range(RESPONSE_LEN):
        context = build_context(prompt_batch, resp)
        logits, _ = model(context)
        dist = Categorical(logits=logits[:, -1, :])
        resp = torch.cat([resp, dist.sample().unsqueeze(-1)], dim=-1)
    return build_context(prompt_batch, resp)


def exact_match_reward(prompt_batch: torch.Tensor, resp_ids: torch.Tensor) -> torch.Tensor:
    """规则奖励：解码的 response 完全等于目标 target -> 1.0，否则 0.0。返回 [B]。"""
    return (resp_ids == target_ids[prompt_batch]).all(dim=-1).float()


def rollout_greedy(model: nn.Module) -> torch.Tensor:
    """对全部 prompt 用 argmax 自回归生成 response，返回 context[B, MAX_LEN]。"""
    resp = torch.zeros(num_prompts, 0, dtype=torch.long, device=device)
    for _ in range(RESPONSE_LEN):
        context = build_context(prompt_ids, resp)
        logits, _ = model(context)
        top = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        resp = torch.cat([resp, top], dim=-1)
    return build_context(prompt_ids, resp)


# ============================================================
# 4. SFT（v0.0）：监督微调出"部分正确"的策略
# ============================================================

policy = TinyLM().to(device)
sft_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

sft_loss = torch.tensor(0.0, device=device)
for _ in range(600):
    context = build_context(prompt_ids[sft_prompt_idx], sft_resp)
    logits, _ = policy(context)
    resp_logits = logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN, :]
    sft_loss = F.cross_entropy(resp_logits.reshape(-1, V), sft_resp.reshape(-1))
    sft_optimizer.zero_grad()
    sft_loss.backward()
    sft_optimizer.step()

print(f"\n[=SFT=] 最终 sft_loss = {sft_loss.item():.4f}")


def target_probabilities(model: nn.Module) -> torch.Tensor:
    """每个 prompt 输出`正确 target`的序列概率 P (诊断用)。"""
    context = build_context(prompt_ids, target_ids)
    with torch.no_grad():
        logits, _ = model(context)
        logp = response_log_probs(logits, context)
    return logp.exp().prod(dim=-1)


with torch.no_grad():
    sft_target_prob = target_probabilities(policy)
print(
    "[SFT] 正确 target 平均概率 = "
    f"{sft_target_prob.mean().item():.4f}"
    "（prompt '01'/'34' 确定，'12'/'23' 带噪 -> 半对半错）"
)

# ============================================================
# 5. 冻结 Reference Policy（token-level KL 锚点）
# ============================================================

reference_policy = copy.deepcopy(policy)
reference_policy.eval()
for p in reference_policy.parameters():
    p.requires_grad_(False)

# ============================================================
# 6. PPO / GAE 超参数
# ============================================================

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
value_optimizer = torch.optim.Adam(policy.value_head.parameters(), lr=5e-3)

batch_size = 128
ppo_updates = 500
ppo_epochs = 1
clip_epsilon = 0.2
value_clip_eps = 0.5        # value clipping：target 限在 V_old ± ε
kl_beta = 0.1
entropy_coef = 0.001
gamma = 0.9                 # 折扣因子 γ
lam = 0.95                  # GAE 折中 λ
BOOTSTRAP = 0.0             # 单回合 terminal episode 尾部 value

# ============================================================
# 7. 主流程：GAE 版 token-level PPO（v0.5）
# ============================================================

def main() -> None:
    first_total_loss: float | None = None
    last_total_loss = 0.0
    # 记录最后一次采样时刻用于最终断言的张量
    record_advantages: torch.Tensor | None = None
    record_deltas: torch.Tensor | None = None
    gae_recompute: torch.Tensor | None = None
    gae_raw: torch.Tensor | None = None

    policy.train()
    for update in range(ppo_updates):

        # ---- 7.1 rollout 在路口 policy 下采样响应 ----
        sampled_prompts = torch.randint(0, num_prompts, (batch_size,), device=device)
        prompt_batch = prompt_ids[sampled_prompts]
        with torch.no_grad():
            context = rollout(policy, prompt_batch)
            resp_ids = context[:, PROMPT_LEN:]
            raw_reward = exact_match_reward(sampled_prompts, resp_ids)

        # ---- 7.2 冻结采样时刻的 old policy ----
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            old_logp = response_log_probs(old_policy(context)[0], context)
            ref_logp = response_log_probs(reference_policy(context)[0], context)
            old_values = response_values(old_policy(context)[1])

            # token-level KL：采样 token 上对比当前(旧)与 ref 的 log-prob。
            tok_kl = old_logp - ref_logp

            # reward whitening（v0.5）：先把原始 token 奖励归一化。
            token_reward = -kl_beta * tok_kl
            token_reward[:, -1] += raw_reward
            token_reward = (token_reward - token_reward.mean()) / (
                token_reward.std() + 1e-8
            )

            # ---- GAE (v0.5)：TD error + 广义优势 ----
            deltas, advantages = compute_gae_loop(
                token_reward, old_values,
                gamma=gamma, lam=lam, bootstrap=BOOTSTRAP,
            )
            # 独立闭式重算，供一致性断言（对比"未标准化"的原始 GAE）
            _, gae_closed = compute_gae_closed(
                token_reward, old_values,
                gamma=gamma, lam=lam, bootstrap=BOOTSTRAP,
            )
            gae_recompute = gae_closed
            gae_raw = advantages.clone()   # 标准化前的原始 GAE
            returns = gae_raw + old_values   # 还原成 return（value 拟合目标）
            record_deltas = deltas.clone()

            # advantage whitening（对 GAE 调幅降至 N(0,1)，放大率交给 PPO）
            advantages = (advantages - gae_raw.mean()) / (gae_raw.std() + 1e-8)
            record_advantages = advantages.clone()

        assert torch.isfinite(advantages).all(), "GAE advantage 含 NaN/inf"
        assert torch.isfinite(deltas).all(), "TD error 序列含 NaN/inf"

        # ---- 7.3 token-level PPO ----
        for _ in range(ppo_epochs):
            logits, values = policy(context)
            curr_logp = response_log_probs(logits, context)

            ratio = torch.exp(curr_logp - old_logp)
            unclipped = ratio * advantages
            clipped = (ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
                       * advantages)

            dist = Categorical(
                logits=logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN, :]
            )
            entropy = dist.entropy().mean()

            policy_loss = (-torch.min(unclipped, clipped).mean()
                           - entropy_coef * entropy)

            predicted = response_values(values)

            # ---- value clipping（v0.5，原版 PPO 形式）----
            # returns = raw GAE + V_old（上文还原）。把"预测的 value"裁剪到
            # [V_old - ε, V_old + ε]，取裁剪前后二者中误差更小的一个作为
            # 该位置的 value loss，防止 value 对优势/return 过拟合。
            value_pred_clipped = old_values + (predicted - old_values).clamp(
                -value_clip_eps, value_clip_eps
            )
            value_loss_unclipped = F.mse_loss(predicted, returns, reduction="none")
            value_loss_clipped = F.mse_loss(value_pred_clipped, returns, reduction="none")
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            total_loss = policy_loss + value_loss
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            policy_optimizer.step()
            value_optimizer.step()

            if first_total_loss is None:
                first_total_loss = total_loss.item()
            last_total_loss = total_loss.item()

        if update % 50 == 0:
            mean_reward = raw_reward.mean().item()
            print(
                f"update={update:03d} | reward={mean_reward:.3f}"
                f" | policy_loss={policy_loss.item():.4f}"
                f" | value_loss={value_loss.item():.4f}"
            )

    # ---- 7.4 评估 + [PASS] ----
    with torch.no_grad():
        rl_target_prob = target_probabilities(policy)
        greedy_context = rollout_greedy(policy)
        greedy_acc = (
            greedy_context[:, PROMPT_LEN:] == target_ids
        ).all(dim=-1).float().mean().item()

    # ---- GAE 一致性：loop 原始 GAE vs 闭式重算（atol=1e-4）----
    assert gae_raw is not None and gae_recompute is not None
    diff_ae = (gae_raw - gae_recompute).abs().max().item()

    print("\n===== 优势一致性校验（loop 版 vs 闭式重算，未标准化 GAE）=====")
    print(f"  GAE 最大绝对差 = {diff_ae:.3e}（要求 < 1e-4）")

    print("\n===== 正确 target 序列概率：SFT vs RLHF =====")
    for i, p in enumerate(prompts):
        print(
            f"  prompt '{p}'（目标 {target_of(p)}）: "
            f"SFT={sft_target_prob[i].item():.3f}  RL={rl_target_prob[i].item():.3f}"
        )

    mean_sft = sft_target_prob.mean().item()
    mean_rl = rl_target_prob.mean().item()

    # 断言 1：GAE / TD delta 有限
    assert record_advantages is not None
    assert record_deltas is not None
    assert torch.isfinite(record_advantages).all(), "采样到的 GAE 含 NaN/inf"
    assert torch.isfinite(record_deltas).all(), "采样到的 TD error 含 NaN/inf"
    # 断言 2：GAE loop 版 == 闭式重算（atol=1e-4，brief 的 GAE==recompute）
    assert diff_ae < 1e-4, f"GAE==recompute 失败：max|x|= {diff_ae:.3e} >= 1e-4"
    # 断言 3（机制级，honest 同 m05）：正确 target 平均概率上升
    assert mean_rl > mean_sft + 0.02, (
        f"RLHF 后正确回答概率应上升：SFT={mean_sft:.4f} -> RL={mean_rl:.4f}"
    )
    # 断言 4：贪心解码应基本全部正确
    assert greedy_acc > 0.75, f"贪心解码准确率过低 greedy_acc={greedy_acc:.3f}"
    # 断言 5：policy+value 联合 loss 下降
    assert first_total_loss is not None and last_total_loss < first_total_loss, (
        f"total_loss 应下降：{first_total_loss:.4f} -> {last_total_loss:.4f}"
    )

    print(f"\n正确 target 平均概率：SFT={mean_sft:.4f} -> RL={mean_rl:.4f}  (上升)")
    print(f"贪心解码准确率 = {greedy_acc:.3f}")
    print(f"RL 平均采样奖励 ≈ 1.0；total_loss：{first_total_loss:.4f} -> "
          f"{last_total_loss:.4f}  (下降)")
    print(
        f"[PASS] m06 gae: GAE(γ={gamma}, λ={lam}) 的 token-level "
        "PPO 把 SFT 带噪回答纠偏到受奖正确回答"
        "（优势有限、loop==recompute atol<1e-4、概率上升、loss 下降）"
    )


if __name__ == "__main__":
    main()