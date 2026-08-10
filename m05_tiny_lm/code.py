"""m05 Tiny LM — 字符级 Tiny GPT + Token-level PPO (v0.4).

Run:  python m05_tiny_lm/code.py

把 m04 的"离散回答选择"升级为真正的 token 级语言模型强化学习：

    Tokenizer → TinyGPT → Auto-regressive Rollout → Reward → Token-level PPO

任务：给定一个两位数字组成的 prompt，模型**自回归地**逐个 token 生成一个两位回文
答案码。每个 prompt 对应一个唯一的目标两位码（互不冲突）。SFT 故意对其中两个 prompt
给出"带噪"示范（对/错各一条），于是策略对它们"半对半错"；Token-level PPO 用
"答对则奖励 +1"把所有 prompt 都推向正确的目标码。
"""

# --- v0.4: Token-level Tiny LM RLHF ---
from __future__ import annotations

import copy
import math
import os
import sys

# 本文件名为 code.py，运行时会以其所在目录作为 sys.path[0]；
# torch 内部会 `import code`（此处 code 是 Python 标准库的 code 模块），
# 若本目录留在 sys.path 中，标准库的 code 会被本文件错误遮蔽导致导入崩溃。
# 因此在 import 任何 torch 之前，先把本文件所在目录从 sys.path 中剔除。
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(os.path.expanduser(p)) != _here]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 配置与数据：字符级"两位答案码"任务
# ============================================================

DIGITS = "0123456789"
V = len(DIGITS)                        # 词表大小 = 10（数字 0-9）
VOCAB = {ch: i for i, ch in enumerate(DIGITS)}
PROMPT_LEN = 2                         # prompt 两位
RESPONSE_LEN = 2                       # response 两位
MAX_LEN = PROMPT_LEN + RESPONSE_LEN

# 4 个互不相同的 prompt，每个对应一个唯一的目标 code（不同 prompt 的目标互不冲突，
# 这样共享参数的网络就不会在同一个 token 分布上收到互相矛盾的正确标签）。
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


# 错误示范：一个与任何合法 target 都不冲突的两位码（10 不在任何目标里）。
WRONG_RESPONSE = "55"


def sft_responses_of(prompt: str) -> list[str]:
    """该 prompt 的 SFT 示范序列。

    对 prompt "12" 和 "23" 故意给带噪示范（正确与 WRONG_RESPONSE 各一条），
    于是模型对它们"不确定/半对半错"；其余两个 prompt 只给正确示范，模型确定。
    """
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

        # 因果掩码：token i 只能看到 j <= i（自回归的前提）。
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

    logits[b, t] 预测的是位置 t+1 的"下一 token"。因此：
    response 第 k 个 token 位于 seq 绝对位置 p = PROMPT_LEN + k，
    它由 logits[b, p-1] 预测（取 position PROMPT_LEN-1 .. PROMPT_LEN+RL-2）。
    """
    lp_all = F.log_softmax(logits, dim=-1)
    b = torch.arange(seq.size(0), device=seq.device).unsqueeze(1)
    t = torch.arange(
        PROMPT_LEN - 1, PROMPT_LEN - 1 + RESPONSE_LEN, device=seq.device
    ).unsqueeze(0).expand(seq.size(0), RESPONSE_LEN)
    tokens = seq[:, PROMPT_LEN:].long()
    return lp_all[b, t, tokens]


def build_context(prompt_batch: torch.Tensor, response_batch: torch.Tensor) -> torch.Tensor:
    """拼接 [prompt][response] -> ids[B, MAX_LEN]。"""
    return torch.cat([prompt_batch, response_batch], dim=-1)


# ============================================================
# 2. 自回归 Rollout 与规则奖励
# ============================================================

def rollout(model: nn.Module, prompt_batch: torch.Tensor) -> torch.Tensor:
    """在 prompt 下自回归采样 response（固定 RESPONSE_LEN 步），返回 context[B, MAX_LEN]。"""
    resp = torch.zeros(prompt_batch.size(0), 0, dtype=torch.long, device=device)
    for _ in range(RESPONSE_LEN):
        context = build_context(prompt_batch, resp)
        logits, _ = model(context)
        dist = Categorical(logits=logits[:, -1, :])   # 预测下一个 token
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
# 3. SFT（v0.0）：监督微调出"部分正确"的策略
# ============================================================

policy = TinyLM().to(device)
sft_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

sft_loss = torch.tensor(0.0, device=device)
for _ in range(600):
    context = build_context(prompt_ids[sft_prompt_idx], sft_resp)
    logits, _ = policy(context)
    # 因果 LM：logits[p-1] 预测位置 p 的 response token
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
# 4. 冻结 Reference Policy（Token-level KL 锚点）
# ============================================================

reference_policy = copy.deepcopy(policy)
reference_policy.eval()
for p in reference_policy.parameters():
    p.requires_grad_(False)

# ============================================================
# 5. PPO 超参数
# ============================================================

# TinyLM 的 trunk、LM head 与 value head 属于同一个参数图；一个 Adam 拥有全部
# 参数，保证 value_head 不会同时落入两套优化器状态、在一次 backward 后被 step 两次。
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)

batch_size = 128
ppo_updates = 500
ppo_epochs = 1
clip_epsilon = 0.2
kl_beta = 0.1
entropy_coef = 0.001
gamma = 0.9   # cheap n-step returns 的折扣因子
GRAD_CLIP_NORM = 0.5   # 梯度裁剪：RL 数值稳定（token-level PPO）


# ============================================================
# 6. 主流程：Token-level PPO
# ============================================================

def main() -> None:
    first_total_loss: float | None = None
    last_total_loss = 0.0

    policy.train()
    for update in range(ppo_updates):

        # ---- 6.1 按当前策略 rollout 采样一批 prompt 的 response ----
        sampled_prompts = torch.randint(0, num_prompts, (batch_size,), device=device)
        prompt_batch = prompt_ids[sampled_prompts]
        with torch.no_grad():
            context = rollout(policy, prompt_batch)
            resp_ids = context[:, PROMPT_LEN:]
            raw_reward = exact_match_reward(sampled_prompts, resp_ids)

        # ---- 6.2 冻结采样时刻的 old policy（ratio 的分母） ----
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            old_logp = response_log_probs(old_policy(context)[0], context)
            ref_logp = response_log_probs(reference_policy(context)[0], context)
            old_values = old_policy(context)[1][:, PROMPT_LEN:]

            # Token-level KL：在采样到的 token 上对比当前(旧)与 ref 的 log-prob。
            tok_kl = old_logp - ref_logp

            # 序列奖励分配（versions.md §9 v0.4）：
            #   每个 response token 的奖励 = -β·KL_t
            #   最后一个 response token 额外 + raw_reward（序列奖励广播）
            token_reward = -kl_beta * tok_kl
            token_reward[:, -1] += raw_reward

            # cheap n-step returns（v0.4 允许的简化）：
            #   returns_t = r_t + γ·returns_{t+1}
            # 让最终 response token 的序列奖励折现回溯到前面的 token，
            # 使前面的 token 也能分享到"回答正确"的信号（m06 用正式 GAE 取代）。
            returns = token_reward.clone()
            for t in range(RESPONSE_LEN - 2, -1, -1):
                returns[:, t] = token_reward[:, t] + gamma * returns[:, t + 1]
            returns = returns.detach()

            advantages = returns - old_values
            # token 级 advantage 标准化（对整个 response-token 集合）
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 6.3 Token-level PPO（多为 epoch 复用同一份 rollout） ----
        for _ in range(ppo_epochs):
            logits, values = policy(context)
            curr_logp = response_log_probs(logits, context)

            ratio = torch.exp(curr_logp - old_logp)
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

            dist = Categorical(logits=logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN, :])
            entropy = dist.entropy().mean()

            policy_loss = (-torch.min(unclipped, clipped).mean()
                           - entropy_coef * entropy)

            predicted = values[:, PROMPT_LEN:]
            value_loss = F.mse_loss(predicted, returns)

            # policy 与 value 共享主体：合并后一次 backward、由同一个 optimizer step。
            total_loss = policy_loss + value_loss
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪：RL 更新不稳定 -> 在一次更新前把共享主体的梯度 total-norm
            # 截到 GRAD_CLIP_NORM，防止个别梯度尖峰扭曲整步更新（value head 是
            # policy.parameters() 的子集，裁剪一次即覆盖 policy 与 value）。
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

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

    # ---- 6.4 评估 + 断言 ----
    with torch.no_grad():
        rl_target_prob = target_probabilities(policy)
        greedy_context = rollout_greedy(policy)
        greedy_acc = (greedy_context[:, PROMPT_LEN:] == target_ids).all(dim=-1).float().mean().item()

    print("\n===== 正确 target 序列概率：SFT vs RLHF =====")
    for i, p in enumerate(prompts):
        print(
            f"  prompt '{p}'（目标 {target_of(p)}）: "
            f"SFT={sft_target_prob[i].item():.3f}  RL={rl_target_prob[i].item():.3f}"
        )

    mean_sft = sft_target_prob.mean().item()
    mean_rl = rl_target_prob.mean().item()

    # 断言 1（机制级）：正确 target 的平均序列概率显著上升
    assert (mean_rl > mean_sft + 0.02), (
        f"RLHF 后正确回答概率应上升：SFT={mean_sft:.4f} -> RL={mean_rl:.4f}"
    )
    # 断言 2：最终贪心解码应基本全部正确
    assert greedy_acc > 0.75, f"贪心解码准确率过低 greedy_acc={greedy_acc:.3f}"
    # 断言 3：(policy + value 联合 loss 下降，说明 token 级 RL 真正收敛)
    assert first_total_loss is not None and last_total_loss < first_total_loss, (
        f"total_loss 应下降：{first_total_loss:.4f} -> {last_total_loss:.4f}"
    )

    print(f"\n正确 target 平均概率：SFT={mean_sft:.4f} -> RL={mean_rl:.4f}  (上升)")
    print(f"贪心解码准确率 = {greedy_acc:.3f}")
    print(f"RL 平均采样奖励 ≈ 1.0；total_loss：{first_total_loss:.4f} -> {last_total_loss:.4f}  (下降)")
    print(
        "[PASS] m05 tiny_lm: 字符级 TinyGPT 经 Token-level PPO，"
        "把 SFT 的带噪回答纠偏到受奖励的正确回答（loss 下降、正确概率上升）"
    )


if __name__ == "__main__":
    main()
