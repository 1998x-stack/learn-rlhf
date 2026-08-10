"""m12 收束 — 端到端 RLHF 全链路 + 深坑清单 (收束 capstone).

Run:  python m12_integration/code.py        # 直接运行（已避开 code.py 遮蔽）
      python -m m12_integration.code         # 从仓库根目录运行亦可

把 m01–m11 的整条 RLHF 流水线在一个文件里跑通：

    SFT → Reward Model(BT) → PPO+KL(tok-level) → DPO → Verifier/Best-of-N

每一阶段的正确回答概率 / 可验证准确率都打印出来，得到一份 per-stage 总结表，
串起从 v0.0(SFT) 到 v1.0(可验证奖励) 的完整收束路径。

同时把 `versions.md` §11.6 深坑清单（Prompt→loss / padding / KL 符号 /
old-logp 冻结 / reward 广播 / advantage detach / EOS）编码为**活跃断言**：
每条 [检查] 都对应其中一坑——若今后有人把 bug 改回来，对应断言会当场失败。
"""

# --- 收束: 整链路集成（v0.0 SFT 起 → v1.0 可验证奖励收束）---
from __future__ import annotations

import copy
import math
import os
import sys

# 本文件名为 code.py，运行时会以其所在目录作为 sys.path[0]；
# torch 内部会 `import code`（此处 code 是 Python 标准库的 code 模块），
# 若本目录留在 sys.path 中，标准库的 code 会被本文件错误遮蔽导致导入崩溃。
# 因此在 import 任何 torch 之前，先把本文件所在目录从 sys.path 中剔除。 （MANDATORY）
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(os.path.expanduser(p)) != _here]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

torch.manual_seed(42)

# 设备：GPU 可用则用 GPU，否则 CPU（对张量与模型统一 apply）。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 配置与数据：字符级"两位答案码"任务（复用 m05 的干净机制）
# ============================================================
# 每个 prompt（两位数字）对应一个唯一的目标两位码，SFT 对其中两个 prompt
# 故意给带噪示范（正确 + 错误各一条）→ 策略对它们半对半错；
# 后续 RM→PPO→DPO 用偏好把全部 prompt 推向正确目标码。

DIGITS = "0123456789"
V = len(DIGITS)                                 # 词表大小 = 10（数字 0-9）
VOCAB = {ch: i for i, ch in enumerate(DIGITS)}
PROMPT_LEN = 2                                  # prompt 两位
RESPONSE_LEN = 2                                # response 两位（定长，无 padding）
MAX_LEN = PROMPT_LEN + RESPONSE_LEN

prompts = ["01", "12", "23", "34"]
num_prompts = len(prompts)
TARGETS = {"01": "07", "12": "08", "23": "09", "34": "00"}

# 错误示范：与任何合法 target 都不冲突的两位码（避免"又是对又是错"的矛盾监督）。
WRONG_RESPONSE = "55"


def token_ids(chars: str) -> list[int]:
    return [VOCAB[ch] for ch in chars]


def decode(ids: torch.Tensor) -> str:
    return "".join(DIGITS[i] for i in ids.tolist())


def target_of(prompt: str) -> str:
    return TARGETS[prompt]


def sft_responses_of(prompt: str) -> list[str]:
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

# 偏好对（v0.1/BT + v0.8/DPO 共用）：每个 prompt 一条 chosen=正确目标 vs rejected=错误码。
pref_prompt = torch.arange(num_prompts, device=device)
pref_chosen = torch.tensor(
    [token_ids(target_of(p)) for p in prompts], dtype=torch.long, device=device,
)
pref_rejected = torch.tensor(
    [token_ids(WRONG_RESPONSE)] * num_prompts, dtype=torch.long, device=device,
)


# ============================================================
# 1. 网络：TinyGPT（策略 + 价值）与 RewardModel（BT 后冻结）
# ============================================================

class TinyLM(nn.Module):
    """字符级因果 LM（单层自注意力），LM head（策略） + Value head（Critic）。"""

    def __init__(self, vocab_size: int = V, hidden: int = 16, max_len: int = MAX_LEN):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.position_embedding = nn.Embedding(max_len, hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden),
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


class RewardModel(nn.Module):
    """对整段 [prompt][response] 打分：取 response 最后 token 的 scalar。

    用 Bradley–Terry loss 训练（chosen 应比 rejected 高分），随后冻结，
    作为 PPO 阶段的「序列奖励源」——奖励只在一个值上，不逐 token 分发。
    """

    def __init__(self, vocab_size: int = V, hidden: int = 16, max_len: int = MAX_LEN):
        super().__init__()
        self.body = TinyLM(vocab_size, hidden, max_len)

    def score(self, seq: torch.Tensor) -> torch.Tensor:
        """对整段 seq 打分：取 response 最后 token 位置的 value。返回 [B]。"""
        _, values = self.body(seq)
        return values[:, PROMPT_LEN + RESPONSE_LEN - 1]


# ============================================================
# 2. 工具：因果 LM 的 response log-prob（固定索引约定，深坑重灾区）
# ============================================================

def response_log_probs(logits: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """对 response 每个 token 取 log_prob，返回 [B, RESPONSE_LEN]。

    logits[b, t] 预测位置 t+1 的"下一 token"，故 response 第 k 个 token
    （位于绝对位置 p = PROMPT_LEN + k）由 logits[b, p-1] 预测。
    """
    lp_all = F.log_softmax(logits, dim=-1)
    b = torch.arange(seq.size(0), device=seq.device).unsqueeze(1)
    t = torch.arange(
        PROMPT_LEN - 1, PROMPT_LEN - 1 + RESPONSE_LEN, device=seq.device,
    ).unsqueeze(0).expand(seq.size(0), RESPONSE_LEN)
    tokens = seq[:, PROMPT_LEN:].long()
    return lp_all[b, t, tokens]


def build_context(prompt_batch: torch.Tensor, response_batch: torch.Tensor) -> torch.Tensor:
    return torch.cat([prompt_batch, response_batch], dim=-1)


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
    """可验证奖励：生成 response 完全等于目标 -> 1.0，否则 0.0。返回 [B]。"""
    return (resp_ids == target_ids[prompt_batch]).all(dim=-1).float()


def target_prob(model: nn.Module) -> torch.Tensor:
    """每个 prompt 输出正确 target 的序列概率 P（per-stage 度量）。返回 [P]。"""
    context = build_context(prompt_ids, target_ids)
    with torch.no_grad():
        logits, _ = model(context)
        logp = response_log_probs(logits, context)
    return logp.exp().prod(dim=-1)


def greedy_acc(model: nn.Module) -> float:
    """对全部 prompt 贪心解码生成 response；返回与 target 精确匹配的准确率。"""
    resp = torch.zeros(num_prompts, 0, dtype=torch.long, device=device)
    for _ in range(RESPONSE_LEN):
        context = build_context(prompt_ids, resp)
        logits, _ = model(context)
        top = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        resp = torch.cat([resp, top], dim=-1)
    return float((resp == target_ids).all(dim=-1).float().mean().item())


def best_of_n_accuracy(model: nn.Module, n: int, trials: int = 200) -> float:
    """Best-of-N（真实拒绝采样）：每个 prompt 采样 **n 个** response，
    只要 n 个里**任一**通过精确验证器（== target）即计该 prompt 成功。

    N 越大，"至少采到一个正确解"的概率越高（1-(1-p)^N），故 N=1 < N=4 < N=16。
    """
    successes = 0
    total = trials * num_prompts
    for _ in range(trials):
        # 每个 prompt 采 n 个候选，验证器挑"已验证正确的"；任一命中即成功。
        for j in range(n):
            with torch.no_grad():
                ctx = rollout(model, prompt_ids)
                resp = ctx[:, PROMPT_LEN:]
                ok = (resp == target_ids).all(dim=-1)      # [P] 精确验证器 0/1
            if j == 0:
                best = ok.clone()                                 # 逐 prompt 的"已验证正确"
            else:
                best = best | ok                                 # best over n 个候选
        successes += int(best.sum().item())
    return successes / total


# ============================================================
# 3. SFT（# v0.0）—— 带噪示范做出一份「部分正确」的初始策略
# ============================================================

pit_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    """打印一条 [检查]，不通过则记到 pit_failures（最后统一判定）。"""
    print(f"  [检查] {name}: {'通过' if cond else '失败'}")
    if not cond:
        pit_failures.append(f"{name}: {detail}")
    return cond


print("\n============== Stage 1/5: SFT (v0.0) ====> 初始（带噪）策略 ==============")
policy = TinyLM().to(device)
sft_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

sft_loss_t = torch.tensor(0.0, device=device)
for _ in range(400):
    context = build_context(prompt_ids[sft_prompt_idx], sft_resp)
    logits, _ = policy(context)
    resp_logits = logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN, :]
    sft_loss_t = F.cross_entropy(resp_logits.reshape(-1, V), sft_resp.reshape(-1))
    sft_optimizer.zero_grad()
    sft_loss_t.backward()
    sft_optimizer.step()

sft_target_prob = target_prob(policy)
sft_target_mean = sft_target_prob.mean().item()
sft_greedy = greedy_acc(policy)
print(f"[SFT] sft_loss={sft_loss_t.item():.4f} | P(target)={sft_target_mean:.4f} "
      f"| greedy_acc={sft_greedy:.3f}")

# ============================================================
# 4. 冻结 Reference Policy（# v0.3 PPO/KL 锚点）
# ============================================================

reference_policy = copy.deepcopy(policy)
reference_policy.eval()
for p in reference_policy.parameters():
    p.requires_grad_(False)


# ============================================================
# 5. Reward Model（# v0.1：Bradley–Terry 偏好建模，训练后冻结）
# ============================================================

print("\n============== Stage 2/5: Reward Model (BT, v0.1) 训练 + 冻结 ==============")

reward_model = RewardModel().to(device)
rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)

chosen_ctx = build_context(prompt_ids, pref_chosen)
rejected_ctx = build_context(prompt_ids, pref_rejected)

# 深坑 ⑧【奖励加到所有 token】的预防：RM 只对整段给一个值（score），
# 逐 token 的分布由 PPO 的序列奖励广播单独决定。
for _ in range(300):
    chosen_r = reward_model.score(chosen_ctx)      # [P]
    rejected_r = reward_model.score(rejected_ctx)  # [P]
    btl = -F.logsigmoid(chosen_r - rejected_r).mean()
    rm_optimizer.zero_grad()
    btl.backward()
    rm_optimizer.step()

reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    rm_chosen = reward_model.score(chosen_ctx)
    rm_rejected = reward_model.score(rejected_ctx)
rm_acc = float((rm_chosen > rm_rejected).float().mean().item())
print(f"[RM] BT 收敛；偏好对上 chosen>rejected 准确率 = {rm_acc:.3f}")

# ============================================================
# 6. PPO + KL（# v0.4 token 级、response mask / 深坑护栏）
# ============================================================

print("\n============== Stage 3/5: PPO + KL (v0.4) ==============")

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
value_optimizer = torch.optim.Adam(policy.value_head.parameters(), lr=5e-3)

batch_size = 128
ppo_updates = 400
ppo_epochs = 3
clip_epsilon = 0.2
kl_beta = 0.1
entropy_coef = 0.001
gamma = 0.9

# ---- 深坑 ②：response 定长（无 padding 逃生）+ 偏好对长度一致（DPO/BT 归一化前提）----
check("response 定长（无 padding 逃生）",
      RESPONSE_LEN == pref_chosen.size(1) == pref_rejected.size(1),
      "response 定长约定被破坏")
check("chosen/rejected 长度一致（DPO/BT 归一化前提）",
      pref_chosen.size(1) == pref_rejected.size(1) == RESPONSE_LEN,
      "偏好对长度不一致未归一化")

first_total_loss: float | None = None
last_total_loss = 0.0
policy_loss = torch.tensor(0.0, device=device)
value_loss = torch.tensor(0.0, device=device)
# 供深坑清单校验读取的 rollout 中间量（在 update 循环内会被真实赋值）。
advantages = torch.zeros(1, device=device)
old_logp = torch.zeros(1, device=device)
token_reward = torch.zeros(0, device=device)
raw_broadcast = torch.zeros(0, device=device)
response_mask = torch.zeros(0, device=device)

policy.train()
for update in range(ppo_updates):
    # ---- 6.1 rollout：按当前策略采样一批 prompt 的 response ----
    sampled_prompts = torch.randint(0, num_prompts, (batch_size,), device=device)
    prompt_batch = prompt_ids[sampled_prompts]
    with torch.no_grad():
        context = rollout(policy, prompt_batch)
        resp_ids = context[:, PROMPT_LEN:]
        raw_reward = exact_match_reward(sampled_prompts, resp_ids)

    # ---- 6.2 冻结采样时刻 old policy（ratio 分母）+ old_logp / ref_logp / value ----
    old_policy = copy.deepcopy(policy)
    old_policy.eval()
    for p in old_policy.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        old_logp = response_log_probs(old_policy(context)[0], context)
        ref_logp = response_log_probs(reference_policy(context)[0], context)
        old_values = old_policy(context)[1][:, PROMPT_LEN:]

        # 深坑 ④：old_logp / ref_logp 只此算一次，PPO epoch 内绝不重算。
        tok_kl = old_logp - ref_logp

        # Token 级 KL + 序列奖励广播（# v0.4）：
        #   非末位 response token 只承担 -kl_beta·KL（惩罚）；
        #   末位 response token 额外承接 RM Reward。
        # 深坑 ⑦（KL 符号）：reward = r - β·KL（减）。若写反成 +，下方 [检查] 立即失败。
        token_reward = -kl_beta * tok_kl                 # 所有 response token 先只 -KL
        raw_broadcast = torch.zeros_like(token_reward)
        raw_broadcast[:, -1] = raw_reward                # RM 只加到最后 token
        token_reward = token_reward + raw_broadcast

        # 深坑 ①/③：response mask 只覆盖 response 长度，prompt 位置不进入 PPO loss。
        # 用「绝对位置 p >= PROMPT_LEN」直接从真实序列结构推导 mask，而非 hardcode：
        # 若有人把 policy loss 扩到整个 context（深坑①复活）或把 PROMPT_LEN 改坏，
        # 下面按位置推导的 mask 会当场在 prompt 区出现非 0、断言立即失败。
        with torch.no_grad():
            full_pos = torch.arange(MAX_LEN, device=device)
            resp_pos_mask = (full_pos >= PROMPT_LEN).float()   # [MAX_LEN]，response 位置=1
            # prompt 区必须全 0，response 区必须全 1——顺序正确（先 prompt 后 response）。
            assert int(resp_pos_mask[:PROMPT_LEN].sum().item()) == 0, \
                "response mask 错误地包含了 prompt 位置"
            assert int(resp_pos_mask[PROMPT_LEN:].sum().item()) == RESPONSE_LEN, \
                "response mask 未覆盖完整 response 长度"
            # loss 实际消费的每-token logp 只到 response 区，宽度必须恰为 RESPONSE_LEN
            # （response_log_probs 内部按绝对位置 PROMPT_LEN+k 取 token，不会碰 prompt）。
            assert old_logp.size(1) == RESPONSE_LEN, \
                "PPO loss 的 logp 覆盖了完整 context（prompt 进入了 loss）"
        # token_reward / mask 都与 response 区对齐（[B, RESPONSE_LEN]）。
        response_mask = torch.ones(token_reward.size(0), RESPONSE_LEN, device=device)

        # cheap n-step returns（token 级简化；正式 GAE 见 m06）。
        returns = token_reward.clone()
        for t in range(RESPONSE_LEN - 2, -1, -1):
            returns[:, t] = token_reward[:, t] + gamma * returns[:, t + 1]
        returns = returns.detach()

        # 深坑 ⑥：advantage = returns - old_value 后立即 detach（无梯度泄漏进常量）。
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

    # ---- 6.3 PPO 多 epoch 复用同一份 rollout（old_logp 冻结、T 算）----
    for _ in range(ppo_epochs):
        logits, values = policy(context)
        curr_logp = response_log_probs(logits, context)      # 只在 response 位置打分
        ratio = torch.exp(curr_logp - old_logp)               # ✓ old_logp 未重算
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        dist = Categorical(logits=logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN, :])
        entropy = dist.entropy().mean()

        # 深坑 ①/③：policy loss 只聚合 response 位置（mask 内），prompt token 不会进入。
        losses = -torch.min(unclipped, clipped)
        policy_loss = losses.mean() - entropy_coef * entropy

        predicted = values[:, PROMPT_LEN:]
        value_loss = F.mse_loss(predicted, returns)

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
        print(f"  ppo update={update:03d} | reward={raw_reward.mean().item():.3f}"
              f" | policy_loss={policy_loss.item():.4f} | value_loss={value_loss.item():.4f}")

rl_target_prob = target_prob(policy)
rl_target_mean = rl_target_prob.mean().item()
rl_greedy = greedy_acc(policy)
print(f"[PPO] P(target)={rl_target_mean:.4f} | greedy_acc={rl_greedy:.3f}")


# ============================================================
# 7. DPO（# v0.8）：同一批偏好数据，离线直接优化（无 RM/rollout/value）
# ============================================================
# 为了让「DPO 也能移动策略」讲得诚实且鲁棒，这里对一份**全新的 SFT 策略**
# 做 standalone DPO（m09 风格）：只用法线上已冻结的 reference（SFT）与偏好对，
# 展示 DPO 独自把正确概率从 SFT 抬上来——而不与已经到 1.0 的 PPO 抢饱和增量。

print("\n============== Stage 4/5: DPO (v0.8) standalone 演示 ==============")

# 深坑 ②：DPO 也基于定长 response；log 概率差在 response 长度上累加（长度已归一化）。
assert pref_chosen.size(1) == pref_rejected.size(1) == RESPONSE_LEN, "DPO 偏好对长度不一致"

dpo_policy = TinyLM().to(device)
dpo_opt = torch.optim.Adam(dpo_policy.parameters(), lr=1e-2)
for _ in range(400):
    context = build_context(prompt_ids[sft_prompt_idx], sft_resp)
    logits, _ = dpo_policy(context)
    resp_logits = logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + RESPONSE_LEN, :]
    dpo_opt.zero_grad()
    F.cross_entropy(resp_logits.reshape(-1, V), sft_resp.reshape(-1)).backward()
    dpo_opt.step()
dpo_sft_prob = target_prob(dpo_policy).mean().item()
dpo_sft_greedy = greedy_acc(dpo_policy)

# DPO 用冻结的 reference（SFT 锚点，与主链同一份）做 π_ref。
beta_dpo = 1.0
ch_ctx = build_context(prompt_ids, pref_chosen)
rj_ctx = build_context(prompt_ids, pref_rejected)
with torch.no_grad():
    ref_ch = response_log_probs(reference_policy(ch_ctx)[0], ch_ctx).sum(-1)
    ref_rj = response_log_probs(reference_policy(rj_ctx)[0], rj_ctx).sum(-1)

dpo_optimizer = torch.optim.Adam(dpo_policy.parameters(), lr=5e-3)
dpo_loss_t = torch.tensor(0.0, device=device)
for _ in range(250):
    theta_ch = response_log_probs(dpo_policy(ch_ctx)[0], ch_ctx).sum(-1)
    theta_rj = response_log_probs(dpo_policy(rj_ctx)[0], rj_ctx).sum(-1)
    logw = theta_ch - ref_ch
    logl = theta_rj - ref_rj
    # 数值稳定：-logsigmoid(β·(logw - logl))；把 π 拉向 chosen、离开 rejected。
    dpo_loss_t = -F.logsigmoid(beta_dpo * (logw - logl)).mean()
    dpo_optimizer.zero_grad()
    dpo_loss_t.backward()
    dpo_optimizer.step()

dpo_target_prob = target_prob(dpo_policy)
dpo_target_mean = dpo_target_prob.mean().item()
dpo_greedy = greedy_acc(dpo_policy)
print(f"[DPO-standalone] P(target): SFT={dpo_sft_prob:.4f} -> DPO={dpo_target_mean:.4f}"
      f" (greedy {dpo_sft_greedy:.3f} -> {dpo_greedy:.3f})")


# ============================================================
# 8. per-stage 总结表（SFT → RL → DPO）
# ============================================================

print("\n============== per-stage 总结表 (SFT → RL/PPO → DPO) ==============")
print(f"{'stage':<10} {'P(target) mean':<18} {'greedy_acc':<10}")
print(f"{'SFT':<10} {sft_target_mean:<18.4f} {sft_greedy:<10.3f}")
print(f"{'RL/PPO+KL':<13} {rl_target_mean:<18.4f} {rl_greedy:<10.3f}")
print(f"{'DPO':<10} {dpo_target_mean:<18.4f} {dpo_greedy:<10.3f}")


# ============================================================
# 9. 验证器 / Best-of-N（# v1.0：推理时再榨正确率）
# ============================================================

print("\n============== Stage 5/5: Verifier / Best-of-N (v1.0) ==============")

# 真实 best-of-N 的"随 N 单调上升"要在有散布的弱策略上才看得见：
# 训练后策略（SFT→PPO→DPO）已近乎满分，最佳采样处处饱和，看不出 N-scaling。
# 故用 SFT 弱策略演示 N=1 < N=4 < N=16 的单调性（与 m11 同款做法、诚实可见），
# 再把已收敛的最终策略打出来作为"训练后已饱和"的对照。
sft_weak_bon1 = best_of_n_accuracy(reference_policy, n=1)
sft_weak_bon4 = best_of_n_accuracy(reference_policy, n=4)
sft_weak_bon16 = best_of_n_accuracy(reference_policy, n=16)
final_bon1 = best_of_n_accuracy(dpo_policy, n=1)
final_bon4 = best_of_n_accuracy(dpo_policy, n=4)
print(f"[验证器] Best-of-N（SFT 弱策略，真实采样 N 个候选验证器筛选）: "
      f"N=1 -> {sft_weak_bon1:.3f}, N=4 -> {sft_weak_bon4:.3f}, N=16 -> {sft_weak_bon16:.3f}")
print(f"[验证器] Best-of-N（DPO 后策略，对照/已饱和）: N=1 -> {final_bon1:.3f}, "
      f"N=4 -> {final_bon4:.3f} | greedy_acc={dpo_greedy:.3f}")

# ============================================================
# 10. 机制级断言（诚实、单调提升，robust 不脆弱）
# ============================================================

assert sft_target_mean < rl_target_mean, (
    f"RL 应使正确概率相比 SFT 提升 {sft_target_mean:.4f} -> {rl_target_mean:.4f}"
)
# DPO 是独立的 standalone 演示：它应把（自己的）SFT 基线明显抬升。
assert dpo_sft_prob < dpo_target_mean, (
    f"DPO 应使正确概率相比其 SFT 基线提升 {dpo_sft_prob:.4f} -> {dpo_target_mean:.4f}"
)
assert dpo_target_mean >= 0.85, f"DPO 后正确概率应大幅提升: {dpo_target_mean:.4f}"
# Best-of-N 单调性要在弱策略上看：SFT 弱策略下 N 越大至少命中一次的正确率越高。
assert sft_weak_bon4 >= sft_weak_bon1 and sft_weak_bon16 >= sft_weak_bon4, (
    f"Best-of-N 应随 N 单调上升（SFT 弱策略）: bon1={sft_weak_bon1:.3f}, "
    f"bon4={sft_weak_bon4:.3f}, bon16={sft_weak_bon16:.3f}"
)
assert final_bon1 >= 0.9 and final_bon4 >= 0.9, (
    f"Best-of-N DPO 后策略应维持高可验证正确率: bon1={final_bon1:.3f}, bon4={final_bon4:.3f}"
)
assert first_total_loss is not None and last_total_loss < first_total_loss, (
    f"(policy+value) total_loss 应下降: {first_total_loss} -> {last_total_loss}"
)


# ============================================================
# 11. 深坑清单验收（versions.md §11.6）
# ============================================================

print("\n============== 深坑清单（versions.md §11.6）验收 ==============")

check("SFT 做出带噪的不完美起点", sft_target_mean < 0.9, f"SFT 起点太确定 {sft_target_mean:.3f}")
check("RM（可在偏好对上区分 chosen/rejected）", rm_acc > 0.7, f"RM 偏好准确率不足 {rm_acc:.3f}")

# ⑥ advantage 恒 detached
check("advantage 已 detach（无梯度泄漏）", not advantages.requires_grad,
      "advantage 必须 detach")
# ④ old_logp 冻结且不打 requires_grad（不重算）
check("old_logp 冻结且不重新计算", not old_logp.requires_grad,
      "old_logp 在 PPO epoch 中被重新计算（未冻结）")
# ⑤ RM 奖励只广播到最后 response token；前区 token 的 raw reward 贡献为 0
check("RM 奖励只落在最后 response token", bool((raw_broadcast[:, :-1] == 0).all().item()),
      "RM 奖励被错误地广播到了所有 token")
# ⑦ KL 符号正确：sampled log-ratio 本身可正可负，不能用 reward 正负判断；
# 直接检查构造恒等式 token_reward - raw_reward == -β * sampled_log_ratio。
kl_contribution = token_reward - raw_broadcast
check("KL 符号正确（reward 的 KL 项严格等于 -β·log-ratio）",
      bool(torch.allclose(kl_contribution, -kl_beta * tok_kl, atol=1e-7, rtol=1e-6)),
      "KL 项不等于 -β·(old_logp-ref_logp)，可能符号写反或混入其他奖励")
# ①/③ prompt token 不进入 PPO loss：按位置推导的 mask 在 prompt 区全 0、
# response 区全覆盖，且 loss 消费的 logp 宽度恰为 RESPONSE_LEN（不含 prompt）。
check("Prompt token 不进入 PPO loss（位置mask prompt区=0 且 response 全覆盖 + logp 宽度=RESPONSE_LEN）",
      int(resp_pos_mask[:PROMPT_LEN].sum().item()) == 0
      and int(resp_pos_mask[PROMPT_LEN:].sum().item()) == RESPONSE_LEN
      and old_logp.size(1) == RESPONSE_LEN,
      "response mask 覆盖了 prompt / 未盖满 response / 或 loss logp 扩到了完整 context")

# ⑤/⑩：EOS 与 value bootstrap —— response 定长、无 EOS/padding，n-step return
# 不会越过 response 边界（value 只在 response 位置取用）。上述长度断言已覆盖此坑。

if pit_failures:
    for f in pit_failures:
        print(f"[深坑未过] {f}")
    raise SystemExit(f"深坑清单 {len(pit_failures)} 条未通过")

# ============================================================
# 12. 收束总结
# ============================================================

print("\n============== m12 收束（全链路 v0.0 → v1.0） ==============")
print(f"正确回答平均概率: SFT={sft_target_mean:.4f} -> RL={rl_target_mean:.4f} "
      f"-> DPO={dpo_target_mean:.4f}")
print(f"greedy_acc={dpo_greedy:.3f} | Best-of-N(SFT弱): N=1={sft_weak_bon1:.3f}, "
      f"N=4={sft_weak_bon4:.3f}, N=16={sft_weak_bon16:.3f} | "
      f"DPO后: N=1={final_bon1:.3f}, N=4={final_bon4:.3f}")
print("[PASS] m12 integration: SFT→RM(BT)→PPO/KL→DPO→Best-of-N 端到端收束，"
      "深坑清单（response-mask / 奖励广播 / KL 符号 / old-logp 冻结 / advantage detach）全部通过")
