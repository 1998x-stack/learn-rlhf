"""m10 RLAIF — AI 反馈 / 规则反馈 (v0.9).

与 m09 纯人类偏好对照：这里的 (chosen, rejected) 偏好对不再需要人手工标注，
而是由"规则/可执行验证器"或"启发式 Judge"自动合成 —— 这就是 RLAIF
（Reinforcement Learning from AI Feedback, Bai et al., 2022）。

任务选"可验证算术题"：正确性可计算，所以能用一台验证器合成偏好。
演示验证器层级：规则 > 可执行 > 异源 Judge > 同源 Judge，以及长度/自我偏置
Judge 比规则衡器更易被欺骗，最终 policy 的真实正确率也更低。

Run:  python m10_rlaf/code.py
"""

# --- v0.9: AI 合成偏好 / RLAIF ---
from __future__ import annotations

import os
import re
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

torch.manual_seed(42)

# --- v0.9: 设备——GPU 可用则用 GPU，否则 CPU（同 m05–m09）---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 0. 可验证算术数据（正确性"可计算"，因此可自动合成偏好）
# ============================================================
# 每条 = (表达式, 通过 eval 计算出的正确整数答案)。
# 正确性可验证 → 无需人工标注就能知道"哪个候选是对的"，
# 这正是 RLAIF 选算术做演示的原因：ground-truth 由机器给出。
ARITH: list[tuple[str, int]] = [
    ("3+5", 8),
    ("12-7", 5),
    ("6*4", 24),
    ("7+8", 15),
    ("20+14", 34),
    ("9*9", 81),
    ("100/25", 4),
    ("28-13", 15),
]


def _verbose_wrong(expr: str, wrong: int) -> str:
    """生成长篇但答案为错误数字的'漂亮'回答（用于演示 verbosity bias）。"""
    return (
        f"这个问题非常有趣！让我们一步一步地拆解题目{expr}，"
        f"先回顾加法和乘法的基本法则，再对多个选项进行反复推敲与交叉验证，"
        f"最终经过严谨的推导，我可以百分之百确定结果的数值就是{wrong}。"
    )


def build_candidates(expr: str, correct: int) -> list[tuple[str, bool]]:
    """每个 prompt 构造三个候选：(文本, 是否携带正确答案)。

    候选0: 正确但极简短（如 "8"）。
    候选1: 错误数字但写成长篇'漂亮'回答（专诱长度偏置的 Judge）。
    候选2: 错误且简短丑陋。
    """
    wrong_v = correct + 1
    wrong_t = wrong_v + 7
    return [
        (str(correct), True),                     # index 0: 正确 简洁
        (_verbose_wrong(expr, wrong_v), False),   # index 1: 错误 冗长漂亮
        (str(wrong_t), False),                    # index 2: 错误 简短
    ]


_prompts: list[str] = []
_candidates: list[list[tuple[str, bool]]] = []
for expr, correct in ARITH:
    _prompts.append(f"{expr}=？")
    _candidates.append(build_candidates(expr, correct))

num_prompts = len(ARITH)
num_actions = 3

# 候选下标语义常量（由 build_candidates 固定）。
CORRECT_IDX = 0
VERBOSE_IDX = 1
SHORT_IDX = 2

prompt_ids = torch.arange(num_prompts, device=device)
# sft_labels：仅一半 prompt 的示范指向正确答案 → 含噪声 SFT 起点。
# prompt 0..3 → 正确答案 0；prompt 4..7 → 错误候选 1，真实可验证正确率约 50%。
sft_labels = torch.tensor(
    [CORRECT_IDX] * 4 + [VERBOSE_IDX] * 4,
    device=device,
)


# ============================================================
# 1. Policy 与 可执行 / 规则衡器（v0.9 顶层：rule ≈ executable）
# ============================================================

class PolicyModel(nn.Module):
    """离散回答级策略：prompt_id → logits[num_actions]。"""

    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        logits = self.policy_head(hidden)
        return logits


def executable_reward(expr: str, candidate: str) -> float:
    """可执行验证器：真正把表达式 eval 出来，再与回答里的数字比对。

    顶层信号：正确性可计算 → 无需人去判。
    """
    correct = int(eval(expr))  # 真正执行一遍"机器"
    matched = re.search(r"\d+", candidate)
    if matched is None:
        return 0.0
    return 1.0 if int(matched.group()) == correct else 0.0


def rule_reward(expr: str, candidate: str) -> float:
    """规则验证器：候选必须是一个紧凑整数且等于真解。

    对算术题，"规则"与"可执行"在此重合；引入只为说明验证器层级中"规则"
    是最强先验。真实场景下规则衡器是精确的形式化准则。
    """
    if re.fullmatch(r"\d+", candidate) is None:
        return 0.0
    return 1.0 if int(candidate) == int(eval(expr)) else 0.0


def length_judge(candidate: str) -> float:
    """启发式'Judge'：只数长度（verbosity bias）——它被"漂亮的长回答"耍弄。

    对应 v0.9 风险之一："偏好长度更长、格式更漂亮的回答"。注意这里
    length_judge 是一个独立于 policy 的"异源但偏置"信号源。
    """
    return float(len(candidate))


# ============================================================
# 2. RLAIF：用衡器自动合成 (chosen, rejected) 偏好数据（无人工标注）
# ============================================================

def synthesise_rule_prefs() -> list[tuple[int, int, int]]:
    """规则衡器偏好：把正确答案选为 chosen，其余错误候选为 rejected。"""
    pairs: list[tuple[int, int, int]] = []
    for p in range(num_prompts):
        pairs.append((p, CORRECT_IDX, VERBOSE_IDX))
        pairs.append((p, CORRECT_IDX, SHORT_IDX))
    return pairs


def synthesise_biased_prefs() -> list[tuple[int, int, int]]:
    """长度偏置'Judge'偏好：长度最长=chosen，长度最短=rejected。

    因候选 1 是故意写长的错误'漂亮'回答，该偏好会反复把错误答案标为 chosen
    → 量化 judge bias 对真实正确性的伤害（vs 规则衡器）。
    """
    pairs: list[tuple[int, int, int]] = []
    for p in range(num_prompts):
        cands = _candidates[p]
        scores = [length_judge(c) for c, _ in cands]  # 由"异源但偏置"的长度 Judge 打分
        argmin = int(torch.tensor(scores).argmin())
        argmax = int(torch.tensor(scores).argmax())
        pairs.append((p, argmax, argmin))
    return pairs


def synthesise_same_prefs(policy: nn.Module) -> list[tuple[int, int, int]]:
    """同源 Judge：把'当前 policy 自己最偏好的答案'标为 chosen（自我偏好）。

    因 Judge 与 Policy 同源、共享同一套错误，偏好只会固化 policy 已有的偏差，
    无法纠正错误 —— 演示验证器层放"同源 Judge"垫底的风险。
    """
    with torch.no_grad():
        log_p = F.log_softmax(policy(prompt_ids), dim=-1)
        argmax = log_p.argmax(dim=-1)
    pairs: list[tuple[int, int, int]] = []
    for p in range(num_prompts):
        ch = int(argmax[p])
        other = 1 if ch != 1 else 0
        pairs.append((p, ch, other))
    return pairs


def dpo_loss(log_pi: torch.Tensor, logref: torch.Tensor,
             prefs: list[tuple[int, int, int]], beta: float) -> torch.Tensor:
    """数值稳定 DPO 损失（v0.8）：-logsigmoid(β(logπ_w - logπ_l))。"""
    total = torch.tensor(0.0, dtype=torch.float32, device=device)
    for prompt_idx, chosen, rejected in prefs:
        log_w = log_pi[prompt_idx, chosen] - logref[prompt_idx, chosen]
        log_l = log_pi[prompt_idx, rejected] - logref[prompt_idx, rejected]
        total = total + -F.logsigmoid(beta * (log_w - log_l))
    return total / len(prefs)


def sft_pretrain(policy: nn.Module, steps: int = 30, lr: float = 1e-2) -> float:
    """含噪声 SFT 起点；因 prompt 4..7 指向错误候选，真实可验证准确率约 50%。"""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss: torch.Tensor | None = None
    for _ in range(steps):
        logits = policy(prompt_ids)
        loss = F.cross_entropy(logits, sft_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss is not None and loss.item() < 1.0, "SFT loss 未下降"
    return float(loss.item())


def policy_accuracy(policy: nn.Module) -> float:
    """真实可验证正确率：argmax 候选是否携带正确答案。"""
    with torch.no_grad():
        probs = F.softmax(policy(prompt_ids), dim=-1)
        argmax = probs.argmax(dim=-1)
    ok = sum(1 for p in range(num_prompts)
             if _candidates[p][int(argmax[p])][1])
    return ok / num_prompts


def train_dpo(policy: nn.Module, ref_state: dict, prefs: list[tuple[int, int, int]],
              beta: float = 1.0, steps: int = 300, lr: float = 1e-2) -> float:
    """在给定离线偏好队上对 policy 做 DPO；ref 冻结为 SFT 起点。"""
    ref = PolicyModel(num_prompts=num_prompts, num_actions=num_actions).to(device)
    ref.load_state_dict(ref_state)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    logref = F.log_softmax(ref(prompt_ids), dim=-1)
    loss: torch.Tensor | None = None
    for _ in range(steps):
        log_p = F.log_softmax(policy(prompt_ids), dim=-1)
        loss = dpo_loss(log_p, logref, prefs, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss is not None and loss.item() < 1.0, "DPO loss 未下降"
    return float(loss.item())


def main() -> None:
    print(f"m10 RLAIF · 多级验证器合成偏好 (v0.9)   device={device}")

    expr_for = {i: ARITH[i][0] for i in range(num_prompts)}

    # ---- v0.9 Step 1: 衡器自检 + SFT 起点 ----
    # 先自证"顶层衡器确实能区分正确候选"。
    rule_ok = sum(
        1 for p in range(num_prompts)
        if rule_reward(expr_for[p], _candidates[p][CORRECT_IDX][0]) == 1.0
    )
    exec_ok = sum(
        1 for p in range(num_prompts)
        if executable_reward(expr_for[p], _candidates[p][CORRECT_IDX][0]) == 1.0
    )
    print(f"[衡器自检] 规则衡器识别正确候选 {rule_ok}/{num_prompts}；"
          f"可执行衡器 {exec_ok}/{num_prompts}")

    sft_policy = PolicyModel(num_prompts=num_prompts, num_actions=num_actions).to(device)
    sft_loss = sft_pretrain(sft_policy)
    sft_state = {k: v.detach().clone() for k, v in sft_policy.state_dict().items()}
    base_acc = policy_accuracy(sft_policy)

    # ---- v0.9 Step 2: 三路 AI/衡器偏好数据 ----
    rule_prefs = synthesise_rule_prefs()
    biased_prefs = synthesise_biased_prefs()
    same_prefs = synthesise_same_prefs(sft_policy)  # 同源 Judge 来自 SFT policy

    # ---- v0.9 Step 3: 分别在各偏好集上 DPO ----
    rule_pol = PolicyModel(num_prompts=num_prompts, num_actions=num_actions).to(device)
    rule_pol.load_state_dict(sft_state)
    train_dpo(rule_pol, sft_state, rule_prefs)

    biased_pol = PolicyModel(num_prompts=num_prompts, num_actions=num_actions).to(device)
    biased_pol.load_state_dict(sft_state)
    train_dpo(biased_pol, sft_state, biased_prefs)

    same_pol = PolicyModel(num_prompts=num_prompts, num_actions=num_actions).to(device)
    same_pol.load_state_dict(sft_state)
    train_dpo(same_pol, sft_state, same_prefs)

    rule_acc = policy_accuracy(rule_pol)
    biased_acc = policy_accuracy(biased_pol)
    same_acc = policy_accuracy(same_pol)

    print(f"\n可验证正确率   SFT基线={base_acc:.2f}   规则衡器RLAIF={rule_acc:.2f}   "
          f"长度偏置Judge={biased_acc:.2f}   同源Judge={same_acc:.2f}")

    # ---- v0.9 Step 4: 机制级断言 ----
    assert rule_acc > base_acc, "规则/AI 偏好 DPO 未提升真实可验证正确率"
    assert rule_acc > biased_acc, "规则衡器被长度偏置 Judge 反超（层级失效）"
    assert rule_acc > same_acc, "规则衡器未胜过同源自我偏好（层级失效）"
    assert base_acc >= same_acc, "同源偏好不应把正确率提到基线以上"

    print("\n[断言] 训练于规则/AI 合成偏好的 policy 真实可验证正确率 高于含噪声 SFT 基线")
    print("[断言] 规则衡器路径的真实正确率 高于 长度偏置 Judge 路径"
          "（量化了 Judge 偏差 + 层级收益）")
    print("[断言] 规则衡器路径的真实正确率 高于 同源自我偏好路径"
          "（消除自我偏好风险）")
    print("[PASS] m10 rlaf: 可计算任务的正确性可由验证器自动合成偏好（无需人工标注），"
          "其 policy 真实正确率高于 SFT 基线，且规则衡器胜过长度偏置与同源 Judge")


if __name__ == "__main__":
    main()