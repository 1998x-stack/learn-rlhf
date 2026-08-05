"""m01 SFT Policy — 离散回答级策略 + 监督微调 (v0.0).

Run:  python m01_sft_policy/code.py
"""

# --- v0.0: 离散 Policy + SFT ---
from __future__ import annotations

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

torch.manual_seed(42)

prompts = [
    "1+1等于多少？",
    "天空为什么是蓝色？",
    "如何安全地过马路？",
    "什么是过拟合？",
]
responses = [
    ["2。", "3。", "这个问题没有答案。"],
    ["因为大气对短波长可见光的散射更强。", "因为海洋把天空染蓝。", "因为蓝色比较好看。"],
    ["看信号灯、左右观察并走斑马线。", "闭眼快速跑过去。", "只要没有喇叭声就可以走。"],
    ["模型记住训练数据而泛化较差。", "模型训练速度太慢。", "模型参数太少。"],
]
num_prompts = len(prompts)
num_actions = len(responses[0])
prompt_ids = torch.arange(num_prompts)
# 故意构造一个带少量噪声的 SFT 数据：
# Prompt 1 和 Prompt 3 的 SFT 演示并不是最佳答案。
sft_labels = torch.tensor([0, 1, 0, 1])

class PolicyModel(nn.Module):
    def __init__(self, num_prompts: int, num_actions: int, hidden_size: int = 16):
        super().__init__()
        self.prompt_embedding = nn.Embedding(num_prompts, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.prompt_embedding(prompt_ids)
        logits = self.policy_head(hidden)
        return logits

def print_policy(title: str, model: nn.Module) -> None:
    print(f"\n===== {title} =====")
    with torch.no_grad():
        probabilities = F.softmax(model(prompt_ids), dim=-1)
    for prompt_id, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt}")
        for action_id, response in enumerate(responses[prompt_id]):
            p = probabilities[prompt_id, action_id].item()
            print(f"  P={p:.4f} | {response}")

def main() -> None:
    policy = PolicyModel(num_prompts=num_prompts, num_actions=num_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    loss = None
    for step in range(60):
        logits = policy(prompt_ids)
        loss = F.cross_entropy(logits, sft_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # shape + loss assertions
    assert policy(prompt_ids).shape == (num_prompts, num_actions)
    assert loss is not None and loss.item() < 1.0, "SFT loss 未下降"
    print(f"\n最终 SFT loss = {loss.item():.4f}")
    print_policy("SFT 后的策略", policy)
    print("[PASS] m01 sft_policy: 离散 Policy 在 SFT 数据上收敛（loss 下降、输出 shape 正确）")

if __name__ == "__main__":
    main()