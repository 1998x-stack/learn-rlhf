# Learn RLHF — 从人类偏好到策略优化（PyTorch 递进式教程）

从零用 PyTorch 搭建 RLHF（Reinforcement Learning from Human Feedback）系统，覆盖 `learn-rlhf/versions.md` 的完整迭代路线：**SFT（v0.0）→ Reward Model（v0.1）→ REINFORCE/KL（v0.2）→ 离散 PPO MVP（v0.3）→ Token-level PPO（v0.4）→ GAE（v0.5）→ 多目标奖励（v0.6）→ 生产分布式 RLHF（v0.7）→ DPO（v0.8）→ RLAIF（v0.9）→ 可验证奖励（v1.0）**。

本教程是 learn-claude-code / learn-llm / learn-seq2seq 系列在"从人类偏好训练语言模型"方向的延伸。每个模块一个目录，含中文教程（Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper）与自包含可运行 `code.py`（CPU 秒级跑通）。

| 模块 | 里程碑 | 覆盖版本 | 描述 |
|------|--------|---------|------|
| **m00_overview** | RLHF 全景 | — | 学习路线 + 版本树导览 |
| **m01_sft_policy** | 离散回答级 Policy + SFT | v0.0 | prompt→候选回答，CE loss，建立参考策略 |
| **m02_reward_model** | Bradley–Terry Reward Model | v0.1 | 偏好对、BT loss、accuracy、margin |
| **m03_reinforce** | REINFORCE + KL | v0.2 | 采样、RM 奖励、`-R·logπ`、KL penalty |
| **m04_ppo_mvp** | PPO + Value（离散 MVP） | v0.3 | old policy、ratio、clip、advantage、entropy |
| **m05_tiny_lm** | 字符级 Tiny LM + Token-level PPO | v0.4 | response mask、token KL、序列奖励分配 |
| **m06_gae** | GAE + 完整 token-level PPO | v0.5 | TD error、GAE、value clip、whitening |
| **m07_multi_objective** | 多目标 Reward 聚合 | v0.6 | 多奖励加权/硬约束、bias 评估 |
| **m08_production_rlhf** | 分布式/生产 RLHF | v0.7 | rollout 解耦、buffer、checkpoint、adaptive KL |
| **m09_dpo** | DPO 离线偏好优化 | v0.8 | 移除 RM/rollout，直接优化 policy |
| **m10_rlaf** | RLAIF + 规则反馈 | v0.9 | Judge 偏好、多级验证器 |
| **m11_verifiable_rl** | 可验证奖励 / 推理 RL | v1.0 | outcome/process reward、GRPO、best-of-N |
| **m12_integration** | 端到端收束 + 深坑清单 | 收束 | SFT→RM→RL→verifier 全链路 |

## Quickstart

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 从 m00 开始阅读
# 每模块目录下均有 README.md（中文教程）+ code.py（可运行代码）

# 3. 运行某个模块（以 m04 为例）
python m04_ppo_mvp/code.py
```

## 链接

- [m00_overview/](m00_overview/)
- [m01_sft_policy/](m01_sft_policy/)
- [m02_reward_model/](m02_reward_model/)
- [m03_reinforce/](m03_reinforce/)
- [m04_ppo_mvp/](m04_ppo_mvp/)
- [m05_tiny_lm/](m05_tiny_lm/)
- [m06_gae/](m06_gae/)
- [m07_multi_objective/](m07_multi_objective/)
- [m08_production_rlhf/](m08_production_rlhf/)
- [m09_dpo/](m09_dpo/)
- [m10_rlaf/](m10_rlaf/)
- [m11_verifiable_rl/](m11_verifiable_rl/)
- [m12_integration/](m12_integration/)