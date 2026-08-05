# learn-rlhf — Learn RLHF from Human Preference to Policy Optimization（PyTorch 递进式教程）

Progressive 13-module tutorial (m00–m12) building an RLHF (Reinforcement Learning from Human Feedback) system from scratch in PyTorch (v0.x → v1.x covers in `versions.md`). Each module is a directory with a Chinese `README.md` tutorial (Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper) + a self-contained, runnable `code.py`（CPU 秒级跑通）.

This is a **teaching repository**, not an application. Modules are designed to be read in order (m00 → m12). The source-of-truth version tree lives in `versions.md`; each `README.md` table links back to the root via `[返回根目录](../README.md)`. Do not modify unless explicitly asked.

## Architecture

### Module dependency graph (runtime)

Every `code.py` is **self-contained** — later modules re-implement the pieces they need rather than importing earlier modules. The dependency is *conceptual*, not import-based:

```
m01_sft_policy       离散回答级 Policy + SFT (v0.0)
        ↓ concept
m02_reward_model     Bradley–Terry Reward Model (v0.1)
        ↓ concept
m03_reinforce        REINFORCE + KL (v0.2)
        ↓ concept
m04_ppo_mvp          PPO + Value（离散 MVP）(v0.3)
        ↓ concept
m05_tiny_lm          字符级 Tiny LM + Token-level PPO (v0.4)
        ↓ concept
m06_gae              GAE + 完整 token-level PPO (v0.5)
        ↓ concept
m07_multi_objective  多目标 Reward 聚合 (v0.6)
        ↓ concept
m08_production_rlhf  分布式/生产 RLHF (v0.7)
        ↓ concept
m09_dpo              DPO 离线偏好优化 (v0.8)
        ↓ concept
m10_rlaf             RLAIF + 规则反馈 (v0.9)
        ↓ concept
m11_verifiable_rl    可验证奖励 / 推理 RL (v1.0)
        ↓ concept
m12_integration      端到端收束 + 深坑清单
```

Data flows conceptually: `prompt → Policy sample →（RM reward / KL / advantage）→ policy updates（SFT→RL→verifier 全链路）`.

### Version coverage

Covered versions (per `versions.md` 迭代路线):

| Series | Modules | Milestone |
|---|---|---|
| v0.x (核心迭代) | m01–m10 | 从 SFT / Reward Model 到生产式 RLHF 与离线偏好 |
| v1.0 (可验证奖励) | m11 | outcome/process reward、GRPO、best-of-N |

m00（全景）是先行基础，不对应具体 v0.x 版本；m12 是端到端收束，覆盖 SFT→RM→RL→verifier 全链路。

## Environment

This teaching series needs **no external API keys**. All modules run locally on CPU with bundled synthetic / small corpora. No `.env` setup required.

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
```

## Commands

```bash
# Setup
pip install -r requirements.txt

# Run a single module
python m04_ppo_mvp/code.py

# Cleaner alternative when the runnable file is `code.py`
# (`code.py` shadows the Python stdlib `code` module, so running as
# `python mXX/code.py` from the module dir can hit the name collision).
# Run from the repo root instead:
python -m m04_ppo_mvp.code
python -m m09_dpo.code

# Syntax check only (no heavy compute)
python -c "import ast; ast.parse(open('m04_ppo_mvp/code.py').read())"
```

## Key conventions

### README template (learn-claude-code style)

All `README.md` files follow: **Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper → 模块定位**.

Every module README must:
- Start from `[返回根目录](../README.md)` back-link so learners can return to the module table.
- List the version(s) it covers (matching the root table's `覆盖版本` column and `versions.md`).
- `code.py` is **self-contained and runnable** (`python mXX_*/code.py`) — it never imports from other modules.
- Each module build step carries a **version-tag comment** (e.g. `# v0.1`, `# v0.4`, `# v1.0`) so the incremental narrative stays anchored to `versions.md`.

## Testing

No separate test framework. Each `code.py` **self-verifies in its `__main__` block** (shape assertions, loss-decreases checks, reward/KL smoke tests). Syntax can still be checked with `ast.parse`. Do not add pytest to this repo.

## Git

- This is an **independent git repo** initialized at `learn-rlhf/.git/` — not the parent `Agent-Tutorials` repo.
- `versions.md` is the source-of-truth and is already committed.
- Commit only module/work files inside `learn-rlhf/`; never stage the parent repo as a nested gitlink.
- `.gitignore` holds off `*.pt` / `*.ckpt` / `checkpoints/` so the m08/m12 checkpoint demos do not bloat the repo.