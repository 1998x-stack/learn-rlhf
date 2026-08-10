# learn-rlhf 深度审查与优化实施计划

日期：2026-08-10  
依据：[深度审查与优化设计](../specs/2026-08-10-learn-rlhf-deep-review-design.md)

## 阶段 0：建立证据基线

- [x] 阅读仓库约束、版本树、根 README 与内部 Git 历史。
- [x] 对照 `learn-claude-code` 的代码分区、可运行入口与“核心机制优先”写法。
- [x] AST 检查 m01–m12。
- [x] 实跑 m01–m12，记录 m12 的稳定失败。
- [x] 静态扫描死变量、重复更新、版本漂移与 code↔doc 矛盾。

提交边界：本设计与计划文档落盘。

## 阶段 1：修复真相源与阻断问题

- [x] 统一 `versions.md`、AGENTS、根表、m00、m07、m08、m12 的 v0.6/v0.7 语义。
- [x] 修复 m12 的最终策略身份、Best-of-N 指标和稳定失败。
- [x] 清除 m12 死变量，并把 KL 符号检查改为代数恒等式。
- [x] 用静态检查和 m07/m08/m12 实跑验证。

提交边界：版本轴与 capstone 恢复可信、可运行。

## 阶段 2：审查并深化核心训练链 m01–m06

- [x] 逐章核对数据、目标函数、梯度流、冻结边界、指标与 README。
- [x] 修复 m05/m06 value head 的重复 optimizer step；采用单 optimizer 联合 loss。
- [x] 检查 REINFORCE/PPO sampled KL 的表述和断言是否准确。
- [x] 清理可读性问题，为关键机制补最小而有力的机制断言。
- [x] 实跑 m01–m06，并核对输出与 README。

提交边界：SFT→RM→REINFORCE→PPO→token PPO→GAE 核心链完成。

## 阶段 3：审查并深化高级主题 m07–m11

- [x] m07：多目标聚合、硬约束、reward hacking 与 bias 指标。
- [x] m08：rollout/buffer/trainer 解耦、adaptive KL、checkpoint 恢复语义。
- [x] m09：DPO log-ratio、reference 冻结、chosen/rejected 诊断。
- [x] m10：反馈合成与 DPO 优化分层，移除死变量并增强衡器自检。
- [x] m11：GRPO 退化组、group advantage 与 Best-of-N 统计口径。
- [x] 实跑 m07–m11，并同步相关 README。

提交边界：多目标/生产/DPO/RLAIF/RLVR 高级链完成。

## 阶段 4：重整 m12 与全局教学文档

- [x] 让 m12 的每个阶段名、实际 reward 来源、policy 身份和总结指标严格一致。
- [x] 明确经典 RM→PPO 与 verifier/RLVR 的边界；两者分别命名并验证。
- [x] 修复共享 value head 的优化器所有权。
- [x] 更新根 README：学习路径、运行/验收命令、教学边界、模块能力矩阵。
- [x] 对所有 README 做模板、版本、命令、指标、前后序链接审计。

提交边界：capstone 与全局文档完成。

## 阶段 5：完成度审计

- [x] AST 全量通过。
- [x] m01–m12 全量从仓库根目录运行，全部退出 0 且打印 `[PASS]`。
- [x] 静态扫描无本计划列出的死变量、重复更新和版本漂移。
- [x] `git diff --check` 通过；ignored checkpoint 不进入索引；工作区只含预期变更。
- [x] 按设计“不变量”逐项核对当前文件与命令证据。
- [x] 仅在所有证据齐全后结束目标。

提交边界：审计中若需要实质修复则单独提交；纯验证不制造空提交。
