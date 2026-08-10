# learn-rlhf 深度审查与优化设计

日期：2026-08-10

## 目标

把 `learn-rlhf` 提升为一套版本叙事一致、算法实现可信、代码可逐段阅读、每章可独立运行的中文 RLHF 渐进教程。代码组织参考 `learn-claude-code` 的教学风格：文件开头先说明本章唯一核心，随后按清晰分区铺开最小机制，主流程打印可观察证据，复杂生产差异放到文档的深入部分，而不是把教学代码伪装成生产实现。

## 不变量

1. `versions.md` 是版本能力树的唯一来源；根表、模块标题、代码版本注释和模块定位必须与它一致。
2. 每个 `mXX/code.py` 自包含，不导入其他章节；从仓库根目录以 `python mXX/code.py` 运行并以非零退出码暴露失败。
3. 每章的断言证明该章的核心机制，而不只证明张量 shape 或一次偶然输出。
4. 教学简化必须明确标注边界，不能把 sampled KL、PPO ratio、RM reward、verifier reward 等不同概念混写。
5. README 保持 Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper → 模块定位，并与实际代码、指标和命令同步。
6. CPU 默认路径无需 API key；固定随机种子下全量模块可重复通过。
7. 大更新按阶段提交到 `learn-rlhf` 自己的 Git 仓库，提交前必须有与变更范围相称的验证证据。

## 已确认的基线问题

### 阻断级

- `m12_integration/code.py` 当前稳定失败：最终 Best-of-N 使用 PPO `policy`，却以 DPO 的 `greedy_acc` 标注并断言采样准确率达到 0.9；实测为 0.75。

### 高优先级

- `versions.md` 的 v0.6/v0.7 章节、总表、建议路线彼此冲突；根 README、AGENTS 与 m07/m08 又采用另一套标签。
- m12 声称演示 RM→PPO，但 rollout 实际使用 `exact_match_reward`，训练出的 RM 没有进入 PPO reward。
- m05、m06、m12 把包含 `value_head` 的 `policy.parameters()` 与 `value_head.parameters()` 分别交给两个 Adam；一次 backward 后 value head 被重复更新并维护两套优化器状态。
- m12 用“非末 token reward 必须非正”判断 KL 符号；sampled log-ratio 可为负，该断言并不能证明代数符号正确。
- m12 有重复前向、死变量；m10 有未使用的 SFT loss；m06 有难读的单字母变量。

### 教学一致性

- 根 README 只有最小链接表，缺少贯穿版本树的学习方式、验证方法和“教学简化 vs 生产系统”边界。
- 部分 README 的指标、模块前后关系和源码已经漂移，需要逐章做 code↔doc 对账。
- 现有自验证覆盖较多，但缺少一个统一、可重复的全仓验收命令与结果口径。

## 设计决策

### 版本轴

保留当前模块的渐进顺序：m06 GAE → m07 多目标 → m08 生产系统。因此统一 `versions.md` 为 `v0.6=多维偏好与安全约束`、`v0.7=生产级 RLHF`，再让总表、建议路线与所有模块锚点服从这一轴。原因是教程目录、模块依赖和概念递进已经按这一顺序形成；交换目录会制造更大的路径断裂，而仅保留原文相互矛盾的标签不能满足“source of truth”。

### 算法审查口径

- SFT：response-only 目标、损失下降与行为指标。
- RM：Bradley–Terry 符号、偏好准确率、margin，并声明域内局限。
- REINFORCE/PPO：采样分布、冻结 old/reference、ratio、clip、KL 估计、baseline/GAE、梯度边界。
- DPO：chosen/rejected 的 policy-reference log-ratio 差，冻结 reference，长度假设明确。
- RLAIF/RLVR：反馈来源与优化算法分离，verifier reward 不冒充 RM reward。
- Production：明确是单进程架构仿真，检查 buffer、adaptive KL、checkpoint round-trip 与恢复元数据。

### 验收层级

1. 静态：全部 Python 可由 `ast.parse` 解析；版本锚点与 README 表一致；无已知死变量/重复语句。
2. 单章：每个 `code.py` 的机制断言通过并打印 `[PASS]`。
3. 全仓：固定顺序运行 m01–m12，全部退出码为 0；m08 生成物保持 ignored。
4. 文档：命令、阶段指标和代码变量逐章抽查；链接与模块定位无断链。

## 非目标

- 不引入外部模型、数据集、API key、GPU 分布式依赖或 pytest 框架。
- 不把教学仿真宣称为真实 vLLM/FSDP/多机训练。
- 不为了追求生产抽象而让章节跨模块 import。

