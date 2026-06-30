# 比赛题目分析和相关资料调研

## 题目理解

本项目围绕 K3/K3x 芯片建立一套 AI 运行时与调度框架。设计重点在于：

- K3/K3x 上的 X100、A100、CPU fallback、DMA、TCM 和实时控制路径的协同。
- AI 推理任务应该在用户态、内核态和硬件执行单元之间的分层。
- 当任务规模、核心负载和实时性要求变化时，系统如何选择 X100、A100 或 CPU fallback。

## K3/K3x AI 算力结构分析

根据 K3/K3x 公开资料和官方 AI 文档，平台可以按 X100 cluster 与 A100 cluster 两类计算域理解：

- X100 侧承担通用计算、系统调度、用户态入口和轻量 AI 任务。资料中 X100 core 旁也标出 AI 模块，因此文档中暂称为 `X100 local tensor path`。
- A100 侧面向重型 AI 计算，包含 Scalar Processor、Vector Processor、Tensor Processor、L2 Cache 和 TCM，更适合大矩阵乘法、LLM 核心算子和长 DAG。
- CPU fallback 用于正确性兜底、异常降级和暂未支持算子的兼容执行。

X100 和 A100 的向量宽度不同，线程一旦使用某类向量/IME 状态，就不能随意跨核心类型迁移。这直接影响系统调度、线程亲和性和上下文保存恢复策略。

## 资料调研

调研主要围绕以下材料展开：

- SpacemiT AI CPU 与 IME 指令资料：确认 A100 Tensor Processor、Integer Tensor、Float Tensor 和 IME 子扩展能力。
- SpacemiT K3 HMP 文档：确认 Linux 侧通过 `/proc/set_ai_thread` 标记 AI 线程，并约束到 AI 核心集合。
- SpacemiT 交叉编译工具链资料：确认 A100 专用 IME 指令需要官方工具链支持，普通 RISC-V 工具链不足以覆盖扩展指令。
- `linux-k3` 公开驱动：检查是否存在在传统 `/dev/npu` 或 `/dev/a100` command-submission 风格驱动,避免后期实现时丢失关键信息。
- StarryOS/tgoskits： 查看os完成K3板级适配的程度,评估后期工作量.

相关资料入口：

- [SpacemiT AI 矩阵扩展指令集](https://github.com/spacemit-com/docs-ai/blob/main/zh/architecture/ime_extension.md)
- [SpacemiT AI CPU 资料目录](https://github.com/spacemit-com/docs-ai/blob/main/zh/index.md)
- [K3 HMP 说明](https://github.com/spacemit-com/docs-buildroot/blob/main/zh/k3_buildroot/device/hmp.md)
- [SpacemiT 交叉编译工具链使用指南](https://www.spacemit.com/community/document/info?lang=zh&nodepath=tools/user_guide/cross_compiler_user_guide.md)
