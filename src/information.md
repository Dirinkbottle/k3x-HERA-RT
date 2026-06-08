# 特别说明和信息

本页用于集中放置项目中的术语说明、硬件信息、阶段性判断和后续需要反复引用的背景材料。

## X100 local tensor path

`X100 local tensor path` 指 X100 核心侧可用于轻量 AI/Tensor/RVV/IME 类任务的本地执行路径。它用于描述小 tensor、短突发、低提交开销的本地 AI 计算候选，而不是 A100 Tensor Processor 的替代品。

在调度策略中，X100 local tensor path 适合优先处理小任务；当该路径任务过多、核心被占用或本地执行成本超过阈值时，任务可以卸载到 A100 candidate。

目前 X100 侧具体支持哪些 tensor/IME 能力仍需要后续通过工具链、hwcap/cpufeature 和真机非法指令探针确认。因此文档中使用 `X100 local tensor path` 作为暂定术语，避免过早绑定到某个确定 ISA 子扩展集合。

## A100 tensor path

`A100 tensor path` 指 A100 AI 核心侧面向重型 AI 推理的 Tensor/IME 执行路径。A100 更适合大矩阵乘法、LLM 核心算子、大 batch、长 DAG 和高吞吐推理场景。

从架构上看，A100 core 具备 Scalar Processor、Vector Processor 和 Tensor Processor。Tensor Processor 对应 SpacemiT IME，包含 Integer Tensor 与 Float Tensor 路径，并配合 A100 cluster 侧的 TCM/缓存资源服务重型 AI 计算。

在调度策略中，A100 tensor path 是重型任务的主要执行候选；当所有 A100 核心都接近满载或队列拥塞时，调度器可以把具备等价 backend 的溢出任务卸载到 X100 local tensor path 或 CPU fallback。

## DirectInline

`DirectInline` 指用户态算子库直接在当前执行上下文中调用 backend，不经过内核调度器进行核心队列调度。它适合小任务、单进程低负载和对提交延迟敏感的场景。

## KernelScheduled

`KernelScheduled` 指用户态将 lowered graph 或 backend job 提交给内核调度器，由调度器根据队列、依赖和资源状态选择目标核心执行。它适合多核心、高负载、多进程竞争、长 DAG 和需要统一资源管理的场景。
