# 开发进度

## 演示视频说明

所有阶段的演示视频都基于阶段一改造后的模型推理后端展开。后续阶段不更换演示基础，而是在同一套或可对照的 LLM、YOLO 等推理场景中，分别展示单核心算力限制、静态多核心分配的残留问题、动态负载均衡的改善效果，以及异步 AI 运行时在长时间高强度压力下的稳定性。

## 阶段一：纯线程最小闭环

| 任务 | 目标 |
|---|---|
| [*] 用户态 frontend 与 ring API | 完成用户态算子结构、最小 UAPI、ring entry 和 uring 用户 API（3 天） |
| [*] 纯线程内核执行路径 | 完成纯线程调度器、核心队列、worker、buffer pin/map/cache 最小路径（3 天） |
| [*] completion 回到用户态 | job/node/graph 完成后能通知用户态等待方（2 天） |
| [*] CPU fallback MatMul | 用 CPU fallback 跑通 MatMul 正确性（2 天） |
| [*] A100/X100 IME probe | 验证 A100/X100 IME backend 的最小执行与结果正确性（2 天） |
| [*] 计划外平台侧适配与驱动补齐 | 原先预期 K3 平台侧适配已有其他人接手，但实际推进中需要临时补齐真板启动、驱动和 V 扩展上下文支持，明显占用了阶段一时间 |
| [ ] ggml `mul_mat` 接入 | llama.cpp 的核心矩阵乘法能走用户态算子库（5 天） |
| [ ] llama.cpp 文本生成跑通 | 改造后的 ggml backend 能跑通一个文本生成模型（5 天） |

### 阶段一子任务拆分

- [x] 用户态 frontend 与 ring API（3 天）
  - [x] 用户态 frontend 小 DAG、算子结构体与调用入口
  - [x] 最小 MatMul UAPI / submit struct / ring entry
  - [x] graph submit 到 ring
  - [x] uring 用户 API
- [x] 纯线程内核执行路径（3 天）
  - [x] 用户态 backend 算子库骨架
  - [x] 纯线程内核调度器
  - [x] 核心队列与 worker
  - [x] buffer pin/map/cache 最小路径
- [x] completion 回到用户态（2 天）
  - [x] completion ring entry
  - [x] token 匹配与错误码返回
  - [x] 用户态等待/唤醒 API
- [*] CPU fallback MatMul（2 天）
  - [x] CPU fallback MatMul 实现
  - [*] MatMul golden case 对比
- [ ] A100/X100 IME probe（2 天）
  - [*] 使用官方支持文档和工具链编写最小 IME probe
  - [*] 验证 A100/X100 IME 执行结果正确性
- [x] 计划外平台侧适配与驱动补齐
  - [x] K3 真板启动路径和 rootfs/UFS 启动验证
  - [x] UFS 驱动迁移和调试
  - [x] RISC-V DMA cache 同步问题定位与补齐
  - [x] `/proc/set_ai_thread` AI 线程控制接口
  - [x] RISC-V V 扩展上下文保存恢复
  - [x] V 扩展同线程、跨线程和真板测试
- [ ] ggml `mul_mat` 接入（5 天）
  - [-] 找到 ggml `mul_mat` backend 接入点
  - [x] 加入 `k3_ggml_bridge` 测试版，用 C ABI 接收 F32 MatMul 请求
  - [x] 将测试版 F32 MatMul 请求转换为阶段一 UAPI graph
  - [x] 接入用户态算子库和 graph submit 测试路径
  - [-] 接入真实 ggml backend
  - [ ] 补充 dtype/layout/quant 支持
  - [ ] 对比 ggml CPU fallback 输出
- [ ] llama.cpp 文本生成跑通（5 天）
  - [X] 固定模型、prompt 和运行参数
  - [ ] 跑通文本生成
  - [ ] 记录 tokens/s、latency 和错误日志

## 阶段一计划外插入：平台侧适配与驱动补齐

阶段一原本主要计划推进 AI runtime、UAPI、调度器和 backend 闭环。当时预期 K3 平台侧适配、真板启动和基础驱动已有其他人接手，因此这些工作没有放进最初的 AI runtime 开发估算里。

实际推进过程中发现，要验证 K3/K3x AI 算力调度，必须先让 StarryOS 能稳定在 K3 真板上启动，并且要补齐 IME 指令依赖的底层系统能力。这部分成为计划外插入项，耗时明显超出最初预期，主要包括：

- K3 真板启动和 UFS rootfs 路径。
- UFS 驱动迁移、初始化序列调试和真板验证。
- StarryOS RISC-V DMA cache 同步空实现的定位与补齐。
- `/proc/set_ai_thread` AI 线程控制接口和核心亲和性验证。
- RISC-V V 扩展上下文保存恢复。
- V 扩展同线程 yield、跨线程隔离和 K3 真板测试。

这部分工作虽然打乱了阶段一原先节奏，但它是后续 A100/X100 IME backend、AI 线程调度和真实硬件测试的前置条件。如果没有这些平台侧基础，AI runtime 只能在 QEMU 或用户态模拟路径中验证，无法证明 K3 真板上的 AI 算力调度链路。

## 阶段二：内核调度器 async 化与多核心简单调度

| 任务 | 目标 |
|---|---|
| [ ] async 调度器与 completion waker | 将阶段一纯线程调度器改造成 Rust async 调度器，并完成异步 completion 唤醒（4天） |
| [ ] 多 A100/X100 核心队列 | 支持多个 A100 和 X100 核心的独立队列（2 天） |
| [ ] 简单核心选择与 backpressure | 先按空闲状态、队列长度或轮询选择目标核心，并支持基础反压（2 天） |
| [ ] cancel/timeout/reset | 任务取消、超时和错误恢复路径可用（2 天） |
| [ ] Buffer planner 与一次性图提交 | 能规划 buffer 复用/布局转换，并支持一个 graph 一次提交（2 天） |
| [ ] 图返回点与可观测性 | 长 graph 可以嵌入检查点返回用户态，并记录基础调度事件（2 天） |
| [ ] 阶段二演示与性能文档 | 展示多核心简单调度改善效果，同时展示静态分配无法处理后台算力占用的问题（3 天） |

## 阶段三：负载均衡调度和性能优化

| 任务 | 目标 |
|---|---|
| [ ] ResourceState 采样与 metrics | 汇总 PMU/Trace、队列深度、DMA backlog 和 latency（3 天） |
| [ ] 保守负载均衡规则 | 实现 DirectInline、KernelScheduled、X100 本地优先、X100 -> A100、A100 -> X100 的第一版规则（3 天） |
| [ ] 多 A100 核选择与异常降级 | 根据队列、PMU、DMA backlog 和历史 latency 选择低负载 A100，并支持 timeout/error fallback（2 天） |
| [ ] 双向卸载演示 | 展示 A100 满载溢出卸载到 X100，以及 X100 过载卸载到 A100（2 天） |
| [ ] benchmark 矩阵 | 完成微基准、多核心吞吐、多进程竞争、DAG 依赖和 llama.cpp end-to-end 对比（3 天） |
| [ ] 阶段三稳定性与最终演示 | 完成长时间负压测试、阶段三改善演示和最终比赛合成视频（3 天） |
| [ ] 用户态性能监控工具（如果时间允许） | 时间允许时实现可视化/命令行监控，展示核心负载、队列深度、latency 和模型运行状态（可选） |
