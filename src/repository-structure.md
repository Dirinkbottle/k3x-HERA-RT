# 提交仓库目录和文件描述

## 根目录

| 路径 | 说明 |
|---|---|
| `README.md` | 项目入口、目标描述、文档索引和仓库结构 |
| `docs/` | 比赛文档、复现说明、开发计划、术语说明和周报 |
| `k3_aiUabi/` | 内核和用户态共享的 ABI crate |
| `k3_aiRuntime/` | 用户态 AI runtime crate |
| `k3_aiScheduler/` | 内核侧 graph scheduler crate |
| `k3_kernel_backend/` | kernel backend 算子分发 crate |
| `k3_ggml_bridge/` | 测试版 ggml/llama.cpp bridge crate |
| `k3_test/` | 最小集成测试程序 |

## `k3_aiUabi/`

| 文件 | 说明 |
|---|---|
| `src/lib.rs` | 导出 UAPI 模块，定义 ABI 版本、最大维度、ioctl 编号和 channel 常量 |
| `src/desc.rs` | 定义 `AiTensorDesc`、`AiKernelDesc`、dtype、layout、format 和 tensor size 计算 |
| `src/graph.rs` | 定义 graph blob、graph header、node、edge、parser 和 `GraphManager` |
| `src/kernel.rs` | 定义 `KernelOp` 和 `AiTargetHint` |
| `src/kernelattr.rs` | 定义 MatMul、RMSNorm、RoPE、Softmax、Conv2d 等算子 attr |
| `src/error.rs` | 定义 runtime、scheduler、backend 的错误类型 |

## `k3_aiRuntime/`

| 文件 | 说明 |
|---|---|
| `src/lib.rs` | 用户态 runtime 入口，导出 frontend 模块 |
| `src/fronted/mod.rs` | frontend 模块组织和 UABI re-export |
| `src/fronted/tensormanager.rs` | `TensorManager` 和 `Tensor`，负责 `MAP_SHARED` tensor 分配与描述符生成 |
| `src/fronted/kd_uring.rs` | `/dev/k3_airunner` 打开、channel 建立、graph 提交和 completion 等待 |
| `src/test/` | runtime 侧测试模块 |

## `k3_aiScheduler/`

| 文件 | 说明 |
|---|---|
| `src/lib.rs` | scheduler crate 入口，定义 OS 适配 trait |
| `src/scheduler.rs` | `GraphScheduler`、ready queue、worker、`run_graph` 和 completion 回包 |
| `src/kd_kring.rs` | graph 到 `TaskLink` 的拓扑解析，以及调度边界设计说明 |

## `k3_kernel_backend/`

| 文件 | 说明 |
|---|---|
| `src/lib.rs` | backend 统一入口 `k3_run_kernel`，将 `AiGraphNode` 转成 `BackendCall` |
| `src/matmul.rs` | MatMul backend 骨架，当前仓库保留 CPU fallback 路径，A100/X100 路径继续接入 |

## `k3_ggml_bridge/`

| 文件 | 说明 |
|---|---|
| `Cargo.toml` | 测试版 bridge crate 配置，产出 `cdylib` 和 `staticlib`，供 C/C++ 侧实验接入 |
| `src/lib.rs` | 暴露 `k3_ggml_matmul_f32` C ABI，将 F32 MatMul 请求转换为 `k3_aiRuntime` graph 提交 |
| `Cargo.lock` | 当前测试构建的依赖锁定文件 |
| `target/` | 本地构建产物目录，包含测试生成的 `.so`、`.a` 等文件，不是核心源码 |

## `k3_test/`

| 文件 | 说明 |
|---|---|
| `src/main.rs` | 最小集成测试，建立 channel，构造 MatMul DAG，提交 graph 并等待 completion |

## `docs/`

| 文件 | 说明 |
|---|---|
| `design-topic-analysis.md` | 比赛题目分析和资料调研 |
| `system-design.md` | 系统框架设计和 Mermaid 架构图 |
| `progress.md` | 开发计划和阶段任务 |
| `milestones.md` | 比赛过程中的重要进展 |
| `testing.md` | 系统测试情况 |
| `problems-and-solutions.md` | 主要问题和解决方法 |
| `team-and-gains.md` | 分工协作和比赛收获 |
| `how_to_use.md` | 复现教程 |
| `information.md` | 术语和补充信息 |
| `SUMMARY.md` | 文档总目录 |
| `开发日志/` | 周报和开发过程记录 |

## 配套仓库说明

本提交仓库不直接保存 StarryOS K3 板级驱动源码。以下内容在配套 [tgoskits dev 分支](https://github.com/Dirinkbottle/tgoskits/tree/dev)中维护：

- K3 COM260 Kit 板级配置。
- `/dev/k3_airunner` 设备驱动。
- `/proc/set_ai_thread` 接口。
- UFS、GMAC、SDMMC 等板级驱动。
- StarryOS 真板启动和 rootfs 集成路径。
