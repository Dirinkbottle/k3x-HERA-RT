# 项目介绍

k3x-HERA-RT 是面向 K3/K3x 芯片的 AI 运行时与实时调度项目。

项目目标是把 X100、A100、CPU fallback、IME、DMA、TCM 和实时控制路径统一看成一个异构计算系统。我们希望在这个系统上打造一套更贴近 K3x 硬件形态的 AI 算力运行通道，让 CPU 与 AI 算力之间可以按任务规模、核心负载、数据依赖和实时性要求动态分配工作。

## 项目目标

- 构建一套纯异构 AI 算力运行体系，使 X100、A100 和 CPU fallback 都能作为可调度计算资源参与推理。
- 建立用户态算子库，把 MatMul、RMSNorm、RoPE、Softmax 等算子转换成可选 backend candidate。
- 建立内核侧调度器，让内核不理解具体算子语义，只根据 DAG 依赖、backend candidate、队列状态和资源状态选择执行位置。
- 支持 CPU 与 AI 算力之间的负载均衡：小任务优先留在 X100 本地路径或 CPU，本地超限时转 A100；A100 拥塞时可回退到等价 X100/CPU backend。
- 保持实时控制路径隔离，避免 AI 推理、缺页、fallback 或高负载调度影响 RT24/CAN 等实时任务。

## 核心思路

项目分为用户态算子库和内核调度器两层。

用户态算子库负责理解算子语义。它知道一个算子是什么、有哪些输入输出、有哪些 dtype/layout/quant 限制，也知道这个算子可以由哪些 backend 实现。例如同一个 `mul_mat` 可能有 A100 IME backend、X100 IME/RVV backend 和 CPU fallback backend。

内核调度器不承接具体算子结构，也不判断某个 node 是 MatMul 还是 RoPE。用户态会先把算子图 lower 成 opaque DAG，内核只看到 node 依赖、backend candidate、buffer handle、优先级和资源提示。

术语和补充说明见[特别说明和信息](./information.md)。

## 负载均衡方向

不同场景下，最优选择可能不同：

- 小算子或单进程低负载场景，用户态 DirectInline 执行可能延迟最低。
- 大矩阵乘法、长 DAG、多核心高负载场景，KernelScheduled 更容易利用多个 X100/A100 核心。
- X100 本地路径适合小 tensor、RVV/IME 轻量计算和低提交开销任务。
- A100 路径适合更大的 tensor/vector 计算和更重的 LLM 核心算子。
- CPU fallback 负责正确性兜底、异常降级和不支持算子的兼容执行。

阶段三通过 benchmark 明确这些边界，包括 DirectInline 与 KernelScheduled 的延迟边界、多核心吞吐、多进程竞争、DAG 依赖调度以及 llama.cpp end-to-end 对比。

## 开发进度

见[开发进度](./progress.md)
