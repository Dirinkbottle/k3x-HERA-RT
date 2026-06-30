# 比赛过程中的重要进展

## 1. 完成 K3/K3x AI 算力形态调研

第一阶段确认了 K3/K3x 更适合按异构计算系统设计，而不是按传统外挂 NPU 设计。调研覆盖 X100、A100、IME、Vector、FPU、DMA、TCM、HMP 和官方 ONNX Runtime EP 后端。

## 2. 建立 AI runtime 与 UAPI 雏形

完成用户态 frontend、tensor desc、kernel desc、graph blob、submit entry 和 completion token 的最小设计。后续将分散在 runtime 中的 ABI 类型拆到独立 `k3_aiUabi` crate，使用户态和内核态共享同一套接口定义。 因为内核也必须至少懂节点这些语义,所以把二者都用的一些数据结构放在uapi里面,并且添加了版本检查.

## 3. 打通共享 channel 与内存保活路径

完成 `/dev/k3_airunner` 的 `BUILD_CHANNEL` 最小通路。用户态先建立 `MAP_SHARED` 内存，内核验证并保活该共享区，再通过 `ov-channels` 发送 graph entry 和 completion。

这一阶段验证了用户态句柄释放后，内核仍能稳定持有共享区引用，为后续无锁队列和异步 completion 打基础。

## 4. StarryOS 成功在 K3 真板上启动

围绕 K3 COM260 Kit 完成 StarryOS 板级启动路径，重点实现 UFS 驱动，使系统可以从 UFS rootfs 启动到 shell。

UFS 驱动迁移为采用了Spacemit的Linux仓库中的驱动copy + bindgen 寄存器定义 + c2rust 初始翻译 + 手工胶水和调试”的方式完成。该工作解决了比赛环境从 QEMU 走向真板验证的问题,同时节省了大量手工翻译的时间。

## 5. 实现 AI 线程亲和性控制

根据 K3 HMP 机制，在 StarryOS 中实现 `/proc/set_ai_thread` 接口。用户态写入 `1` 后可以把当前线程标记为 AI 线程，并进入 AI 核心集合调度。

**注意**:目前set_ai_thread接口不支持设置其他线程为ai线程,并且留有写入0取消为ai线程,并且立即将自己调度到core0-7的x100核心. 这时可用的vlen已经变了,如果要做测试请注意! 目前这个接口是开发测试用

这处理 X100/A100 向量宽度不同的问题，避免使用某类向量状态的线程被随意迁移到另一类核心。

## 6. 补齐 RISC-V V 扩展上下文保存恢复

在 StarryOS 任务上下文中补齐 V 扩展状态保存恢复，包括 32 个 vector register group 和相关 CSR 状态。由于 K3 上向量状态较大，采用堆内存保存上下文。

完成同线程 yield、跨线程 yield、新线程初始状态和跨线程隔离测试,确保任务切换时32个通用向量寄存器+7个csr非特权级控制寄存器+1个特权级寄存器能够正确保存和恢复。

## 7. 完成 DAG 调度器最小形态

`k3_aiScheduler` 从占位骨架推进到可工作的最小原型：

- 完成 `GraphScheduler`、ready queue 和 worker。
- 实现 DAG 到稳定拓扑序的解析。
- 支持 `(a || b) -> d` (d依赖a和b,必须在a和b都完成后才能执行d)形式的依赖图。
- graph 完成后通过 completion token 回到用户态。

## 8. 打通 MatMul CPU fallback 数据链路

`k3_test` 从共享内存保活验证升级为真实数据通路测试。测试构造 MatMul DAG，经 UABI、runtime、driver、scheduler 和 backend 执行，最后回读输出数据。

## 9. 完成 A100 IME MatMul 探针验证

在配套真板测试环境中，使用 SpacemiT 工具链编译 IME 指令测试程序，完成 A100 `smt.vmadot` int8 MatMul tile 验证。测试输出与参考结果一致，pad 区域保持未写入状态，说明基本 IME 执行路径正确。

后续工作会把该探针路径逐步收敛到正式 backend candidate 中。

## 10. 推进 llama.cpp bridge

已尝试使用静态链接和 RISC-V V 扩展编译 llama.cpp，使用 `tinyllama-15M-stories` 的模型转化为 fp32 做 test。当前仓库已加入 `k3_ggml_bridge` 测试版 crate，暴露 `k3_ggml_matmul_f32` C ABI，用于把 ggml/llama.cpp 侧的 F32 MatMul 请求转换为 `k3_aiRuntime` graph 提交。

这个 bridge 目前仍是测试版本，主要用于验证接入方式、shape/stride 转换、channel 提交和 completion 链路，不代表稳定可用的完整 ggml backend。这里完成后，原本能在 K3 Linux 上跑的大部分模型都有机会获得部分加速。
