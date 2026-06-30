# 遇到的主要问题和解决方法

## 1. K3/K3x 不是传统 NPU 提交模型

问题：调研 `linux-k3` 公开驱动时，没有找到传统 `/dev/npu` 或 `/dev/a100` command-submission 风格的统一驱动。这不像我之前接触过的传统npu驱动.
解决方法：转向“用户态 runtime + 内核资源管理 + 异构核心调度”的方案。用户态理解算子和 backend candidate，内核负责 graph、队列、映射、核心类型和 completion。

## 2. K3 真板启动依赖 UFS 驱动

问题：StarryOS 要在 K3 真板上进入可用 shell，需要从 UFS rootfs 启动。UFS 驱动涉及 UFSHCI、UIC、MPHY 上电序列和 UTP 请求列表，迁移成本高。

解决方法：以 SpacemiT Linux 驱动为准，使用 bindgen 生成寄存器定义，然后将驱动文件copy过来,**结合ai把linuxkernel的依赖剔除和替换**然后使用 c2rust 翻译 C 代码，再手工补胶水代码和 Rust 化边界。这样降低了寄存器抄写错误概率，也让调试重点集中在初始化序列和内存同步上。

## 3. DMA cache 同步为空实现

问题：UFS 驱动逻辑按 Linux 参考实现迁移后仍然启动异常。进一步在调用前检查内存内容，发现 StarryOS RISC-V DMA cache 同步接口存在空实现，导致设备和 CPU 看到的数据不一致。

解决方法：补齐 StarryOS RISC-V 侧 DMA cache clean/invalidate/sync 路径，使 UFS 描述符和数据 buffer 在设备访问前后保持一致。


## 4. V 扩展上下文没有保存恢复

问题：K3 的 X100/A100 IME 路径依赖 RISC-V V 扩展。StarryOS 早期任务切换没有保存 32 个向量寄存器和相关 CSR，多个线程使用向量单元时会互相污染。

解决方法：为任务上下文增加 V 状态保存区，覆盖 32 个 vector register group 和相关 CSR。由于 K3 上向量状态较大，使用堆内存保存。然后编写同线程、跨线程和新线程初始化测试，并在 QEMU 与 K3 真板验证通过。

## 5. IME MatMul 形状和文档理解存在偏差

问题：接入 A100 IME MatMul 探针时，最初结果与参考值不一致，说明对 operand layout、tile shape 或 column-major 约定理解有偏差。

解决方法：通过错误输出反推实际参与计算的数据布局，重新整理 4x8x4 tile 的输入排布、输出位置和 pad 区域，再与 CPU reference 逐项对比。最终 A100 IME 输出与参考结果一致。


## 6. QEMU 上的向量程序性能异常

问题：llama.cpp 的 V 扩展版本在 QEMU 上出现负优化，多核 SMP 下如果不限制线程数，运行速度会明显下降。

解决方法：将 QEMU 结果定位为功能验证，不作为真实性能结论；在 QEMU 上使用 `-t 1` 降低原子同步开销，真实性能判断以 K3 真板为准。
