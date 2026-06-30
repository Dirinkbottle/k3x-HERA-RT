# 系统测试情况

## 当前的不足

- 算子支持数量仍少，当前只支持matmul。
- 多核心并发、backpressure、timeout、cancel 和长期压力测试仍在后续阶段。



## 测试概览

| 测试项 | 环境 | 结论 |
|---|---|---|
| StarryOS QEMU 启动 | QEMU | 可启动并验证基础调度路径 |
| StarryOS K3 真板启动 | K3真板 | 已通过 UFS rootfs 启动到 shell |
| `/dev/k3_airunner` channel | K3真板/QEMU | `BUILD_CHANNEL` 可建立共享区并由内核保活 |
| DAG MatMul CPU fallback | QEMU/K3真板 | graph/UAPI/runtime/scheduler/backend 链路可跑通 |
| `/proc/set_ai_thread` | K3真板/QEMU | 可将当前线程标记为 AI 线程并**立刻**调度到 AI 核心 |
| V 扩展上下文保存恢复 | QEMU/K3真板 | 同线程、跨线程、隔离测试全部 PASS |
| A100 IME MatMul 探针(**只有板子支持**) | K3真板 | int8 tile MatMul 输出与参考结果一致 |
| llama.cpp V 扩展 smoke test | QEMU/K3真板 | 程序可运行 |
| `k3_ggml_bridge` 测试版 | 本仓库测试构建 | 已加入测试版 F32 MatMul bridge，仍需继续接入真实 ggml backend |

## 共享 channel 测试

测试是确认用户态建立的 `MAP_SHARED` channel 能被内核验证和保活。流程如下：

1. 用户态调用 `build_channel()` 打开 `/dev/k3_airunner`。
2. 用户态 `mmap(MAP_SHARED | MAP_ANONYMOUS)` 建立共享区。
3. 内核通过 `BUILD_CHANNEL` ioctl 校验共享区并保存强引用。
4. 用户态写入 marker 后释放表层句柄。**这里是为了验证用户释放内存后物理内存不会被回收**
5. 再次读取共享区，确认数据仍存在。
6. 使用 `MAP_FIXED_NOREPLACE` 验证原地址范围仍被占用。

结果：成功。

## DAG 与 CPU fallback 测试

`k3_test` 构造 `(a || b) -> d` 的 MatMul DAG：

- `a` 和 `b` 无依赖，可并列进入 ready 集合。
- `d` 依赖 `a` 和 `b` 的输出。
- 用户态通过 `GraphManager` 冻结 graph blob。
- 内核侧解析 graph，生成稳定拓扑序并调用 backend。
- completion token 返回用户态后读取输出 tensor。

结果：成功。

## AI 线程亲和性测试

通过 `/proc/set_ai_thread` 验证线程类型控制：

```text
echo 1 > /proc/set_ai_thread
cat /proc/set_ai_thread
Current AI threads (1):
  core=8   tid=1      pid=1      busybox
```

结果:成功
## V 扩展上下文测试

测试程序读取硬件 `vlenb`，以 m8 寄存器组覆盖全部 32 个向量寄存器，并执行四类验证：

- 同线程 `sched_yield` 后向量寄存器能恢复。
- 新线程初始向量状态为零，不泄漏父线程数据。
- worker 线程 yield 后自己的向量状态能恢复。
- worker 退出后，主线程向量状态不被污染。

真板结果为：

```text
=== Vector Register Context Switch Test ===
Test-1 PASS
Test-A PASS
Test-C PASS
Test-3 PASS
=== ALL PASS ===
```

结果：成功

## `k3_ggml_bridge` 测试版

当前仓库加入了 `k3_ggml_bridge` 测试版 crate。该模块暴露 `k3_ggml_matmul_f32` C ABI，接收 ggml/llama.cpp 侧的 F32 MatMul 请求，完成 shape/stride 校验后分配 `k3_aiRuntime` tensor，构造单节点 MatMul graph，并通过 `/dev/k3_airunner` 等待 completion。

当前定位：

- 用于验证 ggml/llama.cpp 到 `k3_aiRuntime` 的 bridge 接口形态。
- 当前只覆盖 F32 MatMul 的测试路径。
- 当前仍会复制输入输出数据，尚不是最终高性能零拷贝路径。
- 仍需继续接入真实 ggml backend、补充更多 dtype/layout/quant case 和端到端模型测试。

## A100 IME MatMul 探针

使用 SpacemiT 工具链编译 A100 IME 指令测试程序，验证 `smt.vmadot` int8 MatMul tile。测试矩阵为 4x8 与 4x8 column-major，输出 4x4 tile。

日志中 IME 输出与参考输出逐项匹配，pad region 保持未写入状态：

```text
C = A x B^T
IME        Ref        Match
140        140        ok
168        168        ok
...
=== PASS ===
```

结果:成功
