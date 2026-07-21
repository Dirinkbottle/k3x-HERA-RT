# k3 kernel backend 算子支持矩阵

本 crate 提供 `k3_run_kernel` C ABI 入口，将调度器传入的 `AiGraphNode`
转换成 `BackendCall`，再按 `KernelOp` 分发到后端算子。公共调用层统一处理
ABI 校验、attr 读取、target 校验、tensor typed slice 与 shape/stride 访问。

## 执行目标约定

| 标记 | 含义 |
| --- | --- |
| CPU | 软件参考核，`AUTO`/`PREFER_CPU` 默认路径 |
| RVV | `PREFER_X100`/`PREFER_A100` 下使用真实 RISC-V VLA RVV，host 单测使用同语义软件镜像 |
| IME int8 | X100/A100 int8 `vmadot` tile 路径 |
| A100 FP16 IME | feature `a100-fp16-ime` 下的 A100 `smt.vfwmadot` 路径，默认关闭 |
| Frontend lowering | metadata/view/init 类 op 在 frontend 降低为 tensor view 或常量，不进入 backend kernel |

`a100-fp16-ime` 默认关闭。关闭时，FP16 MatMul/Conv2d 在 `PREFER_A100` 下明确返回
`UnsupportedOp`，避免在 MCPM.BF16 控制尚未接入前静默回退 CPU。

## YOLO26x ONNX 清单

| ONNX 算子 | 次数 | ggml/后端映射 | 当前落点 | 状态 |
| --- | ---: | --- | --- | --- |
| Conv | 166 | `CONV2D` = streaming/im2col + MatMul | CPU F32/F16, IME int8, A100 FP16 IME gated | 已实现 |
| Mul | 154 | `MUL` / `BinaryAttr` | CPU/RVV F32/F16, CPU integer MOD only | 已实现 |
| Sigmoid | 148 | `SIGMOID` / `UnaryAttr` | CPU/RVV F32/F16 | 已实现 |
| Constant | 49 | 编译期常量 | Frontend lowering | 无需 backend |
| Add | 42 | `ADD` / `BinaryAttr` | CPU/RVV F32/F16 | 已实现 |
| Concat | 35 | `CONCAT` / `ConcatAttr` | RVV/memcpy 逻辑拷贝 | 已实现 |
| Reshape | 17 | view reshape | Frontend lowering | 无需 backend |
| Split | 13 | view/slice | Frontend lowering | 无需 backend |
| Transpose | 7 | `TRANSPOSE` / `TransposeAttr` | RVV gather/memcpy | 已实现 |
| MatMul | 6 | `MAT_MUL` / `MatMulAttr` | CPU F32/F16, IME int8, A100 FP16 IME gated | 已实现 |
| Gather | 4 | `GATHER` / `GatherAttr` | RVV gather/CPU index handling | 已实现 |
| MaxPool | 3 | `MAX_POOL` / `Pool2dAttr` | CPU/RVV F32/F16, optional I64 indices | 已实现 |
| Softmax | 3 | `SOFTMAX` / `SoftmaxAttr` | CPU/RVV F32/F16 | 已实现 |
| Shape | 3 | 编译期 shape | Frontend lowering | 无需 backend |
| Unsqueeze | 3 | view reshape | Frontend lowering | 无需 backend |
| Cast | 3 | `CAST` / `CastAttr` | CPU/RVV F32/I32 common path, F16/I8/U8/BOOL coverage | 已实现 |
| Resize | 2 | `RESIZE` / `Resize2dAttr` | CPU/RVV nearest/linear NCHW 2D | 已实现 |
| Div | 2 | `DIV` / `BinaryAttr` | CPU/RVV F32/F16 | 已实现 |
| Slice | 2 | view slice | Frontend lowering | 无需 backend |
| TopK | 2 | `TOP_K` / `TopKAttr` | CPU stable top-k, lower index wins ties | 已实现 |
| ConstantOfShape | 2 | init/fill | Frontend lowering | 无需 backend |
| Expand | 2 | `EXPAND` / `ExpandAttr` | RVV/memcpy broadcast materialization | 已实现 |
| Tile | 2 | `TILE` / `TileAttr` | RVV/memcpy repeat materialization | 已实现 |
| GatherElements | 2 | `GATHER_ELEMENTS` / `GatherElementsAttr` | RVV gather/CPU index handling | 已实现 |
| Flatten | 2 | view reshape | Frontend lowering | 无需 backend |
| Sub | 1 | `SUB` / `BinaryAttr` | CPU/RVV F32/F16 | 已实现 |
| ReduceMax | 1 | `REDUCE_MAX` / `ReduceMaxAttr` | CPU/RVV F32/F16 | 已实现 |
| Mod | 1 | `MOD` / `BinaryAttr` | CPU F32/F16 fmod or I32/I64 remainder | 已实现 |

## GPT / 语音模型常用补充

| 算子 | ABI | 当前落点 | 说明 |
| --- | --- | --- | --- |
| RMSNorm | `RMS_NORM` / `RmsNormAttr` | CPU/RVV F32/F16 | hidden-size 固定在 attr，F16 经 F32 累加后写回 |
| RoPE | `ROPE` / `RopeAttr` | CPU F32/F16 | dense/strided BSHD，支持 GPT-J interleaved 和 NeoX half-split |
| Scale | `SCALE` / `UnaryAttr` | CPU/RVV F32/F16 | `alpha * x + beta` |
| SiLU | `SILU` / `UnaryAttr` | CPU/RVV F32/F16 | Sigmoid 与 fast-exp 路径复用 |

## 当前边界

- 所有 backend tensor 访问经 `call.rs` 做 ABI、arity、attr size、dtype size、shape/stride
  和 typed buffer 校验；输入输出别名默认拒绝。
- rank 上限为 `MAX_DIM = 8`，axis 支持 ONNX 负数语义。
- 空间算子当前实现常用 NCHW 2D 语义，不覆盖 cubic resize、动态 TopK/shape/scales、
  BF16、量化块布局、cache 同步、DMA 地址和原地别名策略。
- Conv2d 支持 groups/depthwise 与可选 `[Cout]` bias；更复杂的量化 bias/scale
  打包格式后续随量化块布局一起补。
- Metadata/view/init 类 op 继续由 frontend lowering 完成，不占用 backend op 编号或执行时间。
