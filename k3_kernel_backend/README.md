# k3 kernel backend

`k3_kernel_backend` 是当前用于验证 K3 图执行链路的 FP16 算子适配层。它由
`k3_run_kernel` 接收一个已映射的图节点，执行固定的静态分发：

```text
k3_run_kernel
  -> <算子 Marker as ComputeKernel>::call
    -> compute_f16_cpu / compute_f16_rvv / compute_f16_a100
```

不存在运行时算子注册表。ABI 校验、属性读取和 dense/strided tensor view 构造由
`call.rs` 统一处理；具体算子只处理已验证的调用数据。

## 数值与目标策略

- 数值输入、权重、bias 和数值输出只接受 `F16`。F32、BF16、I8、U8 以及量化 dtype
  会返回 `UnsupportedDtype`。
- `MAT_MUL` 和 `CONV2D` 的 CPU 路径允许内部以 F32 累加，但只写回 F16；A100 路径始终
  使用 `smt.vfwmadot` 的 FP16 tile 语义。非 RISC-V host 使用等价软件镜像，覆盖同一条
  A100 路由。
- `AUTO` 在公共调用层归一化为 `PREFER_A100`。
- `PREFER_CPU` 使用 FP16 软件参考实现；`PREFER_A100` 使用 A100 FP16 或 RVV 路径。
- `PREFER_X100` 对逐元素、NN 和变换算子使用 FP16 RVV 路径；MatMul 与 Conv2D 返回
  `UnsupportedOp`，不会回退到 CPU。
- `Gather`/`GatherElements` 的索引允许 I32/I64；`TopK` 和可选 MaxPool 索引输出为 I64；
  `Cast` 只保留 F16、I32、I64 的必要转换。

## 覆盖算子

| 类别 | 算子 |
| --- | --- |
| 矩阵与卷积 | `MAT_MUL`、`CONV2D` |
| 逐元素 | `ADD`、`SUB`、`MUL`、`DIV`、`MOD`、`SILU`、`SIGMOID`、`SCALE` |
| NN | `RMS_NORM`、`ROPE`、`SOFTMAX`、`GLU`、`MAX_POOL`、`REDUCE_MAX`、`TOP_K` |
| 变换 | `CONCAT`、`TRANSPOSE`、`GATHER`、`GATHER_ELEMENTS`、`COPY`、`CAST`、`RESIZE`、`EXPAND`、`TILE` |

`GET_ROWS` 和 `SET_ROWS` 已删除，操作码 25、26 保留为空洞，不会被 `KernelOp::is_known()`
接受。

## 量化 ABI 边界

UABI 继续保留量化 tensor 的结构布局、dtype 编号、`AiQuantDesc`、`GGML_QUANT` layout、
buffer 构造器和 `TensorManager::alloc_ggml_quant_tensor`，以保持外部 ABI 兼容。

backend 不读取、不校验、不反量化也不执行这类 tensor。这里没有量化 MatMul、量化行索引、
模型加载或相关测试路径。

## 临时性质

这个 crate 的职责是验证图任务能够正确进入目标计算单元，而不是项目的长期核心。后续可能
直接改用 K3 提供的计算 API。长期需要保持稳定的是图任务组织、通道提交、worker、完成事件
和核心绑定等调度链路。
