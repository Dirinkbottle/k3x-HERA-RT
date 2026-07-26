//! 算子内联参数（attr）定义。
//!
//! 每个算子对应一个 `#[repr(C)] + Copy` 的 attr 结构，
//! 通过 `AiKernelDesc::set_inline_attr` 写入单算子描述的 `attr_inline` 区域。
//! 所有 attr 结构必须在编译期断言为固定大小，方便内核侧做大小校验。

use super::{desc::AiDtype, kernel::KernelOp};
use crate::{DimCount, DimSize, ElemStride, KernelStride, OpFlags, TensorAxis};

/// 能唯一映射到一个语义级 op 的算子 attr。
///
/// `AiKernelDesc::new` 通过这个 trait 从 attr 类型解析出 `KernelOp`。
/// ADD/MUL、SILU/SCALE 这类复用同一个 attr 的算子不能实现这个 trait，
/// 需要通过显式 op 构造入口创建 desc。
pub trait AiKernelAttr: Copy {
    /// 该 attr 唯一对应的语义级算子编号。
    const OP: KernelOp;
}

/// MatMul 算子参数。
///
/// 包含完整的矩阵维度和 stride 信息，无需从 TensorView 提取。
///
/// 张量约定：
/// - tensors[0] = lhs (m×k 或 k×m if transposed)
/// - tensors[1] = rhs (k×n 或 n×k if transposed)
/// - tensors[2] = output (m×n)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MatMulAttr {
    /// 输出矩阵行数
    pub m: DimSize,
    /// 输出矩阵列数
    pub n: DimSize,
    /// 内积维度
    pub k: DimSize,
    /// batch 数量，0 表示单个矩阵
    pub batch: DimSize,

    /// lhs 行 stride（元素数）
    pub lhs_row_stride: ElemStride,
    /// lhs 列 stride（元素数）
    pub lhs_col_stride: ElemStride,
    /// lhs batch stride（元素数）
    pub lhs_batch_stride: ElemStride,

    /// rhs 行 stride（元素数）
    pub rhs_row_stride: ElemStride,
    /// rhs 列 stride（元素数）
    pub rhs_col_stride: ElemStride,
    /// rhs batch stride（元素数）
    pub rhs_batch_stride: ElemStride,

    /// out 行 stride（元素数）
    pub out_row_stride: ElemStride,
    /// out 列 stride（元素数）
    pub out_col_stride: ElemStride,
    /// out batch stride（元素数）
    pub out_batch_stride: ElemStride,

    /// flags: bit0=lhs_transposed, bit1=rhs_transposed
    pub flags: OpFlags,
    /// 累加器数据类型
    pub accum_dtype: AiDtype,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 3],
}

/// RMSNorm 算子参数。
///
/// 张量约定：
/// - tensors[0] = input
/// - tensors[1] = weight
/// - tensors[2] = output
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RmsNormAttr {
    /// 归一化的隐藏维大小。
    pub hidden_size: DimSize,
    /// 数值稳定项 epsilon。
    pub eps: f32,
    /// 算子 flags，具体含义由 backend 约定。
    pub flags: OpFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 13],
}

/// RoPE 算子参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RopeAttr {
    /// 参与旋转的维度数。
    pub n_dims: DimSize,
    /// RoPE 模式（如 NEOX/GPT-J 变体）。
    pub mode: u32,
    /// 上下文长度。
    pub n_ctx: DimSize,
    /// 注意力头数量。
    pub head_count: DimSize,
    /// 频率基数 (theta)。
    pub freq_base: f32,
    /// 频率缩放系数。
    pub freq_scale: f32,
    /// YaRN 外插因子。
    pub ext_factor: f32,
    /// 注意力缩放因子。
    pub attn_factor: f32,
    /// YaRN 快速修正区间下界。
    pub beta_fast: f32,
    /// YaRN 慢速修正区间上界。
    pub beta_slow: f32,
    /// 算子 flags，具体含义由 backend 约定。
    pub flags: OpFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 5],
}

/// Softmax 算子参数。
///
/// 张量约定：
/// - tensors[0] = input
/// - tensors[1] = output
///   可选 mask 后续可以作为额外 input tensor 放在 tensors[1]，output 顺延。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SoftmaxAttr {
    /// 归一化所沿的轴，负数表示从末尾计数。
    pub axis: TensorAxis,
    /// 输入缩放系数（如 attention 的 1/sqrt(d)）。
    pub scale: f32,
    /// GGML attention 的 ALiBi/max-bias 参数；v1 只接受 0。
    pub max_bias: f32,
    /// 算子 flags，具体含义由 backend 约定。
    pub flags: OpFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 12],
}

/// 二元 elementwise 算子参数。
///
/// ADD/MUL 可以共用该 attr，具体语义由 KernelOp 区分。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BinaryAttr {
    /// 广播方式编号。
    pub broadcast_kind: u32,
    /// 第一个操作数的缩放系数。
    pub alpha: f32,
    /// 第二个操作数的缩放系数。
    pub beta: f32,
    /// 算子 flags，具体含义由 backend 约定。
    pub flags: OpFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 12],
}

/// GGML GetRows 参数。
///
/// 张量约定：
/// - tensors[0] = data
/// - tensors[1] = I32/I64 row indices
/// - tensors[2] = output
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GetRowsAttr {
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 15],
}

/// GGML SetRows 参数。
///
/// 张量约定：
/// - tensors[0] = source rows
/// - tensors[1] = I32/I64 row indices
/// - tensors[2] = destination/base tensor
/// - tensors[3] = output view of destination/base tensor
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SetRowsAttr {
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 15],
}

/// GLU 参数。
///
/// v1 仅支持 `OP_SWIGLU`，用于 llama.cpp `ggml_swiglu_split`。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GluAttr {
    /// GLU 变体编号，对齐 ggml `enum ggml_glu_op`。
    pub op: u32,
    /// 非零表示无第二输入时交换 src0 的左右半区。
    pub swapped: u32,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 13],
}

/// 通用 copy/materialize 参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CopyAttr {
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 15],
}

/// 单输入 elementwise 算子参数。
///
/// SILU/SCALE 这类轻量 op 可以先共用该 attr。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct UnaryAttr {
    /// 输入缩放系数。
    pub alpha: f32,
    /// 偏置/位移系数。
    pub beta: f32,
    /// 算子 flags，具体含义由 backend 约定。
    pub flags: OpFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 13],
}

/// Conv2d 算子参数。
///
/// 张量约定：
/// - tensors[0] = input
/// - tensors[1] = weight
/// - tensors[2] = output
/// - tensors[3] = bias/quant 参数，可选
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Conv2dAttr {
    /// batch 大小。
    pub batch: DimSize,
    /// 输入通道数。
    pub in_channels: DimSize,
    /// 输出通道数。
    pub out_channels: DimSize,
    /// 输入高度。
    pub input_h: DimSize,
    /// 输入宽度。
    pub input_w: DimSize,
    /// 输出高度。
    pub output_h: DimSize,
    /// 输出宽度。
    pub output_w: DimSize,
    /// 卷积核高度。
    pub kernel_h: DimSize,
    /// 卷积核宽度。
    pub kernel_w: DimSize,
    /// 垂直方向步长。
    pub stride_h: KernelStride,
    /// 水平方向步长。
    pub stride_w: KernelStride,
    /// 垂直方向 padding。
    pub pad_h: DimSize,
    /// 水平方向 padding。
    pub pad_w: DimSize,
    /// 垂直方向膨胀系数。
    pub dilation_h: KernelStride,
    /// 水平方向膨胀系数。
    pub dilation_w: KernelStride,
    /// 分组卷积的分组数。
    pub groups: DimSize,
    /// 算子 flags，具体含义由 backend 约定。
    pub flags: OpFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 15],
}

/// Concat 参数，所有输入必须同 rank 且仅 `axis` 维允许不同。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ConcatAttr {
    /// 拼接轴，负值从末维计数。
    pub axis: TensorAxis,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 14],
}

/// Transpose 参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TransposeAttr {
    /// permutation 中有效维度数量。
    pub rank: DimCount,
    /// `perm[out_axis]` 指定对应的输入轴。
    pub perm: [TensorAxis; crate::MAX_DIM],
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 6],
}

/// ONNX Gather 参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GatherAttr {
    /// 在 data tensor 上执行索引的轴。
    pub axis: TensorAxis,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 14],
}

/// 二维 MaxPool 参数，空间 tensor 使用 NCHW 布局。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Pool2dAttr {
    /// kernel 高度。
    pub kernel_h: DimSize,
    /// kernel 宽度。
    pub kernel_w: DimSize,
    /// 垂直 stride。
    pub stride_h: KernelStride,
    /// 水平 stride。
    pub stride_w: KernelStride,
    /// 垂直 dilation。
    pub dilation_h: KernelStride,
    /// 水平 dilation。
    pub dilation_w: KernelStride,
    /// 顶部 padding。
    pub pad_top: DimSize,
    /// 左侧 padding。
    pub pad_left: DimSize,
    /// 底部 padding。
    pub pad_bottom: DimSize,
    /// 右侧 padding。
    pub pad_right: DimSize,
    /// bit0 表示 ONNX ceil_mode。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 5],
}

impl Pool2dAttr {
    /// `flags` 中的 ceil-mode 位。
    pub const CEIL_MODE: u32 = 1 << 0;
}

/// Cast 参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CastAttr {
    /// 目标 dtype。
    pub to: AiDtype,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 14],
}

/// 静态二维 Resize 参数，空间 tensor 使用 NCHW 布局。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Resize2dAttr {
    /// 插值模式：0=nearest，1=linear。
    pub mode: u32,
    /// 坐标模式：0=half_pixel，1=asymmetric，2=align_corners。
    pub coordinate_mode: u32,
    /// nearest 取整：0=round_prefer_floor，1=floor，2=ceil。
    pub nearest_mode: u32,
    /// lowering 时确定的输入高度。
    pub input_h: DimSize,
    /// lowering 时确定的输入宽度。
    pub input_w: DimSize,
    /// lowering 时确定的输出高度。
    pub output_h: DimSize,
    /// lowering 时确定的输出宽度。
    pub output_w: DimSize,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 8],
}

impl Resize2dAttr {
    /// 最近邻插值。
    pub const MODE_NEAREST: u32 = 0;
    /// 双线性插值。
    pub const MODE_LINEAR: u32 = 1;
    /// ONNX half-pixel 坐标变换。
    pub const COORD_HALF_PIXEL: u32 = 0;
    /// ONNX asymmetric 坐标变换。
    pub const COORD_ASYMMETRIC: u32 = 1;
    /// ONNX align-corners 坐标变换。
    pub const COORD_ALIGN_CORNERS: u32 = 2;
    /// 最近值相同时优先较小坐标。
    pub const NEAREST_ROUND_PREFER_FLOOR: u32 = 0;
    /// 向下取整。
    pub const NEAREST_FLOOR: u32 = 1;
    /// 向上取整。
    pub const NEAREST_CEIL: u32 = 2;
}

/// TopK 参数，K 在 lowering 时固定进 attr。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TopKAttr {
    /// 选择轴，负值从末维计数。
    pub axis: TensorAxis,
    /// 选择元素数量。
    pub k: DimSize,
    /// 非零表示选择最大值，否则选择最小值。
    pub largest: u32,
    /// 非零表示按结果值排序。
    pub sorted: u32,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 11],
}

/// Expand 参数，目标 shape 在 lowering 时固定进 attr。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ExpandAttr {
    /// `target_shape` 中的有效维度数量。
    pub rank: DimCount,
    /// ONNX Expand 的目标 shape。
    pub target_shape: [DimSize; crate::MAX_DIM],
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 6],
}

/// Tile 参数，repeats 在 lowering 时固定进 attr。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TileAttr {
    /// `repeats` 中的有效维度数量。
    pub rank: DimCount,
    /// 每一维的重复次数。
    pub repeats: [DimSize; crate::MAX_DIM],
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 6],
}

/// ONNX GatherElements 参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GatherElementsAttr {
    /// 在 data tensor 上执行逐元素索引的轴。
    pub axis: TensorAxis,
    /// 算子 flags，当前必须为 0。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 14],
}

/// ReduceMax 参数，axes 在 lowering 时固定进 attr。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ReduceMaxAttr {
    /// `axes` 中的有效轴数量。
    pub axis_count: DimCount,
    /// 归约轴，支持负数。
    pub axes: [TensorAxis; crate::MAX_DIM],
    /// bit0=keepdims，bit1=noop_with_empty_axes。
    pub flags: OpFlags,
    /// 预留字段，保持固定 ABI 大小。
    pub reserved: [u32; 6],
}

impl ReduceMaxAttr {
    /// `flags` 中的 keepdims 位。
    pub const KEEP_DIMS: u32 = 1 << 0;
    /// 空 axes 时保持输入不变。
    pub const NOOP_WITH_EMPTY_AXES: u32 = 1 << 1;
}

impl BinaryAttr {
    /// MOD 使用浮点 fmod 语义的 flag。
    pub const MOD_FMOD: u32 = 1 << 0;
    /// GGML 广播语义：一维权重沿最后逻辑维广播。
    pub const BROADCAST_GGML_LAST_DIM: u32 = 1;
}

impl SoftmaxAttr {
    /// 输入 1 是 attention mask。
    pub const HAS_MASK: u32 = 1 << 0;
}

impl GluAttr {
    /// 对齐 ggml `GGML_GLU_OP_SWIGLU`。
    pub const OP_SWIGLU: u32 = 2;
}

impl RopeAttr {
    /// GPT-J 交错偶/奇维旋转模式。
    pub const MODE_GPT_J: u32 = 0;
    /// GPT-NeoX 前后半区配对旋转模式。
    pub const MODE_NEOX: u32 = 1;
}

impl AiKernelAttr for MatMulAttr {
    const OP: KernelOp = KernelOp::MAT_MUL;
}

impl AiKernelAttr for RmsNormAttr {
    const OP: KernelOp = KernelOp::RMS_NORM;
}

impl AiKernelAttr for RopeAttr {
    const OP: KernelOp = KernelOp::ROPE;
}

impl AiKernelAttr for SoftmaxAttr {
    const OP: KernelOp = KernelOp::SOFTMAX;
}

impl AiKernelAttr for GetRowsAttr {
    const OP: KernelOp = KernelOp::GET_ROWS;
}

impl AiKernelAttr for SetRowsAttr {
    const OP: KernelOp = KernelOp::SET_ROWS;
}

impl AiKernelAttr for GluAttr {
    const OP: KernelOp = KernelOp::GLU;
}

impl AiKernelAttr for CopyAttr {
    const OP: KernelOp = KernelOp::COPY;
}

impl AiKernelAttr for Conv2dAttr {
    const OP: KernelOp = KernelOp::CONV2D;
}

impl AiKernelAttr for ConcatAttr {
    const OP: KernelOp = KernelOp::CONCAT;
}

impl AiKernelAttr for TransposeAttr {
    const OP: KernelOp = KernelOp::TRANSPOSE;
}

impl AiKernelAttr for GatherAttr {
    const OP: KernelOp = KernelOp::GATHER;
}

impl AiKernelAttr for Pool2dAttr {
    const OP: KernelOp = KernelOp::MAX_POOL;
}

impl AiKernelAttr for CastAttr {
    const OP: KernelOp = KernelOp::CAST;
}

impl AiKernelAttr for Resize2dAttr {
    const OP: KernelOp = KernelOp::RESIZE;
}

impl AiKernelAttr for TopKAttr {
    const OP: KernelOp = KernelOp::TOP_K;
}

impl AiKernelAttr for ExpandAttr {
    const OP: KernelOp = KernelOp::EXPAND;
}

impl AiKernelAttr for TileAttr {
    const OP: KernelOp = KernelOp::TILE;
}

impl AiKernelAttr for GatherElementsAttr {
    const OP: KernelOp = KernelOp::GATHER_ELEMENTS;
}

impl AiKernelAttr for ReduceMaxAttr {
    const OP: KernelOp = KernelOp::REDUCE_MAX;
}

// ── 编译期大小断言 ──────────────────────────────────────────────

const _: () = assert!(core::mem::size_of::<MatMulAttr>() == 72);
const _: () = assert!(core::mem::size_of::<RmsNormAttr>() == 64);
const _: () = assert!(core::mem::size_of::<RopeAttr>() == 64);
const _: () = assert!(core::mem::size_of::<SoftmaxAttr>() == 64);
const _: () = assert!(core::mem::size_of::<BinaryAttr>() == 64);
const _: () = assert!(core::mem::size_of::<GetRowsAttr>() == 64);
const _: () = assert!(core::mem::size_of::<SetRowsAttr>() == 64);
const _: () = assert!(core::mem::size_of::<GluAttr>() == 64);
const _: () = assert!(core::mem::size_of::<CopyAttr>() == 64);
const _: () = assert!(core::mem::size_of::<UnaryAttr>() == 64);
const _: () = assert!(core::mem::size_of::<Conv2dAttr>() == 128);
const _: () = assert!(core::mem::size_of::<ConcatAttr>() == 64);
const _: () = assert!(core::mem::size_of::<TransposeAttr>() == 64);
const _: () = assert!(core::mem::size_of::<GatherAttr>() == 64);
const _: () = assert!(core::mem::size_of::<Pool2dAttr>() == 64);
const _: () = assert!(core::mem::size_of::<CastAttr>() == 64);
const _: () = assert!(core::mem::size_of::<Resize2dAttr>() == 64);
const _: () = assert!(core::mem::size_of::<TopKAttr>() == 64);
const _: () = assert!(core::mem::size_of::<ExpandAttr>() == 64);
const _: () = assert!(core::mem::size_of::<TileAttr>() == 64);
const _: () = assert!(core::mem::size_of::<GatherElementsAttr>() == 64);
const _: () = assert!(core::mem::size_of::<ReduceMaxAttr>() == 64);

/// ABI raw mirror layout checks for attr structures touched by transparent newtypes.
#[allow(dead_code, missing_docs, clippy::missing_docs_in_private_items)]
mod abi_layout {
    use super::*;

    macro_rules! assert_attr_layout {
        ($actual:ty, $raw:ty, [$($field:ident),+ $(,)?]) => {
            const _: () = assert!(core::mem::size_of::<$actual>() == core::mem::size_of::<$raw>());
            const _: () = assert!(core::mem::align_of::<$actual>() == core::mem::align_of::<$raw>());
            $(
                const _: () = assert!(
                    core::mem::offset_of!($actual, $field)
                        == core::mem::offset_of!($raw, $field)
                );
            )+
        };
    }

    #[repr(C)]
    struct RawMatMulAttr {
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
        lhs_row_stride: u32,
        lhs_col_stride: u32,
        lhs_batch_stride: u32,
        rhs_row_stride: u32,
        rhs_col_stride: u32,
        rhs_batch_stride: u32,
        out_row_stride: u32,
        out_col_stride: u32,
        out_batch_stride: u32,
        flags: u32,
        accum_dtype: u32,
        reserved: [u32; 3],
    }

    #[repr(C)]
    struct RawRmsNormAttr {
        hidden_size: u32,
        eps: f32,
        flags: u32,
        reserved: [u32; 13],
    }

    #[repr(C)]
    struct RawRopeAttr {
        n_dims: u32,
        mode: u32,
        n_ctx: u32,
        head_count: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        flags: u32,
        reserved: [u32; 5],
    }

    #[repr(C)]
    struct RawSoftmaxAttr {
        axis: i32,
        scale: f32,
        max_bias: f32,
        flags: u32,
        reserved: [u32; 12],
    }

    #[repr(C)]
    struct RawBinaryAttr {
        broadcast_kind: u32,
        alpha: f32,
        beta: f32,
        flags: u32,
        reserved: [u32; 12],
    }

    #[repr(C)]
    struct RawGetRowsAttr {
        flags: u32,
        reserved: [u32; 15],
    }

    #[repr(C)]
    struct RawSetRowsAttr {
        flags: u32,
        reserved: [u32; 15],
    }

    #[repr(C)]
    struct RawGluAttr {
        op: u32,
        swapped: u32,
        flags: u32,
        reserved: [u32; 13],
    }

    #[repr(C)]
    struct RawCopyAttr {
        flags: u32,
        reserved: [u32; 15],
    }

    #[repr(C)]
    struct RawUnaryAttr {
        alpha: f32,
        beta: f32,
        flags: u32,
        reserved: [u32; 13],
    }

    #[repr(C)]
    struct RawConv2dAttr {
        batch: u32,
        in_channels: u32,
        out_channels: u32,
        input_h: u32,
        input_w: u32,
        output_h: u32,
        output_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        groups: u32,
        flags: u32,
        reserved: [u32; 15],
    }

    #[repr(C)]
    struct RawConcatAttr {
        axis: i32,
        flags: u32,
        reserved: [u32; 14],
    }

    #[repr(C)]
    struct RawTransposeAttr {
        rank: u32,
        perm: [i32; crate::MAX_DIM],
        flags: u32,
        reserved: [u32; 6],
    }

    #[repr(C)]
    struct RawGatherAttr {
        axis: i32,
        flags: u32,
        reserved: [u32; 14],
    }

    #[repr(C)]
    struct RawPool2dAttr {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        pad_top: u32,
        pad_left: u32,
        pad_bottom: u32,
        pad_right: u32,
        flags: u32,
        reserved: [u32; 5],
    }

    #[repr(C)]
    struct RawCastAttr {
        to: u32,
        flags: u32,
        reserved: [u32; 14],
    }

    #[repr(C)]
    struct RawResize2dAttr {
        mode: u32,
        coordinate_mode: u32,
        nearest_mode: u32,
        input_h: u32,
        input_w: u32,
        output_h: u32,
        output_w: u32,
        flags: u32,
        reserved: [u32; 8],
    }

    #[repr(C)]
    struct RawTopKAttr {
        axis: i32,
        k: u32,
        largest: u32,
        sorted: u32,
        flags: u32,
        reserved: [u32; 11],
    }

    #[repr(C)]
    struct RawExpandAttr {
        rank: u32,
        target_shape: [u32; crate::MAX_DIM],
        flags: u32,
        reserved: [u32; 6],
    }

    #[repr(C)]
    struct RawTileAttr {
        rank: u32,
        repeats: [u32; crate::MAX_DIM],
        flags: u32,
        reserved: [u32; 6],
    }

    #[repr(C)]
    struct RawGatherElementsAttr {
        axis: i32,
        flags: u32,
        reserved: [u32; 14],
    }

    #[repr(C)]
    struct RawReduceMaxAttr {
        axis_count: u32,
        axes: [i32; crate::MAX_DIM],
        flags: u32,
        reserved: [u32; 6],
    }

    assert_attr_layout!(
        MatMulAttr,
        RawMatMulAttr,
        [
            m,
            n,
            k,
            batch,
            lhs_row_stride,
            lhs_col_stride,
            lhs_batch_stride,
            rhs_row_stride,
            rhs_col_stride,
            rhs_batch_stride,
            out_row_stride,
            out_col_stride,
            out_batch_stride,
            flags,
            accum_dtype,
            reserved,
        ]
    );
    assert_attr_layout!(
        RmsNormAttr,
        RawRmsNormAttr,
        [hidden_size, eps, flags, reserved]
    );
    assert_attr_layout!(
        RopeAttr,
        RawRopeAttr,
        [
            n_dims,
            mode,
            n_ctx,
            head_count,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            flags,
            reserved,
        ]
    );
    assert_attr_layout!(
        SoftmaxAttr,
        RawSoftmaxAttr,
        [axis, scale, max_bias, flags, reserved]
    );
    assert_attr_layout!(
        BinaryAttr,
        RawBinaryAttr,
        [broadcast_kind, alpha, beta, flags, reserved]
    );
    assert_attr_layout!(GetRowsAttr, RawGetRowsAttr, [flags, reserved]);
    assert_attr_layout!(SetRowsAttr, RawSetRowsAttr, [flags, reserved]);
    assert_attr_layout!(GluAttr, RawGluAttr, [op, swapped, flags, reserved]);
    assert_attr_layout!(CopyAttr, RawCopyAttr, [flags, reserved]);
    assert_attr_layout!(UnaryAttr, RawUnaryAttr, [alpha, beta, flags, reserved]);
    assert_attr_layout!(
        Conv2dAttr,
        RawConv2dAttr,
        [
            batch,
            in_channels,
            out_channels,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            flags,
            reserved,
        ]
    );
    assert_attr_layout!(ConcatAttr, RawConcatAttr, [axis, flags, reserved]);
    assert_attr_layout!(
        TransposeAttr,
        RawTransposeAttr,
        [rank, perm, flags, reserved]
    );
    assert_attr_layout!(GatherAttr, RawGatherAttr, [axis, flags, reserved]);
    assert_attr_layout!(
        Pool2dAttr,
        RawPool2dAttr,
        [
            kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left,
            pad_bottom, pad_right, flags, reserved,
        ]
    );
    assert_attr_layout!(CastAttr, RawCastAttr, [to, flags, reserved]);
    assert_attr_layout!(
        Resize2dAttr,
        RawResize2dAttr,
        [
            mode,
            coordinate_mode,
            nearest_mode,
            input_h,
            input_w,
            output_h,
            output_w,
            flags,
            reserved,
        ]
    );
    assert_attr_layout!(
        TopKAttr,
        RawTopKAttr,
        [axis, k, largest, sorted, flags, reserved]
    );
    assert_attr_layout!(
        ExpandAttr,
        RawExpandAttr,
        [rank, target_shape, flags, reserved]
    );
    assert_attr_layout!(TileAttr, RawTileAttr, [rank, repeats, flags, reserved]);
    assert_attr_layout!(
        GatherElementsAttr,
        RawGatherElementsAttr,
        [axis, flags, reserved]
    );
    assert_attr_layout!(
        ReduceMaxAttr,
        RawReduceMaxAttr,
        [axis_count, axes, flags, reserved]
    );
}
