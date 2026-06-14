//! 算子内联参数（attr）定义。
//!
//! 每个算子对应一个 `#[repr(C)] + Copy` 的 attr 结构，
//! 通过 `AiKernelDesc::set_inline_attr` 写入单算子描述的 `attr_inline` 区域。
//! 所有 attr 结构必须在编译期断言为固定大小，方便内核侧做大小校验。

use super::{desc::AiDtype, kernel::KernelOp};

/// 能唯一映射到一个语义级 op 的算子 attr。
///
/// `AiKernelDesc::new` 通过这个 trait 从 attr 类型解析出 `KernelOp`。
/// ADD/MUL、SILU/SCALE 这类复用同一个 attr 的算子不能实现这个 trait，
/// 需要通过显式 op 构造入口创建 desc。
pub trait AiKernelAttr: Copy {
    const OP: KernelOp;
}

/// MatMul 最小算子参数。
///
/// 张量约定：
/// - tensors[0] = lhs
/// - tensors[1] = rhs
/// - tensors[2] = output
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MatMulAttr {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub lhs_batch_stride_bytes: u64,
    pub rhs_batch_stride_bytes: u64,
    pub out_batch_stride_bytes: u64,
    pub flags: u32,
    pub accum_dtype: AiDtype,
    pub reserved: [u32; 4],
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
    pub hidden_size: u32,
    pub eps: f32,
    pub flags: u32,
    pub reserved: [u32; 13],
}

/// RoPE 算子参数。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RopeAttr {
    pub n_dims: u32,
    pub mode: u32,
    pub n_ctx: u32,
    pub head_count: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub flags: u32,
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
    pub axis: i32,
    pub scale: f32,
    pub flags: u32,
    pub reserved: [u32; 13],
}

/// 二元 elementwise 算子参数。
///
/// ADD/MUL 可以共用该 attr，具体语义由 KernelOp 区分。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BinaryAttr {
    pub broadcast_kind: u32,
    pub alpha: f32,
    pub beta: f32,
    pub flags: u32,
    pub reserved: [u32; 12],
}

/// 单输入 elementwise 算子参数。
///
/// SILU/SCALE 这类轻量 op 可以先共用该 attr。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct UnaryAttr {
    pub alpha: f32,
    pub beta: f32,
    pub flags: u32,
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
    pub batch: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub input_h: u32,
    pub input_w: u32,
    pub output_h: u32,
    pub output_w: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub groups: u32,
    pub flags: u32,
    pub reserved: [u32; 15],
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

impl AiKernelAttr for Conv2dAttr {
    const OP: KernelOp = KernelOp::CONV2D;
}

// ── 编译期大小断言 ──────────────────────────────────────────────

const _: () = assert!(core::mem::size_of::<MatMulAttr>() == 64);
const _: () = assert!(core::mem::size_of::<RmsNormAttr>() == 64);
const _: () = assert!(core::mem::size_of::<RopeAttr>() == 64);
const _: () = assert!(core::mem::size_of::<SoftmaxAttr>() == 64);
const _: () = assert!(core::mem::size_of::<BinaryAttr>() == 64);
const _: () = assert!(core::mem::size_of::<UnaryAttr>() == 64);
const _: () = assert!(core::mem::size_of::<Conv2dAttr>() == 128);
