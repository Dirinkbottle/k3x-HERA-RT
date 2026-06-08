//! 算子内联参数（attr）定义。
//!
//! 每个算子对应一个 `#[repr(C)] + Copy` 的 attr 结构，
//! 通过 `AiSubmitEntry::set_inline_attr` 写入 submit entry 的 `attr_inline` 区域。
//! 所有 attr 结构必须在编译期断言为固定大小，方便内核侧做大小校验。

use super::desc::AiDtype;

/// MatMul 算子参数。
///
/// 张量约定：
/// - tensors[0] = lhs
/// - tensors[1] = rhs
/// - tensors[2] = output
///
/// 第一阶段可以先支持 batch=1、无转置的基础路径。
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
///
/// 参数名尽量贴近 ggml rope 的语义，但第一阶段 backend 可以只实现最小子集。
/// 未支持字段必须要求用户态填默认值，而不是让 backend 猜。
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

// ── 编译期大小断言 ──────────────────────────────────────────────

const _: () = assert!(core::mem::size_of::<MatMulAttr>() == 64);
const _: () = assert!(core::mem::size_of::<RmsNormAttr>() == 64);
const _: () = assert!(core::mem::size_of::<RopeAttr>() == 64);
const _: () = assert!(core::mem::size_of::<SoftmaxAttr>() == 64);
const _: () = assert!(core::mem::size_of::<BinaryAttr>() == 64);
const _: () = assert!(core::mem::size_of::<UnaryAttr>() == 64);
const _: () = assert!(core::mem::size_of::<Conv2dAttr>() == 128);
