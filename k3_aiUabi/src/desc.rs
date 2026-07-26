//! 提交给内核的 tensor 描述、单算子描述和 completion entry。
//!
//! 当前阶段 tensor 数据直接放在用户态虚拟地址空间。
//! `AiTensorDesc` 只描述这块用户态 buffer 的形状、布局和地址，本身不拥有内存。
//! 内核或 guard 程序收到 graph 后必须重新校验 `user_va..user_va+size_bytes`，
//! 不能把用户态 desc 当成可信对象直接使用。

use crate::{MAX_DIM, MAX_SUBMIT_TENSORS};

use super::kernel::{AiTargetHint, KernelOp};
use super::kernelattr::AiKernelAttr;
use super::types::{
    AttrByteSize, ByteSize, ByteStride, CompletionStatus, CompletionUserToken, DimCount, DimSize,
    KernelVa, TensorCount, TensorFlags, UserVa,
};

/// tensor 元素类型。
///
/// ggml 量化格式先作为 dtype 编号保留下来。
/// 量化块内部布局由 `AiTensorLayout::GGML_QUANT` 和 `AiQuantDesc` 补充说明。
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq, Debug)]
pub struct AiDtype(pub u32);

/// 内联算子参数区大小。
pub const ATTR_INLINE_SIZE: usize = 128;
impl AiDtype {
    /// 32 位单精度浮点。
    pub const F32: Self = Self(0);
    /// 16 位半精度浮点 (IEEE)。
    pub const F16: Self = Self(1);
    /// 16 位 bfloat。
    pub const BF16: Self = Self(2);
    /// 8 位有符号整数。
    pub const I8: Self = Self(3);
    /// 8 位无符号整数。
    pub const U8: Self = Self(4);
    /// 32 位有符号整数。
    pub const I32: Self = Self(5);
    /// 64 位有符号整数。
    pub const I64: Self = Self(6);
    /// 布尔值，占 1 字节。
    pub const BOOL: Self = Self(7);

    /// ggml Q4_0 量化格式。
    pub const Q4_0: Self = Self(100);
    /// ggml Q4_K 量化格式。
    pub const Q4_K: Self = Self(101);
    /// ggml Q8_0 量化格式。
    pub const Q8_0: Self = Self(102);
    /// ggml Q3_K 量化格式。
    pub const Q3_K: Self = Self(103);
    /// ggml IQ4_NL 量化格式。
    pub const IQ4_NL: Self = Self(104);

    /// 固定宽度 dtype 的单元素字节数。
    ///
    /// 量化 dtype 的物理尺寸依赖 block 格式，所以这里返回 None。
    pub const fn element_size_bytes(self) -> Option<u32> {
        match self.0 {
            0 => Some(4),
            1 | 2 => Some(2),
            3 | 4 | 7 => Some(1),
            5 => Some(4),
            6 => Some(8),
            _ => None,
        }
    }

    /// 返回 ggml 量化 dtype 的 block 内逻辑元素数。
    pub const fn ggml_quant_block_size(self) -> Option<u32> {
        match self.0 {
            102 => Some(32),
            103 => Some(256),
            104 => Some(32),
            _ => None,
        }
    }

    /// 返回 ggml 量化 dtype 单个 block 的物理字节数。
    pub const fn ggml_quant_block_bytes(self) -> Option<u32> {
        match self.0 {
            102 => Some(34),
            103 => Some(110),
            104 => Some(18),
            _ => None,
        }
    }

    /// 判断 dtype 是否为本 ABI 可描述的 ggml 量化块格式。
    pub const fn is_ggml_quant(self) -> bool {
        self.ggml_quant_block_size().is_some()
    }
}

/// tensor 的逻辑格式。
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct AiTensorFormat(pub u32);

impl AiTensorFormat {
    /// 不限格式，由 backend 自行推断。
    pub const ANY: Self = Self(0);
    /// batch-channel-height-width，最常见的 CNN 格式。
    pub const NCHW: Self = Self(1);
    /// batch-height-width-channel，移动端友好格式。
    pub const NHWC: Self = Self(2);
    /// batch-channel-depth-height-width，3D 卷积格式。
    pub const NCDHW: Self = Self(3);
}

/// tensor 的物理布局。
///
/// format 描述逻辑维度含义，layout 描述内存怎么摆。
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct AiTensorLayout(pub u32);

impl AiTensorLayout {
    /// 密集布局，元素连续排列。
    pub const DENSE: Self = Self(0);
    /// 带 stride 布局，允许子张量视图。
    pub const STRIDED: Self = Self(1);
    /// 分块布局，用于 blocked matmul 等优化场景。
    pub const BLOCKED: Self = Self(2);
    /// ggml 量化格式，需配合 `AiQuantDesc` 解释。
    pub const GGML_QUANT: Self = Self(3);
}

/// tensor 量化补充描述。
///
/// 此处主要记录 block_size 和 scale dtype，方便 lowering/backend 做校验。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiQuantDesc {
    /// 反量化缩放系数。
    pub scale: f32,
    /// 零点偏移，对称量化时为 0。
    pub zero_point: i32,
    /// 量化 block 内元素个数。
    pub block_size: DimSize,
    /// scale 值本身的数据类型。
    pub scale_dtype: AiDtype,
    /// 量化 flags，具体含义由 backend 约定。
    pub flags: TensorFlags,
    /// 预留字段，保持 ABI 可扩展。
    pub reserved: [u32; 3],
}

/// 提交给内核的 tensor 描述。
///
/// 这个结构只描述用户态 buffer，不拥有内存。
/// tensor 的实际生命周期由用户态 `Tensor` / `TensorManager` 管理。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiTensorDesc {
    /// 用户态虚拟地址。
    pub user_va: UserVa,

    /// 内核态虚拟地址。 由内核创建alias后填写
    pub kernel_va: KernelVa,

    /// 该 tensor 可访问的完整 buffer 字节数。
    pub size_bytes: ByteSize,

    /// 元素类型，未知值必须被内核拒绝或走 fallback。
    pub dtype: AiDtype,

    /// 逻辑格式，例如 NCHW/NHWC。
    pub format: AiTensorFormat,

    /// 物理布局，例如 dense/strided/blocked/ggml quant。
    pub layout: AiTensorLayout,

    /// 实际维度数量，必须满足 `ndim <= MAX_DIM`。
    pub ndim: DimCount,

    /// tensor flags。具体含义由 frontend/backend 约定。
    /// TODO:
    pub flags: TensorFlags,

    /// 预留字段，ABI 扩展留空间。
    pub reserved0: u32,

    /// 每个维度的元素数量。
    pub shape: [DimSize; MAX_DIM],

    /// 每个维度前进 1 个元素时跨过的字节数。
    pub stride_bytes: [ByteStride; MAX_DIM],

    /// 量化补充信息。
    pub quant: AiQuantDesc,
}

impl AiTensorDesc {
    /// 从一块已有的用户态 buffer 构造 tensor 描述。
    ///
    /// 这里不分配内存，只把用户态地址、shape、dtype 等语义收敛成稳定 ABI。
    pub fn from_user_buffer(
        user_va: UserVa,
        size_bytes: ByteSize,
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[DimSize],
        flags: TensorFlags,
    ) -> Self {
        assert!(shape.len() <= MAX_DIM);

        assert!(
            layout != AiTensorLayout::GGML_QUANT,
            "ggml quant tensor needs from_user_quant_buffer"
        );
        let element_size = dtype
            .element_size_bytes()
            .expect("unknown dtype needs explicit size path");
        let required_size = tensor_size_bytes(shape, element_size);
        assert!(size_bytes.get() >= required_size.get());

        let mut desc = Self {
            user_va,
            size_bytes,
            dtype,
            format,
            layout,
            ndim: DimCount::new(shape.len() as u32),
            flags,
            ..Self::default()
        };

        desc.shape[..shape.len()].copy_from_slice(shape);
        if !shape.is_empty() {
            let mut stride = element_size as u64;
            for dim_idx in (0..shape.len()).rev() {
                desc.stride_bytes[dim_idx] = ByteStride::new(stride);
                stride = stride
                    .checked_mul(shape[dim_idx].get() as u64)
                    .expect("tensor stride overflow");
            }
        }
        desc
    }

    /// 从一块已有的 ggml 量化 buffer 构造 tensor 描述。
    ///
    /// shape 使用逻辑元素数；第 0 维必须是量化 block 的整数倍。stride 采用
    /// ggml 约定：第 0 维 stride 为 block 字节数，其余维度按完整逻辑行递增。
    pub fn from_user_quant_buffer(
        user_va: UserVa,
        size_bytes: ByteSize,
        dtype: AiDtype,
        format: AiTensorFormat,
        shape: &[DimSize],
        flags: TensorFlags,
    ) -> Self {
        assert!(shape.len() <= MAX_DIM);
        let required_size = ggml_quant_tensor_size_bytes(dtype, shape)
            .expect("invalid ggml quant tensor shape or dtype");
        assert!(size_bytes.get() >= required_size.get());

        let block_size = dtype
            .ggml_quant_block_size()
            .expect("invalid ggml quant dtype");
        let block_bytes = dtype
            .ggml_quant_block_bytes()
            .expect("invalid ggml quant dtype");

        let mut desc = Self {
            user_va,
            size_bytes,
            dtype,
            format,
            layout: AiTensorLayout::GGML_QUANT,
            ndim: DimCount::new(shape.len() as u32),
            flags,
            quant: AiQuantDesc {
                block_size: DimSize::new(block_size),
                scale_dtype: AiDtype::F16,
                flags,
                ..AiQuantDesc::default()
            },
            ..Self::default()
        };

        desc.shape[..shape.len()].copy_from_slice(shape);
        if !shape.is_empty() {
            desc.stride_bytes[0] = ByteStride::new(block_bytes as u64);
            let row_blocks = shape[0].get() as u64 / block_size as u64;
            let mut stride = row_blocks
                .checked_mul(block_bytes as u64)
                .expect("quant tensor stride overflow");
            for dim_idx in 1..shape.len() {
                desc.stride_bytes[dim_idx] = ByteStride::new(stride);
                stride = stride
                    .checked_mul(shape[dim_idx].get() as u64)
                    .expect("quant tensor stride overflow");
            }
        }
        desc
    }
}

/// 计算 tensor 数据总字节数：各维度元素数之积 × 单元素字节数。
pub fn tensor_size_bytes(shape: &[DimSize], element_size: u32) -> ByteSize {
    let element_count = shape.iter().fold(1_u64, |acc, dim| {
        acc.checked_mul(dim.get() as u64)
            .expect("tensor element count overflow")
    });
    ByteSize::new(
        element_count
            .checked_mul(element_size as u64)
            .expect("tensor byte size overflow"),
    )
}

/// 计算 ggml 量化 tensor 数据总字节数。
pub fn ggml_quant_tensor_size_bytes(dtype: AiDtype, shape: &[DimSize]) -> Option<ByteSize> {
    if shape.is_empty() {
        return None;
    }
    let block_size = dtype.ggml_quant_block_size()? as u64;
    let block_bytes = dtype.ggml_quant_block_bytes()? as u64;
    let first = shape[0].get() as u64;
    if first == 0 || !first.is_multiple_of(block_size) {
        return None;
    }
    let row_bytes = first.checked_div(block_size)?.checked_mul(block_bytes)?;
    let rows = shape[1..]
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(dim.get() as u64))?;
    row_bytes.checked_mul(rows).map(ByteSize::new)
}

/// 单个 lowered 算子的稳定描述。
/// 内核调度器按 `op` 解释 `attr_inline`，按 input/output count 解释 tensors。
/// 对齐cacheline大小
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct AiKernelDesc {
    /// 语义级 op。
    pub op: KernelOp,

    /// 用户态对目标的倾向(hint)。实际调度由调度器决定。
    pub target_hint: AiTargetHint,

    /// 输入 tensor 数量。输入必须放在 tensors 数组前部。
    pub input_count: TensorCount,

    /// 输出 tensor 数量。输出紧跟在输入 tensor 后面。
    pub output_count: TensorCount,

    /// 输入和输出 tensor 描述数组。
    /// 从[0..input_count)是输入tensor,[input_count..input_count+output_count)是输出tensor
    pub tensors: [AiTensorDesc; MAX_SUBMIT_TENSORS],

    /// 预留字段，保持后续 ABI 可扩展。
    pub reserved0: u32,

    /// attr_inline 中有效字节数。
    pub attr_size: AttrByteSize,

    /// 内联算子参数。
    ///
    /// 按 op 解释为 MatMulAttr/RmsNormAttr/RopeAttr 等。
    pub attr_inline: [u8; ATTR_INLINE_SIZE],
}

impl Default for AiKernelDesc {
    fn default() -> Self {
        Self {
            op: KernelOp::INVALID,
            target_hint: AiTargetHint::AUTO,

            input_count: TensorCount::new(0),
            output_count: TensorCount::new(0),
            tensors: [AiTensorDesc::default(); MAX_SUBMIT_TENSORS],
            reserved0: 0,
            attr_size: AttrByteSize::new(0),
            attr_inline: [0; ATTR_INLINE_SIZE],
        }
    }
}

impl AiKernelDesc {
    /// 通过 attr 类型自动构造单算子 desc。
    ///
    /// `attr` 的类型必须能唯一映射到一个 `KernelOp`，例如 `MatMulAttr` 会解析为
    /// `KernelOp::MAT_MUL`。输入 tensor 必须放在 `tensors` 前部，输出 tensor 紧跟其后。
    pub fn new<T: AiKernelAttr>(
        attr: &T,
        target_hint: AiTargetHint,
        input_count: TensorCount,
        output_count: TensorCount,
        tensors: &[AiTensorDesc],
    ) -> Self {
        Self::new_with_op(T::OP, attr, target_hint, input_count, output_count, tensors)
    }

    /// 显式指定 op 构造 desc。
    ///
    /// ADD/MUL、SILU/SCALE 这类多个 op 共用同一个 attr 的算子走这个入口。
    pub fn new_with_op<T: Copy>(
        op: KernelOp,
        attr: &T,
        target_hint: AiTargetHint,
        input_count: TensorCount,
        output_count: TensorCount,
        tensors: &[AiTensorDesc],
    ) -> Self {
        let total_count = input_count
            .checked_total(output_count)
            .expect("tensor count overflow");

        assert!(total_count <= MAX_SUBMIT_TENSORS);
        assert!(tensors.len() == total_count);

        let mut desc = Self {
            op,
            target_hint,
            input_count,
            output_count,
            ..Self::default()
        };
        desc.tensors[..total_count].copy_from_slice(tensors);
        desc.set_inline_attr(attr);
        desc
    }

    /// 写入内联 attr。
    ///
    /// 只应该传入本模块内定义的 `#[repr(C)] + Copy` attr 结构。
    fn set_inline_attr<T: Copy>(&mut self, attr: &T) {
        let size = core::mem::size_of::<T>();
        assert!(size <= ATTR_INLINE_SIZE);

        self.attr_size = AttrByteSize::new(size as u32);
        self.attr_inline = [0; ATTR_INLINE_SIZE];

        unsafe {
            core::ptr::copy_nonoverlapping(
                (attr as *const T).cast::<u8>(),
                self.attr_inline.as_mut_ptr(),
                size,
            );
        }
    }
}

/// AiGraphSubmitEntry对应的执行结果描述。
#[repr(C, align(64))]
#[derive(Clone, Copy, Default)]
pub struct AiCompletionEntry {
    /// submit 时传入的 token，完成时原样返回。
    pub user_token: CompletionUserToken,

    /// 0 表示成功；负数可以对齐内核 errno 风格错误码。
    pub status: CompletionStatus,

    /// 预留扩展字段。
    pub reserved0: u8,
}

// ── 编译期大小/对齐断言 ──────────────────────────────────────

const _: () = assert!(core::mem::align_of::<AiKernelDesc>() == 64);
const _: () = assert!(core::mem::align_of::<AiCompletionEntry>() == 64);
const _: () = assert!(core::mem::offset_of!(AiKernelDesc, attr_inline) % 8 == 0);

/// ABI raw mirror layout checks for transparent newtype field replacements.
#[allow(dead_code, missing_docs, clippy::missing_docs_in_private_items)]
mod abi_layout {
    use super::*;

    #[repr(C)]
    struct RawAiQuantDesc {
        scale: f32,
        zero_point: i32,
        block_size: u32,
        scale_dtype: u32,
        flags: u8,
        reserved: [u32; 3],
    }

    #[repr(C)]
    struct RawAiTensorDesc {
        user_va: u64,
        kernel_va: u64,
        size_bytes: u64,
        dtype: u32,
        format: u32,
        layout: u32,
        ndim: u32,
        flags: u8,
        reserved0: u32,
        shape: [u32; MAX_DIM],
        stride_bytes: [u64; MAX_DIM],
        quant: RawAiQuantDesc,
    }

    #[repr(C, align(64))]
    struct RawAiKernelDesc {
        op: u8,
        target_hint: u8,
        input_count: u32,
        output_count: u32,
        tensors: [RawAiTensorDesc; MAX_SUBMIT_TENSORS],
        reserved0: u32,
        attr_size: u32,
        attr_inline: [u8; ATTR_INLINE_SIZE],
    }

    #[repr(C, align(64))]
    struct RawAiCompletionEntry {
        user_token: u64,
        status: i32,
        reserved0: u8,
    }

    const _: () =
        assert!(core::mem::size_of::<AiQuantDesc>() == core::mem::size_of::<RawAiQuantDesc>());
    const _: () =
        assert!(core::mem::align_of::<AiQuantDesc>() == core::mem::align_of::<RawAiQuantDesc>());
    const _: () = assert!(
        core::mem::offset_of!(AiQuantDesc, block_size)
            == core::mem::offset_of!(RawAiQuantDesc, block_size)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiQuantDesc, flags) == core::mem::offset_of!(RawAiQuantDesc, flags)
    );

    const _: () =
        assert!(core::mem::size_of::<AiTensorDesc>() == core::mem::size_of::<RawAiTensorDesc>());
    const _: () =
        assert!(core::mem::align_of::<AiTensorDesc>() == core::mem::align_of::<RawAiTensorDesc>());
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, user_va)
            == core::mem::offset_of!(RawAiTensorDesc, user_va)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, kernel_va)
            == core::mem::offset_of!(RawAiTensorDesc, kernel_va)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, size_bytes)
            == core::mem::offset_of!(RawAiTensorDesc, size_bytes)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, ndim) == core::mem::offset_of!(RawAiTensorDesc, ndim)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, flags) == core::mem::offset_of!(RawAiTensorDesc, flags)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, shape) == core::mem::offset_of!(RawAiTensorDesc, shape)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, stride_bytes)
            == core::mem::offset_of!(RawAiTensorDesc, stride_bytes)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiTensorDesc, quant) == core::mem::offset_of!(RawAiTensorDesc, quant)
    );

    const _: () =
        assert!(core::mem::size_of::<AiKernelDesc>() == core::mem::size_of::<RawAiKernelDesc>());
    const _: () =
        assert!(core::mem::align_of::<AiKernelDesc>() == core::mem::align_of::<RawAiKernelDesc>());
    const _: () = assert!(
        core::mem::offset_of!(AiKernelDesc, input_count)
            == core::mem::offset_of!(RawAiKernelDesc, input_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiKernelDesc, output_count)
            == core::mem::offset_of!(RawAiKernelDesc, output_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiKernelDesc, tensors)
            == core::mem::offset_of!(RawAiKernelDesc, tensors)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiKernelDesc, attr_size)
            == core::mem::offset_of!(RawAiKernelDesc, attr_size)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiKernelDesc, attr_inline)
            == core::mem::offset_of!(RawAiKernelDesc, attr_inline)
    );

    const _: () = assert!(
        core::mem::size_of::<AiCompletionEntry>() == core::mem::size_of::<RawAiCompletionEntry>()
    );
    const _: () = assert!(
        core::mem::align_of::<AiCompletionEntry>() == core::mem::align_of::<RawAiCompletionEntry>()
    );
    const _: () = assert!(
        core::mem::offset_of!(AiCompletionEntry, user_token)
            == core::mem::offset_of!(RawAiCompletionEntry, user_token)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiCompletionEntry, status)
            == core::mem::offset_of!(RawAiCompletionEntry, status)
    );
}
