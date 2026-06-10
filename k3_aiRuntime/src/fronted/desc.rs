//! 提交给内核的 tensor 描述、单算子描述和 completion entry。
//!
//! 这些结构只描述用户态 buffer，不拥有内存。
//! 内核收到 graph 后必须校验 `user_va..user_va+size_bytes` 是否有效，
//! 并在执行期间 pin 住相关页帧，不能相信"用户态库保证有效"。

use super::kernel::{AiTargetHint, KernelOp};
use super::kernelattr::AiKernelAttr;
use super::{ATTR_INLINE_SIZE, MAX_DIM, MAX_SUBMIT_TENSORS};

/// tensor 元素类型。
///
/// ggml 量化格式先作为 dtype 编号保留下来。
/// 量化块内部布局由 `AiTensorLayout::GGML_QUANT` 和 `AiQuantDesc` 补充说明。
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct AiDtype(pub u32);

impl AiDtype {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const BF16: Self = Self(2);
    pub const I8: Self = Self(3);
    pub const U8: Self = Self(4);
    pub const I32: Self = Self(5);
    pub const I64: Self = Self(6);
    pub const BOOL: Self = Self(7);

    pub const Q4_0: Self = Self(100);
    pub const Q4_K: Self = Self(101);
    pub const Q8_0: Self = Self(102);

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
}

/// tensor 的逻辑格式。
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct AiTensorFormat(pub u32);

impl AiTensorFormat {
    pub const ANY: Self = Self(0);
    pub const NCHW: Self = Self(1);
    pub const NHWC: Self = Self(2);
    pub const NCDHW: Self = Self(3);
}

/// tensor 的物理布局。
///
/// format 描述逻辑维度含义，layout 描述内存怎么摆。
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct AiTensorLayout(pub u32);

impl AiTensorLayout {
    pub const DENSE: Self = Self(0);
    pub const STRIDED: Self = Self(1);
    pub const BLOCKED: Self = Self(2);
    pub const GGML_QUANT: Self = Self(3);
}

/// tensor 量化补充描述。
///
/// 此处主要记录 block_size 和 scale dtype，方便 lowering/backend 做校验。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiQuantDesc {
    pub scale: f32,
    pub zero_point: i32,
    pub block_size: u32,
    pub scale_dtype: AiDtype,
    pub flags: u32,
    pub reserved: [u32; 3],
}

/// 提交给内核的 tensor 描述。
///
/// 这个结构只描述用户态 buffer，不拥有内存。
/// 目前由内核驱动分配连续的物理页帧
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiTensorDesc {
    /// 由内核驱动分配的内存凭证token
    pub kernel_token:u64,

    /// 用户态虚拟地址。
    pub user_va: u64,

    /// 该 tensor 可访问的完整 buffer 字节数。
    pub size_bytes: u64,

    /// 元素类型，未知值必须被内核拒绝或走 fallback。
    pub dtype: AiDtype,

    /// 逻辑格式，例如 NCHW/NHWC。
    pub format: AiTensorFormat,

    /// 物理布局，例如 dense/strided/blocked/ggml quant。
    pub layout: AiTensorLayout,

    /// 实际维度数量，必须满足 `ndim <= MAX_DIM`。
    pub ndim: u32,

    /// tensor flags。具体含义由 frontend/backend 约定。
    /// TODO:
    pub flags: u8,

    /// 预留字段，保持 8 字节对齐，ABI 扩展留空间。
    pub reserved0: u32,

    /// 每个维度的元素数量。
    pub shape: [u32; MAX_DIM],

    /// 每个维度前进 1 个元素时跨过的字节数。
    pub stride_bytes: [u64; MAX_DIM],

    /// 量化补充信息。
    pub quant: AiQuantDesc,
}

impl AiTensorDesc {
    /// 从内核 allocator 返回的信息构造 tensor 描述。
    ///
    /// 由用户态 tensor allocator 包装层调用。真实 tensor 空间由内核
    /// allocator 分配，`kernel_token` 是后续 submit/bind/free 的可信身份。
    pub(crate) fn from_kernel_alloc(
        kernel_token: u64,
        user_va: u64,
        size_bytes: u64,
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[u32],
        _flags:u8
    ) -> Self {
        assert!(shape.len() <= MAX_DIM);

        let mut desc = Self {
            kernel_token,
            user_va,
            size_bytes,
            dtype,
            format,
            layout,
            ndim: shape.len() as u32,
            flags: _flags,
            ..Self::default()
        };

        desc.shape[..shape.len()].copy_from_slice(shape);
        desc
    }

    /// 调用内核 allocator 分配 tensor desc 和实际 tensor 空间。
    pub(crate) fn alloc_from_kernel(
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[u32],
        flags: u8,
    ) -> Self {
        let element_size = dtype
            .element_size_bytes()
            .expect("quantized or unknown dtype needs explicit allocator path");
        let size_bytes = tensor_size_bytes(shape, element_size);

        let alloc = kernel_alloc_tensor(size_bytes);
        Self::from_kernel_alloc(
            alloc.kernel_token,
            alloc.user_va,
            alloc.actual_size,
            dtype,
            format,
            layout,
            shape,
            flags,
        )
    }

    /// 释放由内核 allocator 分配的 tensor。
    pub(crate) fn free(&mut self) {
        if self.kernel_token == 0 {
            return;
        }

        kernel_free_tensor(*self);
        *self = Self::default();
    }
}

struct KernelTensorAllocResult {
    kernel_token: u64,
    user_va: u64,
    actual_size: u64,
}

fn kernel_alloc_tensor(_size_bytes: u64) -> KernelTensorAllocResult {
    todo!("call StarryOS tensor allocator ioctl")
}

fn kernel_free_tensor(_desc: AiTensorDesc) {
    // TODO: call StarryOS tensor allocator free ioctl.
    todo!()
}

fn tensor_size_bytes(shape: &[u32], element_size: u32) -> u64 {
    let element_count = shape.iter().fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim as u64)
            .expect("tensor element count overflow")
    });
    element_count
        .checked_mul(element_size as u64)
        .expect("tensor byte size overflow")
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
    pub input_count: u32,

    /// 输出 tensor 数量。输出紧跟在输入 tensor 后面。
    pub output_count: u32,

    /// 输入和输出 tensor 描述数组。
    /// 从[0..input_count)是输入tensor,[input_count..input_count+output_count)是输出tensor
    pub tensors: [AiTensorDesc; MAX_SUBMIT_TENSORS],

    /// 预留字段，保持后续 ABI 可扩展。
    pub reserved0: u32,

    /// attr_inline 中有效字节数。
    pub attr_size: u32,

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

            input_count: 0,
            output_count: 0,
            tensors: [AiTensorDesc::default(); MAX_SUBMIT_TENSORS],
            reserved0: 0,
            attr_size: 0,
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
        input_count: u32,
        output_count: u32,
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
        input_count: u32,
        output_count: u32,
        tensors: &[AiTensorDesc],
    ) -> Self {
        let total_count = input_count
            .checked_add(output_count)
            .expect("tensor count overflow");
        let total_count = usize::try_from(total_count).expect("tensor count does not fit usize");

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

        self.attr_size = size as u32;
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
    pub user_token: u64,

    /// 0 表示成功；负数可以对齐内核 errno 风格错误码。
    pub status: i32,

    /// 预留扩展字段。
    pub reserved0: u8,
}

// ── 编译期大小/对齐断言 ──────────────────────────────────────

const _: () = assert!(core::mem::align_of::<AiKernelDesc>() == 64);
const _: () = assert!(core::mem::align_of::<AiCompletionEntry>() == 64);
const _: () = assert!(core::mem::offset_of!(AiKernelDesc, attr_inline) % 8 == 0);
