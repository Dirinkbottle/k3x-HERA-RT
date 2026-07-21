//! Backend 算子公共调用层。

use core::mem::{align_of, size_of};
use core::ops::RangeInclusive;
use core::slice;

use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{
    AiDtype, AiTargetHint, AiTensorDesc, AiTensorFormat, AiTensorLayout, AttrByteSize, ByteSize,
    ByteStride, DimCount, DimSize, KernelOp, MAX_DIM, TensorAxis, TensorCount, TensorFlags,
};
use log::error;

/// backend 算子的 tensor 视图，`data` 指向当前地址空间可访问的连续内存。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BackendTensorView {
    /// 指向当前地址空间可访问的连续 tensor 数据。
    pub data: *mut u8,
    /// 数据区字节数。
    pub byte_len: ByteSize,
    /// 各维度元素数量。
    pub shape: [DimSize; MAX_DIM],
    /// 各维度前进 1 个元素的字节跨度。
    pub stride_bytes: [ByteStride; MAX_DIM],
    /// 实际维度数量。
    pub ndim: DimCount,
    /// 元素数据类型。
    pub dtype: AiDtype,
    /// 逻辑格式（NCHW/NHWC 等）。
    pub format: AiTensorFormat,
    /// 物理布局（dense/strided 等）。
    pub layout: AiTensorLayout,
    /// tensor flags，含义由 frontend/backend 约定。
    pub flags: TensorFlags,
}

impl Default for BackendTensorView {
    fn default() -> Self {
        Self {
            data: core::ptr::null_mut(),
            byte_len: ByteSize::new(0),
            shape: [DimSize::new(0); MAX_DIM],
            stride_bytes: [ByteStride::new(0); MAX_DIM],
            ndim: DimCount::new(0),
            dtype: AiDtype::default(),
            format: AiTensorFormat::default(),
            layout: AiTensorLayout::default(),
            flags: TensorFlags::new(0),
        }
    }
}

impl BackendTensorView {
    /// 从 `AiTensorDesc` 构造 view，`data` 取自内核映射后的 `kernel_va`。
    pub(crate) fn from_desc(desc: &AiTensorDesc) -> Self {
        if desc.kernel_va == 0 {
            error!("desc tensor has null kernel_va!");
        }

        Self {
            data: desc.kernel_va.get() as *mut u8,
            byte_len: desc.size_bytes,
            shape: desc.shape,
            stride_bytes: desc.stride_bytes,
            ndim: desc.ndim,
            dtype: desc.dtype,
            format: desc.format,
            layout: desc.layout,
            flags: desc.flags,
        }
    }

    /// 将 tensor buffer 解释为只读 typed slice。
    ///
    /// # Safety
    ///
    /// 调用方必须已经校验 dtype 与 `T` 匹配，`data..data+byte_len` 在返回 slice
    /// 生命周期内可读，且遵守输入/输出别名规则。
    pub unsafe fn as_slice<T>(&self) -> Result<&[T], BackendErr> {
        let len = self.typed_len::<T>()?;
        Ok(unsafe { slice::from_raw_parts(self.data.cast::<T>(), len) })
    }

    /// 将 tensor buffer 解释为可写 typed slice。
    ///
    /// # Safety
    ///
    /// 调用方必须已经校验 dtype 与 `T` 匹配，`data..data+byte_len` 在返回 slice
    /// 生命周期内可写，且当前调用独占这段输出内存。
    pub unsafe fn as_mut_slice<T>(&mut self) -> Result<&mut [T], BackendErr> {
        let len = self.typed_len::<T>()?;
        Ok(unsafe { slice::from_raw_parts_mut(self.data.cast::<T>(), len) })
    }

    /// 校验非空、字节数整除元素大小、地址对齐后，返回按 `T` 计的元素个数。
    fn typed_len<T>(&self) -> Result<usize, BackendErr> {
        if self.data.is_null() {
            error!("BackendTensorView: tensor data is null");
            return Err(BackendErr::InvalidTensor);
        }

        let byte_len = self.byte_len.try_as_usize().map_err(|_| {
            error!(
                "BackendTensorView: byte_len {} does not fit usize",
                self.byte_len
            );
            BackendErr::InvalidTensor
        })?;
        let element_size = size_of::<T>();
        if element_size == 0 {
            error!("BackendTensorView: zero-sized typed buffers are unsupported");
            return Err(BackendErr::InvalidTensor);
        }
        if !byte_len.is_multiple_of(element_size) {
            error!(
                "BackendTensorView: byte_len {} is not divisible by element size {}",
                byte_len, element_size
            );
            return Err(BackendErr::InvalidTensor);
        }

        let align = align_of::<T>();
        if align > 1 && !(self.data as usize).is_multiple_of(align) {
            error!(
                "BackendTensorView: tensor data {:p} is not aligned to {}",
                self.data, align
            );
            return Err(BackendErr::InvalidTensor);
        }

        Ok(byte_len / element_size)
    }

    /// 校验 tensor 的 dtype、rank、shape、stride 和 buffer bounds，并返回内部元数据。
    pub(crate) fn checked_meta(&self) -> Result<TensorMeta, BackendErr> {
        let rank = self
            .ndim
            .try_under_max(MAX_DIM)
            .map_err(|_| BackendErr::InvalidTensor)?;
        let element_size = self
            .dtype
            .element_size_bytes()
            .ok_or(BackendErr::UnsupportedDtype)? as usize;
        let byte_len = self
            .byte_len
            .try_as_usize()
            .map_err(|_| BackendErr::InvalidTensor)?;
        if self.data.is_null() {
            return Err(BackendErr::InvalidTensor);
        }
        if self.layout != AiTensorLayout::DENSE && self.layout != AiTensorLayout::STRIDED {
            return Err(BackendErr::InvalidTensor);
        }

        let mut shape = [0_usize; MAX_DIM];
        let mut strides = [0_usize; MAX_DIM];
        let mut element_count = 1_usize;
        let mut max_offset = 0_usize;
        for axis in 0..rank {
            shape[axis] = self.shape[axis]
                .try_as_usize()
                .map_err(|_| BackendErr::InvalidTensor)?;
            let stride_bytes = self.stride_bytes[axis]
                .try_as_usize()
                .map_err(|_| BackendErr::InvalidTensor)?;
            if !stride_bytes.is_multiple_of(element_size) {
                return Err(BackendErr::InvalidTensor);
            }
            strides[axis] = stride_bytes / element_size;
            element_count = element_count
                .checked_mul(shape[axis])
                .ok_or(BackendErr::InvalidTensor)?;
            if shape[axis] > 1 && strides[axis] == 0 {
                return Err(BackendErr::InvalidTensor);
            }
            max_offset = max_offset
                .checked_add(
                    shape[axis]
                        .saturating_sub(1)
                        .checked_mul(stride_bytes)
                        .ok_or(BackendErr::InvalidTensor)?,
                )
                .ok_or(BackendErr::InvalidTensor)?;
        }

        if rank == 0 {
            element_count = 1;
        }
        if element_count != 0
            && max_offset
                .checked_add(element_size)
                .is_none_or(|required| required > byte_len)
        {
            return Err(BackendErr::InvalidTensor);
        }

        if self.layout == AiTensorLayout::DENSE {
            let mut expected = 1_usize;
            for axis in (0..rank).rev() {
                if shape[axis] > 1 && strides[axis] != expected {
                    return Err(BackendErr::InvalidTensor);
                }
                expected = expected
                    .checked_mul(shape[axis])
                    .ok_or(BackendErr::InvalidTensor)?;
            }
        }

        Ok(TensorMeta {
            rank,
            shape,
            strides,
            element_size,
            element_count,
        })
    }
}

/// 已校验、以元素为单位的 tensor 元数据。
#[derive(Clone, Copy, Debug)]
pub(crate) struct TensorMeta {
    /// 有效维度数。
    pub(crate) rank: usize,
    /// 逻辑 shape。
    pub(crate) shape: [usize; MAX_DIM],
    /// 每个逻辑轴的元素 stride。
    pub(crate) strides: [usize; MAX_DIM],
    /// 单元素字节数。
    pub(crate) element_size: usize,
    /// 逻辑元素数量。
    pub(crate) element_count: usize,
}

impl TensorMeta {
    /// 把 row-major 逻辑线性下标转换为底层 typed slice 元素下标。
    pub(crate) fn offset_for_linear(&self, linear: usize) -> Result<usize, BackendErr> {
        if linear >= self.element_count {
            return Err(BackendErr::InvalidTensor);
        }
        let mut coordinates = [0_usize; MAX_DIM];
        self.coordinates(linear, &mut coordinates)?;
        self.offset_for_coordinates(&coordinates)
    }

    /// 把逻辑线性下标拆成坐标。
    pub(crate) fn coordinates(
        &self,
        linear: usize,
        coordinates: &mut [usize; MAX_DIM],
    ) -> Result<(), BackendErr> {
        if linear >= self.element_count {
            return Err(BackendErr::InvalidTensor);
        }
        let mut remaining = linear;
        for axis in (0..self.rank).rev() {
            let dim = self.shape[axis];
            if dim == 0 {
                return Err(BackendErr::InvalidTensor);
            }
            coordinates[axis] = remaining % dim;
            remaining /= dim;
        }
        Ok(())
    }

    /// 把逻辑坐标转换为底层 typed slice 元素下标。
    pub(crate) fn offset_for_coordinates(
        &self,
        coordinates: &[usize; MAX_DIM],
    ) -> Result<usize, BackendErr> {
        let mut offset = 0_usize;
        for (axis, &coordinate) in coordinates.iter().enumerate().take(self.rank) {
            if coordinate >= self.shape[axis] {
                return Err(BackendErr::InvalidTensor);
            }
            offset = offset
                .checked_add(
                    coordinate
                        .checked_mul(self.strides[axis])
                        .ok_or(BackendErr::InvalidTensor)?,
                )
                .ok_or(BackendErr::InvalidTensor)?;
        }
        Ok(offset)
    }

    /// 判断 tensor 是否为 row-major contiguous。
    pub(crate) fn is_contiguous(&self) -> bool {
        let mut expected = 1_usize;
        for axis in (0..self.rank).rev() {
            if self.shape[axis] > 1 && self.strides[axis] != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(self.shape[axis]) else {
                return false;
            };
            expected = next;
        }
        true
    }
}

/// 把支持负数的 ONNX axis 归一化到 `0..rank`。
pub(crate) fn normalize_axis(axis: TensorAxis, rank: usize) -> Result<usize, BackendErr> {
    if rank == 0 || rank > MAX_DIM {
        return Err(BackendErr::InvalidTensor);
    }
    let raw = axis.get();
    let rank_i32 = i32::try_from(rank).map_err(|_| BackendErr::InvalidAttr)?;
    let normalized = if raw < 0 {
        raw.checked_add(rank_i32).ok_or(BackendErr::InvalidAttr)?
    } else {
        raw
    };
    if normalized < 0 || normalized >= rank_i32 {
        return Err(BackendErr::InvalidAttr);
    }
    usize::try_from(normalized).map_err(|_| BackendErr::InvalidAttr)
}

/// 单次 backend 算子调用描述。
#[repr(C)]
pub struct BackendCall {
    /// 操作类型。
    pub op: KernelOp,
    /// 执行目标（CPU/X100/A100）。
    pub target: u8,
    /// 输入 tensor view 数组首指针。
    pub inputs: *const BackendTensorView,
    /// 输入 tensor 数量。
    pub input_count: TensorCount,
    /// 输出 tensor view 数组首指针。
    pub outputs: *mut BackendTensorView,
    /// 输出 tensor 数量。
    pub output_count: TensorCount,
    /// kernel attr 结构的地址。
    pub attr: *const u8,
    /// kernel attr 结构的字节大小。
    pub attr_size: AttrByteSize,
}

/// 受生命周期约束的 backend 调用上下文，从 `BackendCall` 安全转换而来。
pub(crate) struct CallContext<'a> {
    /// 执行目标倾向。
    pub(crate) target: AiTargetHint,
    /// 输入 tensor view 切片。
    pub(crate) inputs: &'a [BackendTensorView],
    /// 输出 tensor view 切片（本次调用独占可变访问）。
    pub(crate) outputs: &'a mut [BackendTensorView],
    /// kernel attr 结构地址。
    attr: *const u8,
    /// kernel attr 结构字节大小。
    attr_size: usize,
}

impl<'a> CallContext<'a> {
    /// 从 C ABI 调用描述构造受生命周期约束的 backend 调用上下文。
    ///
    /// # Safety
    ///
    /// `call` 必须指向一个在返回上下文生命周期内有效的 `BackendCall`。当对应
    /// count 非零时，`inputs` 和 `outputs` 必须分别指向有效数组，且 `outputs`
    /// 在本次调用期间可被唯一可变访问。
    pub(crate) unsafe fn from_call(call: *const BackendCall) -> Result<Self, BackendErr> {
        if call.is_null() {
            error!("CallContext: call is null");
            return Err(BackendErr::NullPointer);
        }

        let call = unsafe { &*call };
        let target = AiTargetHint(call.target);
        if !target.is_known() {
            error!("CallContext: unknown target {}", call.target);
            return Err(BackendErr::UnsupportedOp);
        }

        let input_count = call
            .input_count
            .try_as_usize()
            .map_err(|_| BackendErr::InvalidInput)?;
        let output_count = call
            .output_count
            .try_as_usize()
            .map_err(|_| BackendErr::InvalidInput)?;

        if input_count > 0 && call.inputs.is_null() {
            error!("CallContext: inputs is null with count {}", input_count);
            return Err(BackendErr::NullPointer);
        }
        if output_count > 0 && call.outputs.is_null() {
            error!("CallContext: outputs is null with count {}", output_count);
            return Err(BackendErr::NullPointer);
        }

        let inputs = if input_count == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(call.inputs, input_count) }
        };
        let outputs = if output_count == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(call.outputs, output_count) }
        };

        Ok(Self {
            target,
            inputs,
            outputs,
            attr: call.attr,
            attr_size: call
                .attr_size
                .try_as_usize()
                .map_err(|_| BackendErr::InvalidAttr)?,
        })
    }

    /// 校验输入/输出 tensor 数量是否与算子期望一致。
    pub(crate) fn expect_io(
        &self,
        input_count: usize,
        output_count: usize,
    ) -> Result<(), BackendErr> {
        if self.inputs.len() != input_count || self.outputs.len() != output_count {
            error!(
                "CallContext: invalid input/output count, input_count={}, output_count={}",
                self.inputs.len(),
                self.outputs.len()
            );
            return Err(BackendErr::InvalidInput);
        }
        Ok(())
    }

    /// 校验输入/输出 tensor 数量是否落在算子允许的闭区间。
    pub(crate) fn expect_io_range(
        &self,
        input_count: RangeInclusive<usize>,
        output_count: RangeInclusive<usize>,
    ) -> Result<(), BackendErr> {
        if !input_count.contains(&self.inputs.len()) || !output_count.contains(&self.outputs.len())
        {
            return Err(BackendErr::InvalidInput);
        }
        Ok(())
    }

    /// 拒绝任何 input/output 地址区间重叠。
    pub(crate) fn reject_input_output_alias(&self) -> Result<(), BackendErr> {
        for input in self.inputs {
            let input_start = input.data as usize;
            let input_len = input
                .byte_len
                .try_as_usize()
                .map_err(|_| BackendErr::InvalidTensor)?;
            let input_end = input_start
                .checked_add(input_len)
                .ok_or(BackendErr::InvalidTensor)?;
            for output in self.outputs.iter() {
                let output_start = output.data as usize;
                let output_len = output
                    .byte_len
                    .try_as_usize()
                    .map_err(|_| BackendErr::InvalidTensor)?;
                let output_end = output_start
                    .checked_add(output_len)
                    .ok_or(BackendErr::InvalidTensor)?;
                if input_start < output_end && output_start < input_end {
                    return Err(BackendErr::InvalidTensor);
                }
            }
        }
        Ok(())
    }

    /// 从 attr 区非对齐读出一个 `T`；attr 为空或长度不足时返回错误。
    pub(crate) fn read_attr<T: Copy>(&self) -> Result<T, BackendErr> {
        let size = size_of::<T>();
        if self.attr.is_null() || self.attr_size < size {
            error!(
                "CallContext: invalid attr, is_null={}, attr_size={}, required={}",
                self.attr.is_null(),
                self.attr_size,
                size
            );
            return Err(BackendErr::InvalidAttr);
        }

        Ok(unsafe { core::ptr::read_unaligned(self.attr.cast::<T>()) })
    }
}

/// ABI raw mirror layout checks for backend public call/view structures.
#[allow(dead_code, missing_docs, clippy::missing_docs_in_private_items)]
mod abi_layout {
    use super::*;

    #[repr(C)]
    struct RawBackendTensorView {
        data: *mut u8,
        byte_len: u64,
        shape: [u32; MAX_DIM],
        stride_bytes: [u64; MAX_DIM],
        ndim: u32,
        dtype: u32,
        format: u32,
        layout: u32,
        flags: u8,
    }

    #[repr(C)]
    struct RawBackendCall {
        op: u8,
        target: u8,
        inputs: *const RawBackendTensorView,
        input_count: u32,
        outputs: *mut RawBackendTensorView,
        output_count: u32,
        attr: *const u8,
        attr_size: u32,
    }

    const _: () = assert!(
        core::mem::size_of::<BackendTensorView>() == core::mem::size_of::<RawBackendTensorView>()
    );
    const _: () = assert!(
        core::mem::align_of::<BackendTensorView>() == core::mem::align_of::<RawBackendTensorView>()
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendTensorView, byte_len)
            == core::mem::offset_of!(RawBackendTensorView, byte_len)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendTensorView, shape)
            == core::mem::offset_of!(RawBackendTensorView, shape)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendTensorView, stride_bytes)
            == core::mem::offset_of!(RawBackendTensorView, stride_bytes)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendTensorView, ndim)
            == core::mem::offset_of!(RawBackendTensorView, ndim)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendTensorView, flags)
            == core::mem::offset_of!(RawBackendTensorView, flags)
    );

    const _: () =
        assert!(core::mem::size_of::<BackendCall>() == core::mem::size_of::<RawBackendCall>());
    const _: () =
        assert!(core::mem::align_of::<BackendCall>() == core::mem::align_of::<RawBackendCall>());
    const _: () = assert!(
        core::mem::offset_of!(BackendCall, inputs) == core::mem::offset_of!(RawBackendCall, inputs)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendCall, input_count)
            == core::mem::offset_of!(RawBackendCall, input_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendCall, outputs)
            == core::mem::offset_of!(RawBackendCall, outputs)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendCall, output_count)
            == core::mem::offset_of!(RawBackendCall, output_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendCall, attr) == core::mem::offset_of!(RawBackendCall, attr)
    );
    const _: () = assert!(
        core::mem::offset_of!(BackendCall, attr_size)
            == core::mem::offset_of!(RawBackendCall, attr_size)
    );
}

/// 调用层（`CallContext`/`BackendTensorView`）的校验路径单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use k3_ai_uabi::MatMulAttr;

    /// 构造一个只设置 data/byte_len 的最小 tensor view。
    fn tensor_view(data: *mut u8, byte_len: u64) -> BackendTensorView {
        BackendTensorView {
            data,
            byte_len: ByteSize::new(byte_len),
            ..BackendTensorView::default()
        }
    }

    /// 返回一个非空但悬垂的指针，用于只走 count/attr 校验、不解引用数据的用例。
    fn dangling_data() -> *mut u8 {
        core::ptr::dangling_mut::<u8>()
    }

    /// 把一个 `Copy` attr 的原始字节暴露成切片。
    fn attr_bytes<T: Copy>(attr: &T) -> &[u8] {
        unsafe { slice::from_raw_parts((attr as *const T).cast::<u8>(), core::mem::size_of::<T>()) }
    }

    /// arity 不匹配时 `expect_io` 应返回 `InvalidInput`。
    #[test]
    fn call_context_rejects_invalid_arity() {
        let attr = MatMulAttr::default();
        let bytes = attr_bytes(&attr);
        let input = tensor_view(dangling_data(), 4);
        let mut output = tensor_view(dangling_data(), 4);
        let call = BackendCall {
            op: KernelOp::MAT_MUL,
            target: AiTargetHint::AUTO.0,
            inputs: &input,
            input_count: TensorCount::new(1),
            outputs: &mut output,
            output_count: TensorCount::new(1),
            attr: bytes.as_ptr(),
            attr_size: AttrByteSize::new(bytes.len() as u32),
        };

        let ctx = unsafe { CallContext::from_call(&call) }.unwrap();
        assert_eq!(ctx.expect_io(2, 1), Err(BackendErr::InvalidInput));
    }

    /// count 非零但数组指针为空时应返回 `NullPointer`。
    #[test]
    fn call_context_rejects_null_arrays_when_count_is_nonzero() {
        let call = BackendCall {
            op: KernelOp::MAT_MUL,
            target: AiTargetHint::AUTO.0,
            inputs: core::ptr::null(),
            input_count: TensorCount::new(1),
            outputs: core::ptr::null_mut(),
            output_count: TensorCount::new(0),
            attr: core::ptr::null(),
            attr_size: AttrByteSize::new(0),
        };

        assert!(matches!(
            unsafe { CallContext::from_call(&call) },
            Err(BackendErr::NullPointer)
        ));
    }

    /// data 为空指针时 typed slice 应返回 `InvalidTensor`。
    #[test]
    fn typed_slice_rejects_null_data_pointer() {
        let view = tensor_view(core::ptr::null_mut(), 4);

        assert_eq!(
            unsafe { view.as_slice::<u32>() },
            Err(BackendErr::InvalidTensor)
        );
    }

    /// attr 长度不足以容纳目标类型时应返回 `InvalidAttr`。
    #[test]
    fn read_attr_rejects_short_attr() {
        let input = tensor_view(dangling_data(), 4);
        let mut output = tensor_view(dangling_data(), 4);
        let one_byte = [0_u8; 1];
        let call = BackendCall {
            op: KernelOp::MAT_MUL,
            target: AiTargetHint::AUTO.0,
            inputs: &input,
            input_count: TensorCount::new(1),
            outputs: &mut output,
            output_count: TensorCount::new(1),
            attr: one_byte.as_ptr(),
            attr_size: AttrByteSize::new(one_byte.len() as u32),
        };

        let ctx = unsafe { CallContext::from_call(&call) }.unwrap();
        assert!(matches!(
            ctx.read_attr::<MatMulAttr>(),
            Err(BackendErr::InvalidAttr)
        ));
    }

    /// 未知 target 应返回 `UnsupportedOp`。
    #[test]
    fn call_context_rejects_unknown_target() {
        let call = BackendCall {
            op: KernelOp::MAT_MUL,
            target: 99,
            inputs: core::ptr::null(),
            input_count: TensorCount::new(0),
            outputs: core::ptr::null_mut(),
            output_count: TensorCount::new(0),
            attr: core::ptr::null(),
            attr_size: AttrByteSize::new(0),
        };

        assert!(matches!(
            unsafe { CallContext::from_call(&call) },
            Err(BackendErr::UnsupportedOp)
        ));
    }

    /// byte_len 不能被元素大小整除时应返回 `InvalidTensor`。
    #[test]
    fn typed_slice_rejects_non_divisible_byte_len() {
        let mut bytes = vec![0_u8; 5];
        let view = tensor_view(bytes.as_mut_ptr(), bytes.len() as u64);

        assert_eq!(
            unsafe { view.as_slice::<u32>() },
            Err(BackendErr::InvalidTensor)
        );
    }

    /// data 未按元素对齐时应返回 `InvalidTensor`。
    #[test]
    fn typed_slice_rejects_unaligned_buffer() {
        let mut bytes = vec![0_u8; 8];
        let view = tensor_view(unsafe { bytes.as_mut_ptr().add(1) }, 4);

        assert_eq!(
            unsafe { view.as_slice::<u32>() },
            Err(BackendErr::InvalidTensor)
        );
    }
}
