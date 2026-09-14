//! Tensor 搬运、索引与 shape 变换算子。

use crate::call::{CallContext, TensorMeta, normalize_axis};
use crate::rvv::{self, BinaryOp};
use crate::{BackendCall, ComputeKernel};
use alloc::vec;
use alloc::vec::Vec;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{
    AiDtype, AiTargetHint, CastAttr, ConcatAttr, CopyAttr, ExpandAttr, GatherAttr,
    GatherElementsAttr, KernelOp, MAX_DIM, Resize2dAttr, TileAttr, TransposeAttr,
};
use log::warn;

/// Concat 的静态分发标记。
pub(crate) struct ConcatKernel;
/// Transpose 的静态分发标记。
pub(crate) struct TransposeKernel;
/// Gather 的静态分发标记。
pub(crate) struct GatherKernel;
/// GatherElements 的静态分发标记。
pub(crate) struct GatherElementsKernel;
/// Copy 的静态分发标记。
pub(crate) struct CopyKernel;
/// Cast 的静态分发标记。
pub(crate) struct CastKernel;
/// Resize 的静态分发标记。
pub(crate) struct ResizeKernel;
/// Expand 的静态分发标记。
pub(crate) struct ExpandKernel;
/// Tile 的静态分发标记。
pub(crate) struct TileKernel;

macro_rules! impl_transform_kernel {
    ($marker:ident, $op:expr, $entry:ident) => {
        impl ComputeKernel for $marker {
            const OP: KernelOp = $op;

            unsafe fn call(call: *const BackendCall) -> Result<(), BackendErr> {
                unsafe { $entry(call) }
            }
        }
    };
}

impl_transform_kernel!(ConcatKernel, KernelOp::CONCAT, call_concat);
impl_transform_kernel!(TransposeKernel, KernelOp::TRANSPOSE, call_transpose);
impl_transform_kernel!(GatherKernel, KernelOp::GATHER, call_gather);
impl_transform_kernel!(
    GatherElementsKernel,
    KernelOp::GATHER_ELEMENTS,
    call_gather_elements
);
impl_transform_kernel!(CopyKernel, KernelOp::COPY, call_copy);
impl_transform_kernel!(CastKernel, KernelOp::CAST, call_cast);
impl_transform_kernel!(ResizeKernel, KernelOp::RESIZE, call_resize);
impl_transform_kernel!(ExpandKernel, KernelOp::EXPAND, call_expand);
impl_transform_kernel!(TileKernel, KernelOp::TILE, call_tile);

/// Concat 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_concat(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io_range(1..=7, 1..=1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<ConcatAttr>()?;
    if ctx.outputs[0].dtype != AiDtype::F16
        || ctx.inputs.iter().any(|view| view.dtype != AiDtype::F16)
    {
        return Err(BackendErr::UnsupportedDtype);
    }
    let output_meta = ctx.outputs[0].checked_meta()?;
    let axis = normalize_axis(attr.axis, output_meta.rank)?;
    let dtype = ctx.outputs[0].dtype;
    let element_size = output_meta.element_size;
    let mut metas = Vec::with_capacity(ctx.inputs.len());
    let mut axis_sum = 0_usize;
    for input in ctx.inputs {
        if input.dtype != dtype {
            return Err(BackendErr::UnsupportedDtype);
        }
        let meta = input.checked_meta()?;
        if meta.rank != output_meta.rank || meta.element_size != element_size {
            return Err(BackendErr::InvalidTensor);
        }
        for current_axis in 0..meta.rank {
            if current_axis != axis && meta.shape[current_axis] != output_meta.shape[current_axis] {
                return Err(BackendErr::InvalidTensor);
            }
        }
        axis_sum = axis_sum
            .checked_add(meta.shape[axis])
            .ok_or(BackendErr::InvalidTensor)?;
        metas.push(meta);
    }
    if axis_sum != output_meta.shape[axis] {
        return Err(BackendErr::InvalidTensor);
    }

    let target = ctx.target;
    let inner = product(&output_meta.shape[axis + 1..output_meta.rank])?;
    let outer = product(&output_meta.shape[..axis])?;

    // 正常 NCHW/NHWC tensor 不需要先按 stride 物化为 logical buffer。
    // 直接按 axis 两侧的连续 block 写入 output，避免逐元素 offset/gather。
    if output_meta.is_contiguous() && metas.iter().all(TensorMeta::is_contiguous) {
        let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
        let output_block_bytes = output_meta.shape[axis]
            .checked_mul(inner)
            .and_then(|elements| elements.checked_mul(element_size))
            .ok_or(BackendErr::InvalidTensor)?;

        for outer_index in 0..outer {
            let mut destination_start = outer_index
                .checked_mul(output_block_bytes)
                .ok_or(BackendErr::InvalidTensor)?;
            for (input, meta) in ctx.inputs.iter().zip(&metas) {
                let block_bytes = meta.shape[axis]
                    .checked_mul(inner)
                    .and_then(|elements| elements.checked_mul(element_size))
                    .ok_or(BackendErr::InvalidTensor)?;
                let source_start = outer_index
                    .checked_mul(block_bytes)
                    .ok_or(BackendErr::InvalidTensor)?;
                let source_end = source_start
                    .checked_add(block_bytes)
                    .ok_or(BackendErr::InvalidTensor)?;
                let destination_end = destination_start
                    .checked_add(block_bytes)
                    .ok_or(BackendErr::InvalidTensor)?;
                let source = unsafe { input.as_slice::<u8>()? }
                    .get(source_start..source_end)
                    .ok_or(BackendErr::InvalidTensor)?;
                let destination = output
                    .get_mut(destination_start..destination_end)
                    .ok_or(BackendErr::InvalidTensor)?;
                copy_for_target(target, source, destination)?;
                destination_start = destination_end;
            }
        }
        return Ok(());
    }

    let mut logical_output = vec![0_u8; output_meta.element_count * element_size];
    for outer_index in 0..outer {
        let mut destination_axis = 0_usize;
        for (input, meta) in ctx.inputs.iter().zip(&metas) {
            let raw = unsafe { input.as_slice::<u8>()? };
            let logical_input = read_logical_bytes(raw, meta, target)?;
            let block_elements = meta.shape[axis] * inner;
            let source_start = outer_index * block_elements * element_size;
            let destination_start = (outer_index * output_meta.shape[axis] * inner
                + destination_axis * inner)
                * element_size;
            copy_for_target(
                target,
                &logical_input[source_start..source_start + block_elements * element_size],
                &mut logical_output
                    [destination_start..destination_start + block_elements * element_size],
            )?;
            destination_axis += meta.shape[axis];
        }
    }
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_output, output, &output_meta, target)
}

/// Transpose 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_transpose(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<TransposeAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || input_meta.rank != output_meta.rank
        || attr.rank.get() as usize != input_meta.rank
    {
        return Err(BackendErr::InvalidTensor);
    }

    let mut seen = [false; MAX_DIM];
    let mut permutation = [0_usize; MAX_DIM];
    for (output_axis, slot) in permutation.iter_mut().enumerate().take(output_meta.rank) {
        let input_axis = normalize_axis(attr.perm[output_axis], input_meta.rank)?;
        if seen[input_axis] || output_meta.shape[output_axis] != input_meta.shape[input_axis] {
            return Err(BackendErr::InvalidAttr);
        }
        seen[input_axis] = true;
        *slot = input_axis;
    }

    let input_raw = unsafe { ctx.inputs[0].as_slice::<u8>()? };
    let logical_input = read_logical_bytes(input_raw, &input_meta, ctx.target)?;
    let mut source_offsets = vec![0_u64; output_meta.element_count];
    for (output_linear, source_offset) in source_offsets
        .iter_mut()
        .enumerate()
        .take(output_meta.element_count)
    {
        let mut output_coordinates = [0_usize; MAX_DIM];
        output_meta.coordinates(output_linear, &mut output_coordinates)?;
        let mut input_coordinates = [0_usize; MAX_DIM];
        for output_axis in 0..output_meta.rank {
            input_coordinates[permutation[output_axis]] = output_coordinates[output_axis];
        }
        let input_linear = row_major_linear(&input_meta, &input_coordinates)?;
        *source_offset = (input_linear * input_meta.element_size) as u64;
    }
    let logical_output = gather_logical(
        &logical_input,
        &source_offsets,
        output_meta.element_size,
        ctx.target,
    )?;
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_output, output, &output_meta, ctx.target)
}

/// Gather 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_gather(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<GatherAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let data_meta = ctx.inputs[0].checked_meta()?;
    let indices_meta = ctx.inputs[1].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype {
        return Err(BackendErr::UnsupportedDtype);
    }
    let axis = normalize_axis(attr.axis, data_meta.rank)?;
    let expected_rank = data_meta.rank - 1 + indices_meta.rank;
    if output_meta.rank != expected_rank {
        return Err(BackendErr::InvalidTensor);
    }
    for out_axis in 0..output_meta.rank {
        let expected = if out_axis < axis {
            data_meta.shape[out_axis]
        } else if out_axis < axis + indices_meta.rank {
            indices_meta.shape[out_axis - axis]
        } else {
            data_meta.shape[out_axis - indices_meta.rank + 1]
        };
        if output_meta.shape[out_axis] != expected {
            return Err(BackendErr::InvalidTensor);
        }
    }

    let indices = read_indices(&ctx.inputs[1], &indices_meta)?;
    let data_raw = unsafe { ctx.inputs[0].as_slice::<u8>()? };
    let logical_data = read_logical_bytes(data_raw, &data_meta, ctx.target)?;
    let mut offsets = vec![0_u64; output_meta.element_count];
    for (output_linear, offset) in offsets
        .iter_mut()
        .enumerate()
        .take(output_meta.element_count)
    {
        let mut output_coordinates = [0_usize; MAX_DIM];
        output_meta.coordinates(output_linear, &mut output_coordinates)?;
        let mut index_coordinates = [0_usize; MAX_DIM];
        index_coordinates[..indices_meta.rank]
            .copy_from_slice(&output_coordinates[axis..axis + indices_meta.rank]);
        let index_linear = row_major_linear(&indices_meta, &index_coordinates)?;
        let gathered = normalize_index(indices[index_linear], data_meta.shape[axis])?;
        let mut data_coordinates = [0_usize; MAX_DIM];
        data_coordinates[..axis].copy_from_slice(&output_coordinates[..axis]);
        data_coordinates[axis] = gathered;
        for data_axis in axis + 1..data_meta.rank {
            data_coordinates[data_axis] = output_coordinates[data_axis + indices_meta.rank - 1];
        }
        *offset =
            (row_major_linear(&data_meta, &data_coordinates)? * data_meta.element_size) as u64;
    }
    let logical_output = gather_logical(
        &logical_data,
        &offsets,
        output_meta.element_size,
        ctx.target,
    )?;
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_output, output, &output_meta, ctx.target)
}

/// GatherElements 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_gather_elements(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<GatherElementsAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let data_meta = ctx.inputs[0].checked_meta()?;
    let indices_meta = ctx.inputs[1].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || data_meta.rank != indices_meta.rank
        || output_meta.shape[..output_meta.rank] != indices_meta.shape[..indices_meta.rank]
    {
        return Err(BackendErr::InvalidTensor);
    }
    let axis = normalize_axis(attr.axis, data_meta.rank)?;
    for current_axis in 0..data_meta.rank {
        if current_axis != axis && indices_meta.shape[current_axis] > data_meta.shape[current_axis]
        {
            return Err(BackendErr::InvalidTensor);
        }
    }
    let indices = read_indices(&ctx.inputs[1], &indices_meta)?;
    let data_raw = unsafe { ctx.inputs[0].as_slice::<u8>()? };
    let logical_data = read_logical_bytes(data_raw, &data_meta, ctx.target)?;
    let mut offsets = vec![0_u64; output_meta.element_count];
    for linear in 0..output_meta.element_count {
        let mut coordinates = [0_usize; MAX_DIM];
        output_meta.coordinates(linear, &mut coordinates)?;
        coordinates[axis] = normalize_index(indices[linear], data_meta.shape[axis])?;
        offsets[linear] =
            (row_major_linear(&data_meta, &coordinates)? * data_meta.element_size) as u64;
    }
    let logical_output = gather_logical(
        &logical_data,
        &offsets,
        output_meta.element_size,
        ctx.target,
    )?;
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_output, output, &output_meta, ctx.target)
}

/// Materialize/copy one tensor into another.
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_copy(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    let attr = ctx.read_attr::<CopyAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    if attr.flags.get() != 0 {
        return Err(BackendErr::InvalidAttr);
    }
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || input_meta.element_count != output_meta.element_count
    {
        return Err(BackendErr::InvalidTensor);
    }
    if ctx.inputs[0].data == ctx.outputs[0].data {
        return Ok(());
    }
    let input = unsafe { ctx.inputs[0].as_slice::<u8>()? };
    let logical = read_logical_bytes(input, &input_meta, ctx.target)?;
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical, output, &output_meta, ctx.target)
}

/// Expand 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_expand(call: *const BackendCall) -> Result<(), BackendErr> {
    let mut ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<ExpandAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || attr.rank.get() as usize != output_meta.rank
        || input_meta.rank > output_meta.rank
    {
        return Err(BackendErr::InvalidTensor);
    }
    for axis in 0..output_meta.rank {
        if attr.target_shape[axis].get() as usize != output_meta.shape[axis] {
            return Err(BackendErr::InvalidAttr);
        }
        let input_dim = aligned_shape(&input_meta, output_meta.rank, axis);
        if input_dim != 1 && input_dim != output_meta.shape[axis] {
            return Err(BackendErr::InvalidTensor);
        }
    }
    remap_single_input(&mut ctx, &input_meta, &output_meta, |coordinates| {
        let mut input_coordinates = [0_usize; MAX_DIM];
        let leading = output_meta.rank - input_meta.rank;
        for (axis, coordinate) in input_coordinates
            .iter_mut()
            .enumerate()
            .take(input_meta.rank)
        {
            let output_axis = leading + axis;
            *coordinate = if input_meta.shape[axis] == 1 {
                0
            } else {
                coordinates[output_axis]
            };
        }
        row_major_linear(&input_meta, &input_coordinates)
    })
}

/// Tile 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_tile(call: *const BackendCall) -> Result<(), BackendErr> {
    let mut ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<TileAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || attr.rank.get() as usize != input_meta.rank
        || output_meta.rank != input_meta.rank
    {
        return Err(BackendErr::InvalidTensor);
    }
    for axis in 0..input_meta.rank {
        let repeat = attr.repeats[axis].get() as usize;
        if repeat == 0
            || input_meta.shape[axis]
                .checked_mul(repeat)
                .is_none_or(|expected| expected != output_meta.shape[axis])
        {
            return Err(BackendErr::InvalidAttr);
        }
    }
    remap_single_input(&mut ctx, &input_meta, &output_meta, |coordinates| {
        let mut input_coordinates = [0_usize; MAX_DIM];
        for axis in 0..input_meta.rank {
            input_coordinates[axis] = coordinates[axis] % input_meta.shape[axis];
        }
        row_major_linear(&input_meta, &input_coordinates)
    })
}

/// Cast 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_cast(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<CastAttr>()?;
    if !matches!(
        ctx.inputs[0].dtype,
        AiDtype::F16 | AiDtype::I32 | AiDtype::I64
    ) || !matches!(
        ctx.outputs[0].dtype,
        AiDtype::F16 | AiDtype::I32 | AiDtype::I64
    ) {
        return Err(BackendErr::UnsupportedDtype);
    }
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if attr.to != ctx.outputs[0].dtype
        || input_meta.rank != output_meta.rank
        || input_meta.shape[..input_meta.rank] != output_meta.shape[..output_meta.rank]
    {
        return Err(BackendErr::InvalidTensor);
    }
    let input_raw = unsafe { ctx.inputs[0].as_slice::<u8>()? };
    let logical_input = read_logical_bytes(input_raw, &input_meta, ctx.target)?;
    let logical_output = cast_logical(
        &logical_input,
        ctx.inputs[0].dtype,
        ctx.outputs[0].dtype,
        input_meta.element_count,
        ctx.target,
    )?;
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_output, output, &output_meta, ctx.target)
}

/// Resize 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
unsafe fn call_resize(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<Resize2dAttr>()?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if input_meta.rank != 4
        || output_meta.rank != 4
        || ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || input_meta.shape[0] != output_meta.shape[0]
        || input_meta.shape[1] != output_meta.shape[1]
        || input_meta.shape[2] != attr.input_h.get() as usize
        || input_meta.shape[3] != attr.input_w.get() as usize
        || output_meta.shape[2] != attr.output_h.get() as usize
        || output_meta.shape[3] != attr.output_w.get() as usize
    {
        return Err(BackendErr::InvalidTensor);
    }
    if attr.mode > Resize2dAttr::MODE_LINEAR
        || attr.coordinate_mode > Resize2dAttr::COORD_ALIGN_CORNERS
        || attr.nearest_mode > Resize2dAttr::NEAREST_CEIL
    {
        return Err(BackendErr::InvalidAttr);
    }

    let logical_input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let logical_output =
        compute_f16_resize(&logical_input, &input_meta, &output_meta, &attr, ctx.target)?;
    write_float_tensor(
        &logical_output,
        &mut ctx.outputs[0],
        &output_meta,
        ctx.target,
    )
}

/// 把单输入 remap 算子的逻辑映射收敛为一次 indexed gather。
fn remap_single_input<F>(
    ctx: &mut CallContext<'_>,
    input_meta: &TensorMeta,
    output_meta: &TensorMeta,
    mut source_linear: F,
) -> Result<(), BackendErr>
where
    F: FnMut(&[usize; MAX_DIM]) -> Result<usize, BackendErr>,
{
    let raw = unsafe { ctx.inputs[0].as_slice::<u8>()? };
    let logical_input = read_logical_bytes(raw, input_meta, ctx.target)?;
    let mut offsets = vec![0_u64; output_meta.element_count];
    for (linear, offset) in offsets.iter_mut().enumerate() {
        let mut coordinates = [0_usize; MAX_DIM];
        output_meta.coordinates(linear, &mut coordinates)?;
        *offset = (source_linear(&coordinates)? * input_meta.element_size) as u64;
    }
    let logical_output = gather_logical(
        &logical_input,
        &offsets,
        output_meta.element_size,
        ctx.target,
    )?;
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_output, output, output_meta, ctx.target)
}

/// 读取 tensor 的逻辑 row-major 字节序列。
pub(crate) fn read_logical_bytes(
    raw: &[u8],
    meta: &TensorMeta,
    target: AiTargetHint,
) -> Result<Vec<u8>, BackendErr> {
    let logical_len = meta
        .element_count
        .checked_mul(meta.element_size)
        .ok_or(BackendErr::InvalidTensor)?;
    // 本来就连续的话就直接返回
    if meta.is_contiguous() {
        return raw
            .get(..logical_len)
            .ok_or(BackendErr::InvalidTensor)
            .map(<[u8]>::to_vec);
    }
    warn!("Use cpu slow path,one by one element");
    let mut offsets = vec![0_u64; meta.element_count];
    for (linear, offset) in offsets.iter_mut().enumerate() {
        *offset = (meta.offset_for_linear(linear)? * meta.element_size) as u64;
    }
    gather_logical(raw, &offsets, meta.element_size, target)
}

/// 从 byte offsets gather 成连续逻辑 tensor。
fn gather_logical(
    source: &[u8],
    offsets: &[u64],
    element_size: usize,
    target: AiTargetHint,
) -> Result<Vec<u8>, BackendErr> {
    let mut output = vec![0_u8; offsets.len() * element_size];
    if vector_target(target) {
        rvv::gather_bytes(source, offsets, &mut output, element_size)?;
    } else {
        for (linear, &offset) in offsets.iter().enumerate() {
            let source_offset = usize::try_from(offset).map_err(|_| BackendErr::InvalidTensor)?;
            let destination_offset = linear * element_size;
            output[destination_offset..destination_offset + element_size]
                .copy_from_slice(&source[source_offset..source_offset + element_size]);
        }
    }
    Ok(output)
}

/// 把连续逻辑字节序列写入可能带 stride 的 output。
pub(crate) fn write_logical_bytes(
    logical: &[u8],
    output: &mut [u8],
    meta: &TensorMeta,
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    if logical.len() != meta.element_count * meta.element_size {
        return Err(BackendErr::InvalidTensor);
    }
    if meta.is_contiguous() {
        return copy_for_target(target, logical, &mut output[..logical.len()]);
    }
    for linear in 0..meta.element_count {
        let source_offset = linear * meta.element_size;
        let destination_offset = meta.offset_for_linear(linear)? * meta.element_size;
        output[destination_offset..destination_offset + meta.element_size]
            .copy_from_slice(&logical[source_offset..source_offset + meta.element_size]);
    }
    Ok(())
}

/// 按 target 选择真实 RVV copy 或 CPU copy。
fn copy_for_target(
    target: AiTargetHint,
    source: &[u8],
    output: &mut [u8],
) -> Result<(), BackendErr> {
    if vector_target(target) {
        rvv::copy_bytes(source, output)
    } else if source.len() == output.len() {
        output.copy_from_slice(source);
        Ok(())
    } else {
        Err(BackendErr::InvalidTensor)
    }
}

/// 判断 target 是否要求 RVV。
pub(crate) fn vector_target(target: AiTargetHint) -> bool {
    matches!(
        target,
        AiTargetHint::AUTO | AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100
    )
}

/// 读取 I32/I64 indices tensor。
pub(crate) fn read_indices(
    view: &crate::BackendTensorView,
    meta: &TensorMeta,
) -> Result<Vec<i64>, BackendErr> {
    match view.dtype {
        AiDtype::I32 => {
            let values = unsafe { view.as_slice::<i32>()? };
            (0..meta.element_count)
                .map(|linear| Ok(values[meta.offset_for_linear(linear)?] as i64))
                .collect()
        }
        AiDtype::I64 => {
            let values = unsafe { view.as_slice::<i64>()? };
            (0..meta.element_count)
                .map(|linear| Ok(values[meta.offset_for_linear(linear)?]))
                .collect()
        }
        _ => Err(BackendErr::UnsupportedDtype),
    }
}

/// 归一化支持负数的 ONNX index。
fn normalize_index(index: i64, dimension: usize) -> Result<usize, BackendErr> {
    let dimension_i64 = i64::try_from(dimension).map_err(|_| BackendErr::InvalidTensor)?;
    let normalized = if index < 0 {
        index + dimension_i64
    } else {
        index
    };
    if normalized < 0 || normalized >= dimension_i64 {
        return Err(BackendErr::InvalidInput);
    }
    usize::try_from(normalized).map_err(|_| BackendErr::InvalidInput)
}

/// 计算 row-major 逻辑坐标的线性下标。
pub(crate) fn row_major_linear(
    meta: &TensorMeta,
    coordinates: &[usize; MAX_DIM],
) -> Result<usize, BackendErr> {
    let mut linear = 0_usize;
    for (axis, &coordinate) in coordinates.iter().enumerate().take(meta.rank) {
        if coordinate >= meta.shape[axis] {
            return Err(BackendErr::InvalidTensor);
        }
        linear = linear
            .checked_mul(meta.shape[axis])
            .and_then(|value| value.checked_add(coordinate))
            .ok_or(BackendErr::InvalidTensor)?;
    }
    Ok(linear)
}

/// 计算 shape 乘积。
fn product(shape: &[usize]) -> Result<usize, BackendErr> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(BackendErr::InvalidTensor)
    })
}

/// 返回右对齐 shape 的指定维大小。
fn aligned_shape(meta: &TensorMeta, output_rank: usize, output_axis: usize) -> usize {
    let leading = output_rank - meta.rank;
    if output_axis < leading {
        1
    } else {
        meta.shape[output_axis - leading]
    }
}

/// Cast 连续逻辑字节。
fn cast_logical(
    input: &[u8],
    input_dtype: AiDtype,
    output_dtype: AiDtype,
    count: usize,
    _target: AiTargetHint,
) -> Result<Vec<u8>, BackendErr> {
    if input_dtype == output_dtype {
        return Ok(input.to_vec());
    }
    let mut output = vec![0_u8; count * dtype_size(output_dtype)?];
    for index in 0..count {
        let value = read_scalar(input, input_dtype, index)?;
        write_scalar(&mut output, output_dtype, index, value)?;
    }
    Ok(output)
}

/// Cast 内部统一标量表示。
#[derive(Clone, Copy)]
enum ScalarValue {
    /// 浮点值。
    Float(f32),
    /// 有符号整数值。
    Signed(i64),
}

/// 从逻辑字节读取一个标量。
fn read_scalar(bytes: &[u8], dtype: AiDtype, index: usize) -> Result<ScalarValue, BackendErr> {
    let offset = index * dtype_size(dtype)?;
    Ok(match dtype {
        AiDtype::F16 => ScalarValue::Float(
            f16::from_bits(u16::from_ne_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32(),
        ),
        AiDtype::I32 => ScalarValue::Signed(i32::from_ne_bytes(
            bytes[offset..offset + 4].try_into().unwrap(),
        ) as i64),
        AiDtype::I64 => ScalarValue::Signed(i64::from_ne_bytes(
            bytes[offset..offset + 8].try_into().unwrap(),
        )),
        _ => return Err(BackendErr::UnsupportedDtype),
    })
}

/// 把统一标量写成目标 dtype。
fn write_scalar(
    bytes: &mut [u8],
    dtype: AiDtype,
    index: usize,
    value: ScalarValue,
) -> Result<(), BackendErr> {
    let offset = index * dtype_size(dtype)?;
    let as_f32 = match value {
        ScalarValue::Float(value) => value,
        ScalarValue::Signed(value) => value as f32,
    };
    let as_i64 = match value {
        ScalarValue::Float(value) => value as i64,
        ScalarValue::Signed(value) => value,
    };
    match dtype {
        AiDtype::F16 => bytes[offset..offset + 2]
            .copy_from_slice(&f16::from_f32(as_f32).to_bits().to_ne_bytes()),
        AiDtype::I32 => bytes[offset..offset + 4].copy_from_slice(&(as_i64 as i32).to_ne_bytes()),
        AiDtype::I64 => bytes[offset..offset + 8].copy_from_slice(&as_i64.to_ne_bytes()),
        _ => return Err(BackendErr::UnsupportedDtype),
    }
    Ok(())
}

/// 返回本轮 Cast 支持的固定宽度。
fn dtype_size(dtype: AiDtype) -> Result<usize, BackendErr> {
    dtype
        .element_size_bytes()
        .map(|size| size as usize)
        .ok_or(BackendErr::UnsupportedDtype)
}

/// 从 native-endian 字节构造 F32 Vec。
fn bytes_as_f32(bytes: &[u8]) -> Result<Vec<f32>, BackendErr> {
    if !bytes.len().is_multiple_of(4) {
        return Err(BackendErr::InvalidTensor);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect())
}

/// 把 F32 slice 转成 native-endian 字节。
fn f32_as_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// 把 F16 tensor 解码成连续 F32 工作缓冲。
pub(crate) fn read_float_tensor(
    view: &crate::BackendTensorView,
    meta: &TensorMeta,
    target: AiTargetHint,
) -> Result<Vec<f32>, BackendErr> {
    let raw = unsafe { view.as_slice::<u8>()? };
    let logical = read_logical_bytes(raw, meta, target)?;
    match view.dtype {
        AiDtype::F16 => Ok(logical
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_ne_bytes(chunk.try_into().unwrap())).to_f32())
            .collect()),
        _ => Err(BackendErr::UnsupportedDtype),
    }
}

/// 把连续 F32 工作缓冲量化写入 F16 tensor。
pub(crate) fn write_float_tensor(
    values: &[f32],
    view: &mut crate::BackendTensorView,
    meta: &TensorMeta,
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    let logical: Vec<u8> = match view.dtype {
        AiDtype::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_ne_bytes())
            .collect(),
        _ => return Err(BackendErr::UnsupportedDtype),
    };
    let raw = unsafe { view.as_mut_slice::<u8>()? };
    write_logical_bytes(&logical, raw, meta, target)
}

/// 执行 NCHW nearest/linear resize。
fn compute_f16_resize(
    input: &[f32],
    input_meta: &TensorMeta,
    output_meta: &TensorMeta,
    attr: &Resize2dAttr,
    target: AiTargetHint,
) -> Result<Vec<f32>, BackendErr> {
    let n = input_meta.shape[0];
    let c = input_meta.shape[1];
    let ih = input_meta.shape[2];
    let iw = input_meta.shape[3];
    let oh = output_meta.shape[2];
    let ow = output_meta.shape[3];
    if attr.mode == Resize2dAttr::MODE_NEAREST {
        let mut offsets = Vec::with_capacity(output_meta.element_count);
        for batch in 0..n {
            for channel in 0..c {
                for oy in 0..oh {
                    let source_y =
                        nearest_coordinate(source_coordinate(oy, ih, oh, attr), ih, attr)?;
                    for ox in 0..ow {
                        let source_x =
                            nearest_coordinate(source_coordinate(ox, iw, ow, attr), iw, attr)?;
                        offsets.push(
                            (((batch * c + channel) * ih + source_y) * iw + source_x) as u64 * 4,
                        );
                    }
                }
            }
        }
        let bytes = gather_logical(&f32_as_bytes(input), &offsets, 4, target)?;
        return bytes_as_f32(&bytes);
    }

    let count = output_meta.element_count;
    let mut p00 = vec![0.0_f32; count];
    let mut p01 = vec![0.0_f32; count];
    let mut p10 = vec![0.0_f32; count];
    let mut p11 = vec![0.0_f32; count];
    let mut wx = vec![0.0_f32; count];
    let mut wy = vec![0.0_f32; count];
    let mut index = 0;
    for batch in 0..n {
        for channel in 0..c {
            for oy in 0..oh {
                let sy = source_coordinate(oy, ih, oh, attr).clamp(0.0, (ih - 1) as f32);
                let y0 = libm::floorf(sy) as usize;
                let y1 = (y0 + 1).min(ih - 1);
                for ox in 0..ow {
                    let sx = source_coordinate(ox, iw, ow, attr).clamp(0.0, (iw - 1) as f32);
                    let x0 = libm::floorf(sx) as usize;
                    let x1 = (x0 + 1).min(iw - 1);
                    p00[index] = input[((batch * c + channel) * ih + y0) * iw + x0];
                    p01[index] = input[((batch * c + channel) * ih + y0) * iw + x1];
                    p10[index] = input[((batch * c + channel) * ih + y1) * iw + x0];
                    p11[index] = input[((batch * c + channel) * ih + y1) * iw + x1];
                    wx[index] = sx - x0 as f32;
                    wy[index] = sy - y0 as f32;
                    index += 1;
                }
            }
        }
    }
    if vector_target(target) {
        let ones = vec![1.0_f32; count];
        let mut one_minus_x = vec![0.0_f32; count];
        let mut one_minus_y = vec![0.0_f32; count];
        let mut a = vec![0.0_f32; count];
        let mut b = vec![0.0_f32; count];
        let mut top = vec![0.0_f32; count];
        let mut bottom = vec![0.0_f32; count];
        let mut output = vec![0.0_f32; count];
        rvv::f16_binary_work(BinaryOp::Sub, &ones, &wx, &mut one_minus_x)?;
        rvv::f16_binary_work(BinaryOp::Sub, &ones, &wy, &mut one_minus_y)?;
        rvv::f16_binary_work(BinaryOp::Mul, &p00, &one_minus_x, &mut a)?;
        rvv::f16_binary_work(BinaryOp::Mul, &p01, &wx, &mut b)?;
        rvv::f16_binary_work(BinaryOp::Add, &a, &b, &mut top)?;
        rvv::f16_binary_work(BinaryOp::Mul, &p10, &one_minus_x, &mut a)?;
        rvv::f16_binary_work(BinaryOp::Mul, &p11, &wx, &mut b)?;
        rvv::f16_binary_work(BinaryOp::Add, &a, &b, &mut bottom)?;
        rvv::f16_binary_work(BinaryOp::Mul, &top, &one_minus_y, &mut a)?;
        rvv::f16_binary_work(BinaryOp::Mul, &bottom, &wy, &mut b)?;
        rvv::f16_binary_work(BinaryOp::Add, &a, &b, &mut output)?;
        Ok(output)
    } else {
        Ok((0..count)
            .map(|idx| {
                let top = p00[idx] * (1.0 - wx[idx]) + p01[idx] * wx[idx];
                let bottom = p10[idx] * (1.0 - wx[idx]) + p11[idx] * wx[idx];
                top * (1.0 - wy[idx]) + bottom * wy[idx]
            })
            .collect())
    }
}

/// ONNX resize 输出坐标映射到输入浮点坐标。
fn source_coordinate(index: usize, input: usize, output: usize, attr: &Resize2dAttr) -> f32 {
    match attr.coordinate_mode {
        Resize2dAttr::COORD_HALF_PIXEL => (index as f32 + 0.5) * input as f32 / output as f32 - 0.5,
        Resize2dAttr::COORD_ASYMMETRIC => index as f32 * input as f32 / output as f32,
        Resize2dAttr::COORD_ALIGN_CORNERS if output > 1 => {
            index as f32 * (input - 1) as f32 / (output - 1) as f32
        }
        Resize2dAttr::COORD_ALIGN_CORNERS => 0.0,
        _ => 0.0,
    }
}

/// 按 nearest_mode 选择并 clamp 输入坐标。
fn nearest_coordinate(
    coordinate: f32,
    dimension: usize,
    attr: &Resize2dAttr,
) -> Result<usize, BackendErr> {
    let rounded = match attr.nearest_mode {
        Resize2dAttr::NEAREST_ROUND_PREFER_FLOOR => libm::ceilf(coordinate - 0.5),
        Resize2dAttr::NEAREST_FLOOR => libm::floorf(coordinate),
        Resize2dAttr::NEAREST_CEIL => libm::ceilf(coordinate),
        _ => return Err(BackendErr::InvalidAttr),
    };
    Ok(rounded.clamp(0.0, (dimension - 1) as f32) as usize)
}

/// Transform 内部算法测试。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendTensorView;
    use k3_ai_uabi::{
        AiTensorLayout, AttrByteSize, ByteSize, ByteStride, DimCount, DimSize, TensorCount,
    };

    /// 构造 contiguous meta。
    fn meta(shape: &[usize], element_size: usize) -> TensorMeta {
        let mut full_shape = [0_usize; MAX_DIM];
        let mut strides = [0_usize; MAX_DIM];
        let mut count = 1;
        for axis in (0..shape.len()).rev() {
            full_shape[axis] = shape[axis];
            strides[axis] = count;
            count *= shape[axis];
        }
        TensorMeta {
            rank: shape.len(),
            shape: full_shape,
            strides,
            element_size,
            element_count: count,
        }
    }

    /// 以连续 f16 slice 构造后端 tensor view。
    fn contiguous_f16_view(data: *mut u16, len: usize, shape: &[usize]) -> BackendTensorView {
        let mut view_shape = [DimSize::new(0); MAX_DIM];
        let mut stride_bytes = [ByteStride::new(0); MAX_DIM];
        let mut stride = core::mem::size_of::<u16>() as u64;
        for axis in (0..shape.len()).rev() {
            view_shape[axis] = DimSize::new(shape[axis] as u32);
            stride_bytes[axis] = ByteStride::new(stride);
            stride *= shape[axis] as u64;
        }
        BackendTensorView {
            data: data.cast::<u8>(),
            byte_len: ByteSize::new((len * core::mem::size_of::<u16>()) as u64),
            shape: view_shape,
            stride_bytes,
            ndim: DimCount::new(shape.len() as u32),
            dtype: AiDtype::F16,
            layout: AiTensorLayout::DENSE,
            ..BackendTensorView::default()
        }
    }

    /// 把 ABI attr 暴露成只读字节。
    fn attr_bytes<T: Copy>(attr: &T) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts((attr as *const T).cast::<u8>(), core::mem::size_of::<T>())
        }
    }

    /// indexed gather 应保持 offset 顺序。
    #[test]
    fn gather_logical_reorders_elements() {
        let source = [1_u8, 2, 3, 4];
        let output = gather_logical(&source, &[3, 1, 0], 1, AiTargetHint::PREFER_A100).unwrap();
        assert_eq!(output, [4, 2, 1]);
    }

    /// Gather 索引只接受 I32/I64，且两种 ABI 表示应得到相同逻辑索引。
    #[test]
    fn gather_indices_accept_i32_and_i64() {
        let i32_values = [-1_i32, 2];
        let i64_values = [-1_i64, 2];
        let i32_view = BackendTensorView {
            data: i32_values.as_ptr().cast_mut().cast(),
            byte_len: ByteSize::new(core::mem::size_of_val(&i32_values) as u64),
            dtype: AiDtype::I32,
            ..BackendTensorView::default()
        };
        let i64_view = BackendTensorView {
            data: i64_values.as_ptr().cast_mut().cast(),
            byte_len: ByteSize::new(core::mem::size_of_val(&i64_values) as u64),
            dtype: AiDtype::I64,
            ..BackendTensorView::default()
        };
        assert_eq!(read_indices(&i32_view, &meta(&[2], 4)).unwrap(), [-1, 2]);
        assert_eq!(read_indices(&i64_view, &meta(&[2], 8)).unwrap(), [-1, 2]);
    }

    /// 连续 concat 应直接按每个 batch 的连续 block 复制。
    #[test]
    fn concat_contiguous_copies_axis_blocks() {
        let c = [1.0_f32, 2.0, 3.0, 4.0].map(|value| f16::from_f32(value).to_bits());
        let b = [10.0_f32, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]
            .map(|value| f16::from_f32(value).to_bits());
        let mut y = [0_u16; 12];
        let inputs = [
            contiguous_f16_view(c.as_ptr() as *mut u16, c.len(), &[2, 1, 2]),
            contiguous_f16_view(b.as_ptr() as *mut u16, b.len(), &[2, 2, 2]),
        ];
        let mut outputs = [contiguous_f16_view(y.as_mut_ptr(), y.len(), &[2, 3, 2])];
        let attr = ConcatAttr {
            axis: k3_ai_uabi::TensorAxis::new(1),
            ..ConcatAttr::default()
        };
        let attr = attr_bytes(&attr);
        let call = BackendCall {
            op: k3_ai_uabi::KernelOp::CONCAT,
            target: AiTargetHint::PREFER_CPU.0,
            inputs: inputs.as_ptr(),
            input_count: TensorCount::new(inputs.len() as u32),
            outputs: outputs.as_mut_ptr(),
            output_count: TensorCount::new(outputs.len() as u32),
            attr: attr.as_ptr(),
            attr_size: AttrByteSize::new(attr.len() as u32),
        };

        unsafe { call_concat(&call) }.unwrap();
        assert_eq!(
            y,
            [
                1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 3.0, 4.0, 20.0, 21.0, 22.0, 23.0
            ]
            .map(|value| f16::from_f32(value).to_bits())
        );
    }

    /// Cast 仅保留 F16 和索引整数间的必要转换。
    #[test]
    fn cast_common_numeric_types() {
        let input = [
            f16::from_f32(1.5).to_bits().to_ne_bytes(),
            f16::from_f32(-2.25).to_bits().to_ne_bytes(),
        ]
        .concat();
        let integers = cast_logical(
            &input,
            AiDtype::F16,
            AiDtype::I32,
            2,
            AiTargetHint::PREFER_A100,
        )
        .unwrap();
        assert_eq!(
            integers
                .chunks_exact(4)
                .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
            [1, -2]
        );
    }

    /// Linear resize 2x2 -> 3x3 的四角应保持不变。
    #[test]
    fn linear_resize_preserves_corners() {
        let input_meta = meta(&[1, 1, 2, 2], 4);
        let output_meta = meta(&[1, 1, 3, 3], 4);
        let attr = Resize2dAttr {
            mode: Resize2dAttr::MODE_LINEAR,
            coordinate_mode: Resize2dAttr::COORD_ALIGN_CORNERS,
            input_h: k3_ai_uabi::DimSize::new(2),
            input_w: k3_ai_uabi::DimSize::new(2),
            output_h: k3_ai_uabi::DimSize::new(3),
            output_w: k3_ai_uabi::DimSize::new(3),
            ..Resize2dAttr::default()
        };
        let output = compute_f16_resize(
            &[1.0, 2.0, 3.0, 4.0],
            &input_meta,
            &output_meta,
            &attr,
            AiTargetHint::PREFER_A100,
        )
        .unwrap();
        assert_eq!(output[0], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[6], 3.0);
        assert_eq!(output[8], 4.0);
    }
}
