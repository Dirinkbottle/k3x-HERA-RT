//! 归约、池化与模型常用 NN 算子。

use crate::BackendCall;
use crate::call::{CallContext, TensorMeta, normalize_axis};
use crate::rvv::{self, BinaryOp};
use crate::transform::{
    read_float_tensor, read_indices, row_major_linear, vector_target, write_float_tensor,
    write_logical_bytes,
};
use alloc::vec;
use alloc::vec::Vec;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{
    AiDtype, MAX_DIM, Pool2dAttr, ReduceMaxAttr, RmsNormAttr, RopeAttr, SoftmaxAttr, TopKAttr,
};

/// Softmax 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
pub(crate) unsafe fn softmax_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<SoftmaxAttr>()?;
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    validate_float_same_shape(&ctx, &input_meta, &output_meta)?;
    let axis = normalize_axis(attr.axis, input_meta.rank)?;
    let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let output = softmax_f32(&input, &input_meta, axis, attr.scale, ctx.target)?;
    write_float_tensor(&output, &mut ctx.outputs[0], &output_meta, ctx.target)
}

/// RMSNorm 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
pub(crate) unsafe fn rms_norm_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<RmsNormAttr>()?;
    let input_meta = ctx.inputs[0].checked_meta()?;
    let weight_meta = ctx.inputs[1].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    validate_float_same_shape(&ctx, &input_meta, &output_meta)?;
    let hidden = attr.hidden_size.get() as usize;
    if hidden == 0
        || input_meta.rank == 0
        || input_meta.shape[input_meta.rank - 1] != hidden
        || weight_meta.element_count != hidden
        || ctx.inputs[1].dtype != ctx.inputs[0].dtype
        || attr.eps <= 0.0
    {
        return Err(BackendErr::InvalidAttr);
    }
    let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let weight = read_float_tensor(&ctx.inputs[1], &weight_meta, ctx.target)?;
    let output = rms_norm_f32(&input, &weight, hidden, attr.eps, ctx.target)?;
    write_float_tensor(&output, &mut ctx.outputs[0], &output_meta, ctx.target)
}

/// RoPE 调用入口，tensor shape 为 dense/strided BSHD。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
pub(crate) unsafe fn rope_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io_range(1..=2, 1..=1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<RopeAttr>()?;
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    validate_float_same_shape(&ctx, &input_meta, &output_meta)?;
    if input_meta.rank != 4 {
        return Err(BackendErr::InvalidTensor);
    }
    let batch = input_meta.shape[0];
    let sequence = input_meta.shape[1];
    let heads = input_meta.shape[2];
    let head_dim = input_meta.shape[3];
    let n_dims = attr.n_dims.get() as usize;
    if n_dims == 0
        || n_dims > head_dim
        || !n_dims.is_multiple_of(2)
        || (attr.head_count.get() != 0 && attr.head_count.get() as usize != heads)
        || !matches!(attr.mode, RopeAttr::MODE_GPT_J | RopeAttr::MODE_NEOX)
    {
        return Err(BackendErr::InvalidAttr);
    }

    let positions = if ctx.inputs.len() == 2 {
        let meta = ctx.inputs[1].checked_meta()?;
        let values = read_indices(&ctx.inputs[1], &meta)?;
        if values.len() != sequence && values.len() != batch * sequence {
            return Err(BackendErr::InvalidTensor);
        }
        Some(values)
    } else {
        None
    };
    let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let output = rope_f32(
        &input,
        batch,
        sequence,
        heads,
        head_dim,
        n_dims,
        &attr,
        positions.as_deref(),
        ctx.target,
    )?;
    write_float_tensor(&output, &mut ctx.outputs[0], &output_meta, ctx.target)
}

/// MaxPool 调用入口，可选第二个 I64 indices 输出。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
pub(crate) unsafe fn max_pool_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io_range(1..=1, 1..=2)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<Pool2dAttr>()?;
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if input_meta.rank != 4
        || output_meta.rank != 4
        || ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || !matches!(ctx.inputs[0].dtype, AiDtype::F32 | AiDtype::F16)
        || input_meta.shape[0] != output_meta.shape[0]
        || input_meta.shape[1] != output_meta.shape[1]
    {
        return Err(BackendErr::InvalidTensor);
    }
    let kernel_h = attr.kernel_h.get() as usize;
    let kernel_w = attr.kernel_w.get() as usize;
    let stride_h = (attr.stride_h.get() as usize).max(1);
    let stride_w = (attr.stride_w.get() as usize).max(1);
    let dilation_h = (attr.dilation_h.get() as usize).max(1);
    let dilation_w = (attr.dilation_w.get() as usize).max(1);
    if kernel_h == 0 || kernel_w == 0 {
        return Err(BackendErr::InvalidAttr);
    }
    let expected_h = pool_output_dim(
        input_meta.shape[2],
        kernel_h,
        stride_h,
        dilation_h,
        attr.pad_top.get() as usize,
        attr.pad_bottom.get() as usize,
        attr.flags.get() & Pool2dAttr::CEIL_MODE != 0,
    )?;
    let expected_w = pool_output_dim(
        input_meta.shape[3],
        kernel_w,
        stride_w,
        dilation_w,
        attr.pad_left.get() as usize,
        attr.pad_right.get() as usize,
        attr.flags.get() & Pool2dAttr::CEIL_MODE != 0,
    )?;
    if output_meta.shape[2] != expected_h || output_meta.shape[3] != expected_w {
        return Err(BackendErr::InvalidTensor);
    }
    let indices_meta = if ctx.outputs.len() == 2 {
        let meta = ctx.outputs[1].checked_meta()?;
        if ctx.outputs[1].dtype != AiDtype::I64
            || meta.shape[..meta.rank] != output_meta.shape[..output_meta.rank]
        {
            return Err(BackendErr::InvalidTensor);
        }
        Some(meta)
    } else {
        None
    };
    let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let (values, indices) = max_pool_f32(&input, &input_meta, &output_meta, &attr, ctx.target)?;
    write_float_tensor(&values, &mut ctx.outputs[0], &output_meta, ctx.target)?;
    if let Some(meta) = indices_meta {
        let logical = i64_bytes(&indices);
        let output = unsafe { ctx.outputs[1].as_mut_slice::<u8>()? };
        write_logical_bytes(&logical, output, &meta, ctx.target)?;
    }
    Ok(())
}

/// ReduceMax 调用入口。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
pub(crate) unsafe fn reduce_max_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<ReduceMaxAttr>()?;
    let input_meta = ctx.inputs[0].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || !matches!(ctx.inputs[0].dtype, AiDtype::F32 | AiDtype::F16)
    {
        return Err(BackendErr::UnsupportedDtype);
    }
    let axis_count = attr.axis_count.get() as usize;
    if axis_count > input_meta.rank {
        return Err(BackendErr::InvalidAttr);
    }
    if axis_count == 0 && attr.flags.get() & ReduceMaxAttr::NOOP_WITH_EMPTY_AXES != 0 {
        if input_meta.shape[..input_meta.rank] != output_meta.shape[..output_meta.rank] {
            return Err(BackendErr::InvalidTensor);
        }
        let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
        return write_float_tensor(&input, &mut ctx.outputs[0], &output_meta, ctx.target);
    }
    let mut reduced = [axis_count == 0; MAX_DIM];
    for axis_index in 0..axis_count {
        let axis = normalize_axis(attr.axes[axis_index], input_meta.rank)?;
        if reduced[axis] {
            return Err(BackendErr::InvalidAttr);
        }
        reduced[axis] = true;
    }
    validate_reduce_output(&input_meta, &output_meta, &reduced, &attr)?;
    let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let output = reduce_max_f32(&input, &input_meta, &output_meta, &reduced, ctx.target)?;
    write_float_tensor(&output, &mut ctx.outputs[0], &output_meta, ctx.target)
}

/// TopK 调用入口，按计划保留 CPU 实现。
///
/// # Safety
///
/// `call` 及 tensor buffer 必须满足 backend ABI 生命周期约束。
pub(crate) unsafe fn top_k_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 2)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<TopKAttr>()?;
    let input_meta = ctx.inputs[0].checked_meta()?;
    let values_meta = ctx.outputs[0].checked_meta()?;
    let indices_meta = ctx.outputs[1].checked_meta()?;
    let axis = normalize_axis(attr.axis, input_meta.rank)?;
    let k = attr.k.get() as usize;
    if k == 0
        || k > input_meta.shape[axis]
        || ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || ctx.outputs[1].dtype != AiDtype::I64
        || values_meta.shape[..values_meta.rank] != indices_meta.shape[..indices_meta.rank]
        || values_meta.rank != input_meta.rank
    {
        return Err(BackendErr::InvalidTensor);
    }
    for current_axis in 0..input_meta.rank {
        let expected = if current_axis == axis {
            k
        } else {
            input_meta.shape[current_axis]
        };
        if values_meta.shape[current_axis] != expected {
            return Err(BackendErr::InvalidTensor);
        }
    }
    let input = read_float_tensor(&ctx.inputs[0], &input_meta, ctx.target)?;
    let (values, indices) = top_k_f32(&input, &input_meta, axis, k, attr.largest != 0)?;
    write_float_tensor(&values, &mut ctx.outputs[0], &values_meta, ctx.target)?;
    let logical_indices = i64_bytes(&indices);
    let output = unsafe { ctx.outputs[1].as_mut_slice::<u8>()? };
    write_logical_bytes(&logical_indices, output, &indices_meta, ctx.target)
}

/// 校验输入输出均为相同 shape 的 F32/F16 tensor。
fn validate_float_same_shape(
    ctx: &CallContext<'_>,
    input: &TensorMeta,
    output: &TensorMeta,
) -> Result<(), BackendErr> {
    if ctx.inputs[0].dtype != ctx.outputs[0].dtype
        || !matches!(ctx.inputs[0].dtype, AiDtype::F32 | AiDtype::F16)
        || input.rank != output.rank
        || input.shape[..input.rank] != output.shape[..output.rank]
    {
        return Err(BackendErr::InvalidTensor);
    }
    Ok(())
}

/// 按 arbitrary axis 执行数值稳定 Softmax。
fn softmax_f32(
    input: &[f32],
    meta: &TensorMeta,
    axis: usize,
    scale: f32,
    target: k3_ai_uabi::AiTargetHint,
) -> Result<Vec<f32>, BackendErr> {
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let outer = shape_product(&meta.shape[..axis])?;
    let axis_len = meta.shape[axis];
    let inner = shape_product(&meta.shape[axis + 1..meta.rank])?;
    let mut output = vec![0.0_f32; input.len()];
    let mut row = vec![0.0_f32; axis_len];
    let mut shifted = vec![0.0_f32; axis_len];
    let mut exponentials = vec![0.0_f32; axis_len];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            for axis_index in 0..axis_len {
                row[axis_index] = input
                    [outer_index * axis_len * inner + axis_index * inner + inner_index]
                    * scale;
            }
            let maximum = if vector_target(target) {
                rvv::reduce_max_f32(&row)
            } else {
                row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            };
            if vector_target(target) {
                rvv::affine_f32(&row, &mut shifted, 1.0, -maximum)?;
                rvv::exp_f32(&shifted, &mut exponentials)?;
            } else {
                for ((shifted, exponential), &value) in
                    shifted.iter_mut().zip(&mut exponentials).zip(&row)
                {
                    *shifted = value - maximum;
                    *exponential = libm::expf(*shifted);
                }
            }
            let sum = if vector_target(target) {
                rvv::reduce_sum_f32(&exponentials)
            } else {
                exponentials.iter().sum()
            };
            if vector_target(target) {
                rvv::affine_f32(&exponentials, &mut shifted, 1.0 / sum, 0.0)?;
            } else {
                for (dst, &value) in shifted.iter_mut().zip(&exponentials) {
                    *dst = value / sum;
                }
            }
            for axis_index in 0..axis_len {
                output[outer_index * axis_len * inner + axis_index * inner + inner_index] =
                    shifted[axis_index];
            }
        }
    }
    Ok(output)
}

/// RMSNorm F32 核心。
fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    hidden: usize,
    epsilon: f32,
    target: k3_ai_uabi::AiTargetHint,
) -> Result<Vec<f32>, BackendErr> {
    let mut output = vec![0.0_f32; input.len()];
    let mut squares = vec![0.0_f32; hidden];
    let mut weighted = vec![0.0_f32; hidden];
    for (row, out_row) in input
        .chunks_exact(hidden)
        .zip(output.chunks_exact_mut(hidden))
    {
        let sum = if vector_target(target) {
            rvv::binary_f32(BinaryOp::Mul, row, row, &mut squares)?;
            rvv::reduce_sum_f32(&squares)
        } else {
            row.iter().map(|value| value * value).sum()
        };
        let normalization = 1.0 / libm::sqrtf(sum / hidden as f32 + epsilon);
        if vector_target(target) {
            rvv::binary_f32(BinaryOp::Mul, row, weight, &mut weighted)?;
            rvv::affine_f32(&weighted, out_row, normalization, 0.0)?;
        } else {
            for ((dst, &value), &scale) in out_row.iter_mut().zip(row).zip(weight) {
                *dst = value * scale * normalization;
            }
        }
    }
    Ok(output)
}

/// RoPE F32 核心，sin/cos 系数按 position 计算，旋转乘加使用 RVV。
#[allow(clippy::too_many_arguments)]
fn rope_f32(
    input: &[f32],
    batch: usize,
    sequence: usize,
    heads: usize,
    head_dim: usize,
    n_dims: usize,
    attr: &RopeAttr,
    positions: Option<&[i64]>,
    target: k3_ai_uabi::AiTargetHint,
) -> Result<Vec<f32>, BackendErr> {
    let pairs = batch * sequence * heads * (n_dims / 2);
    let mut x = vec![0.0_f32; pairs];
    let mut y = vec![0.0_f32; pairs];
    let mut cosine = vec![0.0_f32; pairs];
    let mut sine = vec![0.0_f32; pairs];
    let base = if attr.freq_base == 0.0 {
        10_000.0
    } else {
        attr.freq_base
    };
    let frequency_scale = if attr.freq_scale == 0.0 {
        1.0
    } else {
        attr.freq_scale
    };
    let attention_scale = if attr.attn_factor == 0.0 {
        1.0
    } else {
        attr.attn_factor
    };
    let mut pair_index = 0;
    for batch_index in 0..batch {
        for sequence_index in 0..sequence {
            let position = positions.map_or(sequence_index as i64, |values| {
                if values.len() == sequence {
                    values[sequence_index]
                } else {
                    values[batch_index * sequence + sequence_index]
                }
            }) as f32;
            for head in 0..heads {
                let base_offset =
                    ((batch_index * sequence + sequence_index) * heads + head) * head_dim;
                for pair in 0..n_dims / 2 {
                    let (first, second) = if attr.mode == RopeAttr::MODE_GPT_J {
                        (pair * 2, pair * 2 + 1)
                    } else {
                        (pair, pair + n_dims / 2)
                    };
                    let frequency = libm::powf(base, -((2 * pair) as f32) / n_dims as f32);
                    let angle = position * frequency * frequency_scale;
                    x[pair_index] = input[base_offset + first];
                    y[pair_index] = input[base_offset + second];
                    cosine[pair_index] = libm::cosf(angle) * attention_scale;
                    sine[pair_index] = libm::sinf(angle) * attention_scale;
                    pair_index += 1;
                }
            }
        }
    }
    let mut first = vec![0.0_f32; pairs];
    let mut second = vec![0.0_f32; pairs];
    if vector_target(target) {
        let mut a = vec![0.0_f32; pairs];
        let mut b = vec![0.0_f32; pairs];
        rvv::binary_f32(BinaryOp::Mul, &x, &cosine, &mut a)?;
        rvv::binary_f32(BinaryOp::Mul, &y, &sine, &mut b)?;
        rvv::binary_f32(BinaryOp::Sub, &a, &b, &mut first)?;
        rvv::binary_f32(BinaryOp::Mul, &x, &sine, &mut a)?;
        rvv::binary_f32(BinaryOp::Mul, &y, &cosine, &mut b)?;
        rvv::binary_f32(BinaryOp::Add, &a, &b, &mut second)?;
    } else {
        for index in 0..pairs {
            first[index] = x[index] * cosine[index] - y[index] * sine[index];
            second[index] = x[index] * sine[index] + y[index] * cosine[index];
        }
    }
    let mut output = input.to_vec();
    pair_index = 0;
    for batch_index in 0..batch {
        for sequence_index in 0..sequence {
            for head in 0..heads {
                let base_offset =
                    ((batch_index * sequence + sequence_index) * heads + head) * head_dim;
                for pair in 0..n_dims / 2 {
                    let (first_axis, second_axis) = if attr.mode == RopeAttr::MODE_GPT_J {
                        (pair * 2, pair * 2 + 1)
                    } else {
                        (pair, pair + n_dims / 2)
                    };
                    output[base_offset + first_axis] = first[pair_index];
                    output[base_offset + second_axis] = second[pair_index];
                    pair_index += 1;
                }
            }
        }
    }
    Ok(output)
}

/// 计算 Pool 输出维大小。
fn pool_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad_before: usize,
    pad_after: usize,
    ceil_mode: bool,
) -> Result<usize, BackendErr> {
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or(BackendErr::InvalidAttr)?;
    let padded = input
        .checked_add(pad_before)
        .and_then(|value| value.checked_add(pad_after))
        .ok_or(BackendErr::InvalidAttr)?;
    if padded < effective_kernel {
        return Ok(0);
    }
    let numerator = padded - effective_kernel;
    Ok(if ceil_mode {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    })
}

/// NCHW MaxPool F32 核心。
fn max_pool_f32(
    input: &[f32],
    input_meta: &TensorMeta,
    output_meta: &TensorMeta,
    attr: &Pool2dAttr,
    target: k3_ai_uabi::AiTargetHint,
) -> Result<(Vec<f32>, Vec<i64>), BackendErr> {
    let n = input_meta.shape[0];
    let c = input_meta.shape[1];
    let ih = input_meta.shape[2];
    let iw = input_meta.shape[3];
    let oh = output_meta.shape[2];
    let ow = output_meta.shape[3];
    let kh = attr.kernel_h.get() as usize;
    let kw = attr.kernel_w.get() as usize;
    let sh = (attr.stride_h.get() as usize).max(1);
    let sw = (attr.stride_w.get() as usize).max(1);
    let dh = (attr.dilation_h.get() as usize).max(1);
    let dw = (attr.dilation_w.get() as usize).max(1);
    let ph = attr.pad_top.get() as usize;
    let pw = attr.pad_left.get() as usize;
    let mut output = vec![f32::NEG_INFINITY; output_meta.element_count];
    let mut indices = vec![0_i64; output_meta.element_count];
    let mut window = Vec::with_capacity(kh * kw);
    for batch in 0..n {
        for channel in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    window.clear();
                    let mut window_indices = Vec::with_capacity(kh * kw);
                    for ky in 0..kh {
                        let raw_y = oy * sh + ky * dh;
                        if raw_y < ph || raw_y - ph >= ih {
                            continue;
                        }
                        let iy = raw_y - ph;
                        for kx in 0..kw {
                            let raw_x = ox * sw + kx * dw;
                            if raw_x < pw || raw_x - pw >= iw {
                                continue;
                            }
                            let ix = raw_x - pw;
                            let input_index = ((batch * c + channel) * ih + iy) * iw + ix;
                            window.push(input[input_index]);
                            window_indices.push(input_index as i64);
                        }
                    }
                    if window.is_empty() {
                        continue;
                    }
                    let maximum = if vector_target(target) {
                        rvv::reduce_max_f32(&window)
                    } else {
                        window.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    };
                    let first = window
                        .iter()
                        .position(|value| *value == maximum)
                        .unwrap_or(0);
                    let output_index = ((batch * c + channel) * oh + oy) * ow + ox;
                    output[output_index] = maximum;
                    indices[output_index] = window_indices[first];
                }
            }
        }
    }
    Ok((output, indices))
}

/// 校验 ReduceMax 输出 shape。
fn validate_reduce_output(
    input: &TensorMeta,
    output: &TensorMeta,
    reduced: &[bool; MAX_DIM],
    attr: &ReduceMaxAttr,
) -> Result<(), BackendErr> {
    let keepdims = attr.flags.get() & ReduceMaxAttr::KEEP_DIMS != 0;
    let expected_rank = if keepdims {
        input.rank
    } else {
        input.rank - reduced[..input.rank].iter().filter(|&&value| value).count()
    };
    if output.rank != expected_rank {
        return Err(BackendErr::InvalidTensor);
    }
    let mut output_axis = 0;
    for (input_axis, &is_reduced) in reduced.iter().enumerate().take(input.rank) {
        if is_reduced {
            if keepdims {
                if output.shape[output_axis] != 1 {
                    return Err(BackendErr::InvalidTensor);
                }
                output_axis += 1;
            }
        } else {
            if output.shape[output_axis] != input.shape[input_axis] {
                return Err(BackendErr::InvalidTensor);
            }
            output_axis += 1;
        }
    }
    Ok(())
}

/// Generic ReduceMax F32 核心。
fn reduce_max_f32(
    input: &[f32],
    input_meta: &TensorMeta,
    output_meta: &TensorMeta,
    reduced: &[bool; MAX_DIM],
    target: k3_ai_uabi::AiTargetHint,
) -> Result<Vec<f32>, BackendErr> {
    let reduced_axes: Vec<usize> = (0..input_meta.rank).filter(|&axis| reduced[axis]).collect();
    let reduced_count = reduced_axes.iter().try_fold(1_usize, |count, &axis| {
        count
            .checked_mul(input_meta.shape[axis])
            .ok_or(BackendErr::InvalidTensor)
    })?;
    let mut output = vec![f32::NEG_INFINITY; output_meta.element_count];
    let mut values = vec![0.0_f32; reduced_count];
    for (output_linear, output_value) in output
        .iter_mut()
        .enumerate()
        .take(output_meta.element_count)
    {
        let mut output_coordinates = [0_usize; MAX_DIM];
        output_meta.coordinates(output_linear, &mut output_coordinates)?;
        let mut input_coordinates = [0_usize; MAX_DIM];
        let mut output_axis = 0;
        for input_axis in 0..input_meta.rank {
            if !reduced[input_axis] {
                input_coordinates[input_axis] = output_coordinates[output_axis];
                output_axis += 1;
            } else if output_meta.rank == input_meta.rank {
                output_axis += 1;
            }
        }
        for (reduced_linear, value) in values.iter_mut().enumerate() {
            let mut remaining = reduced_linear;
            for &axis in reduced_axes.iter().rev() {
                input_coordinates[axis] = remaining % input_meta.shape[axis];
                remaining /= input_meta.shape[axis];
            }
            *value = input[row_major_linear(input_meta, &input_coordinates)?];
        }
        *output_value = if vector_target(target) {
            rvv::reduce_max_f32(&values)
        } else {
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        };
    }
    Ok(output)
}

/// TopK CPU 核心。
fn top_k_f32(
    input: &[f32],
    meta: &TensorMeta,
    axis: usize,
    k: usize,
    largest: bool,
) -> Result<(Vec<f32>, Vec<i64>), BackendErr> {
    let outer = shape_product(&meta.shape[..axis])?;
    let axis_len = meta.shape[axis];
    let inner = shape_product(&meta.shape[axis + 1..meta.rank])?;
    let mut values = vec![0.0_f32; outer * k * inner];
    let mut indices = vec![0_i64; outer * k * inner];
    let mut candidates = Vec::with_capacity(axis_len);
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            candidates.clear();
            for axis_index in 0..axis_len {
                let value =
                    input[outer_index * axis_len * inner + axis_index * inner + inner_index];
                candidates.push((value, axis_index));
            }
            candidates.sort_by(|left, right| {
                let value_order = left.0.total_cmp(&right.0);
                let value_order = if largest {
                    value_order.reverse()
                } else {
                    value_order
                };
                value_order.then_with(|| left.1.cmp(&right.1))
            });
            for (selected, &(value, index)) in candidates.iter().take(k).enumerate() {
                let output_index = outer_index * k * inner + selected * inner + inner_index;
                values[output_index] = value;
                indices[output_index] = index as i64;
            }
        }
    }
    Ok((values, indices))
}

/// 计算 shape 乘积。
fn shape_product(shape: &[usize]) -> Result<usize, BackendErr> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(BackendErr::InvalidTensor)
    })
}

/// 把 I64 values 转为 native-endian bytes。
fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// NN 算子核心测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 构造 contiguous meta。
    fn meta(shape: &[usize]) -> TensorMeta {
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
            element_size: 4,
            element_count: count,
        }
    }

    /// Softmax 行和应接近 1。
    #[test]
    fn softmax_is_normalized() {
        let output = softmax_f32(
            &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0],
            &meta(&[2, 3]),
            1,
            1.0,
            k3_ai_uabi::AiTargetHint::PREFER_A100,
        )
        .unwrap();
        assert!((output[..3].iter().sum::<f32>() - 1.0).abs() < 5.0e-4);
        assert!((output[3..].iter().sum::<f32>() - 1.0).abs() < 5.0e-4);
    }

    /// RMSNorm 应匹配直接参考公式。
    #[test]
    fn rms_norm_matches_reference() {
        let output = rms_norm_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 1.0],
            2,
            1.0e-5,
            k3_ai_uabi::AiTargetHint::PREFER_A100,
        )
        .unwrap();
        let scale = 1.0 / libm::sqrtf(2.5 + 1.0e-5);
        assert!((output[0] - scale).abs() < 1.0e-5);
        assert!((output[1] - 2.0 * scale).abs() < 1.0e-5);
    }

    /// TopK 平局时应优先较低索引。
    #[test]
    fn top_k_is_stable_for_ties() {
        let (values, indices) = top_k_f32(&[3.0, 5.0, 5.0, 1.0], &meta(&[4]), 0, 2, true).unwrap();
        assert_eq!(values, [5.0, 5.0]);
        assert_eq!(indices, [1, 2]);
    }

    /// ReduceMax 多轴结果应正确。
    #[test]
    fn reduce_max_handles_multiple_axes() {
        let input_meta = meta(&[2, 2, 2]);
        let output_meta = meta(&[2]);
        let reduced = [false, true, true, false, false, false, false, false];
        let output = reduce_max_f32(
            &[1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0],
            &input_meta,
            &output_meta,
            &reduced,
            k3_ai_uabi::AiTargetHint::PREFER_A100,
        )
        .unwrap();
        assert_eq!(output, [4.0, 8.0]);
    }
}
