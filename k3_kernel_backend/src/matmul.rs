//! 矩阵乘法算子。

use core::cmp::min;
use core::default::Default;
use core::fmt::Debug;
use core::ops::{Add, Mul};

use crate::BackendCall;
use crate::call::CallContext;
use crate::quant;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, DimSize, ElemStride, MatMulAttr};
use log::error;

/// matmul 算子的输入输出参数集合，供 a100/x100/cpu 分发使用。
struct MatmulParameter<'a, Lhs, Rhs, Out> {
    /// 左矩阵元素切片。
    lhs: &'a [Lhs],
    /// 右矩阵元素切片。
    rhs: &'a [Rhs],
    /// 输出矩阵元素切片。
    out: &'a mut [Out],
    /// matmul 维度与 stride 参数。
    attr: MatMulAttr,
}

/// IME int8 路径直接按原始 byte 打包，signedness 由指令变体决定。
pub(crate) struct Int8MatmulParameter<'a> {
    /// 左矩阵原始字节。
    pub(crate) lhs: &'a [u8],
    /// 右矩阵原始字节。
    pub(crate) rhs: &'a [u8],
    /// i32 累加输出切片。
    pub(crate) out: &'a mut [i32],
    /// matmul 维度与 stride 参数。
    pub(crate) attr: MatMulAttr,
    /// 左右操作数的符号组合，决定 vmadot 指令变体。
    pub(crate) signedness: VmadotSignedness,
}

/// FP16 输入、FP32 累加的 matmul 参数。
pub(crate) struct Fp16MatmulParameter<'a> {
    /// FP16 lhs bits。
    pub(crate) lhs: &'a [u16],
    /// FP16 rhs bits。
    pub(crate) rhs: &'a [u16],
    /// F16 或 F32 输出。
    pub(crate) output: Fp16MatmulOutput<'a>,
    /// matmul 维度、stride 和 transpose 参数。
    pub(crate) attr: MatMulAttr,
}

/// FP16 matmul 支持的输出存储类型。
pub(crate) enum Fp16MatmulOutput<'a> {
    /// 写回 FP16。
    F16(&'a mut [u16]),
    /// 保留 FP32 累加结果。
    F32(&'a mut [f32]),
}

impl Fp16MatmulOutput<'_> {
    /// 写入一个 FP32 累加值，并按目标存储类型转换。
    fn write(&mut self, index: usize, value: f32) {
        match self {
            Self::F16(output) => output[index] = f16::from_f32(value).to_bits(),
            Self::F32(output) => output[index] = value,
        }
    }
}

/// vmadot 点积指令的操作数符号组合。
#[derive(Clone, Copy, Debug)]
pub(crate) enum VmadotSignedness {
    /// 左有符号 × 右有符号。
    SignedSigned,
    /// 左无符号 × 右无符号。
    UnsignedUnsigned,
    /// 左无符号 × 右有符号。
    UnsignedSigned,
    /// 左有符号 × 右无符号。
    SignedUnsigned,
}

/// IME int8 分块的 tile 尺寸（M×K×N）。
#[derive(Clone, Copy)]
pub(crate) struct ImeInt8Tile {
    /// tile 的行数（M 方向）。
    m: usize,
    /// tile 的列数（N 方向）。
    n: usize,
    /// tile 的内积深度（K 方向）。
    k: usize,
}

/// `flags` 位：lhs 转置。
const MATMUL_LHS_TRANSPOSED: u32 = 1 << 0;
/// `flags` 位：rhs 转置。
const MATMUL_RHS_TRANSPOSED: u32 = 1 << 1;

/// 单个 IME 向量寄存器可容纳的最大字节数。
const MAX_IME_VREG_BYTES: usize = 128;
/// 单个 IME C 累加 tile 可容纳的最大 i32 元素数。
const MAX_IME_C_ELEMS: usize = 64;

/// A100 FP16 `smt.vfwmadot` 的固定 M/K/N tile。
#[cfg(feature = "a100-fp16-ime")]
pub(crate) const A100_FP16_TILE: usize = 8;
/// 一个 FP16 IME source tile 的元素数。
#[cfg(feature = "a100-fp16-ime")]
const A100_FP16_SOURCE_ELEMS: usize = A100_FP16_TILE * A100_FP16_TILE;

/// X100: VLEN=256, Int8 tile = M=4, K=8, N=4.
const X100_INT8_TILE: ImeInt8Tile = ImeInt8Tile { m: 4, n: 4, k: 8 };

/// A100 手册 v0.6 写的是 M=8, K=16, N=8；当前硅前芯片实测仍按 A60/X100
/// 的 M=4, K=8, N=4 执行。这里默认选择实测 tile，保证真实硬件结果正确。
#[allow(dead_code)]
const A100_MANUAL_INT8_TILE: ImeInt8Tile = ImeInt8Tile { m: 8, n: 8, k: 16 };
/// A100 硅前芯片实测采用的 int8 tile，与 A60/X100 一致。
const A100_SILICON_INT8_TILE: ImeInt8Tile = ImeInt8Tile { m: 4, n: 4, k: 8 };

/// 将 UABI 维度 newtype 转成内部索引用的 `usize`。
fn dim(value: DimSize) -> usize {
    value.get() as usize
}

/// 将 UABI 元素 stride newtype 转成内部索引用的 `usize`。
fn elem_stride(value: ElemStride) -> usize {
    value.get() as usize
}

/// matmul 算子执行器，将 `BackendCall` 解析为内部参数后按 dtype 分发到具体实现。
pub(crate) unsafe fn matmul_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;

    let attr = ctx.read_attr::<MatMulAttr>()?;
    let target = ctx.target;
    let input0_dtype = ctx.inputs[0].dtype;
    let input1_dtype = ctx.inputs[1].dtype;
    let output_dtype = ctx.outputs[0].dtype;

    match (input0_dtype, input1_dtype, output_dtype) {
        (AiDtype::F32, AiDtype::F32, AiDtype::F32) => {
            if attr.accum_dtype != AiDtype::F32 {
                error!(
                    "matmul_caller: f32 matmul requires f32 accumulator, got {:?}",
                    attr.accum_dtype
                );
                return Err(BackendErr::UnsupportedDtype);
            }

            let lhs = unsafe { ctx.inputs[0].as_slice::<f32>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<f32>()? };
            let out = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            validate_matmul_bounds(&attr, lhs.len(), rhs.len(), out.len())?;

            let parameter = MatmulParameter {
                lhs,
                rhs,
                out,
                attr,
            };

            match target {
                AiTargetHint::AUTO | AiTargetHint::PREFER_CPU => cpu(parameter),
                AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100 => {
                    error!("matmul_caller: IME f32 matmul is not implemented yet");
                    Err(BackendErr::UnsupportedDtype)
                }
                _ => unreachable!("CallContext rejects unknown targets"),
            }
        }
        (AiDtype::F16, AiDtype::F16, output_dtype)
            if output_dtype == AiDtype::F16 || output_dtype == AiDtype::F32 =>
        {
            if attr.accum_dtype != AiDtype::F32 {
                error!(
                    "matmul_caller: f16 IME matmul requires f32 accumulator, got {:?}",
                    attr.accum_dtype
                );
                return Err(BackendErr::UnsupportedDtype);
            }
            let lhs = unsafe { ctx.inputs[0].as_slice::<u16>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<u16>()? };
            let output_len = if output_dtype == AiDtype::F16 {
                ctx.outputs[0]
                    .byte_len
                    .try_as_usize()
                    .map_err(|_| BackendErr::InvalidTensor)?
                    / core::mem::size_of::<u16>()
            } else {
                ctx.outputs[0]
                    .byte_len
                    .try_as_usize()
                    .map_err(|_| BackendErr::InvalidTensor)?
                    / core::mem::size_of::<f32>()
            };
            validate_matmul_bounds(&attr, lhs.len(), rhs.len(), output_len)?;

            let output = if output_dtype == AiDtype::F16 {
                Fp16MatmulOutput::F16(unsafe { ctx.outputs[0].as_mut_slice::<u16>()? })
            } else {
                Fp16MatmulOutput::F32(unsafe { ctx.outputs[0].as_mut_slice::<f32>()? })
            };
            let parameter = Fp16MatmulParameter {
                lhs,
                rhs,
                output,
                attr,
            };
            match target {
                AiTargetHint::AUTO | AiTargetHint::PREFER_CPU => cpu_f16_f32(parameter),
                AiTargetHint::PREFER_X100 => Err(BackendErr::UnsupportedDtype),
                AiTargetHint::PREFER_A100 => {
                    #[cfg(feature = "a100-fp16-ime")]
                    {
                        ime_f16_f32_matmul(parameter)
                    }
                    #[cfg(not(feature = "a100-fp16-ime"))]
                    {
                        let _ = parameter;
                        error!(
                            "matmul_caller: A100 FP16 IME is disabled until MCPM.BF16 is controlled"
                        );
                        Err(BackendErr::UnsupportedOp)
                    }
                }
                _ => unreachable!("CallContext rejects unknown targets"),
            }
        }
        (AiDtype::F16, AiDtype::F32, AiDtype::F32) => {
            if attr.accum_dtype != AiDtype::F32 {
                return Err(BackendErr::UnsupportedDtype);
            }
            let lhs = unsafe { ctx.inputs[0].as_slice::<u16>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<f32>()? };
            let out = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            validate_matmul_bounds(&attr, lhs.len(), rhs.len(), out.len())?;
            cpu_f16_f32_rhs_f32(lhs, rhs, out, &attr)
        }
        (lhs_dtype, rhs_dtype, AiDtype::I32)
            if is_int8_dtype(lhs_dtype) && is_int8_dtype(rhs_dtype) =>
        {
            if attr.accum_dtype != AiDtype::I32 {
                error!(
                    "matmul_caller: int8 IME matmul requires i32 accumulator, got {:?}",
                    attr.accum_dtype
                );
                return Err(BackendErr::UnsupportedDtype);
            }

            let lhs = unsafe { ctx.inputs[0].as_slice::<u8>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<u8>()? };
            let out = unsafe { ctx.outputs[0].as_mut_slice::<i32>()? };
            validate_matmul_bounds(&attr, lhs.len(), rhs.len(), out.len())?;

            let parameter = Int8MatmulParameter {
                lhs,
                rhs,
                out,
                attr,
                signedness: signedness_for(lhs_dtype, rhs_dtype),
            };

            match target {
                AiTargetHint::AUTO | AiTargetHint::PREFER_CPU => cpu_int8_i32(parameter),
                AiTargetHint::PREFER_X100 => x100(parameter),
                AiTargetHint::PREFER_A100 => a100(parameter),
                _ => unreachable!("CallContext rejects unknown targets"),
            }
        }
        (lhs_dtype, AiDtype::F32, AiDtype::F32) if lhs_dtype.is_ggml_quant() => {
            let lhs_meta = ctx.inputs[0].checked_quant_meta()?;
            let rhs = unsafe { ctx.inputs[1].as_slice::<f32>()? };
            let out = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            quantized_lhs_f32_matmul(&ctx.inputs[0], &lhs_meta, rhs, out, &attr)
        }
        (lhs_dtype, AiDtype::F16, AiDtype::F32) if lhs_dtype.is_ggml_quant() => {
            let lhs_meta = ctx.inputs[0].checked_quant_meta()?;
            let rhs = unsafe { ctx.inputs[1].as_slice::<u16>()? };
            let out = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            quantized_lhs_f16_matmul(&ctx.inputs[0], &lhs_meta, rhs, out, &attr)
        }
        _ => {
            error!(
                "matmul_caller: unsupported dtype, input0={:?}, input1={:?}, output={:?}",
                input0_dtype, input1_dtype, output_dtype
            );
            Err(BackendErr::UnsupportedDtype)
        }
    }
}

/// FP16 lhs × F32 rhs -> F32 output.
fn cpu_f16_f32_rhs_f32(
    lhs: &[u16],
    rhs: &[f32],
    out: &mut [f32],
    attr: &MatMulAttr,
) -> Result<(), BackendErr> {
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);
    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);
    let lhs_batch_stride = elem_stride(attr.lhs_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    let lhs = f16::from_bits(lhs[lhs_index(attr, lhs_base, i, p)]).to_f32();
                    let rhs = rhs[rhs_index(attr, rhs_base, p, j)];
                    sum += lhs * rhs;
                }
                out[out_base + i * out_row_stride + j * out_col_stride] = sum;
            }
        }
    }
    Ok(())
}

/// ggml quant lhs × F32 rhs -> F32 output.
fn quantized_lhs_f32_matmul(
    lhs_view: &crate::BackendTensorView,
    lhs_meta: &crate::call::QuantTensorMeta,
    rhs: &[f32],
    out: &mut [f32],
    attr: &MatMulAttr,
) -> Result<(), BackendErr> {
    validate_quantized_lhs_matmul(lhs_meta, rhs.len(), out.len(), attr)?;
    execute_quantized_lhs_matmul(lhs_view, lhs_meta, rhs, out, attr, |slice, index| {
        Ok(slice[index])
    })
}

/// ggml quant lhs × F16 rhs -> F32 output.
fn quantized_lhs_f16_matmul(
    lhs_view: &crate::BackendTensorView,
    lhs_meta: &crate::call::QuantTensorMeta,
    rhs: &[u16],
    out: &mut [f32],
    attr: &MatMulAttr,
) -> Result<(), BackendErr> {
    validate_quantized_lhs_matmul(lhs_meta, rhs.len(), out.len(), attr)?;
    execute_quantized_lhs_matmul(lhs_view, lhs_meta, rhs, out, attr, |slice, index| {
        Ok(f16::from_bits(slice[index]).to_f32())
    })
}

/// Shared quantized lhs matmul loop.
fn execute_quantized_lhs_matmul<Rhs, F>(
    lhs_view: &crate::BackendTensorView,
    lhs_meta: &crate::call::QuantTensorMeta,
    rhs: &[Rhs],
    out: &mut [f32],
    attr: &MatMulAttr,
    mut rhs_value: F,
) -> Result<(), BackendErr>
where
    F: FnMut(&[Rhs], usize) -> Result<f32, BackendErr>,
{
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);
    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let mut lhs_coordinates = [0_usize; k3_ai_uabi::MAX_DIM];

    for b in 0..batch {
        set_quant_batch_coordinates(lhs_meta, b, &mut lhs_coordinates)?;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;
        for i in 0..m {
            lhs_coordinates[1] = i;
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    lhs_coordinates[0] = p;
                    let lhs = quant::read_quant_f32(lhs_view, lhs_meta, &lhs_coordinates)?;
                    let rhs = rhs_value(rhs, rhs_index(attr, rhs_base, p, j))?;
                    sum += lhs * rhs;
                }
                out[out_base + i * out_row_stride + j * out_col_stride] = sum;
            }
        }
    }
    Ok(())
}

/// Validate quantized lhs matmul bounds and supported flags.
fn validate_quantized_lhs_matmul(
    lhs_meta: &crate::call::QuantTensorMeta,
    rhs_len: usize,
    out_len: usize,
    attr: &MatMulAttr,
) -> Result<(), BackendErr> {
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);
    if m == 0
        || n == 0
        || k == 0
        || lhs_transposed(attr)
        || lhs_meta.rank < 2
        || lhs_meta.shape[0] < k
        || lhs_meta.shape[1] < m
        || attr.accum_dtype != AiDtype::F32
    {
        return Err(BackendErr::InvalidAttr);
    }
    let lhs_batches = lhs_meta.shape[2..lhs_meta.rank]
        .iter()
        .try_fold(1_usize, |product, &dim| product.checked_mul(dim))
        .ok_or(BackendErr::InvalidTensor)?;
    if lhs_batches != 1 && lhs_batches < batch {
        return Err(BackendErr::InvalidTensor);
    }
    let last_rhs = rhs_index(
        attr,
        (batch - 1) * elem_stride(attr.rhs_batch_stride),
        k - 1,
        n - 1,
    );
    let last_out = (batch - 1)
        .checked_mul(elem_stride(attr.out_batch_stride))
        .and_then(|base| {
            let with_row =
                base.checked_add((m - 1).checked_mul(elem_stride(attr.out_row_stride))?)?;
            with_row.checked_add((n - 1).checked_mul(elem_stride(attr.out_col_stride))?)
        })
        .ok_or(BackendErr::InvalidTensor)?;
    if last_rhs >= rhs_len || last_out >= out_len {
        return Err(BackendErr::InvalidTensor);
    }
    Ok(())
}

/// Map flattened batch index to quant tensor axes 2..rank, with batch broadcast.
fn set_quant_batch_coordinates(
    meta: &crate::call::QuantTensorMeta,
    batch_index: usize,
    coordinates: &mut [usize; k3_ai_uabi::MAX_DIM],
) -> Result<(), BackendErr> {
    for coordinate in coordinates.iter_mut().take(meta.rank).skip(2) {
        *coordinate = 0;
    }
    let batch_count = meta.shape[2..meta.rank]
        .iter()
        .try_fold(1_usize, |product, &dim| product.checked_mul(dim))
        .ok_or(BackendErr::InvalidTensor)?;
    if batch_count <= 1 {
        return Ok(());
    }
    if batch_index >= batch_count {
        return Err(BackendErr::InvalidTensor);
    }
    let mut remaining = batch_index;
    for (axis, coordinate) in coordinates.iter_mut().enumerate().take(meta.rank).skip(2) {
        *coordinate = remaining % meta.shape[axis];
        remaining /= meta.shape[axis];
    }
    Ok(())
}

/// A100 加速器 matmul 实现。
fn a100(parameter: Int8MatmulParameter<'_>) -> Result<(), BackendErr> {
    error!(
        "run int8 matmul on A100 IME, tile={}x{}x{} (M x K x N)",
        A100_SILICON_INT8_TILE.m, A100_SILICON_INT8_TILE.k, A100_SILICON_INT8_TILE.n
    );
    ime_int8_i32_matmul(parameter, A100_SILICON_INT8_TILE)
}

/// X100 加速器 matmul 实现。
fn x100(parameter: Int8MatmulParameter<'_>) -> Result<(), BackendErr> {
    error!(
        "run int8 matmul on X100 IME, tile={}x{}x{} (M x K x N)",
        X100_INT8_TILE.m, X100_INT8_TILE.k, X100_INT8_TILE.n
    );
    ime_int8_i32_matmul(parameter, X100_INT8_TILE)
}

/// CPU fallback matmul 实现。
fn cpu<T>(parameter: MatmulParameter<'_, T, T, T>) -> Result<(), BackendErr>
where
    T: Debug + Default + Copy + Add<Output = T> + Mul<Output = T>,
{
    error!("run mutmal in cpu");

    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = if attr.batch == 0 { 1 } else { dim(attr.batch) };

    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);

    let lhs_batch_stride = elem_stride(attr.lhs_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for p in 0..k {
                    let lhs_idx = lhs_index(attr, lhs_base, i, p);
                    let rhs_idx = rhs_index(attr, rhs_base, p, j);
                    sum = sum + parameter.lhs[lhs_idx] * parameter.rhs[rhs_idx];
                }
                let out_idx = out_base + i * out_row_stride + j * out_col_stride;
                parameter.out[out_idx] = sum;
            }
        }
    }

    // TODO: 后面必须实现写回同步。

    log::info!("[kernel backend matmul: log]:");
    log::info!("  shape: {}x{} @ {}x{} -> {}x{}", m, k, k, n, m, n);
    log::info!("  batch: {}", batch);

    // 打印计算结果
    log::info!("  cpu matmul output ({}x{}):", m, n);
    for i in 0..m {
        let mut row_str = alloc::string::String::from("    [");
        for j in 0..n {
            let out_idx = i * out_row_stride + j * out_col_stride;
            row_str.push_str(&alloc::format!(" {:?}", parameter.out[out_idx]));
        }
        row_str.push_str(" ]");
    }

    Ok(())
}

/// CPU fallback for FP16×FP16 with FP32 accumulation and F16/F32 output.
fn cpu_f16_f32(mut parameter: Fp16MatmulParameter<'_>) -> Result<(), BackendErr> {
    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);

    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);
    let lhs_batch_stride = elem_stride(attr.lhs_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    let lhs =
                        f16::from_bits(parameter.lhs[lhs_index(attr, lhs_base, i, p)]).to_f32();
                    let rhs =
                        f16::from_bits(parameter.rhs[rhs_index(attr, rhs_base, p, j)]).to_f32();
                    sum += lhs * rhs;
                }
                let out_idx = out_base + i * out_row_stride + j * out_col_stride;
                parameter.output.write(out_idx, sum);
            }
        }
    }
    Ok(())
}

/// A100 FP16 IME tiled matmul using 8×8×8 `smt.vfwmadot` semantics.
#[cfg(feature = "a100-fp16-ime")]
pub(crate) fn ime_f16_f32_matmul(mut parameter: Fp16MatmulParameter<'_>) -> Result<(), BackendErr> {
    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);

    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);
    let lhs_batch_stride = elem_stride(attr.lhs_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;
        let mut row = 0;
        while row < m {
            let valid_m = min(A100_FP16_TILE, m - row);
            let mut col = 0;
            while col < n {
                let valid_n = min(A100_FP16_TILE, n - col);
                let mut acc_tile = [0.0_f32; A100_FP16_SOURCE_ELEMS];
                let mut depth = 0;
                while depth < k {
                    let valid_k = min(A100_FP16_TILE, k - depth);
                    let mut lhs_tile = [0_u16; A100_FP16_SOURCE_ELEMS];
                    let mut rhs_tile = [0_u16; A100_FP16_SOURCE_ELEMS];
                    let mut partial = [0.0_f32; A100_FP16_SOURCE_ELEMS];
                    pack_fp16_tiles(
                        &parameter,
                        lhs_base,
                        rhs_base,
                        row,
                        col,
                        depth,
                        valid_m,
                        valid_n,
                        valid_k,
                        &mut lhs_tile,
                        &mut rhs_tile,
                    );
                    vfwmadot_tile(&lhs_tile, &rhs_tile, &mut partial);
                    for i in 0..valid_m {
                        for j in 0..valid_n {
                            let idx = i * A100_FP16_TILE + j;
                            acc_tile[idx] += partial[idx];
                        }
                    }
                    depth += A100_FP16_TILE;
                }
                for i in 0..valid_m {
                    for j in 0..valid_n {
                        let out_idx =
                            out_base + (row + i) * out_row_stride + (col + j) * out_col_stride;
                        parameter
                            .output
                            .write(out_idx, acc_tile[i * A100_FP16_TILE + j]);
                    }
                }
                col += A100_FP16_TILE;
            }
            row += A100_FP16_TILE;
        }
    }
    Ok(())
}

/// Pack current FP16 tile as A[M,K] and B[N,K] for `C += A × B^T`.
#[cfg(feature = "a100-fp16-ime")]
#[allow(clippy::too_many_arguments)]
fn pack_fp16_tiles(
    parameter: &Fp16MatmulParameter<'_>,
    lhs_base: usize,
    rhs_base: usize,
    row: usize,
    col: usize,
    depth: usize,
    valid_m: usize,
    valid_n: usize,
    valid_k: usize,
    lhs_tile: &mut [u16; A100_FP16_SOURCE_ELEMS],
    rhs_tile: &mut [u16; A100_FP16_SOURCE_ELEMS],
) {
    for i in 0..valid_m {
        for kk in 0..valid_k {
            lhs_tile[i * A100_FP16_TILE + kk] =
                parameter.lhs[lhs_index(&parameter.attr, lhs_base, row + i, depth + kk)];
        }
    }
    for j in 0..valid_n {
        for kk in 0..valid_k {
            rhs_tile[j * A100_FP16_TILE + kk] =
                parameter.rhs[rhs_index(&parameter.attr, rhs_base, depth + kk, col + j)];
        }
    }
}

/// Execute one FP16 tile: RISC-V feature builds use hardware, all others use software mirror.
#[cfg(feature = "a100-fp16-ime")]
fn vfwmadot_tile(
    lhs_tile: &[u16; A100_FP16_SOURCE_ELEMS],
    rhs_tile: &[u16; A100_FP16_SOURCE_ELEMS],
    output: &mut [f32; A100_FP16_SOURCE_ELEMS],
) {
    #[cfg(all(
        feature = "a100-fp16-ime",
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    unsafe {
        vfwmadot_tile_hw(lhs_tile.as_ptr(), rhs_tile.as_ptr(), output.as_mut_ptr());
    }
    #[cfg(not(all(
        feature = "a100-fp16-ime",
        any(target_arch = "riscv32", target_arch = "riscv64")
    )))]
    {
        vfwmadot_tile_sw(lhs_tile, rhs_tile, output);
    }
}

/// Software mirror for `smt.vfwmadot`.
#[cfg(all(
    feature = "a100-fp16-ime",
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
fn vfwmadot_tile_sw(
    lhs_tile: &[u16; A100_FP16_SOURCE_ELEMS],
    rhs_tile: &[u16; A100_FP16_SOURCE_ELEMS],
    output: &mut [f32; A100_FP16_SOURCE_ELEMS],
) {
    for i in 0..A100_FP16_TILE {
        for j in 0..A100_FP16_TILE {
            let mut sum = 0.0_f32;
            for kk in 0..A100_FP16_TILE {
                let lhs = f16::from_bits(lhs_tile[i * A100_FP16_TILE + kk]).to_f32();
                let rhs = f16::from_bits(rhs_tile[j * A100_FP16_TILE + kk]).to_f32();
                sum += lhs * rhs;
            }
            output[i * A100_FP16_TILE + j] = sum;
        }
    }
}

/// Hardware `smt.vfwmadot v16, v2, v8`; requires MCPM.BF16=0 for FP16.
///
/// # Safety
///
/// Must run on A100 with vector/IME enabled and MCPM configured for FP16.
#[cfg(all(
    feature = "a100-fp16-ime",
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
unsafe fn vfwmadot_tile_hw(lhs_tile: *const u16, rhs_tile: *const u16, output: *mut f32) {
    unsafe {
        core::arch::asm!(
            ".option push",
            ".option arch, +v",
            "vsetvli        t0, zero, e16, m1",
            "vle16.v        v2, ({lhs})",
            "vle16.v        v8, ({rhs})",
            "vsetvli        t0, zero, e32, m2",
            "vmv.v.i        v16, 0",
            "vsetvli        t0, zero, e16, m1",
            ".word          0x9E81482B",
            "vsetvli        t0, zero, e32, m2",
            "vse32.v        v16, ({output})",
            ".option pop",
            lhs = in(reg) lhs_tile,
            rhs = in(reg) rhs_tile,
            output = in(reg) output,
            out("t0") _,
            out("v2") _,
            out("v8") _,
            out("v16") _,
            out("v17") _,
            options(nostack),
        );
    }
}

/// CPU fallback for `[U]Int8 x [U]Int8 -> Int32`，语义与 `smt.vmadot*` 一致。
fn cpu_int8_i32(parameter: Int8MatmulParameter<'_>) -> Result<(), BackendErr> {
    error!("run int8 matmul in cpu");

    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);

    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);

    let lhs_batch_stride = elem_stride(attr.lhs_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0_i32;
                for p in 0..k {
                    let lhs_idx = lhs_index(attr, lhs_base, i, p);
                    let rhs_idx = rhs_index(attr, rhs_base, p, j);
                    let lhs =
                        byte_to_i32(parameter.lhs[lhs_idx], parameter.signedness.lhs_signed());
                    let rhs =
                        byte_to_i32(parameter.rhs[rhs_idx], parameter.signedness.rhs_signed());
                    sum = sum.wrapping_add(lhs.wrapping_mul(rhs));
                }
                let out_idx = out_base + i * out_row_stride + j * out_col_stride;
                parameter.out[out_idx] = sum;
            }
        }
    }

    Ok(())
}

/// 通用 IME int8→i32 分块 matmul：按 `tile` 尺寸遍历 M/N/K，逐 tile 累加写回。
pub(crate) fn ime_int8_i32_matmul(
    parameter: Int8MatmulParameter<'_>,
    tile: ImeInt8Tile,
) -> Result<(), BackendErr> {
    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    let batch = normalized_batch(attr);

    let out_row_stride = elem_stride(attr.out_row_stride);
    let out_col_stride = elem_stride(attr.out_col_stride);

    let lhs_batch_stride = elem_stride(attr.lhs_batch_stride);
    let rhs_batch_stride = elem_stride(attr.rhs_batch_stride);
    let out_batch_stride = elem_stride(attr.out_batch_stride);

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;

        let mut row = 0;
        while row < m {
            let mut col = 0;
            while col < n {
                let valid_m = min(tile.m, m - row);
                let valid_n = min(tile.n, n - col);
                let mut acc_tile = [0_i32; MAX_IME_C_ELEMS];

                let mut depth = 0;
                while depth < k {
                    let valid_k = min(tile.k, k - depth);
                    let mut a_tile = [0_u8; MAX_IME_VREG_BYTES];
                    let mut b_tile = [0_u8; MAX_IME_VREG_BYTES];
                    let mut partial = [0_i32; MAX_IME_C_ELEMS];

                    pack_ime_tiles(
                        &parameter,
                        lhs_base,
                        rhs_base,
                        row,
                        col,
                        depth,
                        valid_m,
                        valid_n,
                        valid_k,
                        tile,
                        &mut a_tile,
                        &mut b_tile,
                    );

                    vmadot_tile(&a_tile, &b_tile, &mut partial, tile, parameter.signedness);

                    for i in 0..valid_m {
                        for j in 0..valid_n {
                            let idx = i * tile.n + j;
                            acc_tile[idx] = acc_tile[idx].wrapping_add(partial[idx]);
                        }
                    }

                    depth += tile.k;
                }

                for i in 0..valid_m {
                    for j in 0..valid_n {
                        let out_idx =
                            out_base + (row + i) * out_row_stride + (col + j) * out_col_stride;
                        parameter.out[out_idx] = acc_tile[i * tile.n + j];
                    }
                }

                col += tile.n;
            }
            row += tile.m;
        }
    }

    Ok(())
}

/// 把 lhs/rhs 的当前 tile 数据按 K 主序打包进 `a_tile`/`b_tile` 向量寄存器缓冲。
#[allow(clippy::too_many_arguments)]
fn pack_ime_tiles(
    parameter: &Int8MatmulParameter<'_>,
    lhs_base: usize,
    rhs_base: usize,
    row: usize,
    col: usize,
    depth: usize,
    valid_m: usize,
    valid_n: usize,
    valid_k: usize,
    tile: ImeInt8Tile,
    a_tile: &mut [u8; MAX_IME_VREG_BYTES],
    b_tile: &mut [u8; MAX_IME_VREG_BYTES],
) {
    for i in 0..valid_m {
        for kk in 0..valid_k {
            let src = lhs_index(&parameter.attr, lhs_base, row + i, depth + kk);
            a_tile[i * tile.k + kk] = parameter.lhs[src];
        }
    }

    for j in 0..valid_n {
        for kk in 0..valid_k {
            let src = rhs_index(&parameter.attr, rhs_base, depth + kk, col + j);
            b_tile[j * tile.k + kk] = parameter.rhs[src];
        }
    }
}

/// 对一个 tile 执行 vmadot 点积：RISC-V 目标走硬件指令，其他目标走软件实现。
fn vmadot_tile(
    a_tile: &[u8; MAX_IME_VREG_BYTES],
    b_tile: &[u8; MAX_IME_VREG_BYTES],
    c_tile: &mut [i32; MAX_IME_C_ELEMS],
    tile: ImeInt8Tile,
    signedness: VmadotSignedness,
) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        let _ = tile;
        vmadot_tile_hw(a_tile, b_tile, c_tile, signedness);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    {
        vmadot_tile_sw(a_tile, b_tile, c_tile, tile, signedness);
    }
}

/// vmadot 的软件参考实现，用于非 RISC-V 目标与测试。
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn vmadot_tile_sw(
    a_tile: &[u8; MAX_IME_VREG_BYTES],
    b_tile: &[u8; MAX_IME_VREG_BYTES],
    c_tile: &mut [i32; MAX_IME_C_ELEMS],
    tile: ImeInt8Tile,
    signedness: VmadotSignedness,
) {
    for i in 0..tile.m {
        for j in 0..tile.n {
            let mut sum = 0_i32;
            for kk in 0..tile.k {
                let lhs = byte_to_i32(a_tile[i * tile.k + kk], signedness.lhs_signed());
                let rhs = byte_to_i32(b_tile[j * tile.k + kk], signedness.rhs_signed());
                sum = sum.wrapping_add(lhs.wrapping_mul(rhs));
            }
            c_tile[i * tile.n + j] = sum;
        }
    }
}

/// 按 signedness 选择对应的 `smt.vmadot*` 指令字，执行硬件 tile 点积。
///
/// # Safety
///
/// 需运行在支持向量扩展的 RISC-V 硬件上，且 tile 缓冲大小满足指令假设。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn vmadot_tile_hw(
    a_tile: &[u8; MAX_IME_VREG_BYTES],
    b_tile: &[u8; MAX_IME_VREG_BYTES],
    c_tile: &mut [i32; MAX_IME_C_ELEMS],
    signedness: VmadotSignedness,
) {
    match signedness {
        VmadotSignedness::UnsignedUnsigned => unsafe {
            vmadot_tile_hw_word::<0xE210_082B>(
                a_tile.as_ptr(),
                b_tile.as_ptr(),
                c_tile.as_mut_ptr(),
            )
        },
        VmadotSignedness::UnsignedSigned => unsafe {
            vmadot_tile_hw_word::<0xE210_182B>(
                a_tile.as_ptr(),
                b_tile.as_ptr(),
                c_tile.as_mut_ptr(),
            )
        },
        VmadotSignedness::SignedUnsigned => unsafe {
            vmadot_tile_hw_word::<0xE210_282B>(
                a_tile.as_ptr(),
                b_tile.as_ptr(),
                c_tile.as_mut_ptr(),
            )
        },
        VmadotSignedness::SignedSigned => unsafe {
            vmadot_tile_hw_word::<0xE210_382B>(
                a_tile.as_ptr(),
                b_tile.as_ptr(),
                c_tile.as_mut_ptr(),
            )
        },
    }
}

/// 以内联汇编发射常量指令字 `VMADOT_WORD`，加载 a/b tile 并写回 i32 结果。
///
/// # Safety
///
/// `a_tile`/`b_tile` 必须指向至少一个向量寄存器长度的有效字节，`c_tile` 必须
/// 可写并容纳输出；调用需运行在支持向量扩展的 RISC-V 硬件上。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
unsafe fn vmadot_tile_hw_word<const VMADOT_WORD: u32>(
    a_tile: *const u8,
    b_tile: *const u8,
    c_tile: *mut i32,
) {
    unsafe {
        core::arch::asm!(
            ".option push",
            ".option arch, +v",
            "vsetvli        t0, zero, e8, m1",
            "vle8.v         v0, ({a})",
            "vle8.v         v1, ({b})",
            "vsetvli        t0, zero, e32, m2",
            "vmv.v.i        v16, 0",
            "vsetvli        t0, zero, e8, m1",
            ".word          {vmadot_word}",
            "vsetvli        t0, zero, e32, m2",
            "vse32.v        v16, ({c})",
            ".option pop",
            a = in(reg) a_tile,
            b = in(reg) b_tile,
            c = in(reg) c_tile,
            vmadot_word = const VMADOT_WORD,
            out("t0") _,
            out("v0") _,
            out("v1") _,
            out("v16") _,
            out("v17") _,
            options(nostack),
        );
    }
}

/// 按 shape/stride 校验 lhs/rhs/out 三块 buffer 的最大访问下标都落在长度内。
fn validate_matmul_bounds(
    attr: &MatMulAttr,
    lhs_len: usize,
    rhs_len: usize,
    out_len: usize,
) -> Result<(), BackendErr> {
    let batch = normalized_batch(attr);
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);

    let lhs_rows = if lhs_transposed(attr) { k } else { m };
    let lhs_cols = if lhs_transposed(attr) { m } else { k };
    let rhs_rows = if rhs_transposed(attr) { n } else { k };
    let rhs_cols = if rhs_transposed(attr) { k } else { n };

    let lhs_ok = tensor_region_fits(
        lhs_len,
        batch,
        lhs_rows,
        lhs_cols,
        elem_stride(attr.lhs_row_stride),
        elem_stride(attr.lhs_col_stride),
        elem_stride(attr.lhs_batch_stride),
    );
    let rhs_ok = tensor_region_fits(
        rhs_len,
        batch,
        rhs_rows,
        rhs_cols,
        elem_stride(attr.rhs_row_stride),
        elem_stride(attr.rhs_col_stride),
        elem_stride(attr.rhs_batch_stride),
    );
    let out_ok = tensor_region_fits(
        out_len,
        batch,
        m,
        n,
        elem_stride(attr.out_row_stride),
        elem_stride(attr.out_col_stride),
        elem_stride(attr.out_batch_stride),
    );

    if lhs_ok && rhs_ok && out_ok {
        Ok(())
    } else {
        error!(
            "matmul_caller: tensor bounds invalid, lhs_ok={}, rhs_ok={}, out_ok={}",
            lhs_ok, rhs_ok, out_ok
        );
        Err(BackendErr::InvalidTensor)
    }
}

/// 判断按给定 batch/行列 stride 访问时的最大线性下标是否严格小于 `len`。
fn tensor_region_fits(
    len: usize,
    batch: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    batch_stride: usize,
) -> bool {
    if batch == 0 || rows == 0 || cols == 0 {
        return true;
    }

    let max_batch = match (batch - 1).checked_mul(batch_stride) {
        Some(value) => value,
        None => return false,
    };
    let max_row = match (rows - 1).checked_mul(row_stride) {
        Some(value) => value,
        None => return false,
    };
    let max_col = match (cols - 1).checked_mul(col_stride) {
        Some(value) => value,
        None => return false,
    };

    max_batch
        .checked_add(max_row)
        .and_then(|idx| idx.checked_add(max_col))
        .is_some_and(|max_idx| max_idx < len)
}

/// 把 `batch == 0` 归一化为 1，其余原样返回。
fn normalized_batch(attr: &MatMulAttr) -> usize {
    if attr.batch == 0 { 1 } else { dim(attr.batch) }
}

/// 判断 dtype 是否为 8 位整型（I8/U8）。
fn is_int8_dtype(dtype: AiDtype) -> bool {
    dtype == AiDtype::I8 || dtype == AiDtype::U8
}

/// 计算 lhs 元素在切片中的线性下标，考虑 base、转置与行列 stride。
fn lhs_index(attr: &MatMulAttr, base: usize, row: usize, depth: usize) -> usize {
    if lhs_transposed(attr) {
        base + depth * elem_stride(attr.lhs_row_stride) + row * elem_stride(attr.lhs_col_stride)
    } else {
        base + row * elem_stride(attr.lhs_row_stride) + depth * elem_stride(attr.lhs_col_stride)
    }
}

/// 计算 rhs 元素在切片中的线性下标，考虑 base、转置与行列 stride。
fn rhs_index(attr: &MatMulAttr, base: usize, depth: usize, col: usize) -> usize {
    if rhs_transposed(attr) {
        base + col * elem_stride(attr.rhs_row_stride) + depth * elem_stride(attr.rhs_col_stride)
    } else {
        base + depth * elem_stride(attr.rhs_row_stride) + col * elem_stride(attr.rhs_col_stride)
    }
}

/// lhs 是否置了转置标志位。
fn lhs_transposed(attr: &MatMulAttr) -> bool {
    attr.flags.get() & MATMUL_LHS_TRANSPOSED != 0
}

/// rhs 是否置了转置标志位。
fn rhs_transposed(attr: &MatMulAttr) -> bool {
    attr.flags.get() & MATMUL_RHS_TRANSPOSED != 0
}

/// 返回 A100 硅前芯片实测采用的 int8 tile，供 conv2d 等复用 IME matmul。
pub(crate) fn a100_int8_tile() -> ImeInt8Tile {
    A100_SILICON_INT8_TILE
}

/// 根据左右操作数 dtype 推导 vmadot 的符号组合。
pub(crate) fn signedness_for(lhs: AiDtype, rhs: AiDtype) -> VmadotSignedness {
    match (lhs, rhs) {
        (AiDtype::U8, AiDtype::U8) => VmadotSignedness::UnsignedUnsigned,
        (AiDtype::U8, AiDtype::I8) => VmadotSignedness::UnsignedSigned,
        (AiDtype::I8, AiDtype::U8) => VmadotSignedness::SignedUnsigned,
        _ => VmadotSignedness::SignedSigned,
    }
}

/// 按 `signed` 把一个字节零扩展或符号扩展为 i32。
fn byte_to_i32(value: u8, signed: bool) -> i32 {
    if signed {
        (value as i8) as i32
    } else {
        value as i32
    }
}

impl VmadotSignedness {
    /// 左操作数是否按有符号解释。
    fn lhs_signed(self) -> bool {
        matches!(
            self,
            VmadotSignedness::SignedSigned | VmadotSignedness::SignedUnsigned
        )
    }

    /// 右操作数是否按有符号解释。
    fn rhs_signed(self) -> bool {
        matches!(
            self,
            VmadotSignedness::SignedSigned | VmadotSignedness::UnsignedSigned
        )
    }
}

/// matmul 各执行路径（CPU/x100/a100、转置、分块边界）的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendTensorView;
    use alloc::vec;
    use alloc::vec::Vec;
    use k3_ai_uabi::{AttrByteSize, ByteSize, KernelOp, OpFlags, TensorCount};

    /// 用只读切片构造一个 tensor view。
    fn tensor_view<T>(slice: &[T], dtype: AiDtype) -> BackendTensorView {
        BackendTensorView {
            data: slice.as_ptr() as *mut u8,
            byte_len: ByteSize::new(core::mem::size_of_val(slice) as u64),
            dtype,
            ..BackendTensorView::default()
        }
    }

    /// 用可写切片构造一个 tensor view。
    fn tensor_view_mut<T>(slice: &mut [T], dtype: AiDtype) -> BackendTensorView {
        BackendTensorView {
            data: slice.as_mut_ptr().cast::<u8>(),
            byte_len: ByteSize::new(core::mem::size_of_val(slice) as u64),
            dtype,
            ..BackendTensorView::default()
        }
    }

    /// 把一个 `Copy` attr 的原始字节暴露成切片。
    fn attr_bytes<T: Copy>(attr: &T) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts((attr as *const T).cast::<u8>(), core::mem::size_of::<T>())
        }
    }

    /// 构造行主序、无转置、单 batch 的 `MatMulAttr`。
    fn row_major_attr(m: u32, n: u32, k: u32, accum_dtype: AiDtype) -> MatMulAttr {
        MatMulAttr {
            m: DimSize::new(m),
            n: DimSize::new(n),
            k: DimSize::new(k),
            batch: DimSize::new(0),
            lhs_row_stride: ElemStride::new(k),
            lhs_col_stride: ElemStride::new(1),
            lhs_batch_stride: ElemStride::new(0),
            rhs_row_stride: ElemStride::new(n),
            rhs_col_stride: ElemStride::new(1),
            rhs_batch_stride: ElemStride::new(0),
            out_row_stride: ElemStride::new(n),
            out_col_stride: ElemStride::new(1),
            out_batch_stride: ElemStride::new(0),
            flags: OpFlags::new(0),
            accum_dtype,
            reserved: [0; 3],
        }
    }

    /// 朴素三重循环的 int8→i32 matmul 参考实现，用于对拍分块结果。
    fn reference_i8_matmul(
        lhs: &[u8],
        rhs: &[u8],
        attr: &MatMulAttr,
        signedness: VmadotSignedness,
    ) -> Vec<i32> {
        let m = dim(attr.m);
        let n = dim(attr.n);
        let k = dim(attr.k);
        let mut out = vec![0_i32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0_i32;
                for p in 0..k {
                    let lhs_idx = lhs_index(attr, 0, i, p);
                    let rhs_idx = rhs_index(attr, 0, p, j);
                    let lhs = byte_to_i32(lhs[lhs_idx], signedness.lhs_signed());
                    let rhs = byte_to_i32(rhs[rhs_idx], signedness.rhs_signed());
                    sum = sum.wrapping_add(lhs.wrapping_mul(rhs));
                }
                out[i * n + j] = sum;
            }
        }

        out
    }

    /// 用固定 2×3·3×2 的 f32 数据跑一次完整 `matmul_caller`，返回输出。
    fn run_f32_backend_call(target: AiTargetHint) -> [f32; 4] {
        let attr = row_major_attr(2, 2, 3, AiDtype::F32);
        let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0_f32; 4];

        let inputs = [
            tensor_view(&lhs, AiDtype::F32),
            tensor_view(&rhs, AiDtype::F32),
        ];
        let mut outputs = [tensor_view_mut(&mut out, AiDtype::F32)];
        let attr = attr_bytes(&attr);
        let call = BackendCall {
            op: KernelOp::MAT_MUL,
            target: target.0,
            inputs: inputs.as_ptr(),
            input_count: TensorCount::new(inputs.len() as u32),
            outputs: outputs.as_mut_ptr(),
            output_count: TensorCount::new(outputs.len() as u32),
            attr: attr.as_ptr(),
            attr_size: AttrByteSize::new(attr.len() as u32),
        };

        unsafe { matmul_caller(&call) }.unwrap();
        out
    }

    /// AUTO 与 PREFER_CPU 目标的 f32 matmul 应得到相同的已知结果。
    #[test]
    fn backend_call_cpu_and_auto_f32_matmul() {
        let expected = [58.0_f32, 64.0, 139.0, 154.0];

        assert_eq!(run_f32_backend_call(AiTargetHint::AUTO), expected);
        assert_eq!(run_f32_backend_call(AiTargetHint::PREFER_CPU), expected);
    }

    /// x100 分块在 M/N/K 均不整除 tile 尺寸时应产生与参考实现相同的结果。
    #[test]
    fn x100_int8_tiling_handles_partial_edges() {
        let attr = row_major_attr(5, 6, 9, AiDtype::I32);
        let lhs: Vec<u8> = (0..45).map(|v| (v as i8 - 22) as u8).collect();
        let rhs: Vec<u8> = (0..54).map(|v| (17 - v as i8) as u8).collect();
        let mut out = vec![0_i32; 30];
        let expected = reference_i8_matmul(&lhs, &rhs, &attr, VmadotSignedness::SignedSigned);

        let parameter = Int8MatmulParameter {
            lhs: &lhs,
            rhs: &rhs,
            out: &mut out,
            attr,
            signedness: VmadotSignedness::SignedSigned,
        };

        ime_int8_i32_matmul(parameter, X100_INT8_TILE).unwrap();
        assert_eq!(out, expected);
    }

    /// a100 分块在 unsigned×signed 组合下应与参考实现一致。
    #[test]
    fn a100_int8_tiling_matches_unsigned_signed_reference() {
        let attr = row_major_attr(6, 5, 10, AiDtype::I32);
        let lhs: Vec<u8> = (0..60).map(|v| (v * 3 % 251) as u8).collect();
        let rhs: Vec<u8> = (0..50).map(|v| (v as i8 - 25) as u8).collect();
        let mut out = vec![0_i32; 30];
        let expected = reference_i8_matmul(&lhs, &rhs, &attr, VmadotSignedness::UnsignedSigned);

        let parameter = Int8MatmulParameter {
            lhs: &lhs,
            rhs: &rhs,
            out: &mut out,
            attr,
            signedness: VmadotSignedness::UnsignedSigned,
        };

        ime_int8_i32_matmul(parameter, A100_SILICON_INT8_TILE).unwrap();
        assert_eq!(out, expected);
    }

    /// 分块实现应正确处理转置标志与自定义 stride。
    #[test]
    fn int8_tiling_honors_transpose_flags_and_strides() {
        let mut attr = row_major_attr(3, 4, 5, AiDtype::I32);
        attr.flags = OpFlags::new(MATMUL_LHS_TRANSPOSED | MATMUL_RHS_TRANSPOSED);
        attr.lhs_row_stride = ElemStride::new(4);
        attr.lhs_col_stride = ElemStride::new(1);
        attr.rhs_row_stride = ElemStride::new(6);
        attr.rhs_col_stride = ElemStride::new(1);
        attr.out_row_stride = ElemStride::new(4);

        let mut lhs = vec![0_u8; 5 * 4];
        for depth in 0..5 {
            for row in 0..3 {
                lhs[depth * 4 + row] = (depth as i8 - row as i8 * 2) as u8;
            }
        }

        let mut rhs = vec![0_u8; 4 * 6];
        for col in 0..4 {
            for depth in 0..5 {
                rhs[col * 6 + depth] = (col as i8 * 3 - depth as i8) as u8;
            }
        }

        let mut out = vec![0_i32; 3 * 4];
        let expected = reference_i8_matmul(&lhs, &rhs, &attr, VmadotSignedness::SignedSigned);
        let parameter = Int8MatmulParameter {
            lhs: &lhs,
            rhs: &rhs,
            out: &mut out,
            attr,
            signedness: VmadotSignedness::SignedSigned,
        };

        ime_int8_i32_matmul(parameter, X100_INT8_TILE).unwrap();
        assert_eq!(out, expected);
    }
}
