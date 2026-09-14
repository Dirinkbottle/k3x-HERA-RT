//! FP16 矩阵乘法算子。

use core::cmp::min;

use crate::call::CallContext;
use crate::{BackendCall, ComputeKernel};
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, DimSize, ElemStride, KernelOp, MatMulAttr};

/// 左操作数转置标志。
const MATMUL_LHS_TRANSPOSED: u32 = 1 << 0;
/// 右操作数转置标志。
const MATMUL_RHS_TRANSPOSED: u32 = 1 << 1;
/// A100 FP16 IME 的固定 tile 边长。
pub(crate) const A100_FP16_TILE: usize = 8;
/// 一个输入或累加 tile 的元素数。
const A100_FP16_TILE_ELEMS: usize = A100_FP16_TILE * A100_FP16_TILE;

/// MatMul 的静态分发标记。
pub(crate) struct MatMulKernel;

impl ComputeKernel for MatMulKernel {
    const OP: KernelOp = KernelOp::MAT_MUL;

    unsafe fn call(call: *const BackendCall) -> Result<(), BackendErr> {
        unsafe { call_matmul(call) }
    }
}

/// FP16 MatMul 执行参数。
pub(crate) struct F16Matmul<'a> {
    /// 左矩阵。
    pub(crate) lhs: &'a [u16],
    /// 右矩阵。
    pub(crate) rhs: &'a [u16],
    /// 输出矩阵。
    pub(crate) output: &'a mut [u16],
    /// 维度、stride 和转置属性。
    pub(crate) attr: MatMulAttr,
}

/// 解析 ABI 并选择 FP16 target 实现。
unsafe fn call_matmul(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;
    if ctx.inputs[0].dtype != AiDtype::F16
        || ctx.inputs[1].dtype != AiDtype::F16
        || ctx.outputs[0].dtype != AiDtype::F16
    {
        return Err(BackendErr::UnsupportedDtype);
    }

    let attr = ctx.read_attr::<MatMulAttr>()?;
    if attr.accum_dtype != AiDtype::F32 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let lhs = unsafe { ctx.inputs[0].as_slice::<u16>()? };
    let rhs = unsafe { ctx.inputs[1].as_slice::<u16>()? };
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
    validate_bounds(&attr, lhs.len(), rhs.len(), output.len())?;
    let parameter = F16Matmul {
        lhs,
        rhs,
        output,
        attr,
    };

    match ctx.target {
        AiTargetHint::PREFER_CPU => compute_f16_cpu(parameter),
        AiTargetHint::PREFER_A100 => compute_f16_a100(parameter),
        AiTargetHint::PREFER_X100 => Err(BackendErr::UnsupportedOp),
        _ => unreachable!("CallContext normalizes and validates target hints"),
    }
}

/// 用 FP32 累加的 FP16 软件参考 MatMul。
pub(crate) fn compute_f16_cpu(parameter: F16Matmul<'_>) -> Result<(), BackendErr> {
    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    for batch in 0..normalized_batch(attr) {
        let lhs_base = batch * elem_stride(attr.lhs_batch_stride);
        let rhs_base = batch * elem_stride(attr.rhs_batch_stride);
        let out_base = batch * elem_stride(attr.out_batch_stride);
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0_f32;
                for depth in 0..k {
                    let lhs = f16::from_bits(parameter.lhs[lhs_index(attr, lhs_base, row, depth)])
                        .to_f32();
                    let rhs = f16::from_bits(parameter.rhs[rhs_index(attr, rhs_base, depth, col)])
                        .to_f32();
                    sum += lhs * rhs;
                }
                parameter.output[out_base
                    + row * elem_stride(attr.out_row_stride)
                    + col * elem_stride(attr.out_col_stride)] = f16::from_f32(sum).to_bits();
            }
        }
    }
    Ok(())
}

/// 使用 A100 `smt.vfwmadot` tile 语义执行 FP16 MatMul。
///
/// 非 RISC-V 构建使用等价的软件镜像，使 host 测试覆盖同一条 A100 路由。
pub(crate) fn compute_f16_a100(parameter: F16Matmul<'_>) -> Result<(), BackendErr> {
    let attr = &parameter.attr;
    let m = dim(attr.m);
    let n = dim(attr.n);
    let k = dim(attr.k);
    for batch in 0..normalized_batch(attr) {
        let lhs_base = batch * elem_stride(attr.lhs_batch_stride);
        let rhs_base = batch * elem_stride(attr.rhs_batch_stride);
        let out_base = batch * elem_stride(attr.out_batch_stride);
        let mut row = 0;
        while row < m {
            let valid_m = min(A100_FP16_TILE, m - row);
            let mut col = 0;
            while col < n {
                let valid_n = min(A100_FP16_TILE, n - col);
                let mut acc = [0.0_f32; A100_FP16_TILE_ELEMS];
                let mut depth = 0;
                while depth < k {
                    let valid_k = min(A100_FP16_TILE, k - depth);
                    let mut lhs_tile = [0_u16; A100_FP16_TILE_ELEMS];
                    let mut rhs_tile = [0_u16; A100_FP16_TILE_ELEMS];
                    let mut partial = [0.0_f32; A100_FP16_TILE_ELEMS];
                    pack_f16_tile(
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
                    for tile_row in 0..valid_m {
                        for tile_col in 0..valid_n {
                            let index = tile_row * A100_FP16_TILE + tile_col;
                            acc[index] += partial[index];
                        }
                    }
                    depth += A100_FP16_TILE;
                }
                for tile_row in 0..valid_m {
                    for tile_col in 0..valid_n {
                        parameter.output[out_base
                            + (row + tile_row) * elem_stride(attr.out_row_stride)
                            + (col + tile_col) * elem_stride(attr.out_col_stride)] =
                            f16::from_f32(acc[tile_row * A100_FP16_TILE + tile_col]).to_bits();
                    }
                }
                col += A100_FP16_TILE;
            }
            row += A100_FP16_TILE;
        }
    }
    Ok(())
}

/// 按 A100 所需布局打包一个 `A[M,K]` 和 `B[N,K]` FP16 tile。
#[allow(clippy::too_many_arguments)]
fn pack_f16_tile(
    parameter: &F16Matmul<'_>,
    lhs_base: usize,
    rhs_base: usize,
    row: usize,
    col: usize,
    depth: usize,
    valid_m: usize,
    valid_n: usize,
    valid_k: usize,
    lhs_tile: &mut [u16; A100_FP16_TILE_ELEMS],
    rhs_tile: &mut [u16; A100_FP16_TILE_ELEMS],
) {
    for tile_row in 0..valid_m {
        for tile_depth in 0..valid_k {
            lhs_tile[tile_row * A100_FP16_TILE + tile_depth] = parameter.lhs[lhs_index(
                &parameter.attr,
                lhs_base,
                row + tile_row,
                depth + tile_depth,
            )];
        }
    }
    for tile_col in 0..valid_n {
        for tile_depth in 0..valid_k {
            rhs_tile[tile_col * A100_FP16_TILE + tile_depth] = parameter.rhs[rhs_index(
                &parameter.attr,
                rhs_base,
                depth + tile_depth,
                col + tile_col,
            )];
        }
    }
}

/// 执行单个 FP16 A100 tile。
fn vfwmadot_tile(
    lhs_tile: &[u16; A100_FP16_TILE_ELEMS],
    rhs_tile: &[u16; A100_FP16_TILE_ELEMS],
    output: &mut [f32; A100_FP16_TILE_ELEMS],
) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        vfwmadot_tile_hw(lhs_tile.as_ptr(), rhs_tile.as_ptr(), output.as_mut_ptr());
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    vfwmadot_tile_sw(lhs_tile, rhs_tile, output);
}

/// `smt.vfwmadot` 的软件镜像，供非 RISC-V host 测试使用。
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn vfwmadot_tile_sw(
    lhs_tile: &[u16; A100_FP16_TILE_ELEMS],
    rhs_tile: &[u16; A100_FP16_TILE_ELEMS],
    output: &mut [f32; A100_FP16_TILE_ELEMS],
) {
    for row in 0..A100_FP16_TILE {
        for col in 0..A100_FP16_TILE {
            let mut sum = 0.0_f32;
            for depth in 0..A100_FP16_TILE {
                sum += f16::from_bits(lhs_tile[row * A100_FP16_TILE + depth]).to_f32()
                    * f16::from_bits(rhs_tile[col * A100_FP16_TILE + depth]).to_f32();
            }
            output[row * A100_FP16_TILE + col] = sum;
        }
    }
}

/// 硬件 `smt.vfwmadot v16, v2, v8` 指令封装。
///
/// # Safety
///
/// 调用方须在已启用 vector/IME 且设为 FP16 模式的 A100 上执行。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
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
        );
    }
}

/// 校验按 attr 的 batch、转置和 stride 访问的区域完全落在三个 tensor 内。
fn validate_bounds(
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
    if region_fits(
        lhs_len,
        batch,
        lhs_rows,
        lhs_cols,
        elem_stride(attr.lhs_row_stride),
        elem_stride(attr.lhs_col_stride),
        elem_stride(attr.lhs_batch_stride),
    ) && region_fits(
        rhs_len,
        batch,
        rhs_rows,
        rhs_cols,
        elem_stride(attr.rhs_row_stride),
        elem_stride(attr.rhs_col_stride),
        elem_stride(attr.rhs_batch_stride),
    ) && region_fits(
        out_len,
        batch,
        m,
        n,
        elem_stride(attr.out_row_stride),
        elem_stride(attr.out_col_stride),
        elem_stride(attr.out_batch_stride),
    ) {
        Ok(())
    } else {
        Err(BackendErr::InvalidTensor)
    }
}

/// 判断 strided tensor 的最大访问下标是否落在 `len` 内。
fn region_fits(
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
    (batch - 1)
        .checked_mul(batch_stride)
        .and_then(|value| value.checked_add((rows - 1).checked_mul(row_stride)?))
        .and_then(|value| value.checked_add((cols - 1).checked_mul(col_stride)?))
        .is_some_and(|index| index < len)
}

/// 将维度 ABI 类型转为索引。
fn dim(value: DimSize) -> usize {
    value.get() as usize
}

/// 将元素 stride ABI 类型转为索引。
fn elem_stride(value: ElemStride) -> usize {
    value.get() as usize
}

/// 将 `batch=0` 归一化为单 batch。
fn normalized_batch(attr: &MatMulAttr) -> usize {
    if attr.batch == 0 { 1 } else { dim(attr.batch) }
}

/// 计算左矩阵元素索引。
fn lhs_index(attr: &MatMulAttr, base: usize, row: usize, depth: usize) -> usize {
    if lhs_transposed(attr) {
        base + depth * elem_stride(attr.lhs_row_stride) + row * elem_stride(attr.lhs_col_stride)
    } else {
        base + row * elem_stride(attr.lhs_row_stride) + depth * elem_stride(attr.lhs_col_stride)
    }
}

/// 计算右矩阵元素索引。
fn rhs_index(attr: &MatMulAttr, base: usize, depth: usize, col: usize) -> usize {
    if rhs_transposed(attr) {
        base + col * elem_stride(attr.rhs_row_stride) + depth * elem_stride(attr.rhs_col_stride)
    } else {
        base + depth * elem_stride(attr.rhs_row_stride) + col * elem_stride(attr.rhs_col_stride)
    }
}

/// 左矩阵是否转置。
fn lhs_transposed(attr: &MatMulAttr) -> bool {
    attr.flags.get() & MATMUL_LHS_TRANSPOSED != 0
}

/// 右矩阵是否转置。
fn rhs_transposed(attr: &MatMulAttr) -> bool {
    attr.flags.get() & MATMUL_RHS_TRANSPOSED != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_ai_uabi::{AiTensorLayout, AttrByteSize, ByteSize, OpFlags, TensorCount};

    /// 构造紧凑的单 batch matmul attr。
    fn attr(m: u32, n: u32, k: u32) -> MatMulAttr {
        MatMulAttr {
            m: DimSize::new(m),
            n: DimSize::new(n),
            k: DimSize::new(k),
            batch: DimSize::new(1),
            lhs_row_stride: ElemStride::new(k),
            lhs_col_stride: ElemStride::new(1),
            lhs_batch_stride: ElemStride::new(m * k),
            rhs_row_stride: ElemStride::new(n),
            rhs_col_stride: ElemStride::new(1),
            rhs_batch_stride: ElemStride::new(k * n),
            out_row_stride: ElemStride::new(n),
            out_col_stride: ElemStride::new(1),
            out_batch_stride: ElemStride::new(m * n),
            flags: OpFlags::new(0),
            accum_dtype: AiDtype::F32,
            reserved: [0; 3],
        }
    }

    /// CPU 与 A100 软件镜像的 FP16 结果必须一致。
    #[test]
    fn a100_f16_matches_cpu() {
        let lhs = [1.0_f32, -2.0, 3.0, 4.0, 0.5, -1.0];
        let rhs = [2.0_f32, 1.0, -1.0, 3.0, 0.5, -2.0];
        let lhs = lhs.map(|value| f16::from_f32(value).to_bits());
        let rhs = rhs.map(|value| f16::from_f32(value).to_bits());
        let mut cpu = [0_u16; 4];
        let mut a100 = [0_u16; 4];
        compute_f16_cpu(F16Matmul {
            lhs: &lhs,
            rhs: &rhs,
            output: &mut cpu,
            attr: attr(2, 2, 3),
        })
        .unwrap();
        compute_f16_a100(F16Matmul {
            lhs: &lhs,
            rhs: &rhs,
            output: &mut a100,
            attr: attr(2, 2, 3),
        })
        .unwrap();
        assert_eq!(cpu, a100);
    }

    /// F32 和量化 dtype 不能进入 FP16 MatMul 路径。
    #[test]
    fn caller_rejects_non_f16_tensors() {
        let mut f32_data = [1.0_f32; 1];
        let mut f16_data = [f16::from_f32(1.0).to_bits(); 1];
        let attr = attr(1, 1, 1);
        for dtype in [AiDtype::F32, AiDtype::I8, AiDtype::Q8_0] {
            let inputs = [
                tensor_view(f32_data.as_mut_ptr().cast(), 4, dtype),
                tensor_view(f16_data.as_mut_ptr().cast(), 2, AiDtype::F16),
            ];
            let mut outputs = [tensor_view(f16_data.as_mut_ptr().cast(), 2, AiDtype::F16)];
            let call = BackendCall {
                op: KernelOp::MAT_MUL,
                target: AiTargetHint::PREFER_CPU.0,
                inputs: inputs.as_ptr(),
                input_count: TensorCount::new(2),
                outputs: outputs.as_mut_ptr(),
                output_count: TensorCount::new(1),
                attr: (&attr as *const MatMulAttr).cast(),
                attr_size: AttrByteSize::new(core::mem::size_of::<MatMulAttr>() as u32),
            };
            assert_eq!(
                unsafe { <MatMulKernel as ComputeKernel>::call(&call) },
                Err(BackendErr::UnsupportedDtype)
            );
        }
    }

    /// 构造仅供 dtype 拒绝测试使用的最小 tensor view。
    fn tensor_view(data: *mut u8, byte_len: u64, dtype: AiDtype) -> crate::BackendTensorView {
        crate::BackendTensorView {
            data,
            byte_len: ByteSize::new(byte_len),
            dtype,
            layout: AiTensorLayout::DENSE,
            ..crate::BackendTensorView::default()
        }
    }
}
