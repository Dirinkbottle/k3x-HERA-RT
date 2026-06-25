//! 矩阵乘法算子。

use core::default::Default;
use core::fmt::Debug;
use core::marker::Copy;
use core::mem::size_of;
use core::ops::{Add, Mul};
use core::slice;

use crate::BackendCall;
use k3_aiUabi::error::BackendErr;
use k3_aiUabi::{AiDtype, MatMulAttr};
use log::error;

/// matmul 算子的输入输出参数集合，供 a100/x100/cpu 分发使用。
struct MatmulParameter<'a, T> {
    lhs: &'a [T],
    rhs: &'a [T],
    out: &'a mut [T],
    attr: MatMulAttr,
}

/// matmul 算子执行器，将 `BackendCall` 解析为内部参数后按 dtype 分发到具体实现。
pub(crate) unsafe fn matmul_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    if call.is_null() {
        error!("matmul_caller: call is null");
        return Err(BackendErr::NullPointer);
    }

    let call = unsafe { &*call };
    if call.input_count != 2 || call.output_count != 1 {
        error!(
            "matmul_caller: invalid input/output count, input_count={}, output_count={}",
            call.input_count, call.output_count
        );
        return Err(BackendErr::InvalidInput);
    }
    if call.inputs.is_null() || call.outputs.is_null() {
        error!("matmul_caller: inputs or outputs is null");
        return Err(BackendErr::NullPointer);
    }

    let inputs = unsafe { slice::from_raw_parts(call.inputs, call.input_count as usize) };
    let outputs = unsafe { slice::from_raw_parts_mut(call.outputs, call.output_count as usize) };

    if inputs[0].data.is_null() || inputs[1].data.is_null() || outputs[0].data.is_null() {
        error!("matmul_caller: tensor data is null");
        return Err(BackendErr::InvalidTensor);
    }

    if call.attr.is_null() || call.attr_size < size_of::<MatMulAttr>() as u32 {
        error!(
            "matmul_caller: invalid attr, is_null={}, attr_size={}",
            call.attr.is_null(),
            call.attr_size
        );
        return Err(BackendErr::InvalidAttr);
    }

    let attr = unsafe { core::ptr::read_unaligned(call.attr.cast::<MatMulAttr>()) };

    // 目前只支持 F32
    if inputs[0].dtype != AiDtype::F32
        || inputs[1].dtype != AiDtype::F32
        || outputs[0].dtype != AiDtype::F32
    {
        error!(
            "matmul_caller: unsupported dtype, input0={:?}, input1={:?}, output={:?}",
            inputs[0].dtype, inputs[1].dtype, outputs[0].dtype
        );
        return Err(BackendErr::UnsupportedDtype);
    }

    let parameter = MatmulParameter {
        lhs: unsafe {
            slice::from_raw_parts(
                inputs[0].data.cast::<f32>(),
                inputs[0].byte_len as usize / size_of::<f32>(),
            )
        },
        rhs: unsafe {
            slice::from_raw_parts(
                inputs[1].data.cast::<f32>(),
                inputs[1].byte_len as usize / size_of::<f32>(),
            )
        },
        out: unsafe {
            slice::from_raw_parts_mut(
                outputs[0].data.cast::<f32>(),
                outputs[0].byte_len as usize / size_of::<f32>(),
            )
        },
        attr,
    };

    match call.target {
        1 => cpu(parameter),
        2 => x100(parameter),
        3 => a100(parameter),
        _ => Err(BackendErr::UnsupportedOp),
    }
}

/// A100 加速器 matmul 实现。
fn a100<T>(_parameter: MatmulParameter<'_, T>) -> Result<(), BackendErr> {
    Err(BackendErr::UnsupportedOp)
}

/// X100 加速器 matmul 实现。
fn x100<T>(_parameter: MatmulParameter<'_, T>) -> Result<(), BackendErr> {
    Err(BackendErr::UnsupportedOp)
}

/// CPU fallback matmul 实现。
fn cpu<T>(parameter: MatmulParameter<'_, T>) -> Result<(), BackendErr>
where
    T: Debug + Default + Copy + Add<Output = T> + Mul<Output = T>,
{
    error!("run mutmal in cpu");

    let attr = &parameter.attr;
    let m = attr.m as usize;
    let n = attr.n as usize;
    let k = attr.k as usize;
    let batch = if attr.batch == 0 {
        1
    } else {
        attr.batch as usize
    };

    let lhs_row_stride = attr.lhs_row_stride as usize;
    let lhs_col_stride = attr.lhs_col_stride as usize;
    let rhs_row_stride = attr.rhs_row_stride as usize;
    let rhs_col_stride = attr.rhs_col_stride as usize;
    let out_row_stride = attr.out_row_stride as usize;
    let out_col_stride = attr.out_col_stride as usize;

    let lhs_batch_stride = attr.lhs_batch_stride as usize;
    let rhs_batch_stride = attr.rhs_batch_stride as usize;
    let out_batch_stride = attr.out_batch_stride as usize;

    for b in 0..batch {
        let lhs_base = b * lhs_batch_stride;
        let rhs_base = b * rhs_batch_stride;
        let out_base = b * out_batch_stride;

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for p in 0..k {
                    let lhs_idx = lhs_base + i * lhs_row_stride + p * lhs_col_stride;
                    let rhs_idx = rhs_base + p * rhs_row_stride + j * rhs_col_stride;
                    sum = sum + parameter.lhs[lhs_idx] * parameter.rhs[rhs_idx];
                }
                let out_idx = out_base + i * out_row_stride + j * out_col_stride;
                parameter.out[out_idx] = sum;
            }
        }
    }

    ///TODO: 后面必须实现写回同步

    log::error!("[kernel backend matmul: log]:");
    log::error!("  shape: {}x{} @ {}x{} -> {}x{}", m, k, k, n, m, n);
    log::error!("  batch: {}", batch);

    // 打印计算结果
    log::error!("  cpu matmul output ({}x{}):", m, n);
    for i in 0..m {
        let mut row_str = alloc::format!("    [");
        for j in 0..n {
            let out_idx = i * out_row_stride + j * out_col_stride;
            row_str.push_str(&alloc::format!(" {:?}", parameter.out[out_idx]));
        }
        row_str.push_str(" ]");
        log::error!("{}", row_str);
    }

    Ok(())
}
