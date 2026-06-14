//! 矩阵乘法算子。

use core::slice;

use crate::BackendCall;

/// matmul 算子执行器，将 `BackendCall` 解析为内部参数后按 dtype 分发到具体实现。
pub(crate) unsafe fn matmul_caller(call: *const BackendCall) -> i32 {
    if call.is_null() {
        return -1;
    }

    let call = unsafe { &*call };
    if call.input_count != 2 || call.output_count != 1 {
        return -2;
    }
    if call.inputs.is_null() || call.outputs.is_null() {
        return -3;
    }

    let inputs = unsafe { slice::from_raw_parts(call.inputs, call.input_count as usize) };
    let outputs = unsafe { slice::from_raw_parts_mut(call.outputs, call.output_count as usize) };

    if inputs[0].data.is_null() || inputs[1].data.is_null() || outputs[0].data.is_null() {
        return -4;
    }

    if inputs[0].dtype != inputs[1].dtype || inputs[0].dtype != outputs[0].dtype {
        return -6;
    }

    match inputs[0].dtype {
        0 => {
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
            };

            match call.target {
                1 => cpu(parameter),
                2 => x100(parameter),
                3 => a100(parameter),
                _ => -5,
            }
        }
        _ => -7,
    }
}

/// matmul 算子的输入输出参数集合，供 a100/x100/cpu 分发使用。
struct MatmulParameter<'a, T> {
    lhs: &'a [T],
    rhs: &'a [T],
    out: &'a mut [T],
}

/// A100 加速器 matmul 实现。
fn a100<T>(parameter: MatmulParameter<'_, T>) -> i32 {
    let _ = parameter;
    todo!()
}

/// X100 加速器 matmul 实现。
fn x100<T>(parameter: MatmulParameter<'_, T>) -> i32 {
    let _ = parameter;
    todo!()
}

/// CPU fallback matmul 实现。
fn cpu<T: Default + Copy>(parameter: MatmulParameter<'_, T>) -> i32 {
    // 阶段一先把 CPU fallback 的调用链打通，后面再按 dtype/attr 实现真实 MatMul。
    for value in parameter.out {
        *value = T::default();
    }
    0
}
