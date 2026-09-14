//! FP16 单输入逐元素算子。

use crate::call::CallContext;
use crate::{BackendCall, ComputeKernel};
use alloc::vec;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, KernelOp, UnaryAttr};

/// 单输入算子的计算语义。
#[derive(Clone, Copy)]
enum UnaryKind {
    /// SiLU。
    Silu,
    /// Sigmoid。
    Sigmoid,
    /// 仿射缩放。
    Scale,
}

/// SiLU 静态分发标记。
pub(crate) struct SiluKernel;
/// Sigmoid 静态分发标记。
pub(crate) struct SigmoidKernel;
/// Scale 静态分发标记。
pub(crate) struct ScaleKernel;

/// Defines a marker that statically routes one unary operator.
macro_rules! impl_kernel {
    ($marker:ident, $op:expr, $kind:expr) => {
        impl ComputeKernel for $marker {
            const OP: KernelOp = $op;

            unsafe fn call(call: *const BackendCall) -> Result<(), BackendErr> {
                unsafe { call_unary(call, $kind) }
            }
        }
    };
}

impl_kernel!(SiluKernel, KernelOp::SILU, UnaryKind::Silu);
impl_kernel!(SigmoidKernel, KernelOp::SIGMOID, UnaryKind::Sigmoid);
impl_kernel!(ScaleKernel, KernelOp::SCALE, UnaryKind::Scale);

/// 解析 ABI，校验 F16 tensor，并按 target 分发。
unsafe fn call_unary(call: *const BackendCall, kind: UnaryKind) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;
    if ctx.inputs[0].dtype != AiDtype::F16 || ctx.outputs[0].dtype != AiDtype::F16 {
        return Err(BackendErr::UnsupportedDtype);
    }
    let attr = ctx.read_attr::<UnaryAttr>()?;
    let input = unsafe { ctx.inputs[0].as_slice::<u16>()? };
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
    if input.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }
    match ctx.target {
        AiTargetHint::PREFER_CPU => compute_f16_cpu(kind, attr, input, output),
        AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100 => {
            compute_f16_rvv(kind, attr, input, output)
        }
        _ => unreachable!("CallContext normalizes and validates target hints"),
    }
}

/// FP16 软件参考实现。
fn compute_f16_cpu(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[u16],
    output: &mut [u16],
) -> Result<(), BackendErr> {
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = f16::from_f32(apply(kind, attr, f16::from_bits(*source).to_f32())).to_bits();
    }
    Ok(())
}

/// FP16 RVV 路径；逐元素算子不匹配 IME 的矩阵乘加语义。
fn compute_f16_rvv(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[u16],
    output: &mut [u16],
) -> Result<(), BackendErr> {
    if !matches!(kind, UnaryKind::Sigmoid) {
        return compute_f16_cpu(kind, attr, input, output);
    }
    let decoded = input
        .iter()
        .map(|bits| f16::from_bits(*bits).to_f32())
        .collect::<alloc::vec::Vec<_>>();
    let mut computed = vec![0.0_f32; decoded.len()];
    match kind {
        UnaryKind::Sigmoid => crate::rvv::f16_sigmoid_work(&decoded, &mut computed)?,
        UnaryKind::Silu => {
            let mut sigmoid = vec![0.0_f32; decoded.len()];
            crate::rvv::f16_sigmoid_work(&decoded, &mut sigmoid)?;
            crate::rvv::f16_binary_work(
                crate::rvv::BinaryOp::Mul,
                &decoded,
                &sigmoid,
                &mut computed,
            )?;
        }
        UnaryKind::Scale => {
            crate::rvv::f16_affine_work(&decoded, &mut computed, attr.alpha, attr.beta)?;
        }
    }
    for (destination, value) in output.iter_mut().zip(computed) {
        *destination = f16::from_f32(value).to_bits();
    }
    Ok(())
}

/// 计算一个 FP32 累加域的 unary 值。
fn apply(kind: UnaryKind, attr: UnaryAttr, value: f32) -> f32 {
    match kind {
        UnaryKind::Silu => value * sigmoid(value),
        UnaryKind::Sigmoid => sigmoid(value),
        UnaryKind::Scale => attr.alpha * value + attr.beta,
    }
}

/// 数值稳定的 sigmoid。
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FP16 SiLU 应与 F32 参考值保持半精度误差内一致。
    #[test]
    fn f16_silu_matches_reference() {
        let values = [-2.0_f32, 0.5, 4.0];
        let input = values.map(|value| f16::from_f32(value).to_bits());
        let mut output = [0_u16; 3];
        compute_f16_cpu(UnaryKind::Silu, UnaryAttr::default(), &input, &mut output).unwrap();
        for (value, output) in values.iter().zip(output) {
            let expected = *value / (1.0 + libm::expf(-*value));
            assert!((f16::from_bits(output).to_f32() - expected).abs() < 1e-2);
        }
    }

    /// A100 路径应对每种 unary 语义使用 RVV 原语，并保留 FP16 可表示精度。
    #[test]
    fn f16_rvv_unary_matches_reference() {
        let values = [-2.0_f32, 0.5, 4.0];
        let input = values.map(|value| f16::from_f32(value).to_bits());
        let scale = UnaryAttr {
            alpha: 1.5,
            beta: -0.25,
            ..UnaryAttr::default()
        };

        for (kind, attr) in [
            (UnaryKind::Silu, UnaryAttr::default()),
            (UnaryKind::Sigmoid, UnaryAttr::default()),
            (UnaryKind::Scale, scale),
        ] {
            let mut output = [0_u16; 3];
            compute_f16_rvv(kind, attr, &input, &mut output).unwrap();
            for (output, expected) in output
                .into_iter()
                .zip(values.map(|value| apply(kind, attr, value)))
            {
                assert!((f16::from_bits(output).to_f32() - expected).abs() < 2.0e-3);
            }
        }
    }
}
