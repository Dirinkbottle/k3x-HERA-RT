//! 单输入 elementwise 算子（SILU / SIGMOID / SCALE）。
//!
//! 这三个算子共用 `UnaryAttr`，靠 `KernelOp` 区分语义，因此分发入口需要显式
//! 传入 `UnaryKind`。计算落点：逐元素运算属 RVV 向量单元（非 IME 矩阵引擎），
//! `cpu` 为纯标量软件实现，`x100`/`a100` 为 RVV 向量实现（当前先复用软件路径）。

use crate::BackendCall;
use crate::call::CallContext;
use crate::rvv;
use alloc::vec;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, UnaryAttr};
use log::error;

/// 单输入 elementwise 算子的语义种类。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UnaryKind {
    /// SiLU/Swish：`x * sigmoid(x)`。
    Silu,
    /// Sigmoid：`1 / (1 + exp(-x))`。
    Sigmoid,
    /// 仿射缩放：`alpha * x + beta`。
    Scale,
}

/// 单输入 elementwise 算子执行器：解析 `BackendCall`，按 dtype/target 分发。
///
/// # Safety
///
/// 与其他 caller 一致：`call` 指向有效 `BackendCall`，其 tensor `data` 已映射且
/// 生命周期覆盖本次调用。
pub(crate) unsafe fn unary_caller(
    call: *const BackendCall,
    kind: UnaryKind,
) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(1, 1)?;

    let attr = ctx.read_attr::<UnaryAttr>()?;
    let target = ctx.target;
    let in_dtype = ctx.inputs[0].dtype;
    let out_dtype = ctx.outputs[0].dtype;

    if in_dtype != out_dtype {
        error!(
            "unary_caller: input/output dtype mismatch, in={:?}, out={:?}",
            in_dtype, out_dtype
        );
        return Err(BackendErr::UnsupportedDtype);
    }

    match in_dtype {
        AiDtype::F32 => {
            let input = unsafe { ctx.inputs[0].as_slice::<f32>()? };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            dispatch_f32(kind, attr, input, output, target)
        }
        AiDtype::F16 => {
            let input = unsafe { ctx.inputs[0].as_slice::<u16>()? };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
            dispatch_f16(kind, attr, input, output, target)
        }
        _ => {
            error!("unary_caller: unsupported dtype {:?}", in_dtype);
            Err(BackendErr::UnsupportedDtype)
        }
    }
}

/// f32 路径按 target 路由。elementwise 无 IME 指令，x100/a100 走 RVV（暂复用软件）。
fn dispatch_f32(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[f32],
    output: &mut [f32],
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        error!(
            "unary: length mismatch, input={}, output={}",
            input.len(),
            output.len()
        );
        return Err(BackendErr::InvalidTensor);
    }
    match target {
        AiTargetHint::AUTO | AiTargetHint::PREFER_CPU => cpu_f32(kind, attr, input, output),
        AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100 => rvv_f32(kind, attr, input, output),
        _ => unreachable!("CallContext rejects unknown targets"),
    }
}

/// f16 路径按 target 路由；计算在 f32 域完成后写回 f16。
fn dispatch_f16(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[u16],
    output: &mut [u16],
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        error!(
            "unary: length mismatch, input={}, output={}",
            input.len(),
            output.len()
        );
        return Err(BackendErr::InvalidTensor);
    }
    match target {
        AiTargetHint::AUTO | AiTargetHint::PREFER_CPU => cpu_f16(kind, attr, input, output),
        AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100 => rvv_f16(kind, attr, input, output),
        _ => unreachable!("CallContext rejects unknown targets"),
    }
}

/// 在 f32 域对单个标量施加对应的 unary 运算。
#[inline]
fn apply_scalar(kind: UnaryKind, attr: &UnaryAttr, x: f32) -> f32 {
    match kind {
        UnaryKind::Silu => x * sigmoid(x),
        UnaryKind::Sigmoid => sigmoid(x),
        UnaryKind::Scale => attr.alpha * x + attr.beta,
    }
}

/// 数值稳定的 sigmoid：`1 / (1 + exp(-x))`。
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// f32 纯软件实现（cpu = 纯软件路径）。
fn cpu_f32(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), BackendErr> {
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = apply_scalar(kind, &attr, src);
    }
    Ok(())
}

/// f16 纯软件实现：解码到 f32、计算、再编码回 f16。
fn cpu_f16(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[u16],
    output: &mut [u16],
) -> Result<(), BackendErr> {
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        let x = f16::from_bits(src).to_f32();
        *dst = f16::from_f32(apply_scalar(kind, &attr, x)).to_bits();
    }
    Ok(())
}

/// x100/a100 的真实 RVV F32 路径。
fn rvv_f32(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), BackendErr> {
    match kind {
        UnaryKind::Silu => rvv::silu_f32(input, output),
        UnaryKind::Sigmoid => rvv::sigmoid_f32(input, output),
        UnaryKind::Scale => rvv::affine_f32(input, output, attr.alpha, attr.beta),
    }
}

/// x100/a100 的 RVV f16 路径：half 编解码后在真实 F32 RVV 核中计算。
fn rvv_f16(
    kind: UnaryKind,
    attr: UnaryAttr,
    input: &[u16],
    output: &mut [u16],
) -> Result<(), BackendErr> {
    let decoded: alloc::vec::Vec<f32> = input
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();
    let mut computed = vec![0.0_f32; decoded.len()];
    rvv_f32(kind, attr, &decoded, &mut computed)?;
    for (dst, value) in output.iter_mut().zip(computed) {
        *dst = f16::from_f32(value).to_bits();
    }
    Ok(())
}

/// unary 算子的数值正确性单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 允许的浮点误差。
    const EPS: f32 = 1e-5;

    /// SiLU 参考值：`x * sigmoid(x)`。
    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + libm::expf(-x))
    }

    #[test]
    fn silu_f32_matches_reference() {
        let input = [-3.0_f32, -0.5, 0.0, 0.5, 3.0, 10.0];
        let mut output = [0.0_f32; 6];
        let attr = UnaryAttr::default();
        cpu_f32(UnaryKind::Silu, attr, &input, &mut output).unwrap();
        for (&x, &y) in input.iter().zip(output.iter()) {
            assert!((y - silu_ref(x)).abs() < EPS, "x={x}, got={y}");
        }
    }

    #[test]
    fn sigmoid_f32_bounds_and_midpoint() {
        let input = [0.0_f32, -100.0, 100.0];
        let mut output = [0.0_f32; 3];
        cpu_f32(
            UnaryKind::Sigmoid,
            UnaryAttr::default(),
            &input,
            &mut output,
        )
        .unwrap();
        assert!((output[0] - 0.5).abs() < EPS);
        assert!(output[1] < EPS);
        assert!((output[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn scale_f32_affine() {
        let input = [1.0_f32, 2.0, -1.0];
        let mut output = [0.0_f32; 3];
        let attr = UnaryAttr {
            alpha: 2.0,
            beta: 1.0,
            ..UnaryAttr::default()
        };
        cpu_f32(UnaryKind::Scale, attr, &input, &mut output).unwrap();
        assert_eq!(output, [3.0, 5.0, -1.0]);
    }

    #[test]
    fn silu_f16_close_to_f32() {
        let input_f32 = [-2.0_f32, 0.5, 4.0];
        let input: [u16; 3] = input_f32.map(|x| f16::from_f32(x).to_bits());
        let mut output = [0_u16; 3];
        cpu_f16(UnaryKind::Silu, UnaryAttr::default(), &input, &mut output).unwrap();
        for (&x, &bits) in input_f32.iter().zip(output.iter()) {
            let y = f16::from_bits(bits).to_f32();
            assert!((y - silu_ref(x)).abs() < 1e-2, "x={x}, got={y}");
        }
    }

    #[test]
    fn rvv_matches_cpu() {
        let input = [-1.5_f32, 0.3, 2.2];
        let mut a = [0.0_f32; 3];
        let mut b = [0.0_f32; 3];
        cpu_f32(UnaryKind::Silu, UnaryAttr::default(), &input, &mut a).unwrap();
        rvv_f32(UnaryKind::Silu, UnaryAttr::default(), &input, &mut b).unwrap();
        assert_eq!(a, b);
    }
}
