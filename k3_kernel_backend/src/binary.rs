//! FP16 ONNX 二元逐元素算子。

use alloc::vec;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, BinaryAttr, KernelOp, MAX_DIM};

use crate::call::{CallContext, TensorMeta};
use crate::rvv::{self, BinaryOp};
use crate::{BackendCall, ComputeKernel};

/// 二元算子的语义种类。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinaryKind {
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Subtraction.
    Sub,
    /// Division.
    Div,
    /// Floating-point remainder.
    Mod,
}

/// 已校验的 ONNX 右对齐广播计划。
struct BroadcastPlan {
    /// Left input metadata.
    lhs: TensorMeta,
    /// Right input metadata.
    rhs: TensorMeta,
    /// Output metadata.
    output: TensorMeta,
}

impl BroadcastPlan {
    /// Validates and creates a right-aligned broadcast plan.
    fn new(lhs: TensorMeta, rhs: TensorMeta, output: TensorMeta) -> Result<Self, BackendErr> {
        let rank = lhs.rank.max(rhs.rank);
        if output.rank != rank {
            return Err(BackendErr::InvalidTensor);
        }
        for axis in 0..rank {
            let lhs_dim = aligned_dim(&lhs, rank, axis);
            let rhs_dim = aligned_dim(&rhs, rank, axis);
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Err(BackendErr::InvalidTensor);
            }
            if output.shape[axis] != lhs_dim.max(rhs_dim) {
                return Err(BackendErr::InvalidTensor);
            }
        }
        Ok(Self { lhs, rhs, output })
    }

    /// Resolves one output index to the corresponding three physical offsets.
    fn offsets(&self, linear: usize) -> Result<(usize, usize, usize), BackendErr> {
        let mut output_coordinates = [0_usize; MAX_DIM];
        self.output.coordinates(linear, &mut output_coordinates)?;
        let lhs_coordinates = broadcast_coordinates(&self.lhs, &self.output, &output_coordinates);
        let rhs_coordinates = broadcast_coordinates(&self.rhs, &self.output, &output_coordinates);
        Ok((
            self.lhs.offset_for_coordinates(&lhs_coordinates)?,
            self.rhs.offset_for_coordinates(&rhs_coordinates)?,
            self.output.offset_for_coordinates(&output_coordinates)?,
        ))
    }
}

/// Defines a marker that statically routes one binary operator.
macro_rules! define_binary_kernel {
    ($name:ident, $op:ident, $kind:ident, $doc:literal) => {
        #[doc = $doc]
        pub(crate) struct $name;

        impl ComputeKernel for $name {
            const OP: KernelOp = KernelOp::$op;

            unsafe fn call(call: *const BackendCall) -> Result<(), BackendErr> {
                unsafe { call_binary(call, BinaryKind::$kind) }
            }
        }
    };
}

define_binary_kernel!(AddKernel, ADD, Add, "`ADD` 的静态 FP16 分发器。");
define_binary_kernel!(MulKernel, MUL, Mul, "`MUL` 的静态 FP16 分发器。");
define_binary_kernel!(SubKernel, SUB, Sub, "`SUB` 的静态 FP16 分发器。");
define_binary_kernel!(DivKernel, DIV, Div, "`DIV` 的静态 FP16 分发器。");
define_binary_kernel!(ModKernel, MOD, Mod, "`MOD` 的静态 FP16 分发器。");

/// 解析二元调用并直接路由到 FP16 的 target 实现。
unsafe fn call_binary(call: *const BackendCall, kind: BinaryKind) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<BinaryAttr>()?;
    if attr.broadcast_kind != 0 {
        return Err(BackendErr::InvalidAttr);
    }
    if ctx.inputs[0].dtype != AiDtype::F16
        || ctx.inputs[1].dtype != AiDtype::F16
        || ctx.outputs[0].dtype != AiDtype::F16
    {
        return Err(BackendErr::UnsupportedDtype);
    }

    let plan = BroadcastPlan::new(
        ctx.inputs[0].checked_meta()?,
        ctx.inputs[1].checked_meta()?,
        ctx.outputs[0].checked_meta()?,
    )?;
    let lhs = unsafe { ctx.inputs[0].as_slice::<u16>()? };
    let rhs = unsafe { ctx.inputs[1].as_slice::<u16>()? };
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
    match ctx.target {
        AiTargetHint::PREFER_CPU => compute_f16_cpu(kind, attr, &plan, lhs, rhs, output),
        AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100 => {
            compute_f16_rvv(kind, attr, &plan, lhs, rhs, output)
        }
        _ => unreachable!("CallContext rejects unknown targets"),
    }
}

/// CPU FP16 参考实现；运算在 F32 域完成后写回 FP16。
fn compute_f16_cpu(
    kind: BinaryKind,
    attr: BinaryAttr,
    plan: &BroadcastPlan,
    lhs: &[u16],
    rhs: &[u16],
    output: &mut [u16],
) -> Result<(), BackendErr> {
    for linear in 0..plan.output.element_count {
        let (lhs_offset, rhs_offset, output_offset) = plan.offsets(linear)?;
        let value = apply(kind, attr, lhs[lhs_offset], rhs[rhs_offset])?;
        output[output_offset] = f16::from_f32(value).to_bits();
    }
    Ok(())
}

/// X100/A100 的 FP16 RVV 实现；MOD 保持标量 fmod 语义。
fn compute_f16_rvv(
    kind: BinaryKind,
    attr: BinaryAttr,
    plan: &BroadcastPlan,
    lhs: &[u16],
    rhs: &[u16],
    output: &mut [u16],
) -> Result<(), BackendErr> {
    if kind == BinaryKind::Mod {
        return compute_f16_cpu(kind, attr, plan, lhs, rhs, output);
    }
    let count = plan.output.element_count;
    let mut packed_lhs = vec![0.0_f32; count];
    let mut packed_rhs = vec![0.0_f32; count];
    for linear in 0..count {
        let (lhs_offset, rhs_offset, _) = plan.offsets(linear)?;
        packed_lhs[linear] = f16::from_bits(lhs[lhs_offset]).to_f32();
        packed_rhs[linear] = f16::from_bits(rhs[rhs_offset]).to_f32();
    }
    let mut packed_output = vec![0.0_f32; count];
    rvv::f16_binary_work(
        binary_op(kind)?,
        &packed_lhs,
        &packed_rhs,
        &mut packed_output,
    )?;
    for (linear, value) in packed_output.into_iter().enumerate() {
        let (_, _, output_offset) = plan.offsets(linear)?;
        output[output_offset] = f16::from_f32(value).to_bits();
    }
    Ok(())
}

/// Applies one binary operator in the FP32 accumulation domain.
fn apply(kind: BinaryKind, attr: BinaryAttr, lhs: u16, rhs: u16) -> Result<f32, BackendErr> {
    let lhs = f16::from_bits(lhs).to_f32();
    let rhs = f16::from_bits(rhs).to_f32();
    match kind {
        BinaryKind::Add => Ok(lhs + rhs),
        BinaryKind::Mul => Ok(lhs * rhs),
        BinaryKind::Sub => Ok(lhs - rhs),
        BinaryKind::Div => Ok(lhs / rhs),
        BinaryKind::Mod if attr.flags.get() & BinaryAttr::MOD_FMOD != 0 => {
            Ok(libm::fmodf(lhs, rhs))
        }
        BinaryKind::Mod => Err(BackendErr::InvalidAttr),
    }
}

/// Maps a binary semantic kind to the corresponding RVV primitive.
fn binary_op(kind: BinaryKind) -> Result<BinaryOp, BackendErr> {
    match kind {
        BinaryKind::Add => Ok(BinaryOp::Add),
        BinaryKind::Mul => Ok(BinaryOp::Mul),
        BinaryKind::Sub => Ok(BinaryOp::Sub),
        BinaryKind::Div => Ok(BinaryOp::Div),
        BinaryKind::Mod => Err(BackendErr::UnsupportedOp),
    }
}

/// Returns one input dimension after right-aligning it to the output rank.
fn aligned_dim(meta: &TensorMeta, output_rank: usize, output_axis: usize) -> usize {
    let leading = output_rank - meta.rank;
    if output_axis < leading {
        1
    } else {
        meta.shape[output_axis - leading]
    }
}

/// Converts output coordinates to an input coordinate under ONNX broadcasting rules.
fn broadcast_coordinates(
    input: &TensorMeta,
    output: &TensorMeta,
    output_coordinates: &[usize; MAX_DIM],
) -> [usize; MAX_DIM] {
    let mut coordinates = [0_usize; MAX_DIM];
    let leading = output.rank - input.rank;
    for axis in 0..input.rank {
        coordinates[axis] = if input.shape[axis] == 1 {
            0
        } else {
            output_coordinates[leading + axis]
        };
    }
    coordinates
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(shape: &[usize]) -> TensorMeta {
        let mut full_shape = [0_usize; MAX_DIM];
        let mut strides = [0_usize; MAX_DIM];
        let mut stride = 1;
        for axis in (0..shape.len()).rev() {
            full_shape[axis] = shape[axis];
            strides[axis] = stride;
            stride *= shape[axis];
        }
        TensorMeta {
            rank: shape.len(),
            shape: full_shape,
            strides,
            element_size: 2,
            element_count: stride,
        }
    }

    #[test]
    fn f16_broadcast_matches_reference() {
        let plan = BroadcastPlan::new(meta(&[2, 3]), meta(&[3]), meta(&[2, 3])).unwrap();
        let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].map(|v| f16::from_f32(v).to_bits());
        let rhs = [10.0_f32, 20.0, 30.0].map(|v| f16::from_f32(v).to_bits());
        let mut output = [0_u16; 6];
        compute_f16_cpu(
            BinaryKind::Add,
            BinaryAttr::default(),
            &plan,
            &lhs,
            &rhs,
            &mut output,
        )
        .unwrap();
        let values: alloc::vec::Vec<f32> = output
            .into_iter()
            .map(|v| f16::from_bits(v).to_f32())
            .collect();
        assert_eq!(values, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
}
