//! ONNX 二元 elementwise 算子。
//!
//! ADD/MUL/SUB/DIV 在 RISC-V target 上使用真实 RVV，MOD 按约定保留 CPU 实现。

use crate::BackendCall;
use crate::call::{CallContext, TensorMeta};
use crate::rvv::{self, BinaryOp};
use alloc::vec;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, BinaryAttr, MAX_DIM};
use log::error;

/// 二元算子的语义种类。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BinaryKind {
    /// 逐元素加法。
    Add,
    /// 逐元素乘法。
    Mul,
    /// 逐元素减法。
    Sub,
    /// 逐元素除法。
    Div,
    /// 逐元素取模。
    Mod,
}

/// 已校验的 ONNX 右对齐广播计划。
struct BroadcastPlan {
    /// lhs tensor 元数据。
    lhs: TensorMeta,
    /// rhs tensor 元数据。
    rhs: TensorMeta,
    /// output tensor 元数据。
    output: TensorMeta,
}

impl BroadcastPlan {
    /// 校验两个输入和输出 shape，并构造广播计划。
    fn new(lhs: TensorMeta, rhs: TensorMeta, output: TensorMeta) -> Result<Self, BackendErr> {
        let rank = lhs.rank.max(rhs.rank);
        if output.rank != rank {
            return Err(BackendErr::InvalidTensor);
        }
        for out_axis in 0..rank {
            let lhs_dim = aligned_dim(&lhs, rank, out_axis);
            let rhs_dim = aligned_dim(&rhs, rank, out_axis);
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Err(BackendErr::InvalidTensor);
            }
            if output.shape[out_axis] != lhs_dim.max(rhs_dim) {
                return Err(BackendErr::InvalidTensor);
            }
        }
        Ok(Self { lhs, rhs, output })
    }

    /// 返回 output 逻辑下标对应的 lhs/rhs/output 底层元素下标。
    fn offsets(&self, linear: usize) -> Result<(usize, usize, usize), BackendErr> {
        let mut out_coordinates = [0_usize; MAX_DIM];
        self.output.coordinates(linear, &mut out_coordinates)?;
        let lhs_coordinates = broadcast_coordinates(&self.lhs, &self.output, &out_coordinates);
        let rhs_coordinates = broadcast_coordinates(&self.rhs, &self.output, &out_coordinates);
        Ok((
            self.lhs.offset_for_coordinates(&lhs_coordinates)?,
            self.rhs.offset_for_coordinates(&rhs_coordinates)?,
            self.output.offset_for_coordinates(&out_coordinates)?,
        ))
    }
}

/// 二元算子执行器。
///
/// # Safety
///
/// `call` 及其 tensor buffer 必须满足 [`BackendCall`] 的 ABI 和生命周期约束。
pub(crate) unsafe fn binary_caller(
    call: *const BackendCall,
    kind: BinaryKind,
) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io(2, 1)?;
    ctx.reject_input_output_alias()?;
    let attr = ctx.read_attr::<BinaryAttr>()?;
    let lhs_meta = ctx.inputs[0].checked_meta()?;
    let rhs_meta = ctx.inputs[1].checked_meta()?;
    let output_meta = ctx.outputs[0].checked_meta()?;
    let plan = BroadcastPlan::new(lhs_meta, rhs_meta, output_meta)?;

    let lhs_dtype = ctx.inputs[0].dtype;
    if lhs_dtype != ctx.inputs[1].dtype || lhs_dtype != ctx.outputs[0].dtype {
        return Err(BackendErr::UnsupportedDtype);
    }

    match lhs_dtype {
        AiDtype::F32 => {
            let lhs = unsafe { ctx.inputs[0].as_slice::<f32>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<f32>()? };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            execute_f32(kind, attr, &plan, lhs, rhs, output, ctx.target)
        }
        AiDtype::F16 => {
            let lhs = unsafe { ctx.inputs[0].as_slice::<u16>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<u16>()? };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
            execute_f16(kind, attr, &plan, lhs, rhs, output, ctx.target)
        }
        AiDtype::I32 if kind == BinaryKind::Mod => {
            let lhs = unsafe { ctx.inputs[0].as_slice::<i32>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<i32>()? };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<i32>()? };
            execute_integer_mod(&plan, lhs, rhs, output)
        }
        AiDtype::I64 if kind == BinaryKind::Mod => {
            let lhs = unsafe { ctx.inputs[0].as_slice::<i64>()? };
            let rhs = unsafe { ctx.inputs[1].as_slice::<i64>()? };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<i64>()? };
            execute_integer_mod(&plan, lhs, rhs, output)
        }
        _ => Err(BackendErr::UnsupportedDtype),
    }
}

/// 返回右对齐后的指定维大小；缺失的高维按 1 处理。
fn aligned_dim(meta: &TensorMeta, output_rank: usize, output_axis: usize) -> usize {
    let leading = output_rank - meta.rank;
    if output_axis < leading {
        1
    } else {
        meta.shape[output_axis - leading]
    }
}

/// 把 output 坐标映射为广播输入坐标。
fn broadcast_coordinates(
    input: &TensorMeta,
    output: &TensorMeta,
    output_coordinates: &[usize; MAX_DIM],
) -> [usize; MAX_DIM] {
    let mut coordinates = [0_usize; MAX_DIM];
    let leading = output.rank - input.rank;
    for (axis, coordinate) in coordinates.iter_mut().enumerate().take(input.rank) {
        let output_axis = leading + axis;
        *coordinate = if input.shape[axis] == 1 {
            0
        } else {
            output_coordinates[output_axis]
        };
    }
    coordinates
}

/// 执行 F32 广播算子；向量 target 先按逻辑顺序 pack，再进入 RVV。
fn execute_f32(
    kind: BinaryKind,
    attr: BinaryAttr,
    plan: &BroadcastPlan,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    let count = plan.output.element_count;
    let mut logical_lhs = vec![0.0_f32; count];
    let mut logical_rhs = vec![0.0_f32; count];
    for linear in 0..count {
        let (lhs_offset, rhs_offset, _) = plan.offsets(linear)?;
        logical_lhs[linear] = lhs[lhs_offset];
        logical_rhs[linear] = rhs[rhs_offset];
    }

    let mut logical_output = vec![0.0_f32; count];
    if kind == BinaryKind::Mod {
        if attr.flags.get() & BinaryAttr::MOD_FMOD == 0 {
            return Err(BackendErr::InvalidAttr);
        }
        for ((dst, &a), &b) in logical_output
            .iter_mut()
            .zip(&logical_lhs)
            .zip(&logical_rhs)
        {
            *dst = libm::fmodf(a, b);
        }
    } else if matches!(
        target,
        AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100
    ) {
        rvv::binary_f32(
            rvv_op(kind)?,
            &logical_lhs,
            &logical_rhs,
            &mut logical_output,
        )?;
    } else {
        for ((dst, &a), &b) in logical_output
            .iter_mut()
            .zip(&logical_lhs)
            .zip(&logical_rhs)
        {
            *dst = scalar_f32(kind, a, b)?;
        }
    }

    for (linear, value) in logical_output.into_iter().enumerate() {
        let (_, _, output_offset) = plan.offsets(linear)?;
        output[output_offset] = value;
    }
    Ok(())
}

/// 执行 F16 广播算子，在 F32 域计算后编码回 half。
fn execute_f16(
    kind: BinaryKind,
    attr: BinaryAttr,
    plan: &BroadcastPlan,
    lhs: &[u16],
    rhs: &[u16],
    output: &mut [u16],
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    let count = plan.output.element_count;
    let mut logical_lhs = vec![0.0_f32; count];
    let mut logical_rhs = vec![0.0_f32; count];
    for linear in 0..count {
        let (lhs_offset, rhs_offset, _) = plan.offsets(linear)?;
        logical_lhs[linear] = f16::from_bits(lhs[lhs_offset]).to_f32();
        logical_rhs[linear] = f16::from_bits(rhs[rhs_offset]).to_f32();
    }
    let mut logical_output = vec![0.0_f32; count];
    if kind == BinaryKind::Mod {
        if attr.flags.get() & BinaryAttr::MOD_FMOD == 0 {
            return Err(BackendErr::InvalidAttr);
        }
        for ((dst, &a), &b) in logical_output
            .iter_mut()
            .zip(&logical_lhs)
            .zip(&logical_rhs)
        {
            *dst = libm::fmodf(a, b);
        }
    } else if matches!(
        target,
        AiTargetHint::PREFER_X100 | AiTargetHint::PREFER_A100
    ) {
        rvv::binary_f32(
            rvv_op(kind)?,
            &logical_lhs,
            &logical_rhs,
            &mut logical_output,
        )?;
    } else {
        for ((dst, &a), &b) in logical_output
            .iter_mut()
            .zip(&logical_lhs)
            .zip(&logical_rhs)
        {
            *dst = scalar_f32(kind, a, b)?;
        }
    }
    for (linear, value) in logical_output.into_iter().enumerate() {
        let (_, _, output_offset) = plan.offsets(linear)?;
        output[output_offset] = f16::from_f32(value).to_bits();
    }
    Ok(())
}

/// 执行 I32/I64 MOD。
fn execute_integer_mod<T>(
    plan: &BroadcastPlan,
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
) -> Result<(), BackendErr>
where
    T: Copy + Default + PartialEq + core::ops::Rem<Output = T>,
{
    for linear in 0..plan.output.element_count {
        let (lhs_offset, rhs_offset, output_offset) = plan.offsets(linear)?;
        if rhs[rhs_offset] == T::default() {
            error!("binary MOD: division by zero");
            return Err(BackendErr::InvalidInput);
        }
        output[output_offset] = lhs[lhs_offset] % rhs[rhs_offset];
    }
    Ok(())
}

/// 映射到 RVV 二元操作。
fn rvv_op(kind: BinaryKind) -> Result<BinaryOp, BackendErr> {
    match kind {
        BinaryKind::Add => Ok(BinaryOp::Add),
        BinaryKind::Sub => Ok(BinaryOp::Sub),
        BinaryKind::Mul => Ok(BinaryOp::Mul),
        BinaryKind::Div => Ok(BinaryOp::Div),
        BinaryKind::Mod => Err(BackendErr::UnsupportedOp),
    }
}

/// F32 软件参考标量操作。
fn scalar_f32(kind: BinaryKind, lhs: f32, rhs: f32) -> Result<f32, BackendErr> {
    match kind {
        BinaryKind::Add => Ok(lhs + rhs),
        BinaryKind::Sub => Ok(lhs - rhs),
        BinaryKind::Mul => Ok(lhs * rhs),
        BinaryKind::Div => Ok(lhs / rhs),
        BinaryKind::Mod => Err(BackendErr::UnsupportedOp),
    }
}

/// 二元算子广播和 dtype 测试。
#[cfg(test)]
mod tests {
    use super::*;
    use k3_ai_uabi::AiTensorLayout;

    /// 构造 dense tensor meta。
    fn meta(shape: &[usize]) -> TensorMeta {
        let mut full_shape = [0_usize; MAX_DIM];
        let mut strides = [0_usize; MAX_DIM];
        let mut stride = 1;
        for axis in (0..shape.len()).rev() {
            full_shape[axis] = shape[axis];
            strides[axis] = stride;
            stride *= shape[axis];
        }
        let _ = AiTensorLayout::DENSE;
        TensorMeta {
            rank: shape.len(),
            shape: full_shape,
            strides,
            element_size: 4,
            element_count: stride,
        }
    }

    /// ONNX 右对齐多维广播应产生预期结果。
    #[test]
    fn multidimensional_broadcast_matches_onnx() {
        let plan = BroadcastPlan::new(meta(&[2, 3]), meta(&[3]), meta(&[2, 3])).unwrap();
        let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [10.0_f32, 20.0, 30.0];
        let mut output = [0.0_f32; 6];
        execute_f32(
            BinaryKind::Add,
            BinaryAttr::default(),
            &plan,
            &lhs,
            &rhs,
            &mut output,
            AiTargetHint::PREFER_A100,
        )
        .unwrap();
        assert_eq!(output, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    /// 不兼容 shape 必须拒绝。
    #[test]
    fn incompatible_broadcast_is_rejected() {
        assert!(BroadcastPlan::new(meta(&[2, 3]), meta(&[4]), meta(&[2, 4])).is_err());
    }

    /// 整数 MOD 必须拒绝除零。
    #[test]
    fn integer_mod_rejects_zero_divisor() {
        let plan = BroadcastPlan::new(meta(&[2]), meta(&[2]), meta(&[2])).unwrap();
        let mut output = [0_i32; 2];
        assert!(execute_integer_mod(&plan, &[3, 4], &[2, 0], &mut output).is_err());
    }
}
