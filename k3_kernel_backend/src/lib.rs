//! k3 芯片 AI 算子库。
//!
//! backend 提供 `k3_run_kernel` 分发入口，把内核调度器传入的 `AiGraphNode`
//! 拆成 tensor view 与调用描述，按 `op` 路由到具体算子实现（当前实现 matmul）。

#![no_std]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

extern crate alloc;

use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiGraphNode, AiTargetHint, KernelOp, MAX_SUBMIT_TENSORS};
use log::{error, info};

pub mod binary;
pub mod call;
pub mod conv2d;
pub mod matmul;
pub mod nn;
mod rvv;
pub mod transform;
pub mod unary;

pub use call::{BackendCall, BackendTensorView};

/// 一个静态可分发的 backend 算子。
///
/// 实现者把一个固定的 [`KernelOp`] 映射到其 ABI 调用入口；注册表和动态分发不属于
/// backend ABI。调用方必须满足 [`BackendCall`] 中指针、长度和生命周期约束。
pub trait ComputeKernel {
    /// 此实现处理的 ABI 操作码。
    const OP: KernelOp;

    /// 解析并执行一次 backend 调用。
    ///
    /// # Safety
    ///
    /// `call` 及其 tensor/attr 指针必须在调用期间保持有效，并符合 `BackendCall` ABI。
    unsafe fn call(call: *const BackendCall) -> Result<(), BackendErr>;
}

/// 内核入口,tensor地址需要已经映射
/// backend 算子分发入口，按 `call.op` 路由到对应算子执行器。
///
/// 成功返回 0；失败返回负错误码并把 `BackendErr` 写入 `node.state.error_flag`，
/// 同时标记 `node.state.complete = 1`。
///
/// # Safety
///
/// `node.desc.tensors[*].kernel_va` 必须已经映射到 backend 当前地址空间可访问的
/// 有效内存，且输入/输出 tensor 的生命周期覆盖本次调用。
pub unsafe extern "C" fn k3_run_kernel(node: &mut AiGraphNode) -> i32 {
    let desc = &node.desc;
    let total_count = match desc.input_count.checked_total(desc.output_count) {
        Ok(total_count) => total_count,
        Err(err) => {
            error!("k3_run_kernel: invalid tensor count: {:?}", err);
            node.state.error_flag = BackendErr::InvalidInput as u8;
            node.state.complete = 1;
            return -(BackendErr::InvalidInput as i32);
        }
    };
    if total_count > MAX_SUBMIT_TENSORS {
        error!(
            "k3_run_kernel: tensor count {} exceeds max {}",
            total_count, MAX_SUBMIT_TENSORS
        );
        node.state.error_flag = BackendErr::InvalidInput as u8;
        node.state.complete = 1;
        return -(BackendErr::InvalidInput as i32);
    }

    let input_count = desc.input_count.get() as usize;
    let output_count = desc.output_count.get() as usize;

    let mut input_views = [BackendTensorView::default(); MAX_SUBMIT_TENSORS];
    let mut output_views = [BackendTensorView::default(); MAX_SUBMIT_TENSORS];

    for (view, tensor) in input_views
        .iter_mut()
        .zip(desc.tensors[..input_count].iter())
    {
        *view = BackendTensorView::from_desc(tensor);
    }
    for (view, tensor) in output_views
        .iter_mut()
        .zip(desc.tensors[input_count..total_count].iter())
        .take(output_count)
    {
        *view = BackendTensorView::from_desc(tensor);
    }

    let call = BackendCall {
        op: desc.op,
        // TODO: X100 bring-up 完成后恢复使用调度器传入的 target hint。
        target: AiTargetHint::PREFER_A100.0,
        inputs: input_views.as_ptr(),
        input_count: desc.input_count,
        outputs: output_views.as_mut_ptr(),
        output_count: desc.output_count,
        attr: desc.attr_inline.as_ptr(),
        attr_size: desc.attr_size,
    };

    info!(
        "k3_run_kernel: node_id={}, op={:?}, target_hint={}",
        node.node_id, desc.op, desc.target_hint.0
    );

    let result = match desc.op {
        KernelOp::MAT_MUL => unsafe { <matmul::MatMulKernel as ComputeKernel>::call(&call) },
        KernelOp::SILU => unsafe { <unary::SiluKernel as ComputeKernel>::call(&call) },
        KernelOp::SIGMOID => unsafe { <unary::SigmoidKernel as ComputeKernel>::call(&call) },
        KernelOp::SCALE => unsafe { <unary::ScaleKernel as ComputeKernel>::call(&call) },
        KernelOp::ADD => unsafe { <binary::AddKernel as ComputeKernel>::call(&call) },
        KernelOp::MUL => unsafe { <binary::MulKernel as ComputeKernel>::call(&call) },
        KernelOp::SUB => unsafe { <binary::SubKernel as ComputeKernel>::call(&call) },
        KernelOp::DIV => unsafe { <binary::DivKernel as ComputeKernel>::call(&call) },
        KernelOp::MOD => unsafe { <binary::ModKernel as ComputeKernel>::call(&call) },
        KernelOp::CONV2D => unsafe { <conv2d::Conv2dKernel as ComputeKernel>::call(&call) },
        KernelOp::RMS_NORM => unsafe { <nn::RmsNormKernel as ComputeKernel>::call(&call) },
        KernelOp::ROPE => unsafe { <nn::RopeKernel as ComputeKernel>::call(&call) },
        KernelOp::SOFTMAX => unsafe { <nn::SoftmaxKernel as ComputeKernel>::call(&call) },
        KernelOp::GLU => unsafe { <nn::GluKernel as ComputeKernel>::call(&call) },
        KernelOp::MAX_POOL => unsafe { <nn::MaxPoolKernel as ComputeKernel>::call(&call) },
        KernelOp::REDUCE_MAX => unsafe { <nn::ReduceMaxKernel as ComputeKernel>::call(&call) },
        KernelOp::TOP_K => unsafe { <nn::TopKKernel as ComputeKernel>::call(&call) },
        KernelOp::CONCAT => unsafe { <transform::ConcatKernel as ComputeKernel>::call(&call) },
        KernelOp::TRANSPOSE => unsafe {
            <transform::TransposeKernel as ComputeKernel>::call(&call)
        },
        KernelOp::GATHER => unsafe { <transform::GatherKernel as ComputeKernel>::call(&call) },
        KernelOp::GATHER_ELEMENTS => unsafe {
            <transform::GatherElementsKernel as ComputeKernel>::call(&call)
        },
        KernelOp::COPY => unsafe { <transform::CopyKernel as ComputeKernel>::call(&call) },
        KernelOp::CAST => unsafe { <transform::CastKernel as ComputeKernel>::call(&call) },
        KernelOp::RESIZE => unsafe { <transform::ResizeKernel as ComputeKernel>::call(&call) },
        KernelOp::EXPAND => unsafe { <transform::ExpandKernel as ComputeKernel>::call(&call) },
        KernelOp::TILE => unsafe { <transform::TileKernel as ComputeKernel>::call(&call) },
        _ => Err(BackendErr::UnsupportedOp),
    };

    match result {
        Ok(()) => {
            node.state.complete = 1;
            0
        }
        Err(e) => {
            error!("k3_run_kernel failed: {:?}", e);
            node.state.error_flag = e as u8;
            node.state.complete = 1;
            -(e as i32)
        }
    }
}

/// `k3_run_kernel` 分发入口的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use k3_ai_uabi::TensorCount;

    /// 所有保留操作码都必须静态绑定到对应 marker，而不是经由运行时注册表。
    #[test]
    fn static_markers_cover_all_preserved_ops() {
        assert_eq!(
            <matmul::MatMulKernel as ComputeKernel>::OP,
            KernelOp::MAT_MUL
        );
        assert_eq!(
            <conv2d::Conv2dKernel as ComputeKernel>::OP,
            KernelOp::CONV2D
        );
        assert_eq!(<binary::AddKernel as ComputeKernel>::OP, KernelOp::ADD);
        assert_eq!(<binary::MulKernel as ComputeKernel>::OP, KernelOp::MUL);
        assert_eq!(<binary::SubKernel as ComputeKernel>::OP, KernelOp::SUB);
        assert_eq!(<binary::DivKernel as ComputeKernel>::OP, KernelOp::DIV);
        assert_eq!(<binary::ModKernel as ComputeKernel>::OP, KernelOp::MOD);
        assert_eq!(<unary::SiluKernel as ComputeKernel>::OP, KernelOp::SILU);
        assert_eq!(
            <unary::SigmoidKernel as ComputeKernel>::OP,
            KernelOp::SIGMOID
        );
        assert_eq!(<unary::ScaleKernel as ComputeKernel>::OP, KernelOp::SCALE);
        assert_eq!(<nn::SoftmaxKernel as ComputeKernel>::OP, KernelOp::SOFTMAX);
        assert_eq!(<nn::RmsNormKernel as ComputeKernel>::OP, KernelOp::RMS_NORM);
        assert_eq!(<nn::RopeKernel as ComputeKernel>::OP, KernelOp::ROPE);
        assert_eq!(<nn::GluKernel as ComputeKernel>::OP, KernelOp::GLU);
        assert_eq!(<nn::MaxPoolKernel as ComputeKernel>::OP, KernelOp::MAX_POOL);
        assert_eq!(
            <nn::ReduceMaxKernel as ComputeKernel>::OP,
            KernelOp::REDUCE_MAX
        );
        assert_eq!(<nn::TopKKernel as ComputeKernel>::OP, KernelOp::TOP_K);
        assert_eq!(
            <transform::ConcatKernel as ComputeKernel>::OP,
            KernelOp::CONCAT
        );
        assert_eq!(
            <transform::TransposeKernel as ComputeKernel>::OP,
            KernelOp::TRANSPOSE
        );
        assert_eq!(
            <transform::GatherKernel as ComputeKernel>::OP,
            KernelOp::GATHER
        );
        assert_eq!(
            <transform::GatherElementsKernel as ComputeKernel>::OP,
            KernelOp::GATHER_ELEMENTS
        );
        assert_eq!(<transform::CopyKernel as ComputeKernel>::OP, KernelOp::COPY);
        assert_eq!(<transform::CastKernel as ComputeKernel>::OP, KernelOp::CAST);
        assert_eq!(
            <transform::ResizeKernel as ComputeKernel>::OP,
            KernelOp::RESIZE
        );
        assert_eq!(
            <transform::ExpandKernel as ComputeKernel>::OP,
            KernelOp::EXPAND
        );
        assert_eq!(<transform::TileKernel as ComputeKernel>::OP, KernelOp::TILE);
        assert!(!KernelOp(25).is_known());
        assert!(!KernelOp(26).is_known());
    }

    /// tensor 总数超限时应在解引用数组前就拒绝并返回 -1。
    #[test]
    fn k3_run_kernel_rejects_too_many_tensors_before_indexing() {
        let mut node = AiGraphNode::default();
        node.desc.op = KernelOp::MAT_MUL;
        node.desc.input_count = TensorCount::new(MAX_SUBMIT_TENSORS as u32 + 1);
        node.desc.output_count = TensorCount::new(0);

        assert_eq!(
            unsafe { k3_run_kernel(&mut node) },
            -(BackendErr::InvalidInput as i32)
        );
    }
}
