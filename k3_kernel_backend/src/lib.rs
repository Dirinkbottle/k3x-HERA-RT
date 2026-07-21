//! k3 芯片 AI 算子库。
//!
//! backend 提供 `k3_run_kernel` 分发入口，把内核调度器传入的 `AiGraphNode`
//! 拆成 tensor view 与调用描述，按 `op` 路由到具体算子实现（当前实现 matmul）。

#![no_std]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

extern crate alloc;

use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiGraphNode, KernelOp, MAX_SUBMIT_TENSORS};
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

/// 内核入口,tensor地址需要已经映射
/// backend 算子分发入口，按 `call.op` 路由到对应算子执行器。
///
/// # Safety
///
/// `node.desc.tensors[*].kernel_va` 必须已经映射到 backend 当前地址空间可访问的
/// 有效内存，且输入/输出 tensor 的生命周期覆盖本次调用。
pub unsafe extern "C" fn k3_run_kernel(node: &AiGraphNode) -> i32 {
    let desc = &node.desc;
    let total_count = match desc.input_count.checked_total(desc.output_count) {
        Ok(total_count) => total_count,
        Err(err) => {
            error!("k3_run_kernel: invalid tensor count: {:?}", err);
            return -1;
        }
    };
    if total_count > MAX_SUBMIT_TENSORS {
        error!(
            "k3_run_kernel: tensor count {} exceeds max {}",
            total_count, MAX_SUBMIT_TENSORS
        );
        return -1;
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
        target: desc.target_hint.0,
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
        KernelOp::MAT_MUL => unsafe { matmul::matmul_caller(&call) },
        KernelOp::SILU => unsafe { unary::unary_caller(&call, unary::UnaryKind::Silu) },
        KernelOp::SIGMOID => unsafe { unary::unary_caller(&call, unary::UnaryKind::Sigmoid) },
        KernelOp::SCALE => unsafe { unary::unary_caller(&call, unary::UnaryKind::Scale) },
        KernelOp::ADD => unsafe { binary::binary_caller(&call, binary::BinaryKind::Add) },
        KernelOp::MUL => unsafe { binary::binary_caller(&call, binary::BinaryKind::Mul) },
        KernelOp::SUB => unsafe { binary::binary_caller(&call, binary::BinaryKind::Sub) },
        KernelOp::DIV => unsafe { binary::binary_caller(&call, binary::BinaryKind::Div) },
        KernelOp::MOD => unsafe { binary::binary_caller(&call, binary::BinaryKind::Mod) },
        KernelOp::CONV2D => unsafe { conv2d::conv2d_caller(&call) },
        KernelOp::RMS_NORM => unsafe { nn::rms_norm_caller(&call) },
        KernelOp::ROPE => unsafe { nn::rope_caller(&call) },
        KernelOp::SOFTMAX => unsafe { nn::softmax_caller(&call) },
        KernelOp::MAX_POOL => unsafe { nn::max_pool_caller(&call) },
        KernelOp::REDUCE_MAX => unsafe { nn::reduce_max_caller(&call) },
        KernelOp::TOP_K => unsafe { nn::top_k_caller(&call) },
        KernelOp::CONCAT => unsafe { transform::concat_caller(&call) },
        KernelOp::TRANSPOSE => unsafe { transform::transpose_caller(&call) },
        KernelOp::GATHER => unsafe { transform::gather_caller(&call) },
        KernelOp::GATHER_ELEMENTS => unsafe { transform::gather_elements_caller(&call) },
        KernelOp::CAST => unsafe { transform::cast_caller(&call) },
        KernelOp::RESIZE => unsafe { transform::resize_caller(&call) },
        KernelOp::EXPAND => unsafe { transform::expand_caller(&call) },
        KernelOp::TILE => unsafe { transform::tile_caller(&call) },
        _ => Err(BackendErr::UnsupportedOp),
    };

    match result {
        Ok(()) => 0,
        Err(e) => {
            error!("k3_run_kernel failed: {:?}", e);
            -1
        }
    }
}

/// `k3_run_kernel` 分发入口的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use k3_ai_uabi::TensorCount;

    /// tensor 总数超限时应在解引用数组前就拒绝并返回 -1。
    #[test]
    fn k3_run_kernel_rejects_too_many_tensors_before_indexing() {
        let mut node = AiGraphNode::default();
        node.desc.op = KernelOp::MAT_MUL;
        node.desc.input_count = TensorCount::new(MAX_SUBMIT_TENSORS as u32 + 1);
        node.desc.output_count = TensorCount::new(0);

        assert_eq!(unsafe { k3_run_kernel(&node) }, -1);
    }
}
