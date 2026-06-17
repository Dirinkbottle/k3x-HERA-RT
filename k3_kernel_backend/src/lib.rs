//! k3 芯片 AI 算子库。

#![no_std]

extern crate alloc;

use core::ptr::read_unaligned;

use k3_aiUabi::{AiDtype, AiGraphNode, AiTensorDesc, AiTensorFormat, AiTensorLayout, KernelOp, MAX_DIM, MAX_SUBMIT_TENSORS};
use k3_aiUabi::error::BackendErr;
use log::error;
pub mod matmul;

/// backend 算子的 tensor 视图，`data` 指向当前地址空间可访问的连续内存。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BackendTensorView {
    pub data: *mut u8,
    pub byte_len: u64,
    pub shape: [u32; MAX_DIM],
    pub stride_bytes: [u64; MAX_DIM],
    pub ndim: u32,
    pub dtype: AiDtype,
    pub format: AiTensorFormat,
    pub layout: AiTensorLayout,
    pub flags: u32,
}

impl Default for BackendTensorView {
    fn default() -> Self {
        Self {
            data: core::ptr::null_mut(),
            byte_len: 0,
            shape: [0; MAX_DIM],
            stride_bytes: [0; MAX_DIM],
            ndim: 0,
            dtype: AiDtype::default(),
            format: AiTensorFormat::default(),
            layout: AiTensorLayout::default(),
            flags: 0,
        }
    }
}

impl BackendTensorView {
    fn from_desc(desc: &AiTensorDesc) -> Self {

        if desc.kernel_va==0 {
            error!("desc tensor has null kernel_va!");
        }

        Self {
            data: desc.kernel_va as *mut u8,
            byte_len: desc.size_bytes,
            shape: desc.shape,
            stride_bytes: desc.stride_bytes,
            ndim: desc.ndim,
            dtype: desc.dtype,
            format: desc.format,
            layout: desc.layout,
            flags: desc.flags as u32,
        }
    }
}

/// 单次 backend 算子调用描述。
#[repr(C)]
pub struct BackendCall {
    /// 操作类型。
    pub op: KernelOp,
    /// 执行目标（CPU/X100/A100）。
    pub target: u8,
    pub inputs: *const BackendTensorView,
    pub input_count: u32,
    pub outputs: *mut BackendTensorView,
    pub output_count: u32,
    // kernel attr类型地址
    pub attr: *const u8,
    // kernel attr大小
    pub attr_size: u32,
}

/// 内核入口,tensor地址需要已经映射
/// backend 算子分发入口，按 `call.op` 路由到对应算子执行器。
pub unsafe extern "C" fn k3_run_kernel(node: &AiGraphNode) -> i32 {
    let desc = &node.desc;
    let mut input_views = [BackendTensorView::default(); MAX_SUBMIT_TENSORS];
    let mut output_views = [BackendTensorView::default(); MAX_SUBMIT_TENSORS];

    for i in 0..desc.input_count as usize {
        input_views[i] = BackendTensorView::from_desc(&desc.tensors[i]);
    }
    for i in 0..desc.output_count as usize {
        output_views[i] = BackendTensorView::from_desc(&desc.tensors[desc.input_count as usize + i]);
    }

    let call = BackendCall {
        op: desc.op,
        target: desc.target_hint.0 as u8,
        inputs: input_views.as_ptr(),
        input_count: desc.input_count,
        outputs: output_views.as_mut_ptr(),
        output_count: desc.output_count,
        attr: desc.attr_inline.as_ptr(),
        attr_size: desc.attr_size,
    };

    error!("k3_run_kernel: node_id={}, op={:?}, target_hint={}", node.node_id, desc.op, desc.target_hint.0);

    let result = match desc.op {
        KernelOp::MAT_MUL => matmul::matmul_caller(&call),
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
