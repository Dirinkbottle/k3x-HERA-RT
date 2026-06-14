//! k3 芯片 AI 算子库。

pub mod matmul;

/// backend 算子的 tensor 视图，`data` 指向当前地址空间可访问的连续内存。
#[repr(C)]
pub struct BackendTensorView {
    pub data: *mut u8,
    pub byte_len: u64,
    pub shape: [u32; 8],
    pub stride_bytes: [u64; 8],
    pub ndim: u32,
    pub dtype: u32,
    pub format: u32,
    pub layout: u32,
    pub flags: u32,
}

/// 单次 backend 算子调用描述。
#[repr(C)]
pub struct BackendCall {
    /// 操作类型。
    pub op: u8,
    /// 执行目标（CPU/X100/A100）。
    pub target: u8,
    pub inputs: *const BackendTensorView,
    pub input_count: u32,
    pub outputs: *mut BackendTensorView,
    pub output_count: u32,
    pub attr: *const u8,
    pub attr_size: u32,
}

/// backend 算子分发入口，按 `call.op` 路由到对应算子执行器。
pub unsafe extern "C" fn k3_run_kernel(call: *const BackendCall) -> i32 {
    // 转回 op，把 call 分发给对应算子执行器处理
    unsafe { matmul::matmul_caller(call) }
}
