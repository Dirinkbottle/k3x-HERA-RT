#![no_std]


extern crate alloc;

pub mod desc;
pub mod graph;
pub mod kernel;
pub mod kernelattr;

pub use desc::*;
pub use graph::*;
pub use kernel::*;
pub use kernelattr::*;
// ── 常量 ──────────────────────────────────────────────────────

/// 当前 AI runtime UAPI 版本。
pub const AI_ABI_VERSION: u32 = 1;

/// 张量最多记录的维度数。
///
/// 第一阶段覆盖 LLM 和 YOLO 的主要路径时
/// 超过该维度的上层张量需要在 frontend lowering 时拒绝提交。
pub const MAX_DIM: usize = 8;

/// 单个维度 stride 的最大字节数 (4GB)
pub const MAX_STRIDE_BYTE: usize = 0x1_0000_0000;

/// 一个 submit entry 最多携带的张量描述数量。
///
/// 约定：`tensors[0..input_count]` 是输入，紧随其后的 `output_count` 个是输出。
pub const MAX_SUBMIT_TENSORS: usize = 8;