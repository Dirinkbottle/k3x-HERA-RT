//! k3 AI runtime 的用户态/内核态共享 ABI (UABI)。
//!
//! 这里集中定义用户态 frontend、内核调度器和 backend 之间传递的稳定结构：
//! tensor 描述、单算子描述、算子参数 (attr)、计算图 blob 以及各层的错误类型。
//! 所有 `#[repr(C)]` 结构都是跨特权级的 ABI 契约；内核侧必须直接依赖本 crate，
//! 不要手写镜像结构体。
#![no_std]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(nonstandard_style)]

extern crate alloc;

pub mod control;
pub mod desc;
pub mod error;
pub mod graph;
pub mod kernel;
pub mod kernelattr;
pub mod types;

pub use control::*;
pub use desc::*;
pub use error::*;
pub use graph::*;
pub use kernel::*;
pub use kernelattr::*;
pub use types::*;
// ── 常量 ──────────────────────────────────────────────────────

include!(concat!(env!("OUT_DIR"), "/ai_abi_version.rs"));

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
