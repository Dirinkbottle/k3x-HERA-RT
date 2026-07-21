//! k3 AI runtime 的用户态/内核态共享 ABI (UABI)。
//!
//! 这里集中定义用户态 frontend、内核调度器和 backend 之间传递的稳定结构：
//! tensor 描述、单算子描述、算子参数 (attr)、计算图 blob 以及各层的错误类型。
//! 所有 `#[repr(C)]` 结构都是跨特权级的 ABI 契约，改动需同步内核侧。
#![no_std]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(nonstandard_style)]

extern crate alloc;

pub mod desc;
pub mod error;
pub mod graph;
pub mod kernel;
pub mod kernelattr;
pub mod types;

pub use desc::*;
pub use graph::*;
pub use kernel::*;
pub use kernelattr::*;
pub use types::*;
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

/// `ioctl` 命令号：建立/校验用户态与内核共享的 channel 共享内存。
pub const K3_AI_IOC_BUILD_CHANNEL: u32 = 0x4B33_0001;

/// `ioctl` 命令号：向内核提交一次 graph 执行请求。
pub const K3_AI_IOC_SUBMIT_GRAPH: u32 = 0x4B33_0002;

/// 最小闭环所需的 channel 数量：一个发送、一个接收。
pub const K3_CHANNEL_COUNT: usize = 2;

/// 发送方 channel 的固定下标。
pub const K3_CHANNEL_SNEDERID: u8 = 0;

/// 接收方 channel 的固定下标。
pub const K3_CHANNEL_RECIVERID: u8 = 1;
