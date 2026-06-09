//! 用户态 frontend 的稳定提交描述。
//!
//! 这一层只描述"用户想执行什么算子、输入输出张量在哪里、算子参数是什么"。
//! 它不保存 Rust 对象指针、不保存 backend 函数指针，也不决定最终跑在 CPU/X100/A100。
//! 这些结构后续会被复制到 ring entry 里，再由内核调度器校验、pin buffer、选择 backend。

pub mod desc;
pub mod graph;
pub mod kd_uring;
pub mod kernel;
pub mod kernelattr;
pub mod tensor;

// ── 子模块 re-export ──────────────────────────────────────────

pub use desc::{
    AiCompletionEntry, AiDtype, AiKernelDesc, AiQuantDesc, AiTensorDesc, AiTensorFormat,
    AiTensorLayout,
};
pub use graph::{
    AI_GRAPH_MAGIC, AiGraphBlob, AiGraphBuildError, AiGraphChainId, AiGraphEdge, AiGraphHeader,
    AiGraphNode, AiGraphNodeId, AiGraphParseError, AiGraphParser, AiGraphSubmitEntry,
    AiParsedGraph, GraphManager, GraphSubmitKind,
};
pub use kernel::{AiTargetHint, KernelOp};
pub use kernelattr::{
    BinaryAttr, Conv2dAttr, MatMulAttr, RmsNormAttr, RopeAttr, SoftmaxAttr, UnaryAttr,
};

// ── 常量 ──────────────────────────────────────────────────────

/// 当前 AI runtime UAPI 版本。
pub const AI_ABI_VERSION: u32 = 1;

/// 张量最多记录的维度数。
///
/// 第一阶段覆盖 LLM 和 YOLO 的主要路径时
/// 超过该维度的上层张量需要在 frontend lowering 时拒绝提交。
pub const MAX_DIM: usize = 8;

/// 一个 submit entry 最多携带的张量描述数量。
///
/// 约定：`tensors[0..input_count]` 是输入，紧随其后的 `output_count` 个是输出。
pub const MAX_SUBMIT_TENSORS: usize = 8;

/// 内联算子参数区大小。
pub const ATTR_INLINE_SIZE: usize = 128;
