//! 用户态 frontend 的稳定提交描述。
//!
//! 描述用户态算子提交所需的 tensor 描述、算子参数和 graph 结构。
//! 这些结构会被复制到 ring entry 里，由内核调度器校验、pin buffer、选择 backend。

pub mod desc;
pub mod graph;
pub mod kd_uring;
pub mod kernel;
pub mod kernelattr;
pub mod tensormanager;

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
    AiKernelAttr, BinaryAttr, Conv2dAttr, MatMulAttr, RmsNormAttr, RopeAttr, SoftmaxAttr,
    UnaryAttr,
};
pub use tensormanager::{Tensor, TensorManager};

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
