//! 用户态 frontend 的稳定提交描述。
//!
//! 描述用户态算子提交所需的 tensor 描述、算子参数和 graph 结构。
//! 这些结构会被复制到 ring entry 里，由内核调度器校验、pin buffer、选择 backend。

pub mod kd_uring;
pub mod tensormanager;

// ── 从 k3_aiUabi 重导出 ──────────────────────────────────────────

pub use k3_aiUabi::*;
pub use tensormanager::{Tensor, TensorManager};
