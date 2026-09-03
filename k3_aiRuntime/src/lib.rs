//! k3 芯片 AI 运行时，适用于 x100 与 a100（对较老的 x60 支持暂无测试）。
//!
//! 分为前后端架构：
//! - `fronted` — 用户态提交通路（channel 建立、tensor 管理、graph 提交）
//! - `test` — 集成/单元测试（仅 `cfg(test)` 编译）
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(nonstandard_style)]
pub mod fronted;
pub mod ort_ffi;
#[cfg(test)]
pub mod test;
