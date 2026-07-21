//! k3 芯片 AI 调度器平台 trait 定义。
//!
//! 本 crate 定义内核调度器依赖的操作系统抽象 (`K3SchedulerOps`)，
//! 以及把用户提交的 graph 收敛成可调度直链、按进程 FIFO 执行的调度逻辑。

#![no_std]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(nonstandard_style)]

extern crate alloc;

pub mod kd_kring;
pub mod scheduler;

/// 调度器需要的操作系统接口。
#[allow(clippy::result_unit_err)]
pub trait K3SchedulerOps: Sync {
    /// 返回当前调用路径实际运行的 CPU core id。
    ///
    /// 调度器用这个 id 选择 per-core queue，避免在一个 core 上提交却写入另一个
    /// core 的无锁环形队列。
    fn current_core_id(&self) -> u32;

    /// 从用户空间拷贝数据到内核空间。
    ///
    /// # Safety
    /// 调用者需确保 `user_va` 和 `len` 是有效的用户空间地址范围。
    unsafe fn copy_from_user(&self, user_va: u64, buf: &mut [u8]) -> Result<(), ()>;

    /// 从内核空间拷贝数据到用户空间。
    ///
    /// # Safety
    /// 调用者需确保 `user_va` 和 `len` 是有效的用户空间地址范围。
    unsafe fn copy_to_user(&self, user_va: u64, buf: &[u8]) -> Result<(), ()>;

    /// 将用户虚拟地址映射为内核可访问的虚拟地址。
    ///
    /// 用于 tensor buffer pin 和地址转换。
    /// 返回内核虚拟地址，失败返回 Err。
    ///
    /// # Safety
    /// 调用者需确保 `user_va..user_va+len` 是有效的用户空间地址范围。
    unsafe fn map_user_to_kernel(&self, user_va: u64, len: usize) -> Result<u64, ()>;

    /// 取消用户地址到内核地址的映射。
    ///
    /// # Safety
    /// 调用者需确保 `kernel_va..kernel_va+len` 是当前宿主先前建立的有效映射。
    unsafe fn unmap_user(&self, kernel_va: u64, len: usize) -> Result<(), ()>;

    /// 在指定 CPU core 上启动新线程。
    ///
    /// 宿主实现必须尽力保证 `f(arg)` 在 `core_id` 对应的 core 上执行。
    fn spawn_thread_on_core(&self, core_id: u32, f: fn(usize), arg: usize);
}

/// 调度器操作系统接口 (`K3SchedulerOps`) 的具体宿主实现占位类型。
pub struct Caller {}
