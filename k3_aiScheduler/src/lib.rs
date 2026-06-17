//! k3 芯片 AI 调度器平台 trait 定义。

#![no_std]

extern crate alloc;

pub mod kd_kring;
pub mod scheduler;

use core::{alloc::Layout, usize};

/// 调度器需要的操作系统接口。
pub trait K3SchedulerOps {
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
    unsafe fn map_user_to_kernel(&self, user_va: u64, len: usize) -> Result<u64, ()>;

    /// 取消用户地址到内核地址的映射。
    unsafe fn unmap_user(&self, kernel_va: u64, len: usize) -> Result<(), ()>;


    ///启动新线程
    fn spawn_thread(&self,f:fn(usize),arg:usize);
}

pub struct Caller{}