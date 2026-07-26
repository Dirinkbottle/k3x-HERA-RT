//! 设备控制面 ABI。
//!
//! 这里的类型会在用户态 runtime 和 Starry 内核之间直接复制传递，只能使用
//! `#[repr(C)]` / `#[repr(transparent)]` 加固定宽度整数。不要把 `Box`、`Vec`、
//! 引用、切片或任意 Rust 私有布局类型放进这里。
use crate::AI_ABI_VERSION;
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

/// ovchannel 共享区注册参数。
///
/// 用户态传入自己 mmap 出来的共享区地址和大小，内核校验 ABI 版本与 shared backend，
/// 然后把对应 SharedPages 保活并回填 owner pid。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct K3AiChannelBuildParam {
    /// ABI 版本，必须等于 `AI_ABI_VERSION`。
    pub abi_version: u32,
    /// 建立标志位，阶段一保留。
    pub flags: u32,
    /// 共享区用户态虚拟地址。
    pub user_va: u64,
    /// 共享区字节数。
    pub size_bytes: u64,
    /// 共享区内 channel 的数量。
    pub channel_count: u32,
    /// 内核回填的持有者 pid。
    pub owner_pid: u32,
    /// 预留字段，保持 8 字节对齐并留给后续扩展。
    pub reserved0: u64,
}

impl K3AiChannelBuildParam {
    /// 构造一次 channel build 请求。
    pub const fn new(user_va: u64, size_bytes: u64, channel_count: u32) -> Self {
        Self {
            abi_version: AI_ABI_VERSION,
            flags: 0,
            user_va,
            size_bytes,
            channel_count,
            owner_pid: 0,
            reserved0: 0,
        }
    }
}

#[allow(dead_code, missing_docs, clippy::missing_docs_in_private_items)]
mod abi_layout {
    use super::*;

    #[repr(C)]
    struct RawK3AiChannelBuildParam {
        abi_version: u32,
        flags: u32,
        user_va: u64,
        size_bytes: u64,
        channel_count: u32,
        owner_pid: u32,
        reserved0: u64,
    }

    const _: () = assert!(
        core::mem::size_of::<K3AiChannelBuildParam>()
            == core::mem::size_of::<RawK3AiChannelBuildParam>()
    );
    const _: () = assert!(
        core::mem::align_of::<K3AiChannelBuildParam>()
            == core::mem::align_of::<RawK3AiChannelBuildParam>()
    );
    const _: () = assert!(
        core::mem::offset_of!(K3AiChannelBuildParam, abi_version)
            == core::mem::offset_of!(RawK3AiChannelBuildParam, abi_version)
    );
    const _: () = assert!(
        core::mem::offset_of!(K3AiChannelBuildParam, user_va)
            == core::mem::offset_of!(RawK3AiChannelBuildParam, user_va)
    );
    const _: () = assert!(
        core::mem::offset_of!(K3AiChannelBuildParam, size_bytes)
            == core::mem::offset_of!(RawK3AiChannelBuildParam, size_bytes)
    );
    const _: () = assert!(
        core::mem::offset_of!(K3AiChannelBuildParam, channel_count)
            == core::mem::offset_of!(RawK3AiChannelBuildParam, channel_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(K3AiChannelBuildParam, owner_pid)
            == core::mem::offset_of!(RawK3AiChannelBuildParam, owner_pid)
    );
    const _: () = assert!(
        core::mem::offset_of!(K3AiChannelBuildParam, reserved0)
            == core::mem::offset_of!(RawK3AiChannelBuildParam, reserved0)
    );
}
