//! 包含内核调度器,内核运行时,用户态运行时的错误类型和display实现

use core::fmt;

/// 用户态 Runtime 错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiRuntimeErr {
    /// ABI 版本与内核不一致。
    InvalidAbiVersion,
    /// 输入参数非法。
    InvalidInput,
    /// channel 尚未初始化就被使用。
    ChannelNotInitialized,
    /// 提交数据序列化失败。
    SerializeFailed,
    /// 向 channel 发送消息失败。
    SendFailed,
    /// `ioctl` 系统调用失败。
    IoctlFailed,
    /// `mmap` 映射共享内存失败。
    MmapFailed,
    /// 打开设备节点失败。
    DeviceOpenFailed,
    /// 内存分配失败。
    AllocFailed,
    /// 张量 shape 非法。
    InvalidShape,
    /// 张量 layout 非法。
    InvalidLayout,
}

impl fmt::Display for AiRuntimeErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAbiVersion => write!(f, "ABI version mismatch"),
            Self::InvalidInput => write!(f, "invalid input parameter"),
            Self::ChannelNotInitialized => write!(f, "channel not initialized"),
            Self::SerializeFailed => write!(f, "serialization failed"),
            Self::SendFailed => write!(f, "failed to send message"),
            Self::IoctlFailed => write!(f, "ioctl failed"),
            Self::MmapFailed => write!(f, "mmap failed"),
            Self::DeviceOpenFailed => write!(f, "failed to open device"),
            Self::AllocFailed => write!(f, "allocation failed"),
            Self::InvalidShape => write!(f, "invalid tensor shape"),
            Self::InvalidLayout => write!(f, "invalid tensor layout"),
        }
    }
}

/// 调度器错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerErr {
    /// graph 结构非法。
    InvalidGraph,
    /// graph 解析失败。
    ParseFailed,
    /// graph 节点映射到 backend 失败。
    NodeMappingFailed,
    /// 算子执行失败。
    ExecutionFailed,
    /// 完成通知发送失败。
    NotificationFailed,
}

impl fmt::Display for SchedulerErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph => write!(f, "invalid graph"),
            Self::ParseFailed => write!(f, "failed to parse graph"),
            Self::NodeMappingFailed => write!(f, "failed to map node"),
            Self::ExecutionFailed => write!(f, "kernel execution failed"),
            Self::NotificationFailed => write!(f, "failed to send notification"),
        }
    }
}

/// 内核态 Runtime 错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnAiRuntimeErr {
    /// 输入参数非法。
    InvalidInput,
    /// ABI 版本不匹配。
    InvalidAbiVersion,
    /// 用户态地址非法或不可访问。
    BadAddress,
    /// 内核内存不足。
    NoMemory,
    /// 资源已存在。
    AlreadyExists,
    /// 操作会阻塞（非阻塞路径返回）。
    WouldBlock,
    /// 该操作暂不支持。
    NotSupported,
    /// 共享内存无效或未注册。
    InvalidSharedMemory,
    /// channel 内无待处理消息。
    ChannelEmpty,
    /// 内存映射失败。
    MapFailed,
}

impl fmt::Display for KnAiRuntimeErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput => write!(f, "invalid input"),
            Self::InvalidAbiVersion => write!(f, "ABI version mismatch"),
            Self::BadAddress => write!(f, "bad address"),
            Self::NoMemory => write!(f, "out of memory"),
            Self::AlreadyExists => write!(f, "already exists"),
            Self::WouldBlock => write!(f, "would block"),
            Self::NotSupported => write!(f, "operation not supported"),
            Self::InvalidSharedMemory => write!(f, "invalid shared memory"),
            Self::ChannelEmpty => write!(f, "channel is empty"),
            Self::MapFailed => write!(f, "memory mapping failed"),
        }
    }
}

/// Backend 算子执行错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendErr {
    /// 输入参数非法。
    InvalidInput,
    /// tensor 描述非法。
    InvalidTensor,
    /// 算子参数非法。
    InvalidAttr,
    /// 数据类型不支持。
    UnsupportedDtype,
    /// 算子不支持。
    UnsupportedOp,
    /// 算子执行失败。
    ExecutionFailed,
    /// 遇到空指针。
    NullPointer,
}

impl fmt::Display for BackendErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput => write!(f, "invalid input"),
            Self::InvalidTensor => write!(f, "invalid tensor"),
            Self::InvalidAttr => write!(f, "invalid attribute"),
            Self::UnsupportedDtype => write!(f, "unsupported data type"),
            Self::UnsupportedOp => write!(f, "unsupported operation"),
            Self::ExecutionFailed => write!(f, "execution failed"),
            Self::NullPointer => write!(f, "null pointer"),
        }
    }
}
