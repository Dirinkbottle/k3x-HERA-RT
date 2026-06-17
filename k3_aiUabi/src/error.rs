//! 包含内核调度器,内核运行时,用户态运行时的错误类型和display实现

use core::fmt;

/// 用户态 Runtime 错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiRuntimeErr {
    InvalidAbiVersion,
    InvalidInput,
    ChannelNotInitialized,
    SerializeFailed,
    SendFailed,
    IoctlFailed,
    MmapFailed,
    DeviceOpenFailed,
    AllocFailed,
    InvalidShape,
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
    InvalidGraph,
    ParseFailed,
    NodeMappingFailed,
    ExecutionFailed,
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
    InvalidInput,
    InvalidAbiVersion,
    BadAddress,
    NoMemory,
    AlreadyExists,
    WouldBlock,
    NotSupported,
    InvalidSharedMemory,
    ChannelEmpty,
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
    InvalidInput,
    InvalidTensor,
    InvalidAttr,
    UnsupportedDtype,
    UnsupportedOp,
    ExecutionFailed,
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


