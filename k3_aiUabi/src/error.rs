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
    /// graph 执行失败，携带失败节点详情。
    GraphExecutionFailed {
        /// 第一个失败节点的 id。
        node_id: u32,
        /// 失败算子的 op 码。
        op: u8,
        /// BackendErr as u8。
        backend_err: u8,
    },
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
            Self::GraphExecutionFailed {
                node_id,
                op,
                backend_err,
            } => write!(
                f,
                "graph execution failed at node {} (op={}): backend error {}",
                node_id, op, backend_err
            ),
        }
    }
}

/// 调度器错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SchedulerErr {
    /// graph 结构非法。
    InvalidGraph = 1,
    /// graph 解析失败。
    ParseFailed = 2,
    /// graph 节点映射到 backend 失败。
    NodeMappingFailed = 3,
    /// 算子执行失败。
    ExecutionFailed = 4,
    /// 完成通知发送失败。
    NotificationFailed = 5,
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
#[repr(u8)]
pub enum KnAiRuntimeErr {
    /// 输入参数非法。
    InvalidInput = 1,
    /// ABI 版本不匹配。
    InvalidAbiVersion = 2,
    /// 用户态地址非法或不可访问。
    BadAddress = 3,
    /// 内核内存不足。
    NoMemory = 4,
    /// 资源已存在。
    AlreadyExists = 5,
    /// 操作会阻塞（非阻塞路径返回）。
    WouldBlock = 6,
    /// 该操作暂不支持。
    NotSupported = 7,
    /// 共享内存无效或未注册。
    InvalidSharedMemory = 8,
    /// channel 内无待处理消息。
    ChannelEmpty = 9,
    /// 内存映射失败。
    MapFailed = 10,
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
///
/// 0 保留表示"无错误"；码值稳定，可直接写入 `AiGraphState.error_flag`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BackendErr {
    /// 输入参数非法。
    InvalidInput = 1,
    /// tensor 描述非法。
    InvalidTensor = 2,
    /// 算子参数非法。
    InvalidAttr = 3,
    /// 数据类型不支持。
    UnsupportedDtype = 4,
    /// 算子不支持。
    UnsupportedOp = 5,
    /// 算子执行失败。
    ExecutionFailed = 6,
    /// 遇到空指针。
    NullPointer = 7,
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

impl BackendErr {
    /// 从 `u8` 码值还原 `BackendErr`，无匹配时返回 `None`。
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::InvalidInput),
            2 => Some(Self::InvalidTensor),
            3 => Some(Self::InvalidAttr),
            4 => Some(Self::UnsupportedDtype),
            5 => Some(Self::UnsupportedOp),
            6 => Some(Self::ExecutionFailed),
            7 => Some(Self::NullPointer),
            _ => None,
        }
    }
}

/// graph 执行完成后的回执，通过 completion channel 的 `Message::data` 发送。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AiCompletion {
    /// 匹配提交的 user token。
    pub user_token: u32,
    /// 第一个失败节点的 node_id，所有节点成功时为 u32::MAX。
    pub failed_node_id: u32,
    /// 整体状态：0=成功，非0=`SchedulerErr as u8`。
    pub status: u8,
    /// 第一个失败节点的 `BackendErr as u8`，无失败时为 0。
    pub failed_node_err: u8,
    /// 第一个失败节点的算子 op 码。
    pub failed_node_op: u8,
    /// 对齐填充。
    pub reserved: [u8; 5],
}

const _: () = assert!(core::mem::size_of::<AiCompletion>() == 16);
const _: () = assert!(core::mem::align_of::<AiCompletion>() == 4);
