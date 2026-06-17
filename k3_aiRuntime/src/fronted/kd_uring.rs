//! k3 ai runtime 的最小 channel 建立与 graph 提交通路。
//!
//! 当前约定：
//! - channel 共享内存由用户态用 `mmap(MAP_SHARED | MAP_ANONYMOUS)` 建立
//! - `/dev/k3_airunner` 的 `BUILD_CHANNEL` 负责让内核验证并保活这块共享区
//! - graph submit 仍然走单独的 ioctl

use std::{
    fs::{File, OpenOptions},
    io,
    mem::size_of,
    os::{
        fd::AsRawFd,
        raw::{c_int, c_ulong, c_void},
    },
    ptr,
    sync::{Arc, Mutex},
};

use lazy_static::lazy_static;
use ov_channels::{ChannelId, Message, Receiver, Sender, SharedMemory};
use crate::fronted::{AI_ABI_VERSION, AiGraphSubmitEntry};
use k3_aiUabi::*;
use k3_aiUabi::error::AiRuntimeErr;


// 共享内存请求/返回参数。
// 用户传入自己 mmap 出来的共享区地址和大小，内核校验它是否是 shared backend，
// 然后把对应 SharedPages 保活并回填 pid / flags 等元信息。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct K3AiChannelBuildParam {
    pub user_va: u64,
    pub size_bytes: u64,
    pub channel_count: u32,
    pub flags: u32,
    pub owner_pid: u32,
}

// 用户态保存下来的共享区句柄。
// `memory` 持有这段 mmap，避免 build_channel 返回后就丢失。
pub struct ChannelMemory {
    pub user_va: usize,
    pub size_bytes: usize,
    _memory: Arc<MmapMemory>,
}

pub struct UringChannel {
    dev: File,
    pub shared: ChannelMemory,
    graph_sender:Option<Sender<'static>>,
    complete_reciver:Option<Receiver<'static>>
}

// 当前进程先只允许建立一个 channel，共享区一直保留到进程退出。
lazy_static! {
    static ref CHANNEL_MEMORY: Mutex<Option<Arc<MmapMemory>>> = Mutex::new(None);
}

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, arg: usize) -> c_int;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: isize,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_FAILED: *mut c_void = !0 as *mut c_void;

// 持有 mmap 映射，Drop 时自动 munmap。
pub(crate) struct MmapMemory {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

unsafe impl Send for MmapMemory {}
unsafe impl Sync for MmapMemory {}

impl Drop for MmapMemory {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr.cast::<c_void>(), self.len);
        }
    }
}

impl MmapMemory {
    /// 建立一段 MAP_SHARED anonymous 内存，供 channel/tensor 交给内核保活和映射。
    pub(crate) fn new_shared(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mmap length is zero",
            ));
        }

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.cast_const()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// 打开 `/dev/k3_airunner`，mmap 共享内存并通过 `BUILD_CHANNEL` ioctl 让内核保活。
pub fn build_channel() -> Result<UringChannel, AiRuntimeErr> {
    let dev = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/k3_airunner")
        .map_err(|_| AiRuntimeErr::DeviceOpenFailed)?;

    let shared_size = size_of::<SharedMemory<K3_CHANNEL_COUNT>>();
    let mut memory = MmapMemory::new_shared(shared_size).map_err(|_| AiRuntimeErr::MmapFailed)?;
    let shared_ptr = memory.as_mut_ptr().cast::<c_void>();

    let shm = unsafe { &*(shared_ptr as *const SharedMemory<K3_CHANNEL_COUNT>) };
    shm.init();

    let mut build_param = K3AiChannelBuildParam {
        user_va: shared_ptr as usize as u64,
        size_bytes: shared_size as u64,
        channel_count: K3_CHANNEL_COUNT as u32,
        flags: 0,
        owner_pid: 0,
    };

    let ret = unsafe {
        ioctl(
            dev.as_raw_fd(),
            K3_AI_IOC_BUILD_CHANNEL as c_ulong,
            (&mut build_param as *mut K3AiChannelBuildParam) as usize,
        )
    };
    if ret < 0 {
        return Err(AiRuntimeErr::IoctlFailed);
    }

    let memory = Arc::new(memory);

    let shared = ChannelMemory {
        user_va: shared_ptr as usize,
        size_bytes: shared_size,
        _memory: memory.clone(),
    };

    {
        let mut slot = CHANNEL_MEMORY
            .lock()
            .expect("channel memory mutex poisoned");
        *slot = Some(memory);
    }

    let shm = unsafe { ov_channels::SharedMemory::<K3_CHANNEL_COUNT>::at(shared_ptr as usize) };
    let sender_channel_0 = shm
        .sender(ChannelId::new(K3_CHANNEL_SNEDERID))
        .map_err(|_| AiRuntimeErr::ChannelNotInitialized)?;
    let reciver_channel_1 = shm
        .receiver(ChannelId::new(K3_CHANNEL_RECIVERID))
        .map_err(|_| AiRuntimeErr::ChannelNotInitialized)?;

    Ok(UringChannel { dev, shared, graph_sender: Some(sender_channel_0), complete_reciver: Some(reciver_channel_1) })
}

/// 用户接口
/// 提交 graph 描述。当前仍然只通过 ioctl 把 `AiGraphSubmitEntry` 指针传给内核。
pub fn submit_graph(channel: &UringChannel, graph_entry: &AiGraphSubmitEntry) -> Result<(), AiRuntimeErr> {
    let va = channel.shared.user_va;

    if va == 0
        || channel.shared.size_bytes == 0
        || channel.shared._memory.len == 0
        || channel.shared._memory.ptr.is_null()
    {
        return Err(AiRuntimeErr::InvalidInput);
    }

    if graph_entry.abi_version != AI_ABI_VERSION {
        return Err(AiRuntimeErr::InvalidAbiVersion);
    }

    let sender = channel.graph_sender.ok_or(AiRuntimeErr::ChannelNotInitialized)?;

    let data = graph_entry.to_le_byte().ok_or(AiRuntimeErr::SerializeFailed)?;
    sender
        .try_send(&Message::data(data))
        .map_err(|_| AiRuntimeErr::SendFailed)?;

    let ret = unsafe {
        ioctl(
            channel.dev.as_raw_fd(),
            K3_AI_IOC_SUBMIT_GRAPH as c_ulong,
            graph_entry as *const _ as usize,
        )
    };
    if ret < 0 {
        return Err(AiRuntimeErr::IoctlFailed);
    }

    Ok(())
}

/// completion 通路后续再接。
pub fn wait_graph_complete(_graph_entry: &AiGraphSubmitEntry, channel: &UringChannel) -> Result<(), AiRuntimeErr> {
    let reciver = channel.complete_reciver.ok_or(AiRuntimeErr::ChannelNotInitialized)?;

    loop {
        if let Some(msg) = reciver.try_recv() {
            if let Some(token) = msg.as_notification() {
                if token == _graph_entry.user_token {
                    return Ok(());
                }
            }
        }
        std::thread::yield_now();
    }
}
