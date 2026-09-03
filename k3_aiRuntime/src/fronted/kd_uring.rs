//! k3 ai runtime 的最小 channel 建立与 graph 提交通路。
//!
//! 当前约定：
//! - channel 共享内存由用户态用 `mmap(MAP_SHARED | MAP_ANONYMOUS)` 建立
//! - `/dev/k3_airunner` 的 `BUILD_CHANNEL` 负责让内核验证并保活这块共享区
//! - graph submit 仍然走单独的 ioctl

use std::{
    fs::{File, OpenOptions},
    io,
    marker::PhantomData,
    mem::size_of,
    os::{
        fd::AsRawFd,
        raw::{c_char, c_int, c_uint, c_ulong, c_void},
    },
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::fronted::{AI_ABI_VERSION, AiGraphSubmitEntry};
use k3_ai_uabi::error::{AiCompletion, AiRuntimeErr};
use k3_ai_uabi::*;
use lazy_static::lazy_static;
use ov_channels::{ChannelId, Message, Receiver, Sender, SharedMemory};

/// 用户态保存下来的共享区句柄。
///
/// `_memory` 持有这段 mmap，避免 build_channel 返回后就丢失。
pub struct ChannelMemory {
    /// 共享区用户态虚拟地址。
    pub user_va: usize,
    /// 共享区字节数。
    pub size_bytes: usize,
    /// 持有底层 mmap，保证共享区在 channel 存活期间不被释放。
    _memory: Arc<MmapMemory>,
}

/// 一条建立好的提交通道：设备句柄 + 共享内存 + 收发端。
pub struct UringChannel {
    /// `/dev/k3_airunner` 的打开句柄。
    dev: File,
    /// channel 共享内存句柄。
    pub shared: ChannelMemory,
    /// graph 提交发送端（channel 0）。
    graph_sender: Option<Sender<'static>>,
    /// 完成通知接收端（channel 1）。
    complete_reciver: Option<Receiver<'static>>,
}

// 当前进程先只允许建立一个 channel，共享区一直保留到进程退出。
lazy_static! {
    /// 进程级共享区持有槽：保证 channel 共享内存直到进程退出前一直存活。
    static ref CHANNEL_MEMORY: Mutex<Option<Arc<MmapMemory>>> = Mutex::new(None);
}

/// 标记本进程是否已经成功完成过 BUILD_CHANNEL ioctl。
/// 内核侧已有幂等保护，用户侧再加一层避免重复 open/mmap/ioctl。
static CHANNEL_BUILT: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    /// libc `ioctl`：向设备发送控制命令。
    fn ioctl(fd: c_int, request: c_ulong, arg: usize) -> c_int;
    /// libc `mmap`：建立内存映射。
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: isize,
    ) -> *mut c_void;
    /// libc `munmap`：解除内存映射。
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    /// libc `printf`：用户态 ABI mismatch 时直接给控制台提示。
    fn printf(fmt: *const c_char, args: ...) -> c_int;
}

/// 用户态本地 ABI mismatch 提醒。
fn printf_abi_mismatch(user_abi: u32, runtime_abi: u32) {
    unsafe {
        let _ = printf(
            b"k3_aiRuntime: rejected abi mismatch user=%u runtime=%u\n\0"
                .as_ptr()
                .cast::<c_char>(),
            user_abi as c_uint,
            runtime_abi as c_uint,
        );
    }
}

/// `mmap` 保护位：页可读。
const PROT_READ: c_int = 0x1;
/// `mmap` 保护位：页可写。
const PROT_WRITE: c_int = 0x2;
/// `mmap` 标志：映射对其他进程/内核可见的共享内存。
const MAP_SHARED: c_int = 0x01;
/// `mmap` 标志：匿名映射，不关联文件。
const MAP_ANONYMOUS: c_int = 0x20;
/// `mmap` 失败时返回的哨兵指针。
const MAP_FAILED: *mut c_void = !0 as *mut c_void;

/// 持有 mmap 映射，Drop 时自动 munmap。
pub(crate) struct MmapMemory {
    /// 映射区首地址。
    pub(crate) ptr: *mut u8,
    /// 映射区字节数。
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

    /// 映射区首地址的只读指针。
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.cast_const()
    }

    /// 映射区首地址的可写指针。
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// 映射区字节数。
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// 对调用方已有用户态地址的非拥有视图。
///
/// 这个类型不调用 `mmap`、不拥有也不会在 Drop 时 `munmap` 这段内存。K3 内核在
/// graph submit 时通过 `map_user_to_kernel` 固定页面并创建 kernel alias；因此 view
/// 的 Rust 生命周期只需要覆盖同步的 submit + completion 区间。
///
/// `PhantomData<*mut ()>` 故意让它不能跨线程发送或共享。FFI 调用者承诺 buffer
/// 在 `k3_ort_run_node` 返回前保持有效，runtime 不应把这个借用扩展到该调用之外。
pub(crate) struct BorrowedMemory {
    /// 借用的用户态首地址。
    ptr: *mut u8,
    /// 借用范围的字节数。
    len: usize,
    /// 禁止将调用方内存借用跨线程移动。
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl BorrowedMemory {
    /// 从调用方拥有的连续用户态 buffer 建立一个非拥有 view。
    ///
    /// # Safety
    ///
    /// `ptr..ptr + len` 必须在返回的 `BorrowedMemory` 生命周期内保持有效、可访问，
    /// 且不得由此 view 之外的并发访问破坏 Rust 的别名规则。
    pub(crate) unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> io::Result<Self> {
        if ptr.is_null() || len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "borrowed buffer is null or empty",
            ));
        }

        let _end = (ptr as usize)
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "buffer range overflows"))?;

        Ok(Self {
            ptr,
            len,
            _not_send_or_sync: PhantomData,
        })
    }

    /// 借用范围的只读首地址。
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.cast_const()
    }

    /// 借用范围的可写首地址。
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// 借用范围的字节数。
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// 打开 `/dev/k3_airunner`，mmap 共享内存并通过 `BUILD_CHANNEL` ioctl 让内核保活。
/// 进程内只允许第一次调用真正执行 ioctl；后续调用复用已建立的共享内存，
/// 不再重复 open/mmap/ioctl，内核侧也有幂等保护作为双重保障。
pub fn build_channel() -> Result<UringChannel, AiRuntimeErr> {
    // 用户态幂等保护：如果已经成功 build 过，复用已有共享内存，跳过 open/mmap/ioctl。
    if CHANNEL_BUILT.load(Ordering::Acquire) {
        let memory = {
            let slot = CHANNEL_MEMORY
                .lock()
                .map_err(|_| AiRuntimeErr::IoctlFailed)?;
            slot.clone().ok_or(AiRuntimeErr::ChannelNotInitialized)?
        };
        let shared_ptr = memory.as_ptr() as usize;

        let shm = unsafe { ov_channels::SharedMemory::<K3_CHANNEL_COUNT>::at(shared_ptr) };
        let sender_channel_0 = shm
            .sender(ChannelId::new(K3_CHANNEL_SNEDERID))
            .map_err(|_| AiRuntimeErr::ChannelNotInitialized)?;
        let reciver_channel_1 = shm
            .receiver(ChannelId::new(K3_CHANNEL_RECIVERID))
            .map_err(|_| AiRuntimeErr::ChannelNotInitialized)?;

        let dev = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/k3_airunner")
            .map_err(|_| AiRuntimeErr::DeviceOpenFailed)?;

        return Ok(UringChannel {
            dev,
            shared: ChannelMemory {
                user_va: shared_ptr,
                size_bytes: memory.len(),
                _memory: memory,
            },
            graph_sender: Some(sender_channel_0),
            complete_reciver: Some(reciver_channel_1),
        });
    }

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

    let mut build_param = K3AiChannelBuildParam::new(
        shared_ptr as usize as u64,
        shared_size as u64,
        K3_CHANNEL_COUNT as u32,
    );

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

    // 标记已成功 build，后续调用跳过 ioctl，复用已有共享内存。
    CHANNEL_BUILT.store(true, Ordering::Release);

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

    Ok(UringChannel {
        dev,
        shared,
        graph_sender: Some(sender_channel_0),
        complete_reciver: Some(reciver_channel_1),
    })
}

/// 用户接口
/// 提交 graph 描述。当前仍然只通过 ioctl 把 `AiGraphSubmitEntry` 指针传给内核。
pub fn submit_graph(
    channel: &UringChannel,
    graph_entry: &AiGraphSubmitEntry,
) -> Result<(), AiRuntimeErr> {
    let va = channel.shared.user_va;

    if va == 0
        || channel.shared.size_bytes == 0
        || channel.shared._memory.len == 0
        || channel.shared._memory.ptr.is_null()
    {
        return Err(AiRuntimeErr::InvalidInput);
    }

    if graph_entry.abi_version != AI_ABI_VERSION {
        printf_abi_mismatch(graph_entry.abi_version, AI_ABI_VERSION);
        return Err(AiRuntimeErr::InvalidAbiVersion);
    }

    let sender = channel
        .graph_sender
        .ok_or(AiRuntimeErr::ChannelNotInitialized)?;

    let data = graph_entry
        .to_le_byte()
        .ok_or(AiRuntimeErr::SerializeFailed)?;
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

/// completion 通道等待 graph 执行完成通知。
/// 成功返回 `Ok(())`；失败返回 `GraphExecutionFailed` 携带失败节点信息。
pub fn wait_graph_complete(
    graph_entry: &AiGraphSubmitEntry,
    channel: &UringChannel,
) -> Result<(), AiRuntimeErr> {
    let reciver = channel
        .complete_reciver
        .ok_or(AiRuntimeErr::ChannelNotInitialized)?;
    loop {
        let Some(msg) = reciver.try_recv() else {
            std::thread::yield_now();
            continue;
        };
        // 新版 completion 走 Message::data，携带 AiCompletion。
        if let Some(payload) = msg.as_data() {
            if payload.len() >= core::mem::size_of::<AiCompletion>() {
                let completion =
                    unsafe { core::ptr::read_unaligned(payload.as_ptr().cast::<AiCompletion>()) };
                if completion.user_token == graph_entry.user_token.get() {
                    if completion.status == 0 {
                        return Ok(());
                    }

                    return Err(AiRuntimeErr::GraphExecutionFailed {
                        node_id: completion.failed_node_id,
                        op: completion.failed_node_op,
                        backend_err: completion.failed_node_err,
                    });
                }
            }
            continue;
        }
        // 兼容旧版 notification（仅 token，无错误信息）。
        if let Some(token) = msg.as_notification()
            && token == graph_entry.user_token.get()
        {
            return Ok(());
        }
        // 没拿到再等一下
        std::thread::yield_now();
    }
}
