//! Per-core AI graph 调度器和常驻 worker。
//!
//! 每个实际运行 core 独占一个 scheduler/ready queue/worker。这样 submit path
//! 不会跨 core 写入另一个 worker 的无锁环形队列，避免真板子上缺少跨核数据同步时
//! 出现消费者读不到或读到旧数据的问题。

use crate::{K3SchedulerOps, kd_kring::TaskLink};
use alloc::boxed::Box;
use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering},
};
use k3_ai_uabi::AiCompletion;
use k3_ai_uabi::{UserToken, error::SchedulerErr};
use k3_kernel_backend::k3_run_kernel;
use log::{error, warn};
use ov_channels::{Message, Sender};

/// 单个 scheduler ready queue 能容纳的最大任务数。
pub const SCHEDULER_QUEUE_CAPACITY: usize = 64;

/// 当前支持缓存的最大 CPU core id 数量。
const MAX_SCHEDULER_CORES: usize = 64;

/// scheduler 尚未初始化。
const SCHEDULER_UNINIT: u8 = 0;
/// 某个提交者正在初始化 scheduler。
const SCHEDULER_INITIALIZING: u8 = 1;
/// scheduler 和队列已经可以访问。
const SCHEDULER_READY: u8 = 2;

/// 等待入队时可能遇到的内部状态。
#[derive(Debug, Eq, PartialEq)]
enum QueuePushError {
    /// 环形队列当前已满。
    Full,
    /// 调用方不属于该队列绑定的实际 core。
    CoreMismatch,
    /// 调用方没有提供待写入的任务。
    MissingTask,
}

/// 出队时可能遇到的内部状态。
#[derive(Debug, Eq, PartialEq)]
enum QueuePopError {
    /// 调用方不属于该队列绑定的实际 core。
    CoreMismatch,
}

/// 环形队列的一个 sequence slot。
struct QueueSlot {
    /// slot 的生产/消费代次。
    sequence: AtomicUsize,
    /// 只有成功占用该代次的 producer 或 consumer 才能访问任务值。
    value: UnsafeCell<MaybeUninit<AINodeUnit>>,
}

impl QueueSlot {
    /// 构造一个初始可写的 slot。
    fn new(sequence: usize) -> Self {
        Self {
            sequence: AtomicUsize::new(sequence),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

// SAFETY: sequence 的 acquire/release 协议保证同一代 slot 只有一个 writer 或 reader。
unsafe impl Sync for QueueSlot {}

/// 绑定实际 CPU core 的有界无锁任务队列。
struct PerCoreTaskQueue {
    /// 允许访问该队列的实际 CPU core id。
    core_id: u32,
    /// 下一个 consumer 竞争的位置。
    head: AtomicUsize,
    /// 下一个 producer 竞争的位置。
    tail: AtomicUsize,
    /// 固定容量的 sequence slots。
    slots: [QueueSlot; SCHEDULER_QUEUE_CAPACITY],
}

impl PerCoreTaskQueue {
    /// 为指定实际 CPU core 创建一个空队列。
    fn new(core_id: u32) -> Self {
        Self {
            core_id,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            slots: core::array::from_fn(QueueSlot::new),
        }
    }

    /// 尝试写入一次；队列满时保留 `task` 的所有权交给调用方重试。
    fn try_push(
        &self,
        expected_core_id: u32,
        task: &mut Option<AINodeUnit>,
    ) -> Result<(), QueuePushError> {
        if expected_core_id != self.core_id {
            return Err(QueuePushError::CoreMismatch);
        }
        if task.is_none() {
            return Err(QueuePushError::MissingTask);
        }

        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let slot = &self.slots[tail % SCHEDULER_QUEUE_CAPACITY];
            let sequence = slot.sequence.load(Ordering::Acquire);
            let delta = sequence.wrapping_sub(tail) as isize;

            if delta == 0 {
                if self
                    .tail
                    .compare_exchange_weak(
                        tail,
                        tail.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    let unit = task.take().ok_or(QueuePushError::MissingTask)?;
                    // SAFETY: the successful tail CAS gives this producer exclusive ownership
                    // of this slot until sequence is published below.
                    unsafe {
                        (*slot.value.get()).write(unit);
                    }
                    slot.sequence.store(tail.wrapping_add(1), Ordering::Release);
                    return Ok(());
                }
            } else if delta < 0 {
                return Err(QueuePushError::Full);
            } else {
                core::hint::spin_loop();
            }
        }
    }

    /// 尝试读出队首任务；空队列返回 `Ok(None)`。
    fn try_pop(&self, expected_core_id: u32) -> Result<Option<AINodeUnit>, QueuePopError> {
        if expected_core_id != self.core_id {
            return Err(QueuePopError::CoreMismatch);
        }

        loop {
            let head = self.head.load(Ordering::Relaxed);
            let slot = &self.slots[head % SCHEDULER_QUEUE_CAPACITY];
            let expected_sequence = head.wrapping_add(1);
            let sequence = slot.sequence.load(Ordering::Acquire);
            let delta = sequence.wrapping_sub(expected_sequence) as isize;

            if delta == 0 {
                if self
                    .head
                    .compare_exchange_weak(
                        head,
                        head.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // SAFETY: the successful head CAS gives this consumer exclusive ownership
                    // of the initialized value published by the producer's release store.
                    let unit = unsafe { (*slot.value.get()).assume_init_read() };
                    slot.sequence.store(
                        head.wrapping_add(SCHEDULER_QUEUE_CAPACITY),
                        Ordering::Release,
                    );
                    return Ok(Some(unit));
                }
            } else if delta < 0 {
                return Ok(None);
            } else {
                core::hint::spin_loop();
            }
        }
    }

    /// 返回用于日志和诊断的近似队列长度。
    fn len_approx(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head).min(SCHEDULER_QUEUE_CAPACITY)
    }
}

/// AI graph 调度任务，包含执行链和完成通知端点。
pub struct AINodeUnit {
    /// 用户提交时携带的 completion token。
    pub user_token: UserToken,
    /// 负责释放本次 graph tensor kernel alias 的宿主回调。
    pub caller: Box<dyn K3SchedulerOps>,
    /// 内核已 pin 的 completion channel sender。
    pub complete_sender: Sender<'static>,
    /// 已按依赖关系收敛的 graph 节点链。
    pub tasklink: TaskLink,
}

/// FIFO graph 调度器。
///
/// 当前一个实例对应一个实际 CPU core，worker 也固定在同一个 core 上。
pub struct GraphScheduler {
    /// 该 scheduler 绑定的实际 CPU core id。
    core_id: u32,
    /// 无锁 ready queue。
    ready_queue: PerCoreTaskQueue,
}

impl GraphScheduler {
    /// 创建一个绑定 core 0 的 scheduler。
    pub fn new() -> Self {
        Self::new_with_core(0)
    }

    /// 使用指定实际 core 构造 scheduler。
    fn new_with_core(core_id: u32) -> Self {
        Self {
            core_id,
            ready_queue: PerCoreTaskQueue::new(core_id),
        }
    }

    /// 返回 scheduler 绑定的实际 CPU core id。
    fn core_id(&self) -> u32 {
        self.core_id
    }

    /// 在 scheduler 自身绑定的队列上阻塞入队。
    pub fn push_task(&self, task: AINodeUnit) -> Result<(), SchedulerErr> {
        self.push_task_for_core(self.core_id, task)
    }

    /// 为指定实际 core 阻塞入队，队列满时持续自旋等待。
    fn push_task_for_core(
        &self,
        expected_core_id: u32,
        task: AINodeUnit,
    ) -> Result<(), SchedulerErr> {
        if expected_core_id != self.core_id {
            warn!(
                "scheduler push core mismatch: scheduler_core={}, expected_core={}",
                self.core_id, expected_core_id
            );
            return Err(SchedulerErr::InvalidGraph);
        }

        let token = task.user_token;
        let node_count = task.tasklink.ordered_nodes.len();
        let before = self.queue_len_approx();
        let mut task = Some(task);
        let mut warned_full = false;

        loop {
            match self.ready_queue.try_push(expected_core_id, &mut task) {
                Ok(()) => {
                    warn!(
                        "scheduler push_task: scheduler={:#x}, core_id={}, token={}, \
                         node_count={}, before_len={}, after_len={}",
                        self as *const _ as usize,
                        expected_core_id,
                        token,
                        node_count,
                        before,
                        self.queue_len_approx()
                    );
                    return Ok(());
                }
                Err(QueuePushError::Full) => {
                    if !warned_full {
                        warn!("Queue full , it's time to move other core!");
                        warned_full = true;
                    }
                    core::hint::spin_loop();
                }
                Err(QueuePushError::CoreMismatch | QueuePushError::MissingTask) => {
                    warn!(
                        "scheduler push rejected: scheduler_core={}, expected_core={}, has_task={}",
                        self.core_id,
                        expected_core_id,
                        task.is_some()
                    );
                    return Err(SchedulerErr::InvalidGraph);
                }
            }
        }
    }

    /// 从 scheduler 自身绑定的队列弹出一个任务。
    pub fn take_task(&self) -> Option<AINodeUnit> {
        self.take_task_for_core(self.core_id)
    }

    /// 为指定实际 core 弹出一个任务。
    fn take_task_for_core(&self, expected_core_id: u32) -> Option<AINodeUnit> {
        match self.ready_queue.try_pop(expected_core_id) {
            Ok(unit) => unit,
            Err(QueuePopError::CoreMismatch) => {
                warn!(
                    "scheduler pop core mismatch: scheduler_core={}, expected_core={}",
                    self.core_id, expected_core_id
                );
                None
            }
        }
    }

    /// 返回用于日志和状态检查的近似 ready queue 长度。
    pub fn queue_len_approx(&self) -> usize {
        self.ready_queue.len_approx()
    }
}

impl Default for GraphScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// 单个实际 CPU core 对应的 scheduler slot。
struct SchedulerSlot {
    /// slot 内 scheduler 的初始化状态。
    init_state: AtomicU8,
    /// slot 内 scheduler 对象；READY 后不再替换或释放。
    scheduler: UnsafeCell<Option<GraphScheduler>>,
    /// worker 是否正在执行 graph。
    worker_busy: AtomicBool,
    /// worker 当前正在执行的用户 token，空闲时为 0。
    worker_token: AtomicU32,
    /// worker 启动时记录的实际 CPU core id。
    worker_core_id: AtomicU32,
}

impl SchedulerSlot {
    /// 构造一个空 scheduler slot。
    const fn new() -> Self {
        Self {
            init_state: AtomicU8::new(SCHEDULER_UNINIT),
            scheduler: UnsafeCell::new(None),
            worker_busy: AtomicBool::new(false),
            worker_token: AtomicU32::new(0),
            worker_core_id: AtomicU32::new(0),
        }
    }
}

// SAFETY: 每个 slot 的 init_state 保证只有 CAS 成功的初始化者写入一次；READY 的 release
// store 与读者的 acquire load 建立 happens-before，之后 scheduler 只通过原子队列访问。
unsafe impl Sync for SchedulerSlot {}

/// 按真实 CPU core id 索引的 scheduler slots。
static SCHEDULER_SLOTS: [SchedulerSlot; MAX_SCHEDULER_CORES] =
    [const { SchedulerSlot::new() }; MAX_SCHEDULER_CORES];

/// 尝试成为指定初始化状态的唯一 owner。
fn try_claim_scheduler_initialization(state: &AtomicU8) -> bool {
    state
        .compare_exchange(
            SCHEDULER_UNINIT,
            SCHEDULER_INITIALIZING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
}

/// 根据实际 core id 取得 scheduler slot。
fn scheduler_slot(core_id: u32) -> Result<&'static SchedulerSlot, SchedulerErr> {
    SCHEDULER_SLOTS
        .get(core_id as usize)
        .ok_or(SchedulerErr::InvalidGraph)
}

/// 等待指定 core 的 scheduler 初始化完成并返回共享引用。
fn scheduler_after_ready(core_id: u32) -> Result<&'static GraphScheduler, SchedulerErr> {
    let slot = scheduler_slot(core_id)?;
    loop {
        match slot.init_state.load(Ordering::Acquire) {
            SCHEDULER_READY => {
                // SAFETY: READY 的 acquire load 观察到初始化者的 release store，cell 中的
                // scheduler 已完成构造且在内核生命周期内不会被替换或释放。
                return unsafe { (&*slot.scheduler.get()).as_ref() }
                    .ok_or(SchedulerErr::InvalidGraph);
            }
            SCHEDULER_INITIALIZING => core::hint::spin_loop(),
            _ => return Err(SchedulerErr::InvalidGraph),
        }
    }
}

/// 初始化指定 core 的 scheduler，或等待并发初始化者完成。
fn ensure_scheduler(
    caller: &dyn K3SchedulerOps,
    core_id: u32,
) -> Result<&'static GraphScheduler, SchedulerErr> {
    let slot = scheduler_slot(core_id)?;
    if try_claim_scheduler_initialization(&slot.init_state) {
        let scheduler = GraphScheduler::new_with_core(core_id);

        // SAFETY: 初始化状态 CAS 保证当前线程是唯一 writer，READY 发布前没有读者访问。
        unsafe {
            *slot.scheduler.get() = Some(scheduler);
        }
        slot.init_state.store(SCHEDULER_READY, Ordering::Release);
        caller.spawn_thread_on_core(core_id, worker, core_id as usize);
        warn!(
            "scheduler initialized: scheduler={:#x}, core_id={}",
            slot.scheduler.get() as usize,
            core_id
        );
    }
    scheduler_after_ready(core_id)
}

/// 内核 graph 提交入口，将已解析任务链放入常驻 worker 的 ready queue。
pub fn run_graph(
    user_token: UserToken,
    caller: Box<dyn K3SchedulerOps>,
    complete_sender: Sender<'static>,
    tasklink: TaskLink,
) -> Result<(), SchedulerErr> {
    let submit_core_id = caller.current_core_id();
    let slot = scheduler_slot(submit_core_id)?;
    warn!(
        "scheduler run_graph enter: token={}, task_nodes={}, submit_core={}, init_state={}, inited={}, \
         worker_busy={}, worker_token={}, worker_core_id={}",
        user_token,
        tasklink.ordered_nodes.len(),
        submit_core_id,
        slot.init_state.load(Ordering::Acquire),
        slot.init_state.load(Ordering::Acquire) == SCHEDULER_READY,
        slot.worker_busy.load(Ordering::Acquire),
        slot.worker_token.load(Ordering::Acquire),
        slot.worker_core_id.load(Ordering::Acquire)
    );

    let scheduler = ensure_scheduler(caller.as_ref(), submit_core_id)?;
    let unit = AINodeUnit {
        user_token,
        caller,
        complete_sender,
        tasklink,
    };

    warn!(
        "scheduler run_graph push begin: scheduler={:#x}, core_id={}, token={}",
        scheduler as *const _ as usize, submit_core_id, user_token
    );
    scheduler.push_task_for_core(submit_core_id, unit)?;
    warn!(
        "scheduler run_graph push done: scheduler={:#x}, core_id={}, token={}, \
         queue_len_approx={}",
        scheduler as *const _ as usize,
        submit_core_id,
        user_token,
        scheduler.queue_len_approx()
    );
    Ok(())
}



/// 释放一张 graph 在 submit 阶段为 tensor 创建的所有 kernel alias。
///
/// 这必须在 worker 停止访问 tensor 后执行。失败时继续释放剩余映射并只记录日志，
/// 因为 caller 仍应收到该 graph 的 completion，不能让一个清理失败卡死 worker。
fn release_tensor_mappings(caller: &dyn K3SchedulerOps, tasklink: &TaskLink, token: UserToken) {
    for node in tasklink.iter() {
        let total_count = match node.desc.input_count.checked_total(node.desc.output_count) {
            Ok(total_count) if total_count <= node.desc.tensors.len() => total_count,
            Ok(total_count) => {
                error!(
                    "worker mapping cleanup rejected oversized tensor count: token={}, node_id={}, \
                     total={}, capacity={}",
                    token,
                    node.node_id,
                    total_count,
                    node.desc.tensors.len()
                );
                continue;
            }
            Err(error) => {
                error!(
                    "worker mapping cleanup rejected invalid tensor count: token={}, node_id={}, \
                     error={:?}",
                    token, node.node_id, error
                );
                continue;
            }
        };

        for tensor in &node.desc.tensors[..total_count] {
            let kernel_va = tensor.kernel_va.get();
            if kernel_va == 0 {
                continue;
            }
            let size_bytes = match tensor.size_bytes.try_as_usize() {
                Ok(size_bytes) if size_bytes != 0 => size_bytes,
                Ok(_) | Err(_) => {
                    error!(
                        "worker mapping cleanup rejected invalid tensor size: token={}, node_id={}, \
                         kernel_va={:#x}, size={:#x}",
                        token,
                        node.node_id,
                        kernel_va,
                        tensor.size_bytes.get()
                    );
                    continue;
                }
            };

            // SAFETY: `kernel_va` and `size_bytes` came from this graph's prior
            // successful `map_user_to_kernel` call. No backend work remains.
            if unsafe { caller.unmap_user(kernel_va, size_bytes) }.is_err() {
                error!(
                    "worker mapping cleanup failed: token={}, node_id={}, kernel_va={:#x}, \
                     size={:#x}",
                    token, node.node_id, kernel_va, size_bytes
                );
            }
        }
    }
}

/// 常驻 graph worker；`arg` 是该 worker 绑定的实际 CPU core id。
pub fn worker(arg: usize) {
    let expected_core_id = match u32::try_from(arg) {
        Ok(core_id) => core_id,
        Err(_) => {
            error!("scheduler worker received oversized core id: arg={}", arg);
            return;
        }
    };
    let slot = match scheduler_slot(expected_core_id) {
        Ok(slot) => slot,
        Err(err) => {
            error!(
                "scheduler worker received invalid core id: core_id={}, err={:?}",
                expected_core_id, err
            );
            return;
        }
    };
    slot.worker_core_id
        .store(expected_core_id, Ordering::Release);
    warn!("worker start: core_id={}, arg={}", expected_core_id, arg);

    warn!(
        "worker waiting for scheduler ready: core_id={}, init_state={}",
        expected_core_id,
        slot.init_state.load(Ordering::Acquire)
    );
    let scheduler = match scheduler_after_ready(expected_core_id) {
        Ok(scheduler) => {
            warn!(
                "worker got scheduler: core_id={}, scheduler={:#x}, queue_len={}",
                expected_core_id,
                scheduler as *const _ as usize,
                scheduler.queue_len_approx()
            );
            scheduler
        }
        Err(err) => {
            error!(
                "scheduler worker started before scheduler ready: core_id={}, err={:?}",
                expected_core_id, err
            );
            return;
        }
    };

    if expected_core_id != scheduler.core_id() {
        error!(
            "scheduler worker core mismatch: scheduler_core={}, worker_core={}",
            scheduler.core_id(),
            expected_core_id
        );
        return;
    }

    warn!(
        "worker entering main loop: core_id={}, scheduler={:#x}",
        expected_core_id, scheduler as *const _ as usize
    );

    let mut idle_spins = 0_u32;
    let mut loop_iters: u64 = 0;
    loop {
        loop_iters = loop_iters.wrapping_add(1);


        if let Some(mut unit) = scheduler.take_task_for_core(expected_core_id) {

            slot.worker_busy.store(true, Ordering::Release);
            slot.worker_token
                .store(unit.user_token.get(), Ordering::Release);

            let mut success = true;
            let mut first_failed_node_id: u32 = u32::MAX;
            let mut first_failed_node_err: u8 = 0;
            let mut first_failed_node_op: u8 = 0;
            for node in unit.tasklink.iter_mut() {
                warn!(
                    "worker run node begin: token={}, node_id={}, op={:?}",
                    unit.user_token, node.node_id, node.desc.op
                );
                let ret = unsafe { k3_run_kernel(node) };
                if ret != 0 {
                    error!(
                        "k3_run_kernel failed: node_id={}, op={:?}, ret={}, error_flag={}",
                        node.node_id, node.desc.op, ret, node.state.error_flag
                    );
                    if first_failed_node_id == u32::MAX {
                        first_failed_node_id = node.node_id.get();
                        first_failed_node_err = node.state.error_flag;
                        first_failed_node_op = node.desc.op.0;
                    }
                    success = false;
                    break;
                }
            }

            warn!(
                "worker all nodes done: token={}, success={}",
                unit.user_token, success
            );

            // k3_run_kernel 已经不再引用 graph 的 tensor。先撤销本次提交创建的
            // kernel alias，完成消息发出后用户即可安全复用或释放原 tensor buffer。
            release_tensor_mappings(unit.caller.as_ref(), &unit.tasklink, unit.user_token);

            warn!(
                "worker completion send begin: token={}, success={}",
                unit.user_token, success
            );
            let completion = AiCompletion {
                user_token: unit.user_token.get(),
                failed_node_id: first_failed_node_id,
                status: if success {
                    0
                } else {
                    SchedulerErr::ExecutionFailed as u8
                },
                failed_node_err: first_failed_node_err,
                failed_node_op: first_failed_node_op,
                reserved: [0; 5],
            };
            let completion_bytes = unsafe {
                core::slice::from_raw_parts(
                    &completion as *const AiCompletion as *const u8,
                    core::mem::size_of::<AiCompletion>(),
                )
            };
            if unit
                .complete_sender
                .try_send(&Message::data(completion_bytes))
                .is_ok()
            {
                warn!(
                    "worker completion send ok: token={}, success={}",
                    unit.user_token, success
                );
            } else {
                error!("Can't notificate caller! token={}", unit.user_token);
            }

            slot.worker_token.store(0, Ordering::Release);
            slot.worker_busy.store(false, Ordering::Release);
            warn!(
                "worker task end: token={}, queue_len={}",
                unit.user_token,
                scheduler.queue_len_approx()
            );
            idle_spins = 0;
        } else {
            idle_spins = idle_spins.wrapping_add(1);
            if idle_spins.is_multiple_of(80_000_000) {
                warn!(
                    "worker idle: core_id={}, idle_spins={}, loop_iter={}, queue_len={}",
                    expected_core_id,
                    idle_spins,
                    loop_iters,
                    scheduler.queue_len_approx()
                );
            }
            core::hint::spin_loop();
        }
    }
}

/// 无锁 ready queue 的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use k3_ai_uabi::{AiGraphNode, AiKernelDesc, ByteSize, KernelVa, TensorCount};
    use ov_channels::{ChannelId, SharedMemory};
    extern crate std;
    use self::std::{sync::Arc, thread};

    /// 测试任务共用的静态 completion channel。
    static TEST_CHANNELS: SharedMemory<2> = SharedMemory::new();
    /// 记录 cleanup 请求数，验证每个有效 tensor alias 都会被释放。
    static TEST_UNMAP_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// 不接触真实地址空间的 scheduler 宿主。
    struct TestCaller;

    impl K3SchedulerOps for TestCaller {
        fn current_core_id(&self) -> u32 {
            0
        }

        unsafe fn copy_from_user(&self, _user_va: u64, _buf: &mut [u8]) -> Result<(), ()> {
            Err(())
        }

        unsafe fn copy_to_user(&self, _user_va: u64, _buf: &[u8]) -> Result<(), ()> {
            Err(())
        }

        unsafe fn map_user_to_kernel(&self, _user_va: u64, _len: usize) -> Result<u64, ()> {
            Err(())
        }

        unsafe fn unmap_user(&self, _kernel_va: u64, _len: usize) -> Result<(), ()> {
            TEST_UNMAP_COUNT.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn spawn_thread_on_core(&self, _core_id: u32, _f: fn(usize), _arg: usize) {}
    }

    /// 构造不含节点的任务，队列测试只观察 token 和所有权移动。
    fn unit(token: u32) -> AINodeUnit {
        let sender = TEST_CHANNELS
            .sender(ChannelId::new(0))
            .expect("test channel id should be valid");
        AINodeUnit {
            user_token: UserToken::new(token),
            caller: Box::new(TestCaller),
            complete_sender: sender,
            tasklink: TaskLink {
                pid: 0,
                head_node: None,
                tail_node: None,
                next_node: Vec::new(),
                node_order: Vec::new(),
                ordered_nodes: Vec::new(),
            },
        }
    }

    /// worker 收尾必须释放所有已成功创建的 input/output kernel alias。
    #[test]
    fn cleanup_releases_every_tensor_mapping() {
        let mut desc = AiKernelDesc {
            input_count: TensorCount::new(1),
            output_count: TensorCount::new(1),
            ..AiKernelDesc::default()
        };
        desc.tensors[0].kernel_va = KernelVa::new(0x1000);
        desc.tensors[0].size_bytes = ByteSize::new(64);
        desc.tensors[1].kernel_va = KernelVa::new(0x2000);
        desc.tensors[1].size_bytes = ByteSize::new(128);

        let tasklink = TaskLink {
            pid: 0,
            head_node: None,
            tail_node: None,
            next_node: Vec::new(),
            node_order: Vec::new(),
            ordered_nodes: Vec::from([AiGraphNode {
                node_id: Default::default(),
                desc,
                state: Default::default(),
            }]),
        };
        let before = TEST_UNMAP_COUNT.load(Ordering::Relaxed);

        release_tensor_mappings(&TestCaller, &tasklink, UserToken::new(1));

        assert_eq!(TEST_UNMAP_COUNT.load(Ordering::Relaxed), before + 2);
    }

    /// 入队三个任务后应按 FIFO 顺序弹出。
    #[test]
    fn queue_preserves_fifo_order() {
        let core_id = 8;
        let queue = PerCoreTaskQueue::new(core_id);

        for token in 1..=3 {
            let mut pending = Some(unit(token));
            assert_eq!(queue.try_push(core_id, &mut pending), Ok(()));
            assert!(pending.is_none());
        }

        for token in 1..=3 {
            let task = queue
                .try_pop(core_id)
                .expect("core id should match")
                .expect("queued task should exist");
            assert_eq!(task.user_token, token);
        }
        assert!(
            queue
                .try_pop(core_id)
                .expect("core id should match")
                .is_none()
        );
    }

    /// 第 65 个任务应报告队列已满且保留任务所有权。
    #[test]
    fn queue_reports_full_at_capacity() {
        let core_id = 9;
        let queue = PerCoreTaskQueue::new(core_id);

        for token in 0..SCHEDULER_QUEUE_CAPACITY as u32 {
            let mut pending = Some(unit(token));
            assert_eq!(queue.try_push(core_id, &mut pending), Ok(()));
        }

        let mut pending = Some(unit(99));
        assert_eq!(
            queue.try_push(core_id, &mut pending),
            Err(QueuePushError::Full)
        );
        assert_eq!(
            pending.as_ref().map(|task| task.user_token),
            Some(UserToken::new(99))
        );
        assert_eq!(queue.len_approx(), SCHEDULER_QUEUE_CAPACITY);

        for _ in 0..SCHEDULER_QUEUE_CAPACITY {
            assert!(
                queue
                    .try_pop(core_id)
                    .expect("core id should match")
                    .is_some()
            );
        }
    }

    /// 消费 slot 后应能跨 ring 边界复用相同 slot。
    #[test]
    fn queue_reuses_slots_after_pop() {
        let core_id = 10;
        let queue = PerCoreTaskQueue::new(core_id);

        for token in 0..SCHEDULER_QUEUE_CAPACITY as u32 {
            let mut pending = Some(unit(token));
            assert_eq!(queue.try_push(core_id, &mut pending), Ok(()));
        }
        for token in 0..8 {
            let task = queue
                .try_pop(core_id)
                .expect("core id should match")
                .expect("queued task should exist");
            assert_eq!(task.user_token, token);
        }
        for token in 64..72 {
            let mut pending = Some(unit(token));
            assert_eq!(queue.try_push(core_id, &mut pending), Ok(()));
        }

        for token in 8..72 {
            let task = queue
                .try_pop(core_id)
                .expect("core id should match")
                .expect("queued task should exist");
            assert_eq!(task.user_token, token);
        }
        assert_eq!(queue.len_approx(), 0);
    }

    /// core 不匹配时 push/pop 都不能推进 head/tail 或丢失任务。
    #[test]
    fn queue_rejects_core_mismatch() {
        let core_id = 11;
        let wrong_core_id = 12;
        let queue = PerCoreTaskQueue::new(core_id);
        let mut pending = Some(unit(7));

        assert_eq!(
            queue.try_push(wrong_core_id, &mut pending),
            Err(QueuePushError::CoreMismatch)
        );
        assert!(pending.is_some());
        assert_eq!(queue.len_approx(), 0);

        assert_eq!(queue.try_push(core_id, &mut pending), Ok(()));
        assert!(matches!(
            queue.try_pop(wrong_core_id),
            Err(QueuePopError::CoreMismatch)
        ));
        assert_eq!(queue.len_approx(), 1);
        assert_eq!(
            queue
                .try_pop(core_id)
                .expect("core id should match")
                .expect("queued task should exist")
                .user_token,
            7
        );
    }

    /// 多个 producer 并发竞争 tail 时不能丢失或重复任务。
    #[test]
    fn queue_accepts_concurrent_producers() {
        const PRODUCERS: u32 = 4;
        const TASKS_PER_PRODUCER: u32 = 128;
        const TOTAL_TASKS: usize = (PRODUCERS * TASKS_PER_PRODUCER) as usize;

        let core_id = 13;
        let queue = Arc::new(PerCoreTaskQueue::new(core_id));
        let consumer_queue = Arc::clone(&queue);
        let consumer = thread::spawn(move || {
            let mut tokens = Vec::with_capacity(TOTAL_TASKS);
            while tokens.len() < TOTAL_TASKS {
                if let Some(task) = consumer_queue
                    .try_pop(core_id)
                    .expect("consumer core id should match")
                {
                    tokens.push(task.user_token.get());
                } else {
                    thread::yield_now();
                }
            }
            tokens
        });

        let mut producers = Vec::new();
        for producer in 0..PRODUCERS {
            let producer_queue = Arc::clone(&queue);
            producers.push(thread::spawn(move || {
                for sequence in 0..TASKS_PER_PRODUCER {
                    let token = producer * TASKS_PER_PRODUCER + sequence;
                    let mut pending = Some(unit(token));
                    loop {
                        match producer_queue.try_push(core_id, &mut pending) {
                            Ok(()) => break,
                            Err(QueuePushError::Full) => thread::yield_now(),
                            Err(err) => panic!("unexpected queue push error: {err:?}"),
                        }
                    }
                }
            }));
        }

        for producer in producers {
            producer.join().expect("producer should finish");
        }
        let mut tokens = consumer.join().expect("consumer should finish");
        tokens.sort_unstable();
        assert_eq!(tokens, (0..TOTAL_TASKS as u32).collect::<Vec<_>>());
        assert_eq!(queue.len_approx(), 0);
    }

    /// 初始化状态 CAS 应只允许一个 owner。
    #[test]
    fn scheduler_init_is_single_owner() {
        let state = AtomicU8::new(SCHEDULER_UNINIT);

        assert!(try_claim_scheduler_initialization(&state));
        assert_eq!(state.load(Ordering::Acquire), SCHEDULER_INITIALIZING);
        assert!(!try_claim_scheduler_initialization(&state));

        state.store(SCHEDULER_READY, Ordering::Release);
        assert!(!try_claim_scheduler_initialization(&state));
    }
}
