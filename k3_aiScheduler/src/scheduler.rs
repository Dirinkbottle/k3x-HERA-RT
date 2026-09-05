//! Per-core AI graph 调度器和常驻 worker。
//!
//! 每个实际运行 core 独占一个 scheduler/ready queue/worker。这样 submit path
//! 不会跨 core 写入另一个 worker 的无锁环形队列，避免真板子上缺少跨核数据同步时
//! 出现消费者读不到或读到旧数据的问题。

use crate::{K3SchedulerOps, K3WaitQueue, kd_kring::TaskLink};
use alloc::boxed::Box;
use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering},
};
use k3_ai_uabi::{UserToken, error::SchedulerErr};
use log::warn;
use ov_channels::Sender;

#[path = "woker.rs"]
mod woker;

use woker::worker;

/// 单个 scheduler ready queue 能容纳的最大任务数。
pub const SCHEDULER_QUEUE_CAPACITY: usize = 64;

/// 当前支持缓存的最大 CPU core id 数量。
const MAX_SCHEDULER_CORES: usize = 64;

/// 绑定到指定 CPU core 的常驻 worker 入口。
type WorkerEntry = fn(usize);

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
    /// 启动该 core 常驻 worker 时传给宿主的函数入口。
    worker_entry: WorkerEntry,
    /// worker 后续阻塞等待任务时使用的内核等待队列。
    wait_queue: K3WaitQueue,
    /// 无锁 ready queue。
    ready_queue: PerCoreTaskQueue,
}

impl GraphScheduler {
    /// 使用指定实际 core、worker 入口和等待队列构造 scheduler。
    fn new_with_core(core_id: u32, wait_queue: K3WaitQueue) -> Self {
        Self {
            core_id,
            worker_entry: worker,
            wait_queue,
            ready_queue: PerCoreTaskQueue::new(core_id),
        }
    }

    /// 返回宿主启动该 scheduler worker 所需的函数入口。
    fn worker_entry(&self) -> WorkerEntry {
        self.worker_entry
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
fn get_or_init_scheduler(
    caller: &dyn K3SchedulerOps,
    core_id: u32,
) -> Result<&'static GraphScheduler, SchedulerErr> {
    let slot = scheduler_slot(core_id)?;
    if try_claim_scheduler_initialization(&slot.init_state) {
        let scheduler = GraphScheduler::new_with_core(core_id, caller.new_wait_queue());
        let worker_entry = scheduler.worker_entry();

        // SAFETY: 初始化状态 CAS 保证当前线程是唯一 writer，READY 发布前没有读者访问。
        unsafe {
            *slot.scheduler.get() = Some(scheduler);
        }
        // READY 的 release store 发布 scheduler、worker entry 和 wait queue；worker 在
        // 自己绑定的 core 上经 scheduler_after_ready 的 acquire load 才会开始访问它们。
        slot.init_state.store(SCHEDULER_READY, Ordering::Release);
        caller.spawn_thread_on_core(core_id, worker_entry, core_id as usize);
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

    let scheduler = get_or_init_scheduler(caller.as_ref(), submit_core_id)?;
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
    // 有新任务了唤醒worker
    scheduler.wait_queue.notify_one();

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
    /// 记录 wait queue 工厂调用次数。
    static TEST_WAIT_QUEUE_COUNT: AtomicUsize = AtomicUsize::new(0);
    /// 记录 scheduler 初始化时请求的 worker core。
    static TEST_SPAWN_CORE: AtomicUsize = AtomicUsize::new(usize::MAX);
    /// 记录 scheduler 初始化时传给宿主的 worker entry。
    static TEST_SPAWN_ENTRY: AtomicUsize = AtomicUsize::new(0);

    /// 不接触真实地址空间的 scheduler 宿主。
    struct TestCaller;

    /// 不阻塞测试线程的等待队列实现。
    struct TestWaitQueue;

    impl crate::K3SchedulerWaitQueue for TestWaitQueue {
        fn wait(&self) {}

        fn wait_until(&self, condition: &dyn Fn() -> bool) {
            let _ = condition();
        }

        fn notify_one(&self) {}

        fn notify_all(&self) {}
    }

    impl K3SchedulerOps for TestCaller {
        fn new_wait_queue(&self) -> crate::K3WaitQueue {
            TEST_WAIT_QUEUE_COUNT.fetch_add(1, Ordering::Relaxed);
            Box::new(TestWaitQueue)
        }

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

        fn spawn_thread_on_core(&self, core_id: u32, worker_entry: fn(usize), _arg: usize) {
            TEST_SPAWN_CORE.store(core_id as usize, Ordering::Release);
            TEST_SPAWN_ENTRY.store(worker_entry as usize, Ordering::Release);
        }
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

        woker::release_tensor_mappings(&TestCaller, &tasklink, UserToken::new(1));

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

    /// scheduler 构造时必须同时绑定 worker entry 和等待队列。
    #[test]
    fn scheduler_constructor_binds_worker_and_wait_queue() {
        let scheduler = GraphScheduler::new_with_core(3, Box::new(TestWaitQueue));

        assert_ne!(scheduler.worker_entry() as usize, 0);
        scheduler.wait_queue.notify_all();
    }

    /// 唯一初始化者应发布 scheduler 后，用其绑定的 worker entry 请求宿主启动。
    #[test]
    fn scheduler_initialization_spawns_bound_worker() {
        let core_id = (MAX_SCHEDULER_CORES - 1) as u32;
        TEST_WAIT_QUEUE_COUNT.store(0, Ordering::Relaxed);
        TEST_SPAWN_CORE.store(usize::MAX, Ordering::Relaxed);
        TEST_SPAWN_ENTRY.store(0, Ordering::Relaxed);

        let scheduler = get_or_init_scheduler(&TestCaller, core_id).unwrap();

        assert_eq!(TEST_WAIT_QUEUE_COUNT.load(Ordering::Acquire), 1);
        assert_eq!(TEST_SPAWN_CORE.load(Ordering::Acquire), core_id as usize);
        assert_eq!(
            TEST_SPAWN_ENTRY.load(Ordering::Acquire),
            scheduler.worker_entry() as usize
        );
    }
}
