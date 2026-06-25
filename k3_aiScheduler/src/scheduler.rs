//! 调度器,根据pid将任务链区分开
//! 然后再拆进程内的任务链来parse任务

use crate::{K3SchedulerOps, kd_kring::TaskLink};
use alloc::vec;
use alloc::{boxed::Box, collections::vec_deque::VecDeque, vec::Vec};
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};
use k3_aiUabi::AiGraphSubmitEntry;
use k3_aiUabi::error::SchedulerErr;
use k3_kernel_backend::k3_run_kernel;
use log::error;
use ov_channels::{Message, Sender};
use spin::Mutex;

/// AI任务单元，包含任务链和完成通知发送器
pub struct AINodeUnit {
    pub user_token: u32,
    // 内核已经pin住channel内存,内核释放前 一直有效
    pub complete_sender: Sender<'static>,
    pub tasklink: TaskLink,
}

/// 调度器
/// 先入先出的顺序调度
/// 这个scheduler应该只跑在某一个x100核心上
pub struct GraphScheduler {
    // 运行队列,运行任务从queue弹出,切换任务时把没运行完的放在tail
    ready_queue: Mutex<VecDeque<AINodeUnit>>,
}

impl GraphScheduler {
    pub fn new() -> Self {
        Self {
            ready_queue: Mutex::new(VecDeque::new()),
        }
    }

    pub fn take_task(&self) -> Option<AINodeUnit> {
        self.ready_queue.lock().pop_front()
    }

    pub fn push_task(&self, task: AINodeUnit) {
        self.ready_queue.lock().push_back(task);
    }
}

struct SchedulerCell(UnsafeCell<Option<GraphScheduler>>);
unsafe impl Sync for SchedulerCell {}

static SCHEDULER: SchedulerCell = SchedulerCell(UnsafeCell::new(None));

static SCHEDULER_INITED: AtomicBool = AtomicBool::new(false);

/// abi安全保证: 随内核编译
/// 内核提交入口
/// 'k_graph' 为内核vddr
pub fn run_graph(
    user_token: u32,
    caller: Box<dyn K3SchedulerOps>,
    complete_sender: Sender<'static>,
    tasklink: TaskLink,
) -> Result<(), SchedulerErr> {
    if !SCHEDULER_INITED.load(Ordering::Acquire) {
        unsafe {
            *SCHEDULER.0.get() = Some(GraphScheduler::new());
        }
        SCHEDULER_INITED.store(true, Ordering::Release);

        caller.spawn_thread(worker, 0);
        error!("scheduler initialized, worker thread should be started");
    }

    let unit = AINodeUnit {
        user_token,
        complete_sender,
        tasklink,
    };

    unsafe {
        if let Some(scheduler) = &*SCHEDULER.0.get() {
            scheduler.push_task(unit);
            Ok(())
        } else {
            Err(SchedulerErr::InvalidGraph)
        }
    }
}

// 核心的常住worker
pub fn worker(arg: usize) {
    loop {
        let unit = unsafe {
            if let Some(scheduler) = &*SCHEDULER.0.get() {
                scheduler.take_task()
            } else {
                None
            }
        };

        if let Some(unit) = unit {
            let mut success = true;

            for node in unit.tasklink.iter() {
                error!("run node: node_id={}, op={:?}", node.node_id, node.desc.op);
                let ret = unsafe { k3_run_kernel(node) };
                if ret != 0 {
                    error!(
                        "k3_run_kernel failed: node_id={}, op={:?}, ret={}",
                        node.node_id, node.desc.op, ret
                    );
                    success = false;
                    break;
                }
                error!(
                    "k3_run_kernel success: node_id={}, op={:?}",
                    node.node_id, node.desc.op
                );
            }

            // graph的所有node已经完成

            if success {
                // 通知调用者完成,回写token
                if let Ok(()) = unit
                    .complete_sender
                    .try_send(&Message::notification(unit.user_token))
                {
                    error!("task completed, token={}", unit.user_token);
                } else {
                    error!("Can't notificate caller! token={}", unit.user_token);
                }
            }
        } else {
            // 队列为空，让出CPU
            core::hint::spin_loop();
        }
    }
}
