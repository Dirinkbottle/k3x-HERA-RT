use super::{scheduler_after_ready, scheduler_slot};
use crate::{K3SchedulerOps, kd_kring::TaskLink};
use core::sync::atomic::Ordering;
use k3_ai_uabi::AiCompletion;
use k3_ai_uabi::{UserToken, error::SchedulerErr};
use k3_kernel_backend::k3_run_kernel;
use log::{error, warn};
use ov_channels::Message;

/// 释放一张 graph 在 submit 阶段为 tensor 创建的所有 kernel alias。
///
/// 这必须在 worker 停止访问 tensor 后执行。失败时继续释放剩余映射并只记录日志，
/// 因为 caller 仍应收到该 graph 的 completion，不能让一个清理失败卡死 worker。
pub(super) fn release_tensor_mappings(
    caller: &dyn K3SchedulerOps,
    tasklink: &TaskLink,
    token: UserToken,
) {
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

    //等待队列
    let waiter_queue = &scheduler.wait_queue;

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

    loop {
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
        } else {
            // 没有任务就睡眠
            waiter_queue.wait_until(&|| scheduler.queue_len_approx() != 0);
        }
    }
}
