//! 调度器,根据pid将任务链区分开
//! 然后再拆进程内的任务链来parse任务


use core::cell::UnsafeCell;

use alloc::collections::vec_deque::VecDeque;
use k3_aiUabi::AiGraphSubmitEntry;
use log::error;

use crate::kd_kring::TaskLink;

/// 调度器
/// 先入先出的顺序调度
/// 这个scheduler应该只跑在某一个x100核心上
pub struct GraphScheduler{
            // 运行队列,运行任务从queue弹出,切换任务时把没运行完的放在tail
            ready_queue:UnsafeCell<VecDeque<TaskLink>>,
}
impl GraphScheduler {
    pub fn new() -> Self {
        Self {
            ready_queue: UnsafeCell::new(VecDeque::new()),
        }
    }

    pub fn take_task(&self) -> Option<TaskLink> {
        unsafe { (*self.ready_queue.get()).pop_front() }
    }

    pub fn push_task(&self, task: TaskLink) {
        unsafe { (*self.ready_queue.get()).push_back(task) }
    }
}





/// 内核提交入口
/// 'k_graph' 为内核vddr
pub fn run_graph(k_graph: &AiGraphSubmitEntry)->Result<(),()>{
    // 检查graph_user_va graph_size
    if k_graph.graph_user_va == 0 || k_graph.graph_size == 0 {
        return Err(());
    }


    error!("You reached Graph");
    return Err(());


    
    // 从graph_user_va 和 graph_size反序列化出parsed graph
    // TODO:必须从user空间copy过来,防止后续被篡改
    let blob_slice = unsafe {
        core::slice::from_raw_parts(
            k_graph.graph_user_va as *const u8,
            k_graph.graph_size as usize
        )
    };

    let parsed_graph = k3_aiUabi::AiGraphParser::parse(blob_slice)
        .map_err(|_| ())?;

    let task_link = crate::kd_kring::resolve_parsed_graph(0, &parsed_graph)
        .map_err(|_| ())?;



     //TODO:我们在这里必须将用户的tensor映射为内核虚拟地址


    // 遍历 AiGraphNode
    for node in task_link.iter() {
        error!("I received the node:");
        error!("  node_id: {}", node.node_id);
        error!("  op: {:?}", node.desc.op);
        error!("  target_hint: {:?}", node.desc.target_hint);
        error!("  input_count: {}, output_count: {}", node.desc.input_count, node.desc.output_count);

        // 打印输入 tensors 信息
        for i in 0..node.desc.input_count as usize {
            let tensor = &node.desc.tensors[i];
            error!("  input[{}]: dtype={:?}, ndim={}, shape={:?}",
                i, tensor.dtype, tensor.ndim, &tensor.shape[..tensor.ndim as usize]);
        }

        // 打印输出 tensors 信息
        for i in 0..node.desc.output_count as usize {
            let tensor = &node.desc.tensors[node.desc.input_count as usize + i];
            error!("  output[{}]: dtype={:?}, ndim={}, shape={:?}",
                i, tensor.dtype, tensor.ndim, &tensor.shape[..tensor.ndim as usize]);
        }

        error!("  attr_size: {} bytes", node.desc.attr_size);
    }

    Ok(())
}