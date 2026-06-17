//! 和用户库的提交通道对接。
//!
//! 阶段一先不做完整 shared SQ/CQ ring。用户态 frontend 通过 ovchannel 把
//! `AiGraphSubmitEntry` 发送给内核调度器，调度器主循环使用非阻塞接收收集提交。
//! 当前先采用单调度器模式，调度器可以固定跑在某个 X100 大核上；每个申请 AI
//! runtime 的进程拥有一个 channel sender。
//!
//! 这里记录今晚已经定下来的内核/用户态边界，后面实现时按这个边界拆代码。
//!
//! ## Graph blob
//!
//! `AiGraphSubmitEntry` 只携带用户态 graph blob 的地址、大小和 user token。
//! channel 里不直接传整张图。
//!
//! 调度器收到提交后必须先做最小地址/大小校验，然后把 graph blob 从用户地址空间
//! copy 成内核私有 blob。之后 parser 只解析这份内核私有 blob，并生成内核 owned
//! `ParsedGraph`。`ParsedGraph` 进入调度队列后，调度器和 worker 不能再读用户态
//! graph blob。
//!
//! 这样做是为了避免 TOCTOU：用户态提交后即使释放、复用或篡改原始 graph blob，
//! 也不能改变已经进入内核调度队列的图语义。
//!
//! ## Tensor buffer
//!
//! tensor data 不 copy。阶段一的 A100/X100 backend 只接受 StarryOS allocator
//! 分配出来的物理连续 tensor buffer，普通用户指针不直接进入 A100/X100 backend。
//! 非连续 tensor 后面再扩展 scatter-gather 或 bounce buffer 路径。
//!
//! allocator 返回给用户态的身份包含：
//! - token：内核可信身份，用于 bind tensor；
//! - user addr：用户态读写 tensor 数据用；
//! - actual size：allocator 对齐后的真实 buffer 大小。
//!
//! token 采用 index + generation，bind 时用 pid + token 查 allocator buffer object。
//! generation 用来防止 stale token / ABA 问题。
//!
//! bind tensor 时内核根据 token 找到 buffer object，检查 offset/len 不越界，确认
//! buffer 物理连续，然后增加 buffer refcount。graph completion 后释放这些引用。
//! 如果用户在 refcount 非零时 free tensor，allocator 返回 Busy；阶段一不做
//! pending_free。
//!
//! ## Cache coherency
//!
//! pin/refcount 只保证物理页和生命周期稳定，不保证 CPU、X100、A100、DMA 看到的
//! 数据内容一致。A100/AIDMA 路径阶段一先按 non-coherent 处理，由底层 DMA allocator
//! trait 提供 clean / invalidate 接口。
//!
//! cache 操作按 tensor 访问方向决定：
//! - Input：backend 前 clean，让设备读到 CPU/用户写回 RAM 的数据；
//! - Output：backend 前 invalidate 或 clean+invalidate，backend 后 invalidate；
//! - InOut：backend 前 clean 或 clean+invalidate，backend 后 invalidate。
//!
//! 纯 input 不需要 invalidate，因为设备读的是 RAM，不读 CPU cache；invalidate
//! 主要用于设备写完后让 CPU 放弃旧 cache 副本。clean 后、敲 A100/DMA 门铃前需要
//! 有同步屏障，避免设备在 cache clean 生效前开始读。
//!
//! 阶段一如果限制整张图走同一种 backend，可以先在 graph 执行前后做粗粒度 sync。
//! 如果图内部混用 CPU/A100/X100 backend，则必须在 node/backend 边界按 input/output
//! 方向做 sync。
//!
//! ## Backend code boundary
//!
//! backend 算子核心采用“共用源码、分别编译”的方式：
//! - 用户态产物：编译成 user `.so` / direct inline 路径，小任务可以不进内核调度器；
//! - 内核态产物：编译进 kernel backend object/module，由 KernelScheduled worker 调用。
//!
//! 不把内核模块机器码原样映射给用户态执行。共享的是算子核心源码和 ABI，不是同一个
//! 二进制。这样可以避免用户态直接执行依赖内核地址、内核锁、内核 allocator 或特权
//! 上下文的代码。
//!
//! backend core 的外部 ABI 使用 `repr(C)` 的 tensor view/call 描述，兼容 C/C++ 和
//! Rust。外部 ABI 只传 raw pointer、长度、shape、stride、dtype、layout、attr 等
//! 稳定字段；backend core 内部再把 view 转成 Rust slice 执行。
//!
//! `BackendTensorView.data` 在不同执行路径里的含义不同：
//! - DirectInline 用户态路径：data 是用户态 allocator 解析出的用户虚拟地址；
//! - KernelScheduled 内核态路径：data 是内核解析出的、内核可以直接访问的虚拟地址。
//!
//! AIDMA/IME 设备执行后面可能还需要 phys addr、DMA addr 或 IOMMU 映射配置。这部分属于
//! kernel backend lowering/任务入口配置，不应该藏在用户态 `.so` 里。阶段一先留下 TODO，
//! 等确认 K3 路径上是否有 IOMMU、AIDMA 需要什么地址格式后再扩展。
//!
//! ## Scheduler shape
//!
//! 内核中真正的执行实体是 graph node。用户态提交的 `AiGraphNode` 进入内核后会变成
//! 调度器私有的 node 状态对象。node token/graph token 用于身份匹配，node 是否完成
//! 应该使用单独的状态位，而不是把 token 当完成标志。
//!
//! 阶段一的 node id 是连续编号，调度结构优先使用 `Vec`/ready queue，而不是一开始就
//! 依赖复杂 map。DAG 调度时维护每个 node 的剩余依赖计数和 dependents 列表：
//! ready node 执行完成后，减少其后继节点的 remaining deps；后继依赖清零时进入 ready
//! queue。这样避免把未满足依赖的 node 反复放回 wait 队列轮询。
//!
//! ## Failure and backpressure
//!
//! graph copy、parse、tensor bind 任一步失败，都需要返回错误 completion 或 submit error。
//! 如果 bind 第 N 个 tensor 失败，前面已经 bind 的 tensor ref 必须全部释放。内核兜底
//! 负责进程退出时释放 allocator object；正常路径由 completion 后释放 graph 本次持有的
//! tensor ref。
//!
//! ovchannel 满时阶段一直接返回 Busy 给 userlib，由 userlib 决定重试、阻塞等待或上报给
//! 应用。
//!
//! ## Completion
//!
//! graph 完成后通过 ovchannel 回包。阶段一只在整张 graph 完全完成后返回 completion，
//! 用户态先轮询；异步等待、callback 或更完整的 CQ 留到第二阶段。
//!
//! completion 至少需要带回 user token、graph 状态和错误码。completion 发生后才能释放
//! 本次 graph bind 住的 tensor ref。用户态看到 completion 后才允许安全复用或 free
//! 相关 tensor。


use alloc::{collections::vec_deque::VecDeque, vec::Vec};
use alloc::vec;
use k3_aiUabi::{AiGraphNode, AiParsedGraph};
use k3_aiUabi::error::SchedulerErr;

/// 内核侧把一张 DAG 收敛成的单条调度链。
///
/// 阶段一先把图按稳定拓扑序解析成一条直链。真正执行时仍然以 edge 语义为准，
/// 这里只负责给调度器一个”当前可接受”的线性顺序。
#[derive(Clone)]
pub struct TaskLink {
    /// 提交该 graph 的进程。
    pub pid: u32,
    /// 链头 node id。空图时为 `None`。
    pub head_node: Option<u32>,
    /// 链尾 node id。空图时为 `None`。
    pub tail_node: Option<u32>,
    /// `next_node[node_id]` 表示当前直链里下一个节点是谁。
    pub next_node: Vec<Option<u32>>,
    /// 解析出的稳定拓扑序。
    pub node_order: Vec<u32>,
    /// 直链对应的节点副本，顺序与 `node_order` 一致。
    pub ordered_nodes: Vec<AiGraphNode>,
}

impl TaskLink {
    pub fn iter(&self) -> impl Iterator<Item = &AiGraphNode> {
        self.ordered_nodes.iter()
    }

    pub fn pop_front(&mut self) -> Option<AiGraphNode> {
        if self.ordered_nodes.is_empty() {
            None
        } else {
            Some(self.ordered_nodes.remove(0))
        }
    }
}

/// 根据 edge 关系把一张解析后的 graph 收成可调度的直链。
///
/// 阶段一先做稳定拓扑排序：
/// - 所有入度为 0 的节点先进入 ready 队列；
/// - ready 队列按 node id 的自然顺序出队；
/// - 节点完成后减少后继入度，新的 ready 节点再入队。
///
/// 这样 `a -> b, c -> b` 会解析成 `a -> c -> b` 或其他满足 edge 约束的稳定顺序。
pub fn resolve_parsed_graph(pid: u32, graph: &AiParsedGraph) -> Result<TaskLink, SchedulerErr> {
    let node_count = graph.nodes.len();
    let mut indegree = vec![0_usize; node_count];
    let mut outgoing = vec![Vec::new(); node_count];

    for (idx, node) in graph.nodes.iter().enumerate() {
        if node.node_id != idx as u32 {
            return Err(SchedulerErr::ParseFailed);
        }
    }

    for edge in &graph.edges {
        let from = edge.from_node as usize;
        let to = edge.to_node as usize;
        if from >= node_count || to >= node_count {
            return Err(SchedulerErr::ParseFailed);
        }

        outgoing[from].push(edge.to_node);
        indegree[to] += 1;
    }

    let mut ready = VecDeque::new();
    for (idx, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.push_back(idx as u32);
        }
    }

    let mut node_order = Vec::with_capacity(node_count);
    while let Some(node_id) = ready.pop_front() {
        node_order.push(node_id);

        for &next_node in &outgoing[node_id as usize] {
            let next_idx = next_node as usize;
            indegree[next_idx] -= 1;
            if indegree[next_idx] == 0 {
                ready.push_back(next_node);
            }
        }
    }

    if node_order.len() != node_count {
        return Err(SchedulerErr::ParseFailed);
    }

    let mut next_node = vec![None; node_count];
    for window in node_order.windows(2) {
        next_node[window[0] as usize] = Some(window[1]);
    }

    let mut ordered_nodes = Vec::with_capacity(node_count);
    for &node_id in &node_order {
        ordered_nodes.push(graph.nodes[node_id as usize]);
    }

    Ok(TaskLink {
        pid,
        head_node: node_order.first().copied(),
        tail_node: node_order.last().copied(),
        next_node,
        node_order,
        ordered_nodes,
    })
}

#[cfg(test)]
mod tests {
    use k3_aiUabi::{
        AiGraphBuildError, AiGraphEdge, AiGraphParser, AiKernelDesc, GraphManager,
    };

    use super::*;

    fn build_graph<F>(f: F) -> AiParsedGraph
    where
        F: FnOnce(&mut GraphManager) -> Result<(), AiGraphBuildError>,
    {
        let mut graph = GraphManager::new();
        f(&mut graph).expect("graph build should succeed");
        let blob = graph.freeze().expect("graph freeze should succeed");
        AiGraphParser::parse(blob.as_bytes()).expect("graph parse should succeed")
    }

    #[test]
    fn resolve_chain_keeps_edge_order() {
        let parsed = build_graph(|graph| {
            let a = graph.push_kernel_no_depend(AiKernelDesc::default())?;
            let b = graph.push_kernel_depend(a, AiKernelDesc::default())?;
            let _c = graph.push_kernel_depend(b, AiKernelDesc::default())?;
            Ok(())
        });

        let link = resolve_parsed_graph(7, &parsed).expect("resolve graph should succeed");

        assert_eq!(link.pid, 7);
        assert_eq!(link.node_order, vec![0, 1, 2]);
        assert_eq!(link.head_node, Some(0));
        assert_eq!(link.tail_node, Some(2));
        assert_eq!(link.next_node[0], Some(1));
        assert_eq!(link.next_node[1], Some(2));
        assert_eq!(link.next_node[2], None);
    }

    #[test]
    fn resolve_join_and_fork_respects_dependencies() {
        let parsed = build_graph(|graph| {
            let a = graph.push_kernel_no_depend(AiKernelDesc::default())?;
            let c = graph.push_kernel_no_depend(AiKernelDesc::default())?;
            let b = graph.push_kernel_depend_many(&[a, c], AiKernelDesc::default())?;
            let _d = graph.push_kernel_depend(b, AiKernelDesc::default())?;
            let _e = graph.push_kernel_depend(b, AiKernelDesc::default())?;
            Ok(())
        });

        let link = resolve_parsed_graph(9, &parsed).expect("resolve graph should succeed");

        assert_eq!(link.head_node, Some(link.node_order[0]));
        assert_eq!(link.tail_node, Some(*link.node_order.last().unwrap()));
        assert_eq!(link.ordered_nodes.len(), parsed.nodes.len());

        for edge in &parsed.edges {
            let from_pos = link
                .node_order
                .iter()
                .position(|node_id| *node_id == edge.from_node)
                .expect("from node should exist in order");
            let to_pos = link
                .node_order
                .iter()
                .position(|node_id| *node_id == edge.to_node)
                .expect("to node should exist in order");
            assert!(
                from_pos < to_pos,
                "edge {} -> {} violated by {:?}",
                edge.from_node,
                edge.to_node,
                link.node_order
            );
        }
    }

    #[test]
    fn reject_out_of_range_edge() {
        let mut parsed = build_graph(|graph| {
            graph.push_kernel_no_depend(AiKernelDesc::default())?;
            Ok(())
        });
        parsed.edges.push(AiGraphEdge {
            from_node: 0,
            to_node: 3,
        });

        match resolve_parsed_graph(1, &parsed) {
            Err(err) => assert_eq!(
                err,
                SchedulerErr::ParseFailed           
            ),
            Ok(_) => panic!("resolve_graph should reject out-of-range edge"),
        }
    }

    #[test]
    fn reject_cycle_by_edges() {
        let mut parsed = build_graph(|graph| {
            let a = graph.push_kernel_no_depend(AiKernelDesc::default())?;
            let b = graph.push_kernel_depend(a, AiKernelDesc::default())?;
            let _c = graph.push_kernel_depend(b, AiKernelDesc::default())?;
            Ok(())
        });
        parsed.edges.push(AiGraphEdge {
            from_node: 2,
            to_node: 0,
        });

        match resolve_parsed_graph(2, &parsed) {
            Err(err) => assert_eq!(err, SchedulerErr::ParseFailed),
            Ok(_) => panic!("resolve_graph should reject cycle graph"),
        }
    }
}
