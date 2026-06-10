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

