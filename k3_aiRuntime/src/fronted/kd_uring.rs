//! kernel desc uring
//! 将实际的算子计算图描述符提交到channel执行任务以及处理后续的返回结果处理
//! 内核需要提前mmap一片专用数据区域用作共享内存来通过ovchannel高效的传递任务
//! 我们写入uring之后就可以直接syscall请求调度执行,然后内核调度器就可以从channel拿取任务进行调度

//! 关于共享内存,这里将共享内存当作图描述符传递channel用来和内核调度器传递任务.
//! 用户库如何知道内存在哪里? 用户库需要向调度器申请一次,直到手动释放或者程序自动结束失效
//! 这个地址就是我们可以给ovchannel的内存
//! 在内核中各个进程的共享内存物理上为连续的,调度器可以直接从每片内存拿,效率较高

//TODO:申请共享内存用来当做ring
fn alloc_uring_memory() {}

//TODO:调用syscall.让任务开始调度
fn kernel_schedule_bell() {
    // 调用syscall
}

//TODO:
/// 提交图描述符到内核调度器
pub fn submit_graph_to_scheduler() {}
