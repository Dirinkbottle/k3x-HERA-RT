//! k3 芯片 AI 调度器平台 trait 定义。

pub mod kd_kring;
pub mod scheduler;

/// 调度器依赖的平台能力，由 OS 层实现。
pub trait K3Platform {
        //mmap
        // 创建线程
}

/// ioctl 入口，接收用户态新任务提交。
pub fn recive_newtask() {

}
