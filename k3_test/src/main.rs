//! K3 backend 的板端 FP16 冒烟测试。

mod test;

use k3_ai_runtime::fronted::{AiTargetHint, kd_uring::build_channel};

/// 提交 CPU 和 A100 两个 FP16 MatMul，并验证数值一致性与 COW 生命周期。
fn main() {
    test::test_f16_cow_live();

    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");
    let cpu = test::submit_f16_matmul(&channel, AiTargetHint::PREFER_CPU, false);
    let a100 = test::submit_f16_matmul(&channel, AiTargetHint::PREFER_A100, false);
    assert_eq!(cpu, a100, "CPU and A100 FP16 MatMul outputs diverged");
    println!("k3_test: FP16 CPU/A100 MatMul smoke tests passed");
}
