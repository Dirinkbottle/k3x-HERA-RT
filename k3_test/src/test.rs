//! 板端 FP16 图提交与 tensor 生命周期测试。

use k3_ai_runtime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, DimSize, ElemStride, GraphManager, KernelOp, MatMulAttr,
    OpFlags, Tensor, TensorCount, TensorManager, UserToken,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};

/// 使用 `AUTO`（归一化到 A100）执行一次 FP16 MatMul，并验证 COW 映射保活。
pub fn test_f16_cow_live() {
    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");
    let output = submit_f16_matmul(&channel, AiTargetHint::AUTO, true);
    assert_eq!(output, [0x4400, 0x4500, 0x4900, 0x4980]);
    println!("FP16 COW lifetime test PASS");
}

/// 在指定 target 提交 `2x3 @ 3x2` FP16 MatMul。
pub fn submit_f16_matmul(
    channel: &UringChannel,
    target: AiTargetHint,
    drop_inputs_before_wait: bool,
) -> [u16; 4] {
    let tensor_mgr = TensorManager::new();
    let mut lhs = tensor_mgr
        .alloc_tensor(AiDtype::F16, &[2, 3])
        .expect("alloc FP16 lhs failed");
    let mut rhs = tensor_mgr
        .alloc_tensor(AiDtype::F16, &[3, 2])
        .expect("alloc FP16 rhs failed");
    let output = tensor_mgr
        .alloc_tensor(AiDtype::F16, &[2, 2])
        .expect("alloc FP16 output failed");
    write_f16(&mut lhs, &[0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600]);
    write_f16(&mut rhs, &[0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00]);

    let desc = AiKernelDesc::new_with_op(
        KernelOp::MAT_MUL,
        &MatMulAttr {
            m: DimSize::new(2),
            n: DimSize::new(2),
            k: DimSize::new(3),
            batch: DimSize::new(1),
            lhs_row_stride: ElemStride::new(3),
            lhs_col_stride: ElemStride::new(1),
            lhs_batch_stride: ElemStride::new(6),
            rhs_row_stride: ElemStride::new(2),
            rhs_col_stride: ElemStride::new(1),
            rhs_batch_stride: ElemStride::new(6),
            out_row_stride: ElemStride::new(2),
            out_col_stride: ElemStride::new(1),
            out_batch_stride: ElemStride::new(4),
            flags: OpFlags::new(0),
            accum_dtype: AiDtype::F32,
            reserved: [0; 3],
        },
        target,
        TensorCount::new(2),
        TensorCount::new(1),
        &[lhs.desc(), rhs.desc(), output.desc()],
    );
    let mut graph = GraphManager::new();
    graph
        .push_kernel_no_depend(desc)
        .expect("push FP16 MatMul node failed");
    let blob = graph.freeze().expect("freeze FP16 MatMul graph failed");
    let entry = blob.submit_entry(UserToken::new(1));
    submit_graph(channel, &entry).expect("submit FP16 MatMul graph failed");

    if drop_inputs_before_wait {
        drop(lhs);
        drop(rhs);
        drop(tensor_mgr);
        drop(blob);
        drop(graph);
    }
    wait_graph_complete(&entry, channel).expect("FP16 MatMul graph execution failed");
    read_f16(&output)
}

/// 将 FP16 bit-pattern 写入 tensor。
fn write_f16(tensor: &mut Tensor, values: &[u16]) {
    assert_eq!(tensor.dtype(), AiDtype::F16);
    assert_eq!(
        tensor.as_mut_slice().len(),
        values.len() * core::mem::size_of::<u16>()
    );
    for (bytes, value) in tensor.as_mut_slice().chunks_exact_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
}

/// 从 FP16 tensor 读取 bit-pattern。
fn read_f16(tensor: &Tensor) -> [u16; 4] {
    assert_eq!(tensor.dtype(), AiDtype::F16);
    let values: Vec<u16> = tensor
        .as_slice()
        .chunks_exact(2)
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    values.try_into().expect("expected four FP16 output values")
}
