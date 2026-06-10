//! 测试 user api。

use crate::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, AiTensorDesc, AiTensorFormat, AiTensorLayout, GraphManager, KernelOp, MatMulAttr, kd_uring::{submit_graph, wait_graph_complete}
};

#[test]
fn build_matmul_kernel_desc() {
    let lhs = AiTensorDesc::from_kernel_alloc(
        1,
        0x1000,
        3 * 4 * 4,
        AiDtype::F32,
        AiTensorFormat::ANY,
        AiTensorLayout::DENSE,
        &[3, 4],
        0,
    );
    let rhs = AiTensorDesc::from_kernel_alloc(
        2,
        0x2000,
        4 * 3 * 4,
        AiDtype::F32,
        AiTensorFormat::ANY,
        AiTensorLayout::DENSE,
        &[4, 3],
        0,
    );
    let output = AiTensorDesc::from_kernel_alloc(
        3,
        0x3000,
        3 * 3 * 4,
        AiDtype::F32,
        AiTensorFormat::ANY,
        AiTensorLayout::DENSE,
        &[3, 3],
        0,
    );

    let matmul_attr = MatMulAttr {
        m: 3,
        n: 3,
        k: 4,
        batch: 1,
        accum_dtype: AiDtype::F32,
        ..Default::default()
    };
    let tensors = [lhs, rhs, output];
    let matmul_desc = AiKernelDesc::new(&matmul_attr, AiTargetHint::AUTO, 2, 1, &tensors);
    let mut graph = GraphManager::new();
    graph.push_kernel_no_depend(matmul_desc).expect("matmul kernel graph build error");
    let submit_blob = graph.freeze().expect("matmul kernel graph build error");
    let submit_entry = submit_blob.submit_entry(2);
    // submit_graph(&submit_entry);
    // wait_graph_complete(&submit_entry);

    assert_eq!(matmul_desc.op, KernelOp::MAT_MUL);
    assert_eq!(matmul_desc.target_hint, AiTargetHint::AUTO);
    assert_eq!(matmul_desc.input_count, 2);
    assert_eq!(matmul_desc.output_count, 1);
    assert_eq!(
        matmul_desc.attr_size as usize,
        core::mem::size_of::<MatMulAttr>()
    );
    assert_eq!(matmul_desc.tensors[0].user_va, 0x1000);
    assert_eq!(matmul_desc.tensors[1].user_va, 0x2000);
    assert_eq!(matmul_desc.tensors[2].user_va, 0x3000);
}
