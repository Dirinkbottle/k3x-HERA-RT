//! 测试 user api。

use crate::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, DimSize, GraphManager, KernelOp, MatMulAttr, TensorCount,
    TensorManager, UserToken,
};

/// 构造 matmul 单算子 graph，校验 desc、tensor 与提交 entry 的关键字段。
#[test]
fn build_matmul_kernel_desc() {
    let tensor_manager = TensorManager::new();
    let lhs = tensor_manager.alloc_tensor(AiDtype::F32, &[3, 4]);
    let rhs = tensor_manager.alloc_tensor(AiDtype::F32, &[4, 3]);
    let output = tensor_manager.alloc_tensor(AiDtype::F32, &[3, 3]);

    let matmul_attr = MatMulAttr {
        m: DimSize::new(3),
        n: DimSize::new(3),
        k: DimSize::new(4),
        batch: DimSize::new(1),
        accum_dtype: AiDtype::F32,
        ..Default::default()
    };
    let tensors = [
        lhs.expect("REASON").desc(),
        rhs.expect("REASON").desc(),
        output.expect("REASON").desc(),
    ];
    let matmul_desc = AiKernelDesc::new(
        &matmul_attr,
        AiTargetHint::AUTO,
        TensorCount::new(2),
        TensorCount::new(1),
        &tensors,
    );
    let mut graph = GraphManager::new();
    graph
        .push_kernel_no_depend(matmul_desc)
        .expect("matmul kernel graph build error");
    let submit_blob = graph.freeze().expect("matmul kernel graph build error");
    let submit_entry = submit_blob.submit_entry(UserToken::new(2));

    assert_eq!(matmul_desc.op, KernelOp::MAT_MUL);
    assert_eq!(matmul_desc.target_hint, AiTargetHint::AUTO);
    assert_eq!(matmul_desc.input_count, 2);
    assert_eq!(matmul_desc.output_count, 1);
    assert_eq!(
        matmul_desc.attr_size.get() as usize,
        core::mem::size_of::<MatMulAttr>()
    );
    assert_eq!(submit_entry.user_token, 2);
    assert_ne!(matmul_desc.tensors[0].user_va, 0);
    assert_ne!(matmul_desc.tensors[1].user_va, 0);
    assert_ne!(matmul_desc.tensors[2].user_va, 0);
    assert_eq!(matmul_desc.tensors[0].size_bytes, 3 * 4 * 4);
    assert_eq!(matmul_desc.tensors[1].size_bytes, 4 * 3 * 4);
    assert_eq!(matmul_desc.tensors[2].size_bytes, 3 * 3 * 4);
}
