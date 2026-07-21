//! k3 AI runtime 集成测试：先跑 CPU matmul，再切 AI thread 后跑 A100 matmul。

use std::fs;

use k3_ai_runtime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, DimSize, ElemStride, GraphManager, MatMulAttr, OpFlags,
    Tensor, TensorCount, TensorManager, UserToken,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};

const M: usize = 4;
const K: usize = 8;
const N: usize = 4;

const LHS: [i8; M * K] = [
    0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10,
    11,
];

const RHS: [i8; K * N] = [
    0, 1, 2, 11, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9,
    10,
];

const EXPECTED: [i32; M * N] = [
    140, 168, 196, 224, 168, 204, 240, 284, 196, 240, 284, 344, 252, 312, 372, 464,
];

fn main() {
    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");

    println!(
        "k3_test: channel built, va={:#x}, size={:#x}",
        channel.shared.user_va, channel.shared.size_bytes
    );

    println!("k3_test: submitting CPU int8 matmul");
    let cpu_result = run_matmul(&channel, AiTargetHint::PREFER_CPU, 42, "cpu");
    print_matrix("cpu result", &cpu_result);
    assert_eq!(cpu_result, EXPECTED);

    fs::write("/proc/set_ai_thread", "1");
    println!("k3_test: submitting CPU int8 matmul on ai core!");
    let cpu_result_2 = run_matmul(&channel, AiTargetHint::PREFER_CPU, 43, "cpu");
    print_matrix("cpu result", &cpu_result_2);
    assert_eq!(cpu_result_2, EXPECTED);


}

fn run_matmul(
    channel: &UringChannel,
    target: AiTargetHint,
    token: u32,
    label: &str,
) -> [i32; M * N] {
    let tensor_mgr = TensorManager::new();

    let mut lhs = tensor_mgr
        .alloc_tensor(AiDtype::I8, &[M as u32, K as u32])
        .expect("alloc lhs failed");
    let mut rhs = tensor_mgr
        .alloc_tensor(AiDtype::I8, &[K as u32, N as u32])
        .expect("alloc rhs failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::I32, &[M as u32, N as u32])
        .expect("alloc out failed");

    copy_i8_to_tensor(&mut lhs, &LHS);
    copy_i8_to_tensor(&mut rhs, &RHS);

    let mut graph = GraphManager::new();
    graph
        .push_kernel_no_depend(AiKernelDesc::new(
            &MatMulAttr {
                m: DimSize::new(M as u32),
                n: DimSize::new(N as u32),
                k: DimSize::new(K as u32),
                batch: DimSize::new(0),
                lhs_row_stride: ElemStride::new(K as u32),
                lhs_col_stride: ElemStride::new(1),
                lhs_batch_stride: ElemStride::new(0),
                rhs_row_stride: ElemStride::new(N as u32),
                rhs_col_stride: ElemStride::new(1),
                rhs_batch_stride: ElemStride::new(0),
                out_row_stride: ElemStride::new(N as u32),
                out_col_stride: ElemStride::new(1),
                out_batch_stride: ElemStride::new(0),
                flags: OpFlags::new(0),
                accum_dtype: AiDtype::I32,
                reserved: [0; 3],
            },
            target,
            TensorCount::new(2),
            TensorCount::new(1),
            &[lhs.desc(), rhs.desc(), out.desc()],
        ))
        .expect("failed to push matmul node");

    let blob = graph.freeze().expect("failed to freeze graph");
    let entry = blob.submit_entry(UserToken::new(token));

    submit_graph(channel, &entry).unwrap_or_else(|err| {
        panic!("failed to submit {label} graph: {err}");
    });
    wait_graph_complete(&entry, channel).unwrap_or_else(|err| {
        panic!("{label} graph execute failed: {err}");
    });

    tensor_i32_array(&out)
}

fn copy_i8_to_tensor(tensor: &mut Tensor, values: &[i8]) {
    assert_eq!(tensor.dtype(), AiDtype::I8);
    assert_eq!(tensor.size_bytes().get() as usize, values.len());

    let bytes = tensor.as_mut_slice();
    for (dst, src) in bytes.iter_mut().zip(values.iter().copied()) {
        *dst = src as u8;
    }
}

fn tensor_i32_array(tensor: &Tensor) -> [i32; M * N] {
    assert_eq!(tensor.dtype(), AiDtype::I32);

    let bytes = tensor.as_slice();
    assert_eq!(bytes.len(), M * N * core::mem::size_of::<i32>());

    let mut values = [0_i32; M * N];
    for (idx, chunk) in bytes.chunks_exact(4).enumerate() {
        values[idx] = i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    values
}

fn print_matrix(label: &str, values: &[i32; M * N]) {
    println!("{label} ({M}x{N}):");
    for i in 0..M {
        print!("  [");
        for j in 0..N {
            print!("{:6}", values[i * N + j]);
        }
        println!(" ]");
    }
}
