//! k3 AI runtime 集成测试：通过 graph 提交通路覆盖 backend 算子调用。

use k3_ai_runtime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, BinaryAttr, CastAttr, ConcatAttr, Conv2dAttr, DimCount,
    DimSize, ElemStride, ExpandAttr, GatherAttr, GatherElementsAttr, GraphManager, KernelOp,
    KernelStride, MAX_DIM, MatMulAttr, OpFlags, Pool2dAttr, ReduceMaxAttr, Resize2dAttr,
    RmsNormAttr, RopeAttr, SoftmaxAttr, Tensor, TensorAxis, TensorCount, TensorManager, TileAttr,
    TopKAttr, TransposeAttr, UnaryAttr, UserToken,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};
mod test;

const M: usize = 4;
const K: usize = 8;
const N: usize = 4;
const EPS: f32 = 1.0e-4;

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
    // cow保活测试
    test::test_cow_live();
    test::test_quantized_ops();

    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");
    let mut tokens = TokenSource::new(42);

    println!(
        "k3_test: channel built, va={:#x}, size={:#x}",
        channel.shared.user_va, channel.shared.size_bytes
    );

    println!("k3_test: submitting CPU int8 matmul");
    let cpu_result = run_matmul(&channel, AiTargetHint::AUTO, &mut tokens, "matmul/cpu");
    print_matrix("cpu matmul result", &cpu_result);
    assert_eq!(cpu_result, EXPECTED);

    run_binary_tests(&channel, &mut tokens);
    run_unary_tests(&channel, &mut tokens);
    run_conv2d_tests(&channel, &mut tokens);
    run_nn_tests(&channel, &mut tokens);
    run_transform_tests(&channel, &mut tokens);
    run_rvv_tests(&channel, &mut tokens);

    println!("k3_test: all backend operator call tests passed");
}

struct TokenSource {
    next: u32,
}

impl TokenSource {
    fn new(first: u32) -> Self {
        Self { next: first }
    }

    fn next(&mut self) -> u32 {
        let token = self.next;
        self.next = self.next.checked_add(1).expect("user token overflow");
        token
    }
}

fn run_kernel<T: Copy>(
    channel: &UringChannel,
    op: KernelOp,
    attr: &T,
    target: AiTargetHint,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
    tokens: &mut TokenSource,
    label: &str,
) {
    let tensors = inputs
        .iter()
        .chain(outputs.iter())
        .map(|tensor| tensor.desc())
        .collect::<Vec<_>>();
    let desc = AiKernelDesc::new_with_op(
        op,
        attr,
        target,
        TensorCount::new(inputs.len() as u32),
        TensorCount::new(outputs.len() as u32),
        &tensors,
    );

    let mut graph = GraphManager::new();
    graph
        .push_kernel_no_depend(desc)
        .unwrap_or_else(|err| panic!("failed to push {label} node: {err:?}"));

    let blob = graph
        .freeze()
        .unwrap_or_else(|err| panic!("failed to freeze {label} graph: {err:?}"));
    let entry = blob.submit_entry(UserToken::new(tokens.next()));

    println!(
        "k3_test: submitting {label}, op={:?}, target={:?}",
        op, target
    );
    submit_graph(channel, &entry).unwrap_or_else(|err| {
        panic!("failed to submit {label} graph: {err}");
    });
    wait_graph_complete(&entry, channel).unwrap_or_else(|err| {
        panic!("{label} graph execute failed: {err}");
    });
}

fn run_matmul(
    channel: &UringChannel,
    target: AiTargetHint,
    tokens: &mut TokenSource,
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

    run_kernel(
        channel,
        KernelOp::MAT_MUL,
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
        &[&lhs, &rhs],
        &[&out],
        tokens,
        label,
    );

    tensor_i32_array(&out)
}

fn run_binary_tests(channel: &UringChannel, tokens: &mut TokenSource) {
    println!("k3_test: binary.rs operator calls");

    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [10.0_f32, 20.0, 30.0];

    let add = run_binary_f32(
        channel,
        KernelOp::ADD,
        &[2, 3],
        &[3],
        &[2, 3],
        &lhs,
        &rhs,
        AiTargetHint::AUTO,
        tokens,
        "binary/add",
    );
    assert_f32_close(
        "binary/add",
        &add,
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        EPS,
    );

    let sub = run_binary_f32(
        channel,
        KernelOp::SUB,
        &[2, 3],
        &[3],
        &[2, 3],
        &lhs,
        &rhs,
        AiTargetHint::AUTO,
        tokens,
        "binary/sub",
    );
    assert_f32_close(
        "binary/sub",
        &sub,
        &[-9.0, -18.0, -27.0, -6.0, -15.0, -24.0],
        EPS,
    );

    let mul = run_binary_f32(
        channel,
        KernelOp::MUL,
        &[2, 3],
        &[3],
        &[2, 3],
        &lhs,
        &rhs,
        AiTargetHint::AUTO,
        tokens,
        "binary/mul",
    );
    assert_f32_close(
        "binary/mul",
        &mul,
        &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0],
        EPS,
    );

    let div = run_binary_f32(
        channel,
        KernelOp::DIV,
        &[2, 3],
        &[3],
        &[2, 3],
        &lhs,
        &rhs,
        AiTargetHint::AUTO,
        tokens,
        "binary/div",
    );
    assert_f32_close("binary/div", &div, &[0.1, 0.1, 0.1, 0.4, 0.25, 0.2], EPS);

    let int_mod = run_binary_i32(
        channel,
        KernelOp::MOD,
        &[3],
        &[3],
        &[3],
        &[7, 8, 9],
        &[2, 3, 4],
        tokens,
        "binary/mod_i32",
    );
    assert_eq!(int_mod, [1, 2, 1]);
}

fn run_unary_tests(channel: &UringChannel, tokens: &mut TokenSource) {
    println!("k3_test: unary.rs operator calls");

    let silu_input = [-3.0_f32, -0.5, 0.0, 0.5, 3.0, 10.0];
    let silu = run_unary_f32(
        channel,
        KernelOp::SILU,
        UnaryAttr::default(),
        &[6],
        &silu_input,
        AiTargetHint::AUTO,
        tokens,
        "unary/silu",
    );
    let silu_expected = silu_input.map(silu_ref);
    assert_f32_close("unary/silu", &silu, &silu_expected, EPS);

    let sigmoid_input = [0.0_f32, -100.0, 100.0];
    let sigmoid = run_unary_f32(
        channel,
        KernelOp::SIGMOID,
        UnaryAttr::default(),
        &[3],
        &sigmoid_input,
        AiTargetHint::AUTO,
        tokens,
        "unary/sigmoid",
    );
    assert_f32_close("unary/sigmoid", &sigmoid, &[0.5, 0.0, 1.0], EPS);

    let scale = run_unary_f32(
        channel,
        KernelOp::SCALE,
        UnaryAttr {
            alpha: 2.0,
            beta: 1.0,
            ..UnaryAttr::default()
        },
        &[3],
        &[1.0, 2.0, -1.0],
        AiTargetHint::AUTO,
        tokens,
        "unary/scale",
    );
    assert_f32_close("unary/scale", &scale, &[3.0, 5.0, -1.0], EPS);
}

fn run_conv2d_tests(channel: &UringChannel, tokens: &mut TokenSource) {
    println!("k3_test: conv2d.rs operator calls");

    let f32_sum = run_conv2d_f32(
        channel,
        conv2d_attr(1, 1, 1, 3, 3, 2, 2, 2, 2, 1, 0, 1),
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1.0; 4],
        None,
        tokens,
        "conv2d/f32_sum",
    );
    assert_f32_close("conv2d/f32_sum", &f32_sum, &[12.0, 16.0, 24.0, 28.0], EPS);

    let f32_depthwise = run_conv2d_f32(
        channel,
        conv2d_attr(1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 2),
        &[1, 2, 2, 2],
        &[2, 1, 1, 1],
        &[1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[10.0, 20.0],
        Some(&[1.0, -1.0]),
        tokens,
        "conv2d/f32_depthwise_bias",
    );
    assert_f32_close(
        "conv2d/f32_depthwise_bias",
        &f32_depthwise,
        &[11.0, 21.0, 31.0, 41.0, 99.0, 119.0, 139.0, 159.0],
        EPS,
    );

    let int8_sum = run_conv2d_i8(
        channel,
        conv2d_attr(1, 1, 1, 3, 3, 2, 2, 2, 2, 1, 0, 1),
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        &[1, 1, 1, 1],
        None,
        tokens,
        "conv2d/int8_sum",
    );
    assert_eq!(int8_sum, [12, 16, 24, 28]);
}

fn run_nn_tests(channel: &UringChannel, tokens: &mut TokenSource) {
    println!("k3_test: nn.rs operator calls");

    let softmax_input = [1.0_f32, 2.0, 3.0, -1.0, 0.0, 1.0];
    let softmax = run_softmax_f32(channel, &[2, 3], &softmax_input, tokens);
    assert_f32_close(
        "nn/softmax",
        &softmax[..3],
        &softmax_ref(&softmax_input[..3]),
        EPS,
    );
    assert_f32_close(
        "nn/softmax",
        &softmax[3..],
        &softmax_ref(&softmax_input[3..]),
        EPS,
    );

    let rms = run_rms_norm_f32(
        channel,
        &[2, 2],
        &[2],
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 1.0],
        tokens,
    );
    let row0 = 1.0 / (2.5_f32 + 1.0e-5).sqrt();
    let row1 = 1.0 / (12.5_f32 + 1.0e-5).sqrt();
    assert_f32_close(
        "nn/rms_norm",
        &rms,
        &[row0, 2.0 * row0, 3.0 * row1, 4.0 * row1],
        EPS,
    );

    let rope_input = [1.0_f32, 2.0, 3.0, 4.0];
    let rope = run_rope_f32(channel, &[1, 1, 1, 4], &rope_input, &[1], tokens);
    assert_f32_close(
        "nn/rope",
        &rope,
        &rope_gptj_ref(&rope_input, 1, 10_000.0),
        EPS,
    );

    let (pool_values, pool_indices) = run_max_pool_f32(
        channel,
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        tokens,
    );
    assert_f32_close(
        "nn/max_pool values",
        &pool_values,
        &[5.0, 6.0, 8.0, 9.0],
        EPS,
    );
    assert_eq!(pool_indices, [4, 5, 7, 8]);

    let reduced = run_reduce_max_f32(
        channel,
        &[2, 2, 2],
        &[2],
        &[1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0],
        tokens,
    );
    assert_f32_close("nn/reduce_max", &reduced, &[4.0, 8.0], EPS);

    let (top_values, top_indices) = run_top_k_f32(channel, &[4], &[3.0, 5.0, 5.0, 1.0], tokens);
    assert_f32_close("nn/top_k values", &top_values, &[5.0, 5.0], EPS);
    assert_eq!(top_indices, [1, 2]);
}

fn run_transform_tests(channel: &UringChannel, tokens: &mut TokenSource) {
    println!("k3_test: transform.rs operator calls");

    let concat = run_concat_f32(
        channel,
        &[&[1.0_f32, 2.0][..], &[3.0_f32, 4.0, 5.0, 6.0][..]],
        &[&[1, 2][..], &[2, 2][..]],
        &[3, 2],
        tokens,
    );
    assert_f32_close(
        "transform/concat",
        &concat,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        EPS,
    );

    let transpose = run_transpose_f32(
        channel,
        &[2, 3],
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        tokens,
    );
    assert_f32_close(
        "transform/transpose",
        &transpose,
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        EPS,
    );

    let gather = run_gather_f32(
        channel,
        &[4],
        &[3],
        &[10.0, 20.0, 30.0, 40.0],
        &[3, 1, 0],
        tokens,
    );
    assert_f32_close("transform/gather", &gather, &[40.0, 20.0, 10.0], EPS);

    let gather_elements = run_gather_elements_f32(
        channel,
        &[2, 3],
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 1, 0, 2],
        tokens,
    );
    assert_f32_close(
        "transform/gather_elements",
        &gather_elements,
        &[3.0, 2.0, 4.0, 6.0],
        EPS,
    );

    let cast = run_cast_f32_to_i32(channel, &[2], &[1.5, -2.25], tokens);
    assert_eq!(cast, [1, -2]);

    let resize = run_resize_f32(
        channel,
        &[1, 1, 2, 2],
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0],
        tokens,
    );
    assert_f32_close(
        "transform/resize",
        &resize,
        &[1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0],
        EPS,
    );

    let expand = run_expand_f32(channel, &[1, 2], &[3, 2], &[1.0, 2.0], tokens);
    assert_f32_close(
        "transform/expand",
        &expand,
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        EPS,
    );

    let tile = run_tile_f32(channel, &[2], &[6], &[7.0, 8.0], &[3], tokens);
    assert_f32_close(
        "transform/tile",
        &tile,
        &[7.0, 8.0, 7.0, 8.0, 7.0, 8.0],
        EPS,
    );
}

fn run_rvv_tests(channel: &UringChannel, tokens: &mut TokenSource) {
    println!("k3_test: rvv.rs operator calls through PREFER_X100 (A100 is not used)");

    let binary = run_binary_f32(
        channel,
        KernelOp::MUL,
        &[3],
        &[3],
        &[3],
        &[-2.0, 0.0, 2.0],
        &[2.0, 3.0, 4.0],
        AiTargetHint::AUTO,
        tokens,
        "rvv/binary_mul_x100",
    );
    assert_f32_close("rvv/binary_mul_x100", &binary, &[-4.0, 0.0, 8.0], EPS);

    let affine = run_unary_f32(
        channel,
        KernelOp::SCALE,
        UnaryAttr {
            alpha: 2.0,
            beta: 1.0,
            ..UnaryAttr::default()
        },
        &[3],
        &[-2.0, 0.0, 2.0],
        AiTargetHint::AUTO,
        tokens,
        "rvv/affine_x100",
    );
    assert_f32_close("rvv/affine_x100", &affine, &[-3.0, 1.0, 5.0], EPS);
}

fn run_binary_f32(
    channel: &UringChannel,
    op: KernelOp,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    output_shape: &[u32],
    lhs_values: &[f32],
    rhs_values: &[f32],
    target: AiTargetHint,
    tokens: &mut TokenSource,
    label: &str,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut lhs = tensor_mgr
        .alloc_tensor(AiDtype::F32, lhs_shape)
        .expect("alloc binary lhs failed");
    let mut rhs = tensor_mgr
        .alloc_tensor(AiDtype::F32, rhs_shape)
        .expect("alloc binary rhs failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc binary out failed");

    copy_f32_to_tensor(&mut lhs, lhs_values);
    copy_f32_to_tensor(&mut rhs, rhs_values);
    run_kernel(
        channel,
        op,
        &BinaryAttr::default(),
        target,
        &[&lhs, &rhs],
        &[&out],
        tokens,
        label,
    );
    tensor_f32_vec(&out)
}

fn run_binary_i32(
    channel: &UringChannel,
    op: KernelOp,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    output_shape: &[u32],
    lhs_values: &[i32],
    rhs_values: &[i32],
    tokens: &mut TokenSource,
    label: &str,
) -> Vec<i32> {
    let tensor_mgr = TensorManager::new();
    let mut lhs = tensor_mgr
        .alloc_tensor(AiDtype::I32, lhs_shape)
        .expect("alloc binary lhs failed");
    let mut rhs = tensor_mgr
        .alloc_tensor(AiDtype::I32, rhs_shape)
        .expect("alloc binary rhs failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::I32, output_shape)
        .expect("alloc binary out failed");

    copy_i32_to_tensor(&mut lhs, lhs_values);
    copy_i32_to_tensor(&mut rhs, rhs_values);
    run_kernel(
        channel,
        op,
        &BinaryAttr::default(),
        AiTargetHint::AUTO,
        &[&lhs, &rhs],
        &[&out],
        tokens,
        label,
    );
    tensor_i32_vec(&out)
}

fn run_unary_f32(
    channel: &UringChannel,
    op: KernelOp,
    attr: UnaryAttr,
    shape: &[u32],
    values: &[f32],
    target: AiTargetHint,
    tokens: &mut TokenSource,
    label: &str,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .expect("alloc unary input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .expect("alloc unary out failed");

    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        op,
        &attr,
        target,
        &[&input],
        &[&out],
        tokens,
        label,
    );
    tensor_f32_vec(&out)
}

fn run_conv2d_f32(
    channel: &UringChannel,
    attr: Conv2dAttr,
    input_shape: &[u32],
    weight_shape: &[u32],
    output_shape: &[u32],
    input_values: &[f32],
    weight_values: &[f32],
    bias_values: Option<&[f32]>,
    tokens: &mut TokenSource,
    label: &str,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc conv2d input failed");
    let mut weight = tensor_mgr
        .alloc_tensor(AiDtype::F32, weight_shape)
        .expect("alloc conv2d weight failed");
    let mut bias = bias_values.map(|values| {
        let mut tensor = tensor_mgr
            .alloc_tensor(AiDtype::F32, &[values.len() as u32])
            .expect("alloc conv2d bias failed");
        copy_f32_to_tensor(&mut tensor, values);
        tensor
    });
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc conv2d out failed");

    copy_f32_to_tensor(&mut input, input_values);
    copy_f32_to_tensor(&mut weight, weight_values);

    let mut inputs = vec![&input, &weight];
    if let Some(bias) = bias.as_ref() {
        inputs.push(bias);
    }
    run_kernel(
        channel,
        KernelOp::CONV2D,
        &attr,
        AiTargetHint::AUTO,
        &inputs,
        &[&out],
        tokens,
        label,
    );
    drop(bias.take());
    tensor_f32_vec(&out)
}

fn run_conv2d_i8(
    channel: &UringChannel,
    attr: Conv2dAttr,
    input_shape: &[u32],
    weight_shape: &[u32],
    output_shape: &[u32],
    input_values: &[i8],
    weight_values: &[i8],
    bias_values: Option<&[i32]>,
    tokens: &mut TokenSource,
    label: &str,
) -> Vec<i32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::I8, input_shape)
        .expect("alloc conv2d input failed");
    let mut weight = tensor_mgr
        .alloc_tensor(AiDtype::I8, weight_shape)
        .expect("alloc conv2d weight failed");
    let mut bias = bias_values.map(|values| {
        let mut tensor = tensor_mgr
            .alloc_tensor(AiDtype::I32, &[values.len() as u32])
            .expect("alloc conv2d bias failed");
        copy_i32_to_tensor(&mut tensor, values);
        tensor
    });
    let out = tensor_mgr
        .alloc_tensor(AiDtype::I32, output_shape)
        .expect("alloc conv2d out failed");

    copy_i8_to_tensor(&mut input, input_values);
    copy_i8_to_tensor(&mut weight, weight_values);

    let mut inputs = vec![&input, &weight];
    if let Some(bias) = bias.as_ref() {
        inputs.push(bias);
    }
    run_kernel(
        channel,
        KernelOp::CONV2D,
        &attr,
        AiTargetHint::AUTO,
        &inputs,
        &[&out],
        tokens,
        label,
    );
    drop(bias.take());
    tensor_i32_vec(&out)
}

fn run_softmax_f32(
    channel: &UringChannel,
    shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .expect("alloc softmax input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .expect("alloc softmax out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::SOFTMAX,
        &SoftmaxAttr {
            axis: TensorAxis::new(1),
            scale: 1.0,
            max_bias: 0.0,
            flags: OpFlags::new(0),
            reserved: [0; 12],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "nn/softmax",
    );
    tensor_f32_vec(&out)
}

fn run_rms_norm_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    weight_shape: &[u32],
    input_values: &[f32],
    weight_values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc rms input failed");
    let mut weight = tensor_mgr
        .alloc_tensor(AiDtype::F32, weight_shape)
        .expect("alloc rms weight failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc rms out failed");
    copy_f32_to_tensor(&mut input, input_values);
    copy_f32_to_tensor(&mut weight, weight_values);
    run_kernel(
        channel,
        KernelOp::RMS_NORM,
        &RmsNormAttr {
            hidden_size: DimSize::new(weight_values.len() as u32),
            eps: 1.0e-5,
            flags: OpFlags::new(0),
            reserved: [0; 13],
        },
        AiTargetHint::AUTO,
        &[&input, &weight],
        &[&out],
        tokens,
        "nn/rms_norm",
    );
    tensor_f32_vec(&out)
}

fn run_rope_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    values: &[f32],
    positions: &[i64],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc rope input failed");
    let mut position = tensor_mgr
        .alloc_tensor(AiDtype::I64, &[positions.len() as u32])
        .expect("alloc rope position failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc rope out failed");
    copy_f32_to_tensor(&mut input, values);
    copy_i64_to_tensor(&mut position, positions);
    run_kernel(
        channel,
        KernelOp::ROPE,
        &RopeAttr {
            n_dims: DimSize::new(4),
            mode: RopeAttr::MODE_GPT_J,
            n_ctx: DimSize::new(4),
            head_count: DimSize::new(1),
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            flags: OpFlags::new(0),
            reserved: [0; 5],
        },
        AiTargetHint::AUTO,
        &[&input, &position],
        &[&out],
        tokens,
        "nn/rope",
    );
    tensor_f32_vec(&out)
}

fn run_max_pool_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    output_shape: &[u32],
    input_values: &[f32],
    tokens: &mut TokenSource,
) -> (Vec<f32>, Vec<i64>) {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc max_pool input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc max_pool out failed");
    let indices = tensor_mgr
        .alloc_tensor(AiDtype::I64, output_shape)
        .expect("alloc max_pool indices failed");
    copy_f32_to_tensor(&mut input, input_values);
    run_kernel(
        channel,
        KernelOp::MAX_POOL,
        &Pool2dAttr {
            kernel_h: DimSize::new(2),
            kernel_w: DimSize::new(2),
            stride_h: KernelStride::new(1),
            stride_w: KernelStride::new(1),
            dilation_h: KernelStride::new(1),
            dilation_w: KernelStride::new(1),
            pad_top: DimSize::new(0),
            pad_left: DimSize::new(0),
            pad_bottom: DimSize::new(0),
            pad_right: DimSize::new(0),
            flags: OpFlags::new(0),
            reserved: [0; 5],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out, &indices],
        tokens,
        "nn/max_pool",
    );
    (tensor_f32_vec(&out), tensor_i64_vec(&indices))
}

fn run_reduce_max_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    output_shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc reduce input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc reduce out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::REDUCE_MAX,
        &ReduceMaxAttr {
            axis_count: DimCount::new(2),
            axes: axis_array(&[1, 2]),
            flags: OpFlags::new(0),
            reserved: [0; 6],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "nn/reduce_max",
    );
    tensor_f32_vec(&out)
}

fn run_top_k_f32(
    channel: &UringChannel,
    shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> (Vec<f32>, Vec<i64>) {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .expect("alloc topk input failed");
    let values_out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[2])
        .expect("alloc topk values failed");
    let indices_out = tensor_mgr
        .alloc_tensor(AiDtype::I64, &[2])
        .expect("alloc topk indices failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::TOP_K,
        &TopKAttr {
            axis: TensorAxis::new(0),
            k: DimSize::new(2),
            largest: 1,
            sorted: 1,
            flags: OpFlags::new(0),
            reserved: [0; 11],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&values_out, &indices_out],
        tokens,
        "nn/top_k",
    );
    (tensor_f32_vec(&values_out), tensor_i64_vec(&indices_out))
}

fn run_concat_f32(
    channel: &UringChannel,
    values: &[&[f32]],
    shapes: &[&[u32]],
    output_shape: &[u32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut inputs = Vec::with_capacity(values.len());
    for (shape, values) in shapes.iter().zip(values.iter()) {
        let mut tensor = tensor_mgr
            .alloc_tensor(AiDtype::F32, shape)
            .expect("alloc concat input failed");
        copy_f32_to_tensor(&mut tensor, values);
        inputs.push(tensor);
    }
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc concat out failed");
    let input_refs = inputs.iter().collect::<Vec<_>>();
    run_kernel(
        channel,
        KernelOp::CONCAT,
        &ConcatAttr {
            axis: TensorAxis::new(0),
            flags: OpFlags::new(0),
            reserved: [0; 14],
        },
        AiTargetHint::AUTO,
        &input_refs,
        &[&out],
        tokens,
        "transform/concat",
    );
    tensor_f32_vec(&out)
}

fn run_transpose_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    output_shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc transpose input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc transpose out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::TRANSPOSE,
        &TransposeAttr {
            rank: DimCount::new(2),
            perm: axis_array(&[1, 0]),
            flags: OpFlags::new(0),
            reserved: [0; 6],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "transform/transpose",
    );
    tensor_f32_vec(&out)
}

fn run_gather_f32(
    channel: &UringChannel,
    data_shape: &[u32],
    output_shape: &[u32],
    data_values: &[f32],
    indices_values: &[i64],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut data = tensor_mgr
        .alloc_tensor(AiDtype::F32, data_shape)
        .expect("alloc gather data failed");
    let mut indices = tensor_mgr
        .alloc_tensor(AiDtype::I64, &[indices_values.len() as u32])
        .expect("alloc gather indices failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc gather out failed");
    copy_f32_to_tensor(&mut data, data_values);
    copy_i64_to_tensor(&mut indices, indices_values);
    run_kernel(
        channel,
        KernelOp::GATHER,
        &GatherAttr {
            axis: TensorAxis::new(0),
            flags: OpFlags::new(0),
            reserved: [0; 14],
        },
        AiTargetHint::AUTO,
        &[&data, &indices],
        &[&out],
        tokens,
        "transform/gather",
    );
    tensor_f32_vec(&out)
}

fn run_gather_elements_f32(
    channel: &UringChannel,
    data_shape: &[u32],
    indices_shape: &[u32],
    data_values: &[f32],
    indices_values: &[i64],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut data = tensor_mgr
        .alloc_tensor(AiDtype::F32, data_shape)
        .expect("alloc gather_elements data failed");
    let mut indices = tensor_mgr
        .alloc_tensor(AiDtype::I64, indices_shape)
        .expect("alloc gather_elements indices failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, indices_shape)
        .expect("alloc gather_elements out failed");
    copy_f32_to_tensor(&mut data, data_values);
    copy_i64_to_tensor(&mut indices, indices_values);
    run_kernel(
        channel,
        KernelOp::GATHER_ELEMENTS,
        &GatherElementsAttr {
            axis: TensorAxis::new(1),
            flags: OpFlags::new(0),
            reserved: [0; 14],
        },
        AiTargetHint::AUTO,
        &[&data, &indices],
        &[&out],
        tokens,
        "transform/gather_elements",
    );
    tensor_f32_vec(&out)
}

fn run_cast_f32_to_i32(
    channel: &UringChannel,
    shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<i32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .expect("alloc cast input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::I32, shape)
        .expect("alloc cast out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::CAST,
        &CastAttr {
            to: AiDtype::I32,
            flags: OpFlags::new(0),
            reserved: [0; 14],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "transform/cast",
    );
    tensor_i32_vec(&out)
}

fn run_resize_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    output_shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc resize input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc resize out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::RESIZE,
        &Resize2dAttr {
            mode: Resize2dAttr::MODE_LINEAR,
            coordinate_mode: Resize2dAttr::COORD_ALIGN_CORNERS,
            nearest_mode: Resize2dAttr::NEAREST_ROUND_PREFER_FLOOR,
            input_h: DimSize::new(input_shape[2]),
            input_w: DimSize::new(input_shape[3]),
            output_h: DimSize::new(output_shape[2]),
            output_w: DimSize::new(output_shape[3]),
            flags: OpFlags::new(0),
            reserved: [0; 8],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "transform/resize",
    );
    tensor_f32_vec(&out)
}

fn run_expand_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    output_shape: &[u32],
    values: &[f32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc expand input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc expand out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::EXPAND,
        &ExpandAttr {
            rank: DimCount::new(output_shape.len() as u32),
            target_shape: dim_size_array(output_shape),
            flags: OpFlags::new(0),
            reserved: [0; 6],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "transform/expand",
    );
    tensor_f32_vec(&out)
}

fn run_tile_f32(
    channel: &UringChannel,
    input_shape: &[u32],
    output_shape: &[u32],
    values: &[f32],
    repeats: &[u32],
    tokens: &mut TokenSource,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut input = tensor_mgr
        .alloc_tensor(AiDtype::F32, input_shape)
        .expect("alloc tile input failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, output_shape)
        .expect("alloc tile out failed");
    copy_f32_to_tensor(&mut input, values);
    run_kernel(
        channel,
        KernelOp::TILE,
        &TileAttr {
            rank: DimCount::new(input_shape.len() as u32),
            repeats: dim_size_array(repeats),
            flags: OpFlags::new(0),
            reserved: [0; 6],
        },
        AiTargetHint::AUTO,
        &[&input],
        &[&out],
        tokens,
        "transform/tile",
    );
    tensor_f32_vec(&out)
}

fn conv2d_attr(
    batch: u32,
    in_channels: u32,
    out_channels: u32,
    input_h: u32,
    input_w: u32,
    output_h: u32,
    output_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    pad: u32,
    groups: u32,
) -> Conv2dAttr {
    Conv2dAttr {
        batch: DimSize::new(batch),
        in_channels: DimSize::new(in_channels),
        out_channels: DimSize::new(out_channels),
        input_h: DimSize::new(input_h),
        input_w: DimSize::new(input_w),
        output_h: DimSize::new(output_h),
        output_w: DimSize::new(output_w),
        kernel_h: DimSize::new(kernel_h),
        kernel_w: DimSize::new(kernel_w),
        stride_h: KernelStride::new(stride),
        stride_w: KernelStride::new(stride),
        pad_h: DimSize::new(pad),
        pad_w: DimSize::new(pad),
        dilation_h: KernelStride::new(1),
        dilation_w: KernelStride::new(1),
        groups: DimSize::new(groups),
        flags: OpFlags::new(0),
        reserved: [0; 15],
    }
}

fn copy_i8_to_tensor(tensor: &mut Tensor, values: &[i8]) {
    assert_eq!(tensor.dtype(), AiDtype::I8);
    assert_eq!(tensor.size_bytes().get() as usize, values.len());

    let bytes = tensor.as_mut_slice();
    for (dst, src) in bytes.iter_mut().zip(values.iter().copied()) {
        *dst = src as u8;
    }
}

fn copy_f32_to_tensor(tensor: &mut Tensor, values: &[f32]) {
    assert_eq!(tensor.dtype(), AiDtype::F32);
    assert_eq!(tensor.as_f32_slice().len(), values.len());
    tensor.as_f32_mut_slice().copy_from_slice(values);
}

fn copy_i32_to_tensor(tensor: &mut Tensor, values: &[i32]) {
    assert_eq!(tensor.dtype(), AiDtype::I32);
    assert_eq!(
        tensor.size_bytes().get() as usize,
        values.len() * core::mem::size_of::<i32>()
    );
    for (chunk, value) in tensor.as_mut_slice().chunks_exact_mut(4).zip(values.iter()) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

fn copy_i64_to_tensor(tensor: &mut Tensor, values: &[i64]) {
    assert_eq!(tensor.dtype(), AiDtype::I64);
    assert_eq!(
        tensor.size_bytes().get() as usize,
        values.len() * core::mem::size_of::<i64>()
    );
    for (chunk, value) in tensor.as_mut_slice().chunks_exact_mut(8).zip(values.iter()) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

fn tensor_i32_array(tensor: &Tensor) -> [i32; M * N] {
    let values = tensor_i32_vec(tensor);
    values.try_into().expect("matmul output shape mismatch")
}

fn tensor_f32_vec(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), AiDtype::F32);
    tensor.as_f32_slice().to_vec()
}

fn tensor_i32_vec(tensor: &Tensor) -> Vec<i32> {
    assert_eq!(tensor.dtype(), AiDtype::I32);
    tensor
        .as_slice()
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn tensor_i64_vec(tensor: &Tensor) -> Vec<i64> {
    assert_eq!(tensor.dtype(), AiDtype::I64);
    tensor
        .as_slice()
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect()
}

fn dim_size_array(values: &[u32]) -> [DimSize; MAX_DIM] {
    let mut dims = [DimSize::new(0); MAX_DIM];
    for (dst, src) in dims.iter_mut().zip(values.iter().copied()) {
        *dst = DimSize::new(src);
    }
    dims
}

fn axis_array(values: &[i32]) -> [TensorAxis; MAX_DIM] {
    let mut axes = [TensorAxis::new(0); MAX_DIM];
    for (dst, src) in axes.iter_mut().zip(values.iter().copied()) {
        *dst = TensorAxis::new(src);
    }
    axes
}

fn silu_ref(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax_ref(values: &[f32]) -> Vec<f32> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = exponentials.iter().sum::<f32>();
    exponentials.into_iter().map(|value| value / sum).collect()
}

fn rope_gptj_ref(values: &[f32; 4], position: i64, base: f32) -> [f32; 4] {
    let mut output = *values;
    for pair in 0..2 {
        let first = pair * 2;
        let second = first + 1;
        let frequency = base.powf(-((2 * pair) as f32) / 4.0);
        let angle = position as f32 * frequency;
        let cosine = angle.cos();
        let sine = angle.sin();
        output[first] = values[first] * cosine - values[second] * sine;
        output[second] = values[first] * sine + values[second] * cosine;
    }
    output
}

fn assert_f32_close(label: &str, actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch, actual={actual:?}, expected={expected:?}"
    );
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= eps,
            "{label}[{idx}]: expected {expected}, got {actual}"
        );
    }
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
