use k3_ai_runtime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, DimSize, ElemStride, GetRowsAttr, GraphManager, KernelOp,
    MatMulAttr, OpFlags, Tensor, TensorCount, TensorManager, UserToken,
    kd_uring::{build_channel, submit_graph, wait_graph_complete},
};

const M: usize = 4;
const K: usize = 8;
const N: usize = 4;
const QUANT_K: usize = 32;
const Q3_K: usize = 256;
const Q_TEST_M: usize = 2;
const Q_TEST_N: usize = 3;
const EPS: f32 = 1.0e-4;
const F16_ONE_LE: [u8; 2] = 0x3c00_u16.to_le_bytes();
const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

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

/// 确保cow映射能够正确保活.
pub fn test_cow_live() {
    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");

    let token = 1;

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

    let desc = AiKernelDesc::new_with_op(
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
        AiTargetHint::AUTO,
        TensorCount::new(2),
        TensorCount::new(1),
        &[lhs.desc(), rhs.desc(), out.desc()],
    );

    let mut graph = GraphManager::new();
    graph
        .push_kernel_no_depend(desc)
        .expect("failed to push matmul node");

    let blob = graph.freeze().expect("failed to freeze graph");
    let entry = blob.submit_entry(UserToken::new(token));

    submit_graph(&channel, &entry).expect("failed to submit graph");

    // 立即释放输入tensor，内核的COW映射必须保活
    drop(lhs);
    drop(rhs);
    drop(tensor_mgr);
    drop(blob);
    drop(graph);

    wait_graph_complete(&entry, &channel).expect("graph execute failed");

    let result = tensor_i32_vec(&out);
    assert_eq!(result, EXPECTED, "cow_live: output mismatch");
    println!("Cow live test PASS!");
}

/// 覆盖 llama.cpp 权重量化实际会走的 ggml quant tensor 路径。
pub fn test_quantized_ops() {
    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");
    let mut token = 10_000_u32;

    test_q8_0_get_rows(&channel, &mut token);
    test_q8_0_matmul(&channel, &mut token);
    test_iq4_nl_matmul(&channel, &mut token);
    test_q3_k_matmul(&channel, &mut token);

    println!("Quantized ggml tensor tests PASS!");
}

fn test_q8_0_get_rows(channel: &k3_ai_runtime::fronted::kd_uring::UringChannel, token: &mut u32) {
    let tensor_mgr = TensorManager::new();
    let mut data = tensor_mgr
        .alloc_ggml_quant_tensor(AiDtype::Q8_0, &[QUANT_K as u32, 3], 0)
        .expect("alloc q8_0 rows failed");
    let mut indices = tensor_mgr
        .alloc_tensor(AiDtype::I32, &[2])
        .expect("alloc get_rows indices failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[QUANT_K as u32, 2, 1, 1])
        .expect("alloc get_rows out failed");

    let rows = [
        quant_pattern_q8(0),
        quant_pattern_q8(1),
        quant_pattern_q8(2),
    ];
    write_q8_0_rows(&mut data, &rows);
    copy_i32_to_tensor(&mut indices, &[2, 0]);

    run_kernel(
        channel,
        KernelOp::GET_ROWS,
        &GetRowsAttr {
            flags: OpFlags::new(0),
            reserved: [0; 15],
        },
        &[&data, &indices],
        &[&out],
        token,
        "quant/q8_0_get_rows",
    );

    let selected_rows = [2_usize, 0_usize];
    let expected = get_rows_expected_logical(&rows, &selected_rows);
    let actual = tensor_f32_vec(&out);
    print_quant_case("quant/q8_0_get_rows", &actual);
    assert_f32_close("quant/q8_0_get_rows", &actual, &expected, EPS);
}

fn test_q8_0_matmul(channel: &k3_ai_runtime::fronted::kd_uring::UringChannel, token: &mut u32) {
    let lhs_rows = [quant_pattern_q8(0), quant_pattern_q8(1)];
    let rhs = rhs_pattern::<QUANT_K, Q_TEST_N>();
    let expected = matmul_ref(&lhs_rows, &rhs);
    let actual = run_quant_matmul(
        channel,
        token,
        AiDtype::Q8_0,
        &pack_q8_0_rows(&lhs_rows),
        QUANT_K,
        &rhs,
        "quant/q8_0_matmul",
    );
    print_quant_case("quant/q8_0_matmul", &actual);
    assert_f32_close("quant/q8_0_matmul", &actual, &expected, EPS);
}

fn test_iq4_nl_matmul(channel: &k3_ai_runtime::fronted::kd_uring::UringChannel, token: &mut u32) {
    let lhs_rows = [quant_pattern_iq4(0), quant_pattern_iq4(1)];
    let rhs = rhs_pattern::<QUANT_K, Q_TEST_N>();
    let expected = matmul_ref(&lhs_rows, &rhs);
    let actual = run_quant_matmul(
        channel,
        token,
        AiDtype::IQ4_NL,
        &pack_iq4_nl_rows(&lhs_rows),
        QUANT_K,
        &rhs,
        "quant/iq4_nl_matmul",
    );
    print_quant_case("quant/iq4_nl_matmul", &actual);
    assert_f32_close("quant/iq4_nl_matmul", &actual, &expected, EPS);
}

fn test_q3_k_matmul(channel: &k3_ai_runtime::fronted::kd_uring::UringChannel, token: &mut u32) {
    let lhs_rows = [quant_pattern_q3(0), quant_pattern_q3(1)];
    let rhs = rhs_pattern::<Q3_K, 2>();
    let expected = matmul_ref(&lhs_rows, &rhs);
    let actual = run_quant_matmul(
        channel,
        token,
        AiDtype::Q3_K,
        &pack_q3_k_rows(&lhs_rows),
        Q3_K,
        &rhs,
        "quant/q3_k_matmul",
    );
    print_quant_case("quant/q3_k_matmul", &actual);
    assert_f32_close("quant/q3_k_matmul", &actual, &expected, EPS);
}

fn run_quant_matmul<const KLEN: usize, const NCOL: usize>(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    token: &mut u32,
    dtype: AiDtype,
    lhs_bytes: &[u8],
    k: usize,
    rhs_values: &[[f32; NCOL]; KLEN],
    label: &str,
) -> Vec<f32> {
    let tensor_mgr = TensorManager::new();
    let mut lhs = tensor_mgr
        .alloc_ggml_quant_tensor(dtype, &[k as u32, Q_TEST_M as u32], 0)
        .expect("alloc quant lhs failed");
    let mut rhs = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[k as u32, NCOL as u32])
        .expect("alloc rhs failed");
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[Q_TEST_M as u32, NCOL as u32])
        .expect("alloc out failed");

    lhs.as_mut_slice().copy_from_slice(lhs_bytes);
    copy_f32_to_tensor(&mut rhs, &flatten_rhs(rhs_values));

    run_kernel(
        channel,
        KernelOp::MAT_MUL,
        &MatMulAttr {
            m: DimSize::new(Q_TEST_M as u32),
            n: DimSize::new(NCOL as u32),
            k: DimSize::new(k as u32),
            batch: DimSize::new(0),
            lhs_row_stride: ElemStride::new(0),
            lhs_col_stride: ElemStride::new(0),
            lhs_batch_stride: ElemStride::new(0),
            rhs_row_stride: ElemStride::new(NCOL as u32),
            rhs_col_stride: ElemStride::new(1),
            rhs_batch_stride: ElemStride::new(0),
            out_row_stride: ElemStride::new(NCOL as u32),
            out_col_stride: ElemStride::new(1),
            out_batch_stride: ElemStride::new(0),
            flags: OpFlags::new(0),
            accum_dtype: AiDtype::F32,
            reserved: [0; 3],
        },
        &[&lhs, &rhs],
        &[&out],
        token,
        label,
    );

    tensor_f32_vec(&out)
}

fn run_kernel<T: Copy>(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    op: KernelOp,
    attr: &T,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
    token: &mut u32,
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
        AiTargetHint::AUTO,
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
    let entry = blob.submit_entry(UserToken::new(*token));
    *token = token.checked_add(1).expect("quant test token overflow");

    submit_graph(channel, &entry).unwrap_or_else(|err| panic!("failed to submit {label}: {err}"));
    wait_graph_complete(&entry, channel)
        .unwrap_or_else(|err| panic!("{label} graph execute failed: {err}"));
}

fn quant_pattern_q8(row: usize) -> [f32; QUANT_K] {
    let mut out = [0.0_f32; QUANT_K];
    for (idx, value) in out.iter_mut().enumerate() {
        *value = match row {
            0 => idx as f32 - 16.0,
            1 => 16.0 - idx as f32,
            _ => (idx as i32 % 7 - 3) as f32,
        };
    }
    out
}

fn quant_pattern_iq4(row: usize) -> [f32; QUANT_K] {
    let mut out = [0.0_f32; QUANT_K];
    for (idx, value) in out.iter_mut().enumerate() {
        let table_idx = match row {
            0 => idx % IQ4_NL_VALUES.len(),
            _ => (IQ4_NL_VALUES.len() - 1) - (idx % IQ4_NL_VALUES.len()),
        };
        *value = IQ4_NL_VALUES[table_idx] as f32;
    }
    out
}

fn quant_pattern_q3(row: usize) -> [f32; Q3_K] {
    let mut out = [0.0_f32; Q3_K];
    for (idx, value) in out.iter_mut().enumerate() {
        let centered = match row {
            0 => (idx % 8) as i8 - 4,
            _ => 3 - (idx % 8) as i8,
        };
        *value = -32.0 * centered as f32;
    }
    out
}

fn rhs_pattern<const KLEN: usize, const NCOL: usize>() -> [[f32; NCOL]; KLEN] {
    let mut out = [[0.0_f32; NCOL]; KLEN];
    for (k, row) in out.iter_mut().enumerate() {
        for (n, value) in row.iter_mut().enumerate() {
            *value = ((k + 2 * n) % 11) as f32 - 5.0;
        }
    }
    out
}

fn matmul_ref<const KLEN: usize, const NCOL: usize>(
    lhs: &[[f32; KLEN]; Q_TEST_M],
    rhs: &[[f32; NCOL]; KLEN],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; Q_TEST_M * NCOL];
    for m in 0..Q_TEST_M {
        for n in 0..NCOL {
            let mut sum = 0.0_f32;
            for k in 0..KLEN {
                sum += lhs[m][k] * rhs[k][n];
            }
            out[m * NCOL + n] = sum;
        }
    }
    out
}

fn get_rows_expected_logical(rows: &[[f32; QUANT_K]; 3], selected_rows: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(QUANT_K * selected_rows.len());
    for col in 0..QUANT_K {
        for &row in selected_rows {
            out.push(rows[row][col]);
        }
    }
    out
}

fn flatten_rhs<const KLEN: usize, const NCOL: usize>(rhs: &[[f32; NCOL]; KLEN]) -> Vec<f32> {
    rhs.iter().flat_map(|row| row.iter().copied()).collect()
}

fn pack_q8_0_rows(rows: &[[f32; QUANT_K]; Q_TEST_M]) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows.len() * 34];
    for (row_idx, row) in rows.iter().enumerate() {
        let block = &mut bytes[row_idx * 34..row_idx * 34 + 34];
        block[0..2].copy_from_slice(&F16_ONE_LE);
        for (dst, value) in block[2..34].iter_mut().zip(row.iter().copied()) {
            *dst = (value as i8) as u8;
        }
    }
    bytes
}

fn write_q8_0_rows(tensor: &mut Tensor, rows: &[[f32; QUANT_K]; 3]) {
    let bytes = tensor.as_mut_slice();
    for (row_idx, row) in rows.iter().enumerate() {
        let block = &mut bytes[row_idx * 34..row_idx * 34 + 34];
        block[0..2].copy_from_slice(&F16_ONE_LE);
        for (dst, value) in block[2..34].iter_mut().zip(row.iter().copied()) {
            *dst = (value as i8) as u8;
        }
    }
}

fn pack_iq4_nl_rows(rows: &[[f32; QUANT_K]; Q_TEST_M]) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows.len() * 18];
    for (row_idx, row) in rows.iter().enumerate() {
        let block = &mut bytes[row_idx * 18..row_idx * 18 + 18];
        block[0..2].copy_from_slice(&F16_ONE_LE);
        for lane in 0..16 {
            let lo = iq4_index(row[lane] as i8);
            let hi = iq4_index(row[lane + 16] as i8);
            block[2 + lane] = lo | (hi << 4);
        }
    }
    bytes
}

fn pack_q3_k_rows(rows: &[[f32; Q3_K]; Q_TEST_M]) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows.len() * 110];
    for (row_idx, row) in rows.iter().enumerate() {
        let block = &mut bytes[row_idx * 110..row_idx * 110 + 110];
        block[108..110].copy_from_slice(&F16_ONE_LE);
        for (idx, value) in row.iter().enumerate() {
            let centered = (-*value / 32.0) as i8;
            let (low, high) = if centered >= 0 {
                (centered as u8, true)
            } else {
                ((centered + 4) as u8, false)
            };
            let half = idx / 128;
            let within = idx % 128;
            let group = within / 32;
            let lane = within % 32;
            let second = lane >= 16;
            let q_index = half * 32 + if second { 16 } else { 0 } + lane % 16;
            let shift = group * 2;
            block[32 + q_index] |= (low & 3) << shift;
            if high {
                block[q_index % 32] |= 1_u8 << (group + half * 4);
            }
        }
    }
    bytes
}

fn iq4_index(value: i8) -> u8 {
    IQ4_NL_VALUES
        .iter()
        .position(|&candidate| candidate == value)
        .expect("value must be in IQ4_NL table") as u8
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

fn tensor_f32_vec(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), AiDtype::F32);
    tensor.as_f32_slice().to_vec()
}

fn print_quant_case(label: &str, values: &[f32]) {
    print!("{{\"source\":\"K3\",\"case\":\"{}\",\"values\":[", label);
    for (idx, value) in values.iter().enumerate() {
        if idx != 0 {
            print!(",");
        }
        print!("{:.6}", value);
    }
    println!("]}}");
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

fn tensor_i32_vec(tensor: &Tensor) -> Vec<i32> {
    assert_eq!(tensor.dtype(), AiDtype::I32);
    tensor
        .as_slice()
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
