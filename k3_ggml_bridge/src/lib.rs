//! C ABI bridge used by the ggml K3 backend.
//!
//! The bridge intentionally exports a small stable C surface and keeps the
//! StarryOS/K3 runtime details on the Rust side. The ggml backend can then be a
//! normal ggml backend instead of patching CPU kernels.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    slice,
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use k3_ai_runtime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, DimSize, ElemStride, GraphManager, MatMulAttr, OpFlags,
    Tensor, TensorCount, TensorManager, UserToken,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};

/// ggml-side success return code.
const K3_GGML_OK: i32 = 1;
/// ggml-side failure return code.
const K3_GGML_FAILED: i32 = 0;

/// MatMul flag indicating that the RHS logical matrix is stored transposed.
const MATMUL_RHS_TRANSPOSED: u32 = 1 << 1;

/// Lazily-created runtime submission channel.
static CHANNEL: Mutex<Option<UringChannel>> = Mutex::new(None);
/// Monotonic token source for bridge-submitted single-kernel graphs.
static NEXT_TOKEN: AtomicU32 = AtomicU32::new(0x4B33_0000);

/// Whole-graph compute request consumed by the native ggml K3 backend.
#[repr(C)]
pub struct K3GgmlGraphCompute {
    /// Pointer to an array of K3 kernel descriptors.
    pub nodes: *const AiKernelDesc,
    /// Number of descriptors in `nodes`.
    pub node_count: u32,
    /// Non-zero means the caller requires strict no-fallback behavior.
    pub strict: u32,
}

/// # Safety
///
/// `req` must point to a valid `K3GgmlGraphCompute`. When `node_count` is
/// non-zero, `nodes` must point to at least that many valid `AiKernelDesc`
/// entries whose referenced tensor buffers remain alive until the call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k3_ggml_graph_compute(req: *const K3GgmlGraphCompute) -> i32 {
    if req.is_null() {
        eprintln!("k3_ggml_graph_compute: null request");
        return K3_GGML_FAILED;
    }
    // SAFETY: null was checked above; the caller owns request validity.
    let req = unsafe { &*req };
    match run_graph(req) {
        Ok(()) => K3_GGML_OK,
        Err(err) => {
            eprintln!("k3_ggml_graph_compute failed: {err}");
            K3_GGML_FAILED
        }
    }
}

/// Flat F32 matmul request consumed by the C ABI entry point.
#[repr(C)]
pub struct K3GgmlMatmulF32 {
    /// Left-hand matrix base pointer.
    pub lhs: *const f32,
    /// Right-hand matrix base pointer.
    pub rhs: *const f32,
    /// Output matrix base pointer.
    pub out: *mut f32,
    /// Logical output row count.
    pub m: u32,
    /// Logical output column count.
    pub n: u32,
    /// Reduction dimension.
    pub k: u32,
    /// LHS stride, in F32 elements, between adjacent rows.
    pub lhs_row_stride: u32,
    /// LHS stride, in F32 elements, between adjacent columns.
    pub lhs_col_stride: u32,
    /// RHS stride, in F32 elements, between adjacent rows.
    pub rhs_row_stride: u32,
    /// RHS stride, in F32 elements, between adjacent columns.
    pub rhs_col_stride: u32,
    /// Output stride, in F32 elements, between adjacent rows.
    pub out_row_stride: u32,
    /// Output stride, in F32 elements, between adjacent columns.
    pub out_col_stride: u32,
    /// MatMul flags shared with the backend attr.
    pub flags: u32,
    /// K3 target hint byte.
    pub target_hint: u8,
}

/// # Safety
///
/// `req` must point to a valid `K3GgmlMatmulF32`. The buffers referenced by
/// the request must stay valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k3_ggml_matmul_f32(req: *const K3GgmlMatmulF32) -> i32 {
    if req.is_null() {
        return K3_GGML_FAILED;
    }

    // SAFETY: null was checked above; the C caller is responsible for passing a
    // valid request pointer for the duration of this call.
    let req = unsafe { &*req };
    match run_matmul_f32(req) {
        Ok(()) => K3_GGML_OK,
        Err(_) => K3_GGML_FAILED,
    }
}

/// Validate, serialize, submit, and wait for a whole K3 graph.
fn run_graph(req: &K3GgmlGraphCompute) -> Result<(), &'static str> {
    if req.node_count == 0 {
        return Ok(());
    }
    if req.nodes.is_null() {
        return Err("nodes is null with non-zero node_count");
    }

    let node_count = req.node_count as usize;
    // SAFETY: request validation above and C ABI contract guarantee this span.
    let nodes = unsafe { slice::from_raw_parts(req.nodes, node_count) };
    let mut graph = GraphManager::new();
    let mut tail = None;
    for (index, node) in nodes.iter().copied().enumerate() {
        tail = Some(if let Some(depend) = tail {
            graph
                .push_kernel_depend(depend, node)
                .map_err(|_| "failed to append dependent graph node")?
        } else {
            graph
                .push_kernel_no_depend(node)
                .map_err(|_| "failed to append graph node")?
        });
        if node.op.0 == 0 {
            eprintln!("k3_ggml_graph_compute: node {index} has invalid op 0");
            return Err("invalid op");
        }
    }

    submit_frozen_graph(graph).map_err(|_| "graph submission failed")
}

/// Validate, stage, submit, and copy back one F32 matmul request.
fn run_matmul_f32(req: &K3GgmlMatmulF32) -> Result<(), ()> {
    validate_request(req)?;

    let lhs_len = strided_len(req.m, req.k, req.lhs_row_stride, req.lhs_col_stride)?;
    let rhs_rows = if req.flags & MATMUL_RHS_TRANSPOSED != 0 {
        req.n
    } else {
        req.k
    };
    let rhs_cols = if req.flags & MATMUL_RHS_TRANSPOSED != 0 {
        req.k
    } else {
        req.n
    };
    let rhs_len = strided_len(rhs_rows, rhs_cols, req.rhs_row_stride, req.rhs_col_stride)?;
    let out_len = strided_len(req.m, req.n, req.out_row_stride, req.out_col_stride)?;
    let lhs_shape_len = usize_to_u32(lhs_len)?;
    let rhs_shape_len = usize_to_u32(rhs_len)?;
    let out_shape_len = usize_to_u32(out_len)?;

    // SAFETY: `validate_request` checked non-null buffers and dimensions. The
    // exported function's safety contract requires the raw buffers to be valid
    // for the computed strided element spans.
    let lhs_src = unsafe { slice::from_raw_parts(req.lhs, lhs_len) };
    // SAFETY: same request contract as for `lhs_src`; `rhs_len` is the minimum
    // span covering every RHS element referenced by the submitted matmul.
    let rhs_src = unsafe { slice::from_raw_parts(req.rhs, rhs_len) };
    // SAFETY: same request contract as for inputs, plus unique mutable access to
    // the output span while the bridge copies the computed result back.
    let out_dst = unsafe { slice::from_raw_parts_mut(req.out, out_len) };

    let tensor_mgr = TensorManager::new();
    let mut lhs = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[lhs_shape_len])
        .map_err(|_| ())?;
    let mut rhs = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[rhs_shape_len])
        .map_err(|_| ())?;
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[out_shape_len])
        .map_err(|_| ())?;

    lhs.as_f32_mut_slice().copy_from_slice(lhs_src);
    rhs.as_f32_mut_slice().copy_from_slice(rhs_src);

    submit_single_matmul(req, &lhs, &rhs, &out)?;

    copy_output(req, out.as_f32_slice(), out_dst);
    Ok(())
}

/// Check ABI-level request invariants before raw pointer conversion.
fn validate_request(req: &K3GgmlMatmulF32) -> Result<(), ()> {
    if req.lhs.is_null()
        || req.rhs.is_null()
        || req.out.is_null()
        || req.m == 0
        || req.n == 0
        || req.k == 0
    {
        return Err(());
    }

    if (req.m > 1 && req.lhs_row_stride == 0)
        || (req.k > 1 && req.lhs_col_stride == 0)
        || (req.n > 1 && req.rhs_row_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED != 0)
        || (req.k > 1 && req.rhs_col_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED != 0)
        || (req.k > 1 && req.rhs_row_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED == 0)
        || (req.n > 1 && req.rhs_col_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED == 0)
        || (req.m > 1 && req.out_row_stride == 0)
        || (req.n > 1 && req.out_col_stride == 0)
    {
        return Err(());
    }

    if req.flags & !MATMUL_RHS_TRANSPOSED != 0 {
        return Err(());
    }

    let target = AiTargetHint(req.target_hint);
    if !target.is_known() {
        return Err(());
    }

    Ok(())
}

/// Submit a single-node K3 graph containing one matmul kernel.
fn submit_single_matmul(
    req: &K3GgmlMatmulF32,
    lhs: &Tensor,
    rhs: &Tensor,
    out: &Tensor,
) -> Result<(), ()> {
    let mut graph = GraphManager::new();
    let attr = MatMulAttr {
        m: DimSize::new(req.m),
        n: DimSize::new(req.n),
        k: DimSize::new(req.k),
        batch: DimSize::new(0),
        lhs_row_stride: ElemStride::new(req.lhs_row_stride),
        lhs_col_stride: ElemStride::new(req.lhs_col_stride),
        lhs_batch_stride: ElemStride::new(0),
        rhs_row_stride: ElemStride::new(req.rhs_row_stride),
        rhs_col_stride: ElemStride::new(req.rhs_col_stride),
        rhs_batch_stride: ElemStride::new(0),
        out_row_stride: ElemStride::new(req.out_row_stride),
        out_col_stride: ElemStride::new(req.out_col_stride),
        out_batch_stride: ElemStride::new(0),
        flags: OpFlags::new(req.flags),
        accum_dtype: AiDtype::F32,
        reserved: [0; 3],
    };

    graph
        .push_kernel_no_depend(AiKernelDesc::new(
            &attr,
            AiTargetHint(req.target_hint),
            TensorCount::new(2),
            TensorCount::new(1),
            &[lhs.desc(), rhs.desc(), out.desc()],
        ))
        .map_err(|_| ())?;

    submit_frozen_graph(graph)
}

/// Freeze and submit a graph through the lazily-created runtime channel.
fn submit_frozen_graph(graph: GraphManager) -> Result<(), ()> {
    let blob = graph.freeze().map_err(|_| ())?;
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let entry = blob.submit_entry(UserToken::new(token));
    let mut channel = CHANNEL.lock().map_err(|_| ())?;
    if channel.is_none() {
        *channel = Some(build_channel().map_err(|_| ())?);
    }
    let channel = channel.as_ref().ok_or(())?;
    submit_graph(channel, &entry).map_err(|_| ())?;
    wait_graph_complete(&entry, channel).map_err(|e| {
        eprintln!("k3_ggml_graph_compute: wait_graph_complete failed: {e}");
    })?;
    Ok(())
}

/// Copy the strided K3 matmul output span back to ggml memory.
fn copy_output(req: &K3GgmlMatmulF32, src: &[f32], dst: &mut [f32]) {
    let m = req.m as usize;
    let n = req.n as usize;
    let row_stride = req.out_row_stride as usize;
    let col_stride = req.out_col_stride as usize;

    for row in 0..m {
        for col in 0..n {
            let idx = row * row_stride + col * col_stride;
            dst[idx] = src[idx];
        }
    }
}

/// Return the contiguous span length needed to cover a strided 2D matrix.
fn strided_len(rows: u32, cols: u32, row_stride: u32, col_stride: u32) -> Result<usize, ()> {
    let rows = rows as usize;
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    (rows - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add((cols - 1).checked_mul(col_stride)?))
        .and_then(|v| v.checked_add(1))
        .ok_or(())
}

/// Convert a Rust `usize` to a K3 ABI `u32` shape extent.
fn usize_to_u32(value: usize) -> Result<u32, ()> {
    u32::try_from(value).map_err(|_| ())
}
