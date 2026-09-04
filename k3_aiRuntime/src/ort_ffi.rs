//! C ABI shim for the ONNX Runtime K3 execution provider.
//!
//! The shim keeps K3 runtime graph construction on the Rust side. ORT passes a
//! single lowered node with raw tensor buffers and inline attr bytes; this module
//! builds zero-copy `TensorManager` views over those buffers, creates a one-node
//! graph, then synchronously submits it to `/dev/k3_airunner`.

use std::sync::Mutex;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{ffi::c_char, fmt};

use crate::fronted::{
    ATTR_INLINE_SIZE, AiKernelDesc, AiTargetHint, AiTensorFormat, AiTensorLayout, AttrByteSize,
    KernelOp, MAX_DIM, MAX_SUBMIT_TENSORS, Tensor, TensorCount, TensorManager,
};
use crate::fronted::{
    GraphManager, UserToken,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};

/// Version of the private ORT-to-K3 node ABI.
pub const K3_ORT_ABI_VERSION: u32 = 1;

/// Stable status codes returned through the C ABI.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum K3OrtStatus {
    /// The node completed successfully.
    Ok = 0,
    /// The request pointer was null.
    NullRequest = -1,
    /// Input/output tensor counts were invalid.
    InvalidCount = -2,
    /// The operator or target hint is unsupported.
    Unsupported = -3,
    /// Inline attributes were malformed.
    InvalidAttribute = -4,
    /// A tensor descriptor was malformed or unsupported.
    InvalidTensor = -5,
    /// Runtime-owned tensor staging allocation failed.
    AllocationFailed = -6,
    /// The selected K3 backend failed to execute the node.
    ExecutionFailed = -7,
    /// Copying tensor data into or out of staging storage failed.
    CopyFailed = -8,
}

/// Monotonic token source for single-node graph submissions.
static NEXT_TOKEN: AtomicU32 = AtomicU32::new(0x4B33_1000);

/// Number of successfully completed ORT-submitted nodes in this process.
static EXECUTED_NODE_COUNT: AtomicU64 = AtomicU64::new(0);

lazy_static::lazy_static! {
    /// Lazily-opened runtime channel for device submissions.
    static ref CHANNEL: Mutex<Option<UringChannel>> = Mutex::new(None);
}

/// Enable detailed node-submission diagnostics with `K3_ORT_DEBUG=1`.
///
/// This is deliberately evaluated for each request so a statically linked ORT
/// application can turn tracing on from its launch environment without a rebuild.
fn debug_enabled() -> bool {
    matches!(
        std::env::var("K3_ORT_DEBUG").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Print one diagnostic event when K3 ORT tracing is enabled.
fn trace(arguments: fmt::Arguments<'_>) {
    if debug_enabled() {
        eprintln!("[k3-ort] {arguments}");
    }
}

/// Flat tensor view passed by the ORT provider.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K3OrtTensor {
    /// Caller-owned tensor buffer pointer.
    pub data: *mut u8,
    /// Caller-owned tensor buffer byte length.
    pub size_bytes: u64,
    /// K3 `AiDtype` raw value.
    pub dtype: u32,
    /// K3 `AiTensorFormat` raw value.
    pub format: u32,
    /// K3 `AiTensorLayout` raw value.
    pub layout: u32,
    /// Number of valid dimensions in `shape`.
    pub ndim: u32,
    /// K3 tensor flags raw value.
    pub flags: u8,
    /// Padding reserved for ABI stability.
    pub reserved0: [u8; 3],
    /// Tensor shape, valid for `ndim` entries.
    pub shape: [u32; MAX_DIM],
    /// Dense/strided byte strides supplied by ORT.
    pub stride_bytes: [u64; MAX_DIM],
}

/// Single lowered K3 node request passed by the ORT provider.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K3OrtRunNode {
    /// K3 `KernelOp` raw value.
    pub op: u8,
    /// K3 `AiTargetHint` raw value.
    pub target_hint: u8,
    /// Reserved alignment bytes.
    pub reserved0: [u8; 2],
    /// Number of input tensors at the beginning of `tensors`.
    pub input_count: u32,
    /// Number of output tensors after the inputs in `tensors`.
    pub output_count: u32,
    /// Input tensors followed by output tensors.
    pub tensors: [K3OrtTensor; MAX_SUBMIT_TENSORS],
    /// Number of valid bytes in `attr_inline`.
    pub attr_size: u32,
    /// K3 inline operator attribute bytes.
    pub attr_inline: [u8; ATTR_INLINE_SIZE],
}

/// Return the private ORT-to-K3 ABI version compiled into the runtime.
#[unsafe(no_mangle)]
pub extern "C" fn k3_ort_abi_version() -> u32 {
    K3_ORT_ABI_VERSION
}

/// Return the number of K3 nodes completed by this process.
#[unsafe(no_mangle)]
pub extern "C" fn k3_ort_executed_node_count() -> u64 {
    EXECUTED_NODE_COUNT.load(Ordering::Relaxed)
}

/// Return a static, null-terminated description for a K3 ORT status code.
#[unsafe(no_mangle)]
pub extern "C" fn k3_ort_status_message(status: i32) -> *const c_char {
    let message: &'static [u8] = match status {
        0 => b"success\0",
        -1 => b"null request\0",
        -2 => b"invalid tensor count\0",
        -3 => b"unsupported operator or target\0",
        -4 => b"invalid operator attributes\0",
        -5 => b"invalid or unsupported tensor\0",
        -6 => b"tensor staging allocation failed\0",
        -7 => b"K3 node execution failed\0",
        -8 => b"tensor staging copy failed\0",
        _ => b"unknown K3 ORT status\0",
    };
    message.as_ptr().cast()
}

/// # Safety
///
/// `req` must be non-null and must point to a valid `K3OrtRunNode`. Every input
/// and output buffer referenced by `req.tensors` must remain valid for the
/// duration of this call, and output buffers must be uniquely writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k3_ort_run_node(req: *const K3OrtRunNode) -> i32 {
    if req.is_null() {
        return K3OrtStatus::NullRequest as i32;
    }

    // SAFETY: The null pointer case is handled above. The C caller owns the
    // lifetime contract documented on this function.
    let req = unsafe { &*req };
    match run_node(req) {
        Ok(()) => {
            trace(format_args!("stage=complete op={}", req.op));
            K3OrtStatus::Ok as i32
        }
        Err(status) => {
            trace(format_args!("stage=failed op={} status={status:?}", req.op));
            status as i32
        }
    }
}

/// Validate, borrow, execute, and complete a single ORT-submitted node.
fn run_node(req: &K3OrtRunNode) -> Result<(), K3OrtStatus> {
    trace_request(req);
    let total = validate_request(req)?;
    trace(format_args!("stage=validated op={}", req.op));
    let tensor_manager = TensorManager::new();
    let tensors = borrow_tensors(req, &tensor_manager, total)?;
    trace(format_args!(
        "stage=tensor-views op={} tensors={}",
        req.op,
        tensors.len()
    ));

    let desc = build_desc(req, &tensors);
    trace(format_args!("stage=descriptor op={}", desc.op.0));
    execute_on_device(&desc)?;
    EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);

    Ok(())
}

/// Emit the exact request shape accepted from ORT, without dereferencing any
/// caller-owned tensor buffer.
fn trace_request(req: &K3OrtRunNode) {
    if !debug_enabled() {
        return;
    }

    let total = match total_tensor_count(req) {
        Ok(total) => total,
        Err(status) => {
            trace(format_args!(
                "stage=request-invalid op={} inputs={} outputs={} status={status:?}",
                req.op, req.input_count, req.output_count
            ));
            return;
        }
    };

    trace(format_args!(
        "stage=request op={} inputs={} outputs={} attr_bytes={}",
        req.op, req.input_count, req.output_count, req.attr_size
    ));
    for (index, tensor) in req.tensors[..total].iter().enumerate() {
        let rank = (tensor.ndim as usize).min(MAX_DIM);
        trace(format_args!(
            "tensor[{index}] bytes={} dtype={} format={} layout={} shape={:?}",
            tensor.size_bytes,
            tensor.dtype,
            tensor.format,
            tensor.layout,
            &tensor.shape[..rank]
        ));
    }
}

/// Check ABI-level request invariants before touching raw tensor pointers.
fn validate_request(req: &K3OrtRunNode) -> Result<usize, K3OrtStatus> {
    let target = AiTargetHint(req.target_hint);
    if !KernelOp(req.op).is_known() || !target.is_known() {
        return Err(K3OrtStatus::Unsupported);
    }
    if req.attr_size as usize > ATTR_INLINE_SIZE {
        return Err(K3OrtStatus::InvalidAttribute);
    }

    let total = total_tensor_count(req)?;
    for raw in &req.tensors[..total] {
        validate_tensor(raw)?;
    }

    Ok(total)
}

/// Validate the v1 dense FP32 tensor contract and calculate its byte size safely.
fn validate_tensor(raw: &K3OrtTensor) -> Result<(), K3OrtStatus> {
    let rank = raw.ndim as usize;
    if rank > MAX_DIM
        || raw.data.is_null()
        || raw.dtype != crate::fronted::AiDtype::F32.0
        || !matches!(raw.format, 0 | 1)
        || raw.layout != AiTensorLayout::DENSE.0
    {
        return Err(K3OrtStatus::InvalidTensor);
    }

    let mut expected_size = core::mem::size_of::<f32>() as u64;
    for &dim in &raw.shape[..rank] {
        if dim == 0 {
            return Err(K3OrtStatus::InvalidTensor);
        }
        expected_size = expected_size
            .checked_mul(dim as u64)
            .ok_or(K3OrtStatus::InvalidTensor)?;
    }
    if raw.size_bytes != expected_size {
        return Err(K3OrtStatus::InvalidTensor);
    }

    let mut expected_stride = core::mem::size_of::<f32>() as u64;
    for dim_idx in (0..rank).rev() {
        if raw.stride_bytes[dim_idx] != expected_stride {
            return Err(K3OrtStatus::InvalidTensor);
        }
        expected_stride = expected_stride
            .checked_mul(raw.shape[dim_idx] as u64)
            .ok_or(K3OrtStatus::InvalidTensor)?;
    }
    Ok(())
}

/// Return the combined input/output tensor count.
fn total_tensor_count(req: &K3OrtRunNode) -> Result<usize, K3OrtStatus> {
    let total = req
        .input_count
        .checked_add(req.output_count)
        .ok_or(K3OrtStatus::InvalidCount)? as usize;
    if req.input_count == 0 || req.output_count == 0 || total > MAX_SUBMIT_TENSORS {
        Err(K3OrtStatus::InvalidCount)
    } else {
        Ok(total)
    }
}

/// Build zero-copy views over every caller-owned ORT tensor buffer.
///
/// The exported C function guarantees every buffer survives this synchronous
/// call. The device backend waits for graph completion before these views drop,
/// so the kernel never observes a released ORT allocation.
fn borrow_tensors(
    req: &K3OrtRunNode,
    manager: &TensorManager,
    total: usize,
) -> Result<Vec<Tensor>, K3OrtStatus> {
    let mut tensors = Vec::with_capacity(total);

    for raw in &req.tensors[..total] {
        let size_bytes = usize::try_from(raw.size_bytes).map_err(|_| K3OrtStatus::InvalidTensor)?;
        // SAFETY: `validate_request` checked the tensor metadata, and the C ABI
        // contract guarantees that ORT owns the buffer until this call returns.
        let tensor = unsafe {
            manager.borrow_tensor_with_layout(
                raw.data,
                size_bytes,
                crate::fronted::AiDtype(raw.dtype),
                AiTensorFormat(raw.format),
                AiTensorLayout(raw.layout),
                &raw.shape[..raw.ndim as usize],
                raw.flags,
            )
        }
        .map_err(|_| K3OrtStatus::InvalidTensor)?;
        tensors.push(tensor);
    }

    Ok(tensors)
}

/// Build a K3 kernel desc from staged runtime tensors and raw attr bytes.
fn build_desc(req: &K3OrtRunNode, staged: &[Tensor]) -> AiKernelDesc {
    let mut desc = AiKernelDesc {
        op: KernelOp(req.op),
        target_hint: AiTargetHint(req.target_hint),
        input_count: TensorCount::new(req.input_count),
        output_count: TensorCount::new(req.output_count),
        attr_size: AttrByteSize::new(req.attr_size),
        ..AiKernelDesc::default()
    };

    for (index, tensor) in staged.iter().enumerate() {
        desc.tensors[index] = tensor.desc();
    }

    desc.attr_inline[..req.attr_size as usize]
        .copy_from_slice(&req.attr_inline[..req.attr_size as usize]);
    desc
}

/// Build a one-node graph and submit it to the shared K3 device channel.
fn execute_on_device(desc: &AiKernelDesc) -> Result<(), K3OrtStatus> {
    let operation = desc.op.0;
    let mut graph = GraphManager::new();
    device_step(
        "graph-build",
        operation,
        None,
        graph.push_kernel_no_depend(*desc),
    )?;
    let blob = device_step("graph-freeze", operation, None, graph.freeze())?;

    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let entry = blob.submit_entry(UserToken::new(token));
    trace(format_args!(
        "stage=graph-ready op={operation} token={token}"
    ));

    let mut channel = device_step("channel-lock", operation, None, CHANNEL.lock())?;
    if channel.is_none() {
        trace(format_args!("stage=channel-build-start op={operation}"));
        *channel = Some(device_step(
            "channel-build",
            operation,
            None,
            build_channel(),
        )?);
        trace(format_args!("stage=channel-build-complete op={operation}"));
    }
    let channel = channel.as_ref().ok_or_else(|| {
        trace(format_args!("stage=channel-unavailable op={operation}"));
        K3OrtStatus::ExecutionFailed
    })?;

    trace(format_args!(
        "stage=submit-start op={operation} token={token}"
    ));
    device_step(
        "submit",
        operation,
        Some(token),
        submit_graph(channel, &entry),
    )?;
    trace(format_args!(
        "stage=submit-complete op={operation} token={token}"
    ));
    device_step(
        "completion",
        operation,
        Some(token),
        wait_graph_complete(&entry, channel),
    )?;
    trace(format_args!(
        "stage=completion-success op={operation} token={token}"
    ));
    Ok(())
}

/// Convert a failed device operation into the stable C ABI execution status.
fn device_step<T, E: fmt::Debug>(
    stage: &str,
    operation: u8,
    token: Option<u32>,
    result: Result<T, E>,
) -> Result<T, K3OrtStatus> {
    result.map_err(|error| {
        if let Some(token) = token {
            trace(format_args!(
                "stage={stage} op={operation} token={token} error={error:?}"
            ));
        } else {
            trace(format_args!("stage={stage} op={operation} error={error:?}"));
        }
        K3OrtStatus::ExecutionFailed
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fronted::{AiDtype, AiTensorFormat, AiTensorLayout};

    /// Construct one dense F32 tensor descriptor around caller-owned storage.
    fn dense_tensor(storage: &mut [f32], shape: &[u32]) -> K3OrtTensor {
        let mut tensor = K3OrtTensor {
            data: storage.as_mut_ptr().cast(),
            size_bytes: core::mem::size_of_val(storage) as u64,
            dtype: AiDtype::F32.0,
            format: AiTensorFormat::ANY.0,
            layout: AiTensorLayout::DENSE.0,
            ndim: shape.len() as u32,
            flags: 0,
            reserved0: [0; 3],
            shape: [0; MAX_DIM],
            stride_bytes: [0; MAX_DIM],
        };
        tensor.shape[..shape.len()].copy_from_slice(shape);
        let mut stride = core::mem::size_of::<f32>() as u64;
        for idx in (0..shape.len()).rev() {
            tensor.stride_bytes[idx] = stride;
            stride *= shape[idx] as u64;
        }
        tensor
    }

    /// Construct an otherwise empty request for validation tests.
    fn request() -> K3OrtRunNode {
        K3OrtRunNode {
            op: KernelOp::SIGMOID.0,
            target_hint: AiTargetHint::AUTO.0,
            reserved0: [0; 2],
            input_count: 1,
            output_count: 1,
            tensors: [K3OrtTensor {
                data: core::ptr::null_mut(),
                size_bytes: 0,
                dtype: 0,
                format: 0,
                layout: 0,
                ndim: 0,
                flags: 0,
                reserved0: [0; 3],
                shape: [0; MAX_DIM],
                stride_bytes: [0; MAX_DIM],
            }; MAX_SUBMIT_TENSORS],
            attr_size: 0,
            attr_inline: [0; ATTR_INLINE_SIZE],
        }
    }

    #[test]
    fn abi_layout_matches_c_header_contract() {
        assert_eq!(core::mem::size_of::<K3OrtTensor>(), 136);
        assert_eq!(core::mem::align_of::<K3OrtTensor>(), 8);
        assert_eq!(core::mem::offset_of!(K3OrtTensor, shape), 36);
        assert_eq!(core::mem::offset_of!(K3OrtTensor, stride_bytes), 72);
        assert_eq!(core::mem::size_of::<K3OrtRunNode>(), 1240);
        assert_eq!(core::mem::offset_of!(K3OrtRunNode, tensors), 16);
        assert_eq!(core::mem::offset_of!(K3OrtRunNode, attr_inline), 1108);
    }

    #[test]
    fn null_request_has_stable_error() {
        // SAFETY: A null request is explicitly accepted and rejected by the ABI.
        assert_eq!(unsafe { k3_ort_run_node(core::ptr::null()) }, -1);
    }

    #[test]
    fn rejects_invalid_counts_and_tensor_layout() {
        let mut req = request();
        req.input_count = 0;
        assert_eq!(run_node(&req), Err(K3OrtStatus::InvalidCount));

        let mut input = [1.0_f32; 2];
        let mut output = [0.0_f32; 2];
        req.input_count = 1;
        req.tensors[0] = dense_tensor(&mut input, &[2]);
        req.tensors[1] = dense_tensor(&mut output, &[2]);
        req.tensors[0].layout = AiTensorLayout::STRIDED.0;
        assert_eq!(run_node(&req), Err(K3OrtStatus::InvalidTensor));
    }
}
