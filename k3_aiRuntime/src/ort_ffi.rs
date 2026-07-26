//! C ABI shim for the ONNX Runtime K3 execution provider.
//!
//! The shim keeps K3 runtime graph construction on the Rust side. ORT passes a
//! single lowered node with raw tensor buffers and inline attr bytes; this module
//! stages those buffers into `TensorManager` tensors, builds a one-node graph,
//! and either submits it to `/dev/k3_airunner` or executes the CPU backend when
//! built with the `host-cpu` feature.

use std::slice;
#[cfg(not(feature = "host-cpu"))]
use std::sync::Mutex;
#[cfg(not(feature = "host-cpu"))]
use std::sync::atomic::{AtomicU32, Ordering};

use crate::fronted::{
    ATTR_INLINE_SIZE, AiKernelDesc, AiTargetHint, AiTensorFormat, AiTensorLayout, AttrByteSize,
    KernelOp, MAX_DIM, MAX_SUBMIT_TENSORS, Tensor, TensorCount, TensorManager,
};
#[cfg(not(feature = "host-cpu"))]
use crate::fronted::{
    GraphManager, UserToken,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};

/// Successful C ABI return value.
const K3_ORT_OK: i32 = 0;
/// Generic C ABI failure return value.
const K3_ORT_ERR: i32 = -1;

/// Monotonic token source for single-node graph submissions.
#[cfg(not(feature = "host-cpu"))]
static NEXT_TOKEN: AtomicU32 = AtomicU32::new(0x4B33_1000);

#[cfg(not(feature = "host-cpu"))]
lazy_static::lazy_static! {
    /// Lazily-opened runtime channel for device submissions.
    static ref CHANNEL: Mutex<Option<UringChannel>> = Mutex::new(None);
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

/// # Safety
///
/// `req` must be non-null and must point to a valid `K3OrtRunNode`. Every input
/// and output buffer referenced by `req.tensors` must remain valid for the
/// duration of this call, and output buffers must be uniquely writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k3_ort_run_node(req: *const K3OrtRunNode) -> i32 {
    if req.is_null() {
        return K3_ORT_ERR;
    }

    // SAFETY: The null pointer case is handled above. The C caller owns the
    // lifetime contract documented on this function.
    let req = unsafe { &*req };
    match run_node(req) {
        Ok(()) => K3_ORT_OK,
        Err(()) => K3_ORT_ERR,
    }
}

/// Validate, stage, execute, and copy back a single ORT-submitted node.
fn run_node(req: &K3OrtRunNode) -> Result<(), ()> {
    validate_request(req)?;

    let tensor_manager = TensorManager::new();
    let total = total_tensor_count(req)?;
    let mut staged = Vec::with_capacity(total);

    for idx in 0..total {
        let raw = &req.tensors[idx];
        let mut tensor = alloc_tensor(&tensor_manager, raw)?;
        if idx < req.input_count as usize {
            copy_input(raw, &mut tensor)?;
        }
        staged.push(tensor);
    }

    let desc = build_desc(req, &staged)?;
    execute_desc(&desc)?;

    for output_idx in 0..req.output_count as usize {
        let raw_idx = req.input_count as usize + output_idx;
        copy_output(&staged[raw_idx], &req.tensors[raw_idx])?;
    }

    Ok(())
}

/// Check ABI-level request invariants before touching raw tensor pointers.
fn validate_request(req: &K3OrtRunNode) -> Result<(), ()> {
    let target = AiTargetHint(req.target_hint);
    if !KernelOp(req.op).is_known()
        || !target.is_known()
        || req.attr_size as usize > ATTR_INLINE_SIZE
        || total_tensor_count(req)? > MAX_SUBMIT_TENSORS
    {
        return Err(());
    }

    for raw in &req.tensors[..total_tensor_count(req)?] {
        if raw.ndim as usize > MAX_DIM || raw.size_bytes == 0 || raw.data.is_null() {
            return Err(());
        }
    }

    Ok(())
}

/// Return the combined input/output tensor count.
fn total_tensor_count(req: &K3OrtRunNode) -> Result<usize, ()> {
    let total = req
        .input_count
        .checked_add(req.output_count)
        .ok_or(())? as usize;
    if total > MAX_SUBMIT_TENSORS {
        Err(())
    } else {
        Ok(total)
    }
}

/// Allocate a runtime-owned tensor matching an ORT tensor view.
fn alloc_tensor(manager: &TensorManager, raw: &K3OrtTensor) -> Result<Tensor, ()> {
    let rank = raw.ndim as usize;
    manager
        .alloc_tensor_with_layout(
            crate::fronted::AiDtype(raw.dtype),
            AiTensorFormat(raw.format),
            AiTensorLayout(raw.layout),
            &raw.shape[..rank],
            raw.flags,
        )
        .map_err(|_| ())
}

/// Copy one ORT input buffer into its runtime staging tensor.
fn copy_input(raw: &K3OrtTensor, tensor: &mut Tensor) -> Result<(), ()> {
    let dst = tensor.as_mut_slice();
    if raw.size_bytes < dst.len() as u64 {
        return Err(());
    }

    // SAFETY: `validate_request` checked the pointer is non-null. The exported
    // function requires the caller to provide at least `dst.len()` readable bytes.
    let src = unsafe { slice::from_raw_parts(raw.data.cast_const(), dst.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

/// Copy one runtime output tensor back to the ORT output buffer.
fn copy_output(tensor: &Tensor, raw: &K3OrtTensor) -> Result<(), ()> {
    let src = tensor.as_slice();
    if raw.size_bytes < src.len() as u64 {
        return Err(());
    }

    // SAFETY: `validate_request` checked the pointer is non-null. The exported
    // function requires the caller to provide at least `src.len()` writable bytes.
    let dst = unsafe { slice::from_raw_parts_mut(raw.data, src.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

/// Build a K3 kernel desc from staged runtime tensors and raw attr bytes.
fn build_desc(req: &K3OrtRunNode, staged: &[Tensor]) -> Result<AiKernelDesc, ()> {
    let total = total_tensor_count(req)?;
    let mut desc = AiKernelDesc {
        op: KernelOp(req.op),
        target_hint: AiTargetHint(req.target_hint),
        input_count: TensorCount::new(req.input_count),
        output_count: TensorCount::new(req.output_count),
        attr_size: AttrByteSize::new(req.attr_size),
        ..AiKernelDesc::default()
    };

    for idx in 0..total {
        let tensor_desc = {
            #[cfg(feature = "host-cpu")]
            {
                let mut tensor_desc = staged[idx].desc();
                tensor_desc.kernel_va = crate::fronted::KernelVa::new(tensor_desc.user_va.get());
                tensor_desc
            }
            #[cfg(not(feature = "host-cpu"))]
            {
                staged[idx].desc()
            }
        };
        desc.tensors[idx] = tensor_desc;
    }

    desc.attr_inline[..req.attr_size as usize]
        .copy_from_slice(&req.attr_inline[..req.attr_size as usize]);
    Ok(desc)
}

/// Execute a one-node kernel desc via host CPU or the runtime device channel.
fn execute_desc(desc: &AiKernelDesc) -> Result<(), ()> {
    #[cfg(feature = "host-cpu")]
    {
        use crate::fronted::AiGraphNode;

        let node = AiGraphNode {
            node_id: crate::fronted::AiGraphNodeId::new(0),
            desc: *desc,
            state: crate::fronted::AiGraphState::default(),
        };
        // SAFETY: `build_desc` filled tensor kernel VAs with runtime-owned
        // buffers whose lifetimes cover this call.
        let ret = unsafe { k3_kernel_backend::k3_run_kernel(&node) };
        if ret == 0 { Ok(()) } else { Err(()) }
    }

    #[cfg(not(feature = "host-cpu"))]
    {
        let mut graph = GraphManager::new();
        graph.push_kernel_no_depend(*desc).map_err(|_| ())?;
        let blob = graph.freeze().map_err(|_| ())?;
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
        let entry = blob.submit_entry(UserToken::new(token));

        let mut channel = CHANNEL.lock().map_err(|_| ())?;
        if channel.is_none() {
            *channel = Some(build_channel().map_err(|_| ())?);
        }
        let channel = channel.as_ref().ok_or(())?;
        submit_graph(channel, &entry).map_err(|_| ())?;
        wait_graph_complete(&entry, channel).map_err(|_| ())?;
        Ok(())
    }
}
