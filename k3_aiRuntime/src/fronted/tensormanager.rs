//! 用户态 tensor 分配和生命周期管理。
//!
//! 当前阶段 tensor 数据直接放在用户态。
//! `TensorManager` 负责分配一块稳定的用户态 buffer，并生成配套 `AiTensorDesc`。

use super::kd_uring::MmapMemory;
use k3_aiUabi::{AiDtype, AiTensorDesc, AiTensorFormat, AiTensorLayout, tensor_size_bytes};

/// 用户态 tensor 句柄。
///
/// `storage` 是 MAP_SHARED mmap 内存，`desc.user_va` 把这块内存暴露给 graph/kernel ABI。
pub struct Tensor {
    desc: AiTensorDesc,
    storage: MmapMemory,
}

impl Tensor {
    /// 返回可提交给 graph/kernel desc 的稳定描述。
    pub fn desc(&self) -> AiTensorDesc {
        self.desc
    }

    /// 数据区用户态虚拟地址。
    pub fn user_va(&self) -> u64 {
        self.desc.user_va
    }

    /// 数据区总字节数。
    pub fn size_bytes(&self) -> u64 {
        self.desc.size_bytes
    }

    /// 当前张量的 dtype。
    pub fn dtype(&self) -> AiDtype {
        self.desc.dtype
    }

    /// 当前张量的维度视图。
    pub fn shape(&self) -> &[u32] {
        &self.desc.shape[..self.desc.ndim as usize]
    }

    /// 原始字节只读视图。
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.storage.as_ptr(), self.storage.len()) }
    }

    /// 原始字节可写视图。
    ///
    /// mmap 生命周期由 `Tensor` 持有，`desc.user_va` 在 drop 前保持稳定。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.storage.as_mut_ptr(), self.storage.len()) }
    }

    /// F32 只读视图。
    ///
    /// 当前 demo 和 matmul 用例只先接 F32，其他 dtype 后面再各自补。
    pub fn as_f32_slice(&self) -> &[f32] {
        assert!(self.desc.dtype == AiDtype::F32);
        assert!(self.storage.len() % core::mem::size_of::<f32>() == 0);

        unsafe {
            core::slice::from_raw_parts(
                self.storage.as_ptr() as *const f32,
                self.storage.len() / core::mem::size_of::<f32>(),
            )
        }
    }

    /// F32 可写视图。
    pub fn as_f32_mut_slice(&mut self) -> &mut [f32] {
        assert!(self.desc.dtype == AiDtype::F32);
        assert!(self.storage.len() % core::mem::size_of::<f32>() == 0);

        unsafe {
            core::slice::from_raw_parts_mut(
                self.storage.as_mut_ptr() as *mut f32,
                self.storage.len() / core::mem::size_of::<f32>(),
            )
        }
    }
}

/// 用户态 tensor allocator，负责分配稳定的用户态 buffer 并生成配套 `AiTensorDesc`。
#[derive(Default)]
pub struct TensorManager;

impl TensorManager {
    /// 创建空的 `TensorManager`。
    pub fn new() -> Self {
        Self
    }

    /// 用默认 format/layout 分配一个 dense tensor。
    pub fn alloc_tensor(&self, dtype: AiDtype, shape: &[u32]) -> Tensor {
        self.alloc_tensor_with_layout(dtype, AiTensorFormat::ANY, AiTensorLayout::DENSE, shape, 0)
    }

    /// 按指定格式和布局分配用户态 tensor。
    pub fn alloc_tensor_with_layout(
        &self,
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[u32],
        flags: u8,
    ) -> Tensor {
        let element_size = dtype
            .element_size_bytes()
            .expect("quantized or unknown dtype needs explicit tensor manager path");
        let size_bytes = tensor_size_bytes(shape, element_size);
        let size_bytes = usize::try_from(size_bytes).expect("tensor size does not fit usize");

        // Tensor 数据必须是 shared mmap，这样 StarryOS 内核能抓 SharedPages 保活。
        let mut storage = MmapMemory::new_shared(size_bytes).expect("failed to mmap tensor");
        let desc = AiTensorDesc::from_user_buffer(
            storage.as_mut_ptr() as u64,
            storage.len() as u64,
            dtype,
            format,
            layout,
            shape,
            flags,
        );

        Tensor { desc, storage }
    }
}
