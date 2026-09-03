//! 用户态 tensor 分配和生命周期管理。
//!
//! 当前阶段 tensor 数据直接放在用户态。
//! `TensorManager` 负责分配一块稳定的用户态 buffer，并生成配套 `AiTensorDesc`。

use super::kd_uring::{BorrowedMemory, MmapMemory};
use k3_ai_uabi::error::AiRuntimeErr;
use k3_ai_uabi::{
    AiDtype, AiTensorDesc, AiTensorFormat, AiTensorLayout, ByteSize, DimSize, MAX_DIM, TensorFlags,
    UserVa, ggml_quant_tensor_size_bytes, tensor_size_bytes,
};

/// Tensor 数据的所有权模式。
///
/// `Mapped` 用于 runtime 自己分配的 buffer；`Borrowed` 用于 ORT 等调用方在一次
/// 同步执行期间借出的 buffer。两种模式共用同一个 `AiTensorDesc`，区别只在 Drop
/// 时是否解除用户态内存映射。
enum TensorStorage {
    /// runtime 通过 `mmap` 分配并拥有的 buffer。
    Mapped(MmapMemory),
    /// 调用方拥有、runtime 只在本次执行中借用的 buffer。
    Borrowed(BorrowedMemory),
}

impl TensorStorage {
    /// 返回数据区的只读首地址。
    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Mapped(memory) => memory.as_ptr(),
            Self::Borrowed(memory) => memory.as_ptr(),
        }
    }

    /// 返回数据区的可写首地址。
    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Mapped(memory) => memory.as_mut_ptr(),
            Self::Borrowed(memory) => memory.as_mut_ptr(),
        }
    }

    /// 返回数据区的总字节数。
    fn len(&self) -> usize {
        match self {
            Self::Mapped(memory) => memory.len(),
            Self::Borrowed(memory) => memory.len(),
        }
    }
}

/// 用户态 tensor 句柄。
///
/// `storage` 既可以是 runtime 自己分配的 MAP_SHARED mmap，也可以是调用方提供的
/// 借用 buffer。`desc.user_va` 始终指向实际数据地址，供 graph/kernel ABI 使用。
pub struct Tensor {
    /// 提交给 graph/kernel ABI 的稳定描述。
    desc: AiTensorDesc,
    /// 承载 tensor 数据的所有权或借用信息。
    storage: TensorStorage,
}

impl Tensor {
    /// 返回可提交给 graph/kernel desc 的稳定描述。
    pub fn desc(&self) -> AiTensorDesc {
        self.desc
    }

    /// 数据区用户态虚拟地址。
    pub fn user_va(&self) -> UserVa {
        self.desc.user_va
    }

    /// 数据区总字节数。
    pub fn size_bytes(&self) -> ByteSize {
        self.desc.size_bytes
    }

    /// 当前张量的 dtype。
    pub fn dtype(&self) -> AiDtype {
        self.desc.dtype
    }

    /// 当前张量的维度视图。
    pub fn shape(&self) -> &[DimSize] {
        &self.desc.shape[..self.desc.ndim.get() as usize]
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
        assert!(
            self.storage
                .len()
                .is_multiple_of(core::mem::size_of::<f32>())
        );

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
        assert!(
            self.storage
                .len()
                .is_multiple_of(core::mem::size_of::<f32>())
        );

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

    /// 从调用方已有的连续 buffer 创建零拷贝 tensor view。
    ///
    /// 该函数不复制数据，也不接管 buffer 的释放。设备路径在 submit 时由内核 pin
    /// 对应页面并建立 kernel alias；因此调用方必须让 buffer 至少存活到本次同步执行
    /// 收到 completion 为止。
    ///
    /// # Safety
    ///
    /// `data..data + size_bytes` 必须是有效的可读写内存，并且在返回的 `Tensor`
    /// 生命周期内不可释放或并发破坏。
    pub unsafe fn borrow_tensor_with_layout(
        &self,
        data: *mut u8,
        size_bytes: usize,
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[u32],
        flags: u8,
    ) -> Result<Tensor, AiRuntimeErr> {
        let element_size = dtype
            .element_size_bytes()
            .ok_or(AiRuntimeErr::InvalidInput)?;
        if shape.len() > MAX_DIM {
            return Err(AiRuntimeErr::InvalidShape);
        }
        let shape_dims: Vec<DimSize> = shape.iter().copied().map(DimSize::new).collect();
        let required_size = tensor_size_bytes(&shape_dims, element_size)
            .try_as_usize()
            .map_err(|_| AiRuntimeErr::InvalidShape)?;
        if size_bytes < required_size {
            return Err(AiRuntimeErr::InvalidInput);
        }

        // SAFETY: The caller upholds the raw buffer lifetime and access contract
        // documented above. The returned view never frees this address.
        let mut storage = unsafe { BorrowedMemory::from_raw_parts(data, size_bytes) }
            .map_err(|_| AiRuntimeErr::InvalidInput)?;
        let desc = AiTensorDesc::from_user_buffer(
            UserVa::new(storage.as_mut_ptr() as u64),
            ByteSize::new(storage.len() as u64),
            dtype,
            format,
            layout,
            &shape_dims,
            TensorFlags::new(flags),
        );

        Ok(Tensor {
            desc,
            storage: TensorStorage::Borrowed(storage),
        })
    }

    /// 用默认 format/layout 分配一个 dense tensor。
    pub fn alloc_tensor(&self, dtype: AiDtype, shape: &[u32]) -> Result<Tensor, AiRuntimeErr> {
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
    ) -> Result<Tensor, AiRuntimeErr> {
        let element_size = dtype
            .element_size_bytes()
            .ok_or(AiRuntimeErr::InvalidInput)?;
        if shape.len() > MAX_DIM {
            return Err(AiRuntimeErr::InvalidShape);
        }
        let shape_dims: Vec<DimSize> = shape.iter().copied().map(DimSize::new).collect();
        let size_bytes = tensor_size_bytes(&shape_dims, element_size);
        let alloc_size = size_bytes
            .try_as_usize()
            .map_err(|_| AiRuntimeErr::InvalidShape)?;

        let mut storage =
            MmapMemory::new_shared(alloc_size).map_err(|_| AiRuntimeErr::AllocFailed)?;
        let desc = AiTensorDesc::from_user_buffer(
            UserVa::new(storage.as_mut_ptr() as u64),
            ByteSize::new(storage.len() as u64),
            dtype,
            format,
            layout,
            &shape_dims,
            TensorFlags::new(flags),
        );

        Ok(Tensor {
            desc,
            storage: TensorStorage::Mapped(storage),
        })
    }

    /// 分配一个 ggml 量化 layout tensor。
    pub fn alloc_ggml_quant_tensor(
        &self,
        dtype: AiDtype,
        shape: &[u32],
        flags: u8,
    ) -> Result<Tensor, AiRuntimeErr> {
        if shape.len() > MAX_DIM {
            return Err(AiRuntimeErr::InvalidShape);
        }
        let shape_dims: Vec<DimSize> = shape.iter().copied().map(DimSize::new).collect();
        let size_bytes =
            ggml_quant_tensor_size_bytes(dtype, &shape_dims).ok_or(AiRuntimeErr::InvalidShape)?;
        let alloc_size = size_bytes
            .try_as_usize()
            .map_err(|_| AiRuntimeErr::InvalidShape)?;

        let mut storage =
            MmapMemory::new_shared(alloc_size).map_err(|_| AiRuntimeErr::AllocFailed)?;
        let desc = AiTensorDesc::from_user_quant_buffer(
            UserVa::new(storage.as_mut_ptr() as u64),
            ByteSize::new(storage.len() as u64),
            dtype,
            AiTensorFormat::ANY,
            &shape_dims,
            TensorFlags::new(flags),
        );

        Ok(Tensor {
            desc,
            storage: TensorStorage::Mapped(storage),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A borrowed tensor must describe and write the caller's original buffer.
    #[test]
    fn borrowed_tensor_view_uses_the_original_address() {
        let manager = TensorManager::new();
        let mut storage = [1.0_f32, 2.0_f32];
        let address = storage.as_mut_ptr().cast::<u8>();

        // SAFETY: `storage` remains allocated and exclusively owned for the
        // lifetime of `tensor` in this test.
        let mut tensor = unsafe {
            manager.borrow_tensor_with_layout(
                address,
                core::mem::size_of_val(&storage),
                AiDtype::F32,
                AiTensorFormat::ANY,
                AiTensorLayout::DENSE,
                &[2],
                0,
            )
        }
        .expect("borrowed tensor view should be valid");

        assert_eq!(tensor.user_va().get(), address as u64);
        tensor.as_f32_mut_slice()[1] = 7.0;
        assert_eq!(storage, [1.0, 7.0]);

        drop(tensor);
        storage[0] = 9.0;
        assert_eq!(storage, [9.0, 7.0]);
    }
}
