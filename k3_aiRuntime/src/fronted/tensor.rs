//! 用户态 Tensor 包装。

use super::{AiDtype, AiTensorDesc, AiTensorFormat, AiTensorLayout};

/// 用户态 tensor 句柄。
///
/// `Tensor` 拥有一个内核 allocator 分配出来的 tensor desc。Drop 时会调用 desc 的
/// free 路径释放内核侧 buffer。数据访问通过 desc 里的用户态虚拟地址临时转成 slice。
pub struct Tensor {
    desc: AiTensorDesc,
}

impl Tensor {
    pub(crate) fn from_desc(desc: AiTensorDesc) -> Self {
        Self { desc }
    }

    /// 返回可提交给 graph/kernel desc 的 tensor 描述。
    pub fn desc(&self) -> AiTensorDesc {
        self.desc
    }

    /// 数据区用户态虚拟地址。
    pub fn user_va(&self) -> u64 {
        self.desc.user_va
    }

    /// 数据区字节数。
    pub fn size_bytes(&self) -> u64 {
        self.desc.size_bytes
    }

    /// 只读访问 tensor 映射。
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.desc.user_va as *const u8, self.len_usize()) }
    }

    /// 可写访问 tensor 映射。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.desc.user_va as *mut u8, self.len_usize()) }
    }

    fn len_usize(&self) -> usize {
        usize::try_from(self.desc.size_bytes).expect("tensor size does not fit usize")
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        self.desc.free();
    }
}

/// 用户态 tensor allocator 入口。
///
/// 负责把 dtype/shape 交给内核 allocator，拿到 desc 后包装成 `Tensor`。
#[derive(Default)]
pub struct TensorManager;

impl TensorManager {
    pub fn new() -> Self {
        Self
    }

    pub fn alloc_tensor(&self, dtype: AiDtype, shape: &[u32]) -> Tensor {
        self.alloc_tensor_with_layout(
            dtype,
            AiTensorFormat::ANY,
            AiTensorLayout::DENSE,
            shape,
            0,
        )
    }

    pub fn alloc_tensor_with_layout(
        &self,
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[u32],
        flags: u8,
    ) -> Tensor {
        let desc = AiTensorDesc::alloc_from_kernel(dtype, format, layout, shape, flags);
        Tensor::from_desc(desc)
    }
}
