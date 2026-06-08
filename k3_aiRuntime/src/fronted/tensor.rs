//! 用户态 Tensor 管理器。
//!
//! 这一层的职责是持有用户态 buffer，并给 frontend 提供稳定的 `AiTensorDesc`。
//! 注意：`Tensor`、`TensorManager`、`Box<[u8]>` 都是用户态库内部对象，不进 ring。

use super::{AiDtype, AiTensorDesc, AiTensorFormat, AiTensorLayout};

/// 用户态 Tensor 数据载体。
///
/// `data` 放在 heap 上，Box 内部 buffer 的地址在 Tensor 移动时不会改变。
/// 从它导出 user_va，交给内核后由内核再做页表校验和 pin。
pub struct Tensor {
    data: Box<[u8]>,
}

impl Tensor {
    /// 创建一块由 runtime 管理的 tensor 数据区。
    ///
    /// `shape` 是逻辑形状；`dtype` 决定元素宽度。
    pub fn new(dtype: AiDtype, shape: &[u32]) -> Self {
        let element_size = dtype
            .element_size_bytes()
            .expect("quantized or unknown dtype needs explicit raw allocation");
        let size_bytes = tensor_size_bytes(shape, element_size);
        let data = vec![0_u8; size_bytes as usize].into_boxed_slice();
        Self { data }
    }

    /// 用现有 bytes 创建 raw tensor。
    ///
    /// 该路径适合 ggml 量化块、权重文件加载结果，或者暂时无法用 element size 描述的 buffer。
    pub fn from_raw_bytes(data: Box<[u8]>) -> Self {
        Self { data }
    }

    /// 数据区用户态虚拟地址。
    pub fn user_va(&self) -> u64 {
        self.data.as_ptr() as u64
    }

    /// 数据区字节数。
    pub fn size_bytes(&self) -> u64 {
        self.data.len() as u64
    }

    /// 只读访问用户态 buffer。
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// 可写访问用户态 buffer。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Tensor 数据令牌。
///
/// token 只在用户态库内部使用，用于从 `TensorManager` 中找回 Tensor。
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct TensorDataToken(u64);

impl TensorDataToken {
    fn new(slot_idx: usize, generation: u32) -> Self {
        assert!(slot_idx <= u32::MAX as usize);
        Self(((generation as u64) << 32) | slot_idx as u64)
    }

    fn slot_idx(self) -> usize {
        (self.0 & 0xffff_ffff) as usize
    }

    fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// 暴露原始 token 值，方便上层 runtime 建映射表。
    pub fn raw(self) -> u64 {
        self.0
    }
}

struct TensorSlot {
    generation: u32,
    // 这个slot是否在使用
    inuse: bool,
    tensor: Option<Tensor>,
}

/// 用户态 Tensor 管理器。
///
/// 第一阶段先用简单 Vec 管理。
/// 后续如果要减少分配开销，可以在这里替换成 slab、arena 或接入共享 buffer allocator。
#[derive(Default)]
pub struct TensorManager {
    slots: Vec<TensorSlot>,
}

impl TensorManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&mut self, dtype: AiDtype, shape: &[u32]) -> TensorDataToken {
        self.alloc_tensor(Tensor::new(dtype, shape))
    }

    pub fn alloc_tensor(&mut self, tensor: Tensor) -> TensorDataToken {
        let next_idx = self
            .slots
            .iter()
            .enumerate()
            .find_map(|(idx, slot)| (!slot.inuse).then_some(idx));

        if let Some(idx) = next_idx {
            // 再次有效
            self.slots[idx].inuse = true;
            // generate迭代
            self.slots[idx].generation = self.slots[idx].generation.wrapping_add(1);
            // 再次填充tensor
            self.slots[idx].tensor = Some(tensor);

            // 构造token
            TensorDataToken::new(idx, self.slots[idx].generation)
        } else {
            // 直接新建slot ，push进去
            self.slots.push(TensorSlot {
                generation: 1,
                inuse: true,
                tensor: Some(tensor),
            });

            let new_slot_idx = self.slots.len() - 1;

            // 构造token
            TensorDataToken::new(new_slot_idx, self.slots[new_slot_idx].generation)
        }
    }

    pub fn remove(&mut self, token: TensorDataToken) -> Option<Tensor> {
        let slot = self.slots.get_mut(token.slot_idx())?;
        if !slot.inuse || slot.generation != token.generation() {
            return None;
        }

        slot.inuse = false;
        slot.tensor.take()
    }

    pub fn get(&self, token: TensorDataToken) -> Option<&Tensor> {
        let slot = self.slots.get(token.slot_idx())?;
        if !slot.inuse || slot.generation != token.generation() {
            return None;
        }

        slot.tensor.as_ref()
    }

    pub fn get_mut(&mut self, token: TensorDataToken) -> Option<&mut Tensor> {
        let slot = self.slots.get_mut(token.slot_idx())?;
        if !slot.inuse || slot.generation != token.generation() {
            return None;
        }

        slot.tensor.as_mut()
    }

    pub fn desc(
        &self,
        token: TensorDataToken,
        dtype: AiDtype,
        format: AiTensorFormat,
        layout: AiTensorLayout,
        shape: &[u32],
    ) -> Option<AiTensorDesc> {
        let tensor = self.get(token)?;
        let element_size = dtype.element_size_bytes()?;
        let desc = AiTensorDesc::from_buffer(
            tensor.user_va(),
            tensor.size_bytes(),
            dtype,
            format,
            layout,
            shape,
            element_size,
        );
        Some(desc)
    }
}

fn tensor_size_bytes(shape: &[u32], element_size_bytes: u32) -> u64 {
    let elements = shape
        .iter()
        .fold(1_u64, |acc, dim| acc.saturating_mul(*dim as u64));
    elements.saturating_mul(element_size_bytes as u64)
}
