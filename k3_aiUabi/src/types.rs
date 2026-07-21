//! ABI 语义 newtype，避免跨层边界继续传递裸整数。

use core::fmt;
use core::ops::Range;

/// ABI newtype 构造或转换时的轻量边界错误。
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AbiTypeError {
    /// 地址或长度要求非零，但传入值为 0。
    Zero,
    /// 值无法无损转换为当前目标平台的 `usize`。
    UsizeOverflow,
    /// count 相加或 count * size 计算溢出。
    CountOverflow,
    /// 值超过调用方传入的最大上限。
    ExceedsMax,
}

/// 生成 `#[repr(transparent)]` ABI newtype，统一派生常用 trait 并转发外部属性。
macro_rules! abi_newtype {
    ($(#[$meta:meta])* pub struct $name:ident($inner:ty);) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
        pub struct $name(pub $inner);

        impl $name {
            /// 用原始 ABI 值构造 newtype。
            pub const fn new(raw: $inner) -> Self {
                Self(raw)
            }

            /// 返回底层原始 ABI 值。
            pub const fn get(self) -> $inner {
                self.0
            }

            /// 构造非零值。
            pub const fn try_nonzero(raw: $inner) -> Result<Self, AbiTypeError> {
                if raw == 0 {
                    Err(AbiTypeError::Zero)
                } else {
                    Ok(Self(raw))
                }
            }

            /// 转换为 `usize`，转换失败时返回 ABI 类型错误。
            pub fn try_as_usize(self) -> Result<usize, AbiTypeError> {
                usize::try_from(self.0).map_err(|_| AbiTypeError::UsizeOverflow)
            }

            /// 转换为 `usize` 后校验小于等于 `max`。
            pub fn try_under_max(self, max: usize) -> Result<usize, AbiTypeError> {
                let value = self.try_as_usize()?;
                if value > max {
                    Err(AbiTypeError::ExceedsMax)
                } else {
                    Ok(value)
                }
            }
        }

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.get()
            }
        }

        impl PartialEq<$inner> for $name {
            fn eq(&self, other: &$inner) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<$name> for $inner {
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

abi_newtype! {
    /// 用户态虚拟地址。
    pub struct UserVa(u64);
}

abi_newtype! {
    /// 内核态虚拟地址。
    pub struct KernelVa(u64);
}

abi_newtype! {
    /// 64 位字节长度。
    pub struct ByteSize(u64);
}

abi_newtype! {
    /// tensor 维度上的字节 stride。
    pub struct ByteStride(u64);
}

abi_newtype! {
    /// graph blob 内 32 位字节偏移；也用于 header 内 32 位 blob 字节大小。
    pub struct ByteOffset(u32);
}

abi_newtype! {
    /// inline attr 有效字节数。
    pub struct AttrByteSize(u32);
}

abi_newtype! {
    /// 用户提交 graph 时携带的 completion token。
    pub struct UserToken(u32);
}

abi_newtype! {
    /// completion entry 中返回的用户 token。
    pub struct CompletionUserToken(u64);
}

abi_newtype! {
    /// 单算子 tensor 输入或输出数量。
    pub struct TensorCount(u32);
}

abi_newtype! {
    /// graph 中的节点数量。
    pub struct NodeCount(u32);
}

abi_newtype! {
    /// graph 中的依赖边数量。
    pub struct EdgeCount(u32);
}

abi_newtype! {
    /// tensor 实际维度数量。
    pub struct DimCount(u32);
}

abi_newtype! {
    /// tensor 或算子参数中的维度大小。
    pub struct DimSize(u32);
}

abi_newtype! {
    /// 以元素数为单位的 stride。
    pub struct ElemStride(u32);
}

abi_newtype! {
    /// 卷积核移动或采样 stride。
    pub struct KernelStride(u32);
}

abi_newtype! {
    /// tensor 轴编号，允许负数表示从末尾倒数。
    pub struct TensorAxis(i32);
}

abi_newtype! {
    /// tensor flags 位图。
    pub struct TensorFlags(u8);
}

abi_newtype! {
    /// 算子 attr flags 位图。
    pub struct OpFlags(u32);
}

abi_newtype! {
    /// graph flags 位图。
    pub struct GraphFlags(u32);
}

abi_newtype! {
    /// graph submit flags 位图。
    pub struct SubmitFlags(u32);
}

abi_newtype! {
    /// completion 状态码，0 表示成功，负数表示错误。
    pub struct CompletionStatus(i32);
}

impl TensorCount {
    /// 合并输入与输出 tensor 数量，并返回可索引的 `usize`。
    pub fn checked_total(self, other: Self) -> Result<usize, AbiTypeError> {
        let total = self
            .0
            .checked_add(other.0)
            .ok_or(AbiTypeError::CountOverflow)?;
        usize::try_from(total).map_err(|_| AbiTypeError::UsizeOverflow)
    }
}

impl ByteOffset {
    /// 以当前偏移和字节长度构造一个落在 `total_size` 内的字节范围。
    pub fn checked_range(
        self,
        byte_len: usize,
        total_size: usize,
    ) -> Result<Range<usize>, AbiTypeError> {
        let start = self.try_as_usize()?;
        let end = start
            .checked_add(byte_len)
            .ok_or(AbiTypeError::CountOverflow)?;
        if end > total_size {
            Err(AbiTypeError::ExceedsMax)
        } else {
            Ok(start..end)
        }
    }
}

/// Compile-time transparent layout checks for ABI newtypes.
#[allow(dead_code, missing_docs, clippy::missing_docs_in_private_items)]
mod abi_layout {
    use super::*;

    macro_rules! assert_transparent {
        ($newtype:ty, $raw:ty) => {
            const _: () = assert!(core::mem::size_of::<$newtype>() == core::mem::size_of::<$raw>());
            const _: () =
                assert!(core::mem::align_of::<$newtype>() == core::mem::align_of::<$raw>());
        };
    }

    assert_transparent!(UserVa, u64);
    assert_transparent!(KernelVa, u64);
    assert_transparent!(ByteSize, u64);
    assert_transparent!(ByteStride, u64);
    assert_transparent!(ByteOffset, u32);
    assert_transparent!(AttrByteSize, u32);
    assert_transparent!(UserToken, u32);
    assert_transparent!(CompletionUserToken, u64);
    assert_transparent!(TensorCount, u32);
    assert_transparent!(NodeCount, u32);
    assert_transparent!(EdgeCount, u32);
    assert_transparent!(DimCount, u32);
    assert_transparent!(DimSize, u32);
    assert_transparent!(ElemStride, u32);
    assert_transparent!(KernelStride, u32);
    assert_transparent!(TensorAxis, i32);
    assert_transparent!(TensorFlags, u8);
    assert_transparent!(OpFlags, u32);
    assert_transparent!(GraphFlags, u32);
    assert_transparent!(SubmitFlags, u32);
    assert_transparent!(CompletionStatus, i32);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 非零地址构造拒绝 0。
    #[test]
    fn nonzero_address_rejects_zero() {
        assert_eq!(UserVa::try_nonzero(0), Err(AbiTypeError::Zero));
        assert_eq!(UserVa::try_nonzero(0x1000).unwrap().get(), 0x1000);
    }

    /// count 上限校验返回可用于索引的 usize。
    #[test]
    fn count_under_max_checks_limit() {
        assert_eq!(TensorCount::new(2).try_under_max(8).unwrap(), 2);
        assert_eq!(
            TensorCount::new(9).try_under_max(8),
            Err(AbiTypeError::ExceedsMax)
        );
    }

    /// input/output count 相加时检测溢出。
    #[test]
    fn tensor_count_checked_total_rejects_overflow() {
        assert_eq!(
            TensorCount::new(2)
                .checked_total(TensorCount::new(1))
                .unwrap(),
            3
        );
        assert_eq!(
            TensorCount::new(u32::MAX).checked_total(TensorCount::new(1)),
            Err(AbiTypeError::CountOverflow)
        );
    }

    /// u64 字节长度转换为 usize 时检测平台相关溢出。
    #[test]
    fn byte_size_try_as_usize_roundtrips_small_values() {
        assert_eq!(ByteSize::new(128).try_as_usize().unwrap(), 128);
    }

    /// 字节 offset helper 校验范围不越界。
    #[test]
    fn byte_offset_checked_range_rejects_out_of_range() {
        assert_eq!(ByteOffset::new(4).checked_range(8, 16).unwrap(), 4..12);
        assert_eq!(
            ByteOffset::new(12).checked_range(8, 16),
            Err(AbiTypeError::ExceedsMax)
        );
    }
}
