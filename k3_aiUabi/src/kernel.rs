//! ring 中的算子编号与目标倾向。

/// ring 中的算子编号。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct KernelOp(pub u8);

impl KernelOp {
    /// 非法/空 op。0 保留出来，避免默认初始化后被误认为有效 op。
    pub const INVALID: Self = Self(0);
    /// 矩阵乘法。
    pub const MAT_MUL: Self = Self(1);
    /// RMS 归一化。
    pub const RMS_NORM: Self = Self(2);
    /// 旋转位置编码 (RoPE)。
    pub const ROPE: Self = Self(3);
    /// Softmax。
    pub const SOFTMAX: Self = Self(4);
    /// 逐元素加法。
    pub const ADD: Self = Self(5);
    /// 逐元素乘法。
    pub const MUL: Self = Self(6);
    /// SiLU 激活。
    pub const SILU: Self = Self(7);
    /// 逐元素缩放。
    pub const SCALE: Self = Self(8);
    /// 二维卷积。
    pub const CONV2D: Self = Self(9);
    /// Sigmoid 激活。
    pub const SIGMOID: Self = Self(10);
    /// 沿指定轴拼接 tensor。
    pub const CONCAT: Self = Self(11);
    /// 按 permutation 重排 tensor 维度。
    pub const TRANSPOSE: Self = Self(12);
    /// ONNX Gather。
    pub const GATHER: Self = Self(13);
    /// 二维最大池化。
    pub const MAX_POOL: Self = Self(14);
    /// tensor dtype 转换。
    pub const CAST: Self = Self(15);
    /// 二维 resize。
    pub const RESIZE: Self = Self(16);
    /// 逐元素除法。
    pub const DIV: Self = Self(17);
    /// 沿指定轴选取前 K 个元素。
    pub const TOP_K: Self = Self(18);
    /// 按 ONNX 广播规则扩展 tensor。
    pub const EXPAND: Self = Self(19);
    /// 按各维 repeats 平铺 tensor。
    pub const TILE: Self = Self(20);
    /// ONNX GatherElements。
    pub const GATHER_ELEMENTS: Self = Self(21);
    /// 逐元素减法。
    pub const SUB: Self = Self(22);
    /// 沿指定轴集合求最大值。
    pub const REDUCE_MAX: Self = Self(23);
    /// 逐元素取模。
    pub const MOD: Self = Self(24);

    /// 最小合法性检查：op 编号是否落在已知算子区间内。
    pub const fn is_known(self) -> bool {
        matches!(self.0, 1..=24)
    }
}

/// 用户态给调度器的目标倾向。
/// hint，最终执行位置由调度器决定。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct AiTargetHint(pub u8);

impl AiTargetHint {
    /// 由调度器自动选择最优执行目标。
    pub const AUTO: Self = Self(0);
    /// 优先在 CPU 上执行。
    pub const PREFER_CPU: Self = Self(1);
    /// 优先在 x100 NPU 上执行。
    pub const PREFER_X100: Self = Self(2);
    /// 优先在 A100 GPU 上执行。
    pub const PREFER_A100: Self = Self(3);

    /// 最小合法性检查：hint 编号是否落在已知目标区间内。
    pub const fn is_known(self) -> bool {
        matches!(self.0, 0..=3)
    }
}
