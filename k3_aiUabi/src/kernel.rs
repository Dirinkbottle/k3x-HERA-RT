//! ring 中的算子编号与目标倾向。


/// ring 中的算子编号。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct KernelOp(pub u8);

impl KernelOp {
    /// 非法/空 op。0 保留出来，避免默认初始化后被误认为有效 op。
    pub const INVALID: Self = Self(0);
    pub const MAT_MUL: Self = Self(1);
    pub const RMS_NORM: Self = Self(2);
    pub const ROPE: Self = Self(3);
    pub const SOFTMAX: Self = Self(4);
    pub const ADD: Self = Self(5);
    pub const MUL: Self = Self(6);
    pub const SILU: Self = Self(7);
    pub const SCALE: Self = Self(8);
    pub const CONV2D: Self = Self(9);

    /// 最小合法性检查。
    pub const fn is_known(self) -> bool {
        matches!(self.0, 1..=9)
    }
}

/// 用户态给调度器的目标倾向。
/// hint，最终执行位置由调度器决定。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct AiTargetHint(pub u8);

impl AiTargetHint {
    pub const AUTO: Self = Self(0);
    pub const PREFER_CPU: Self = Self(1);
    pub const PREFER_X100: Self = Self(2);
    pub const PREFER_A100: Self = Self(3);

    pub const fn is_known(self) -> bool {
        matches!(self.0, 0..=3)
    }
}
