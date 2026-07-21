//! 聚合的 RVV 向量原语。
//!
//! RISC-V 目标使用 VLA inline assembly；其他目标执行同语义软件镜像，便于单测。

use alloc::vec;
use k3_ai_uabi::error::BackendErr;

/// 二元 F32 向量运算。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BinaryOp {
    /// 加法。
    Add,
    /// 减法。
    Sub,
    /// 乘法。
    Mul,
    /// 除法。
    Div,
}

/// 对等长 F32 切片执行向量二元运算。
pub(crate) fn binary_f32(
    op: BinaryOp,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
) -> Result<(), BackendErr> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        binary_f32_hw(op, lhs, rhs, output);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    for ((dst, &a), &b) in output.iter_mut().zip(lhs).zip(rhs) {
        *dst = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        };
    }

    Ok(())
}

/// 对 F32 向量执行 `alpha * x + beta`。
pub(crate) fn affine_f32(
    input: &[f32],
    output: &mut [f32],
    alpha: f32,
    beta: f32,
) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        affine_f32_hw(input, output, alpha, beta);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = alpha * src + beta;
    }

    Ok(())
}

/// 计算向量 exp；硬件路径使用范围缩减和六阶多项式。
pub(crate) fn exp_f32(input: &[f32], output: &mut [f32]) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        exp_f32_hw(input, output);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = fast_exp_scalar(src);
    }

    Ok(())
}

/// 计算 Sigmoid。
pub(crate) fn sigmoid_f32(input: &[f32], output: &mut [f32]) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }
    let mut negated = vec![0.0_f32; input.len()];
    affine_f32(input, &mut negated, -1.0, 0.0)?;
    exp_f32(&negated, output)?;

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        reciprocal_one_plus_f32_hw(output);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    for value in output {
        *value = 1.0 / (1.0 + *value);
    }
    Ok(())
}

/// 计算 SiLU。
pub(crate) fn silu_f32(input: &[f32], output: &mut [f32]) -> Result<(), BackendErr> {
    let mut sigmoid = vec![0.0_f32; input.len()];
    sigmoid_f32(input, &mut sigmoid)?;
    binary_f32(BinaryOp::Mul, input, &sigmoid, output)
}

/// 使用 RVV 或软件镜像复制任意字节区间。
pub(crate) fn copy_bytes(source: &[u8], destination: &mut [u8]) -> Result<(), BackendErr> {
    if source.len() != destination.len() {
        return Err(BackendErr::InvalidTensor);
    }

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        copy_bytes_hw(source, destination);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    destination.copy_from_slice(source);
    Ok(())
}

/// 按 byte offsets 从一个 buffer gather 固定宽度元素。
pub(crate) fn gather_bytes(
    source: &[u8],
    offsets: &[u64],
    destination: &mut [u8],
    element_size: usize,
) -> Result<(), BackendErr> {
    if !matches!(element_size, 1 | 2 | 4 | 8)
        || destination.len() != offsets.len().saturating_mul(element_size)
    {
        return Err(BackendErr::InvalidTensor);
    }
    for &offset in offsets {
        let offset = usize::try_from(offset).map_err(|_| BackendErr::InvalidTensor)?;
        if offset
            .checked_add(element_size)
            .is_none_or(|end| end > source.len())
        {
            return Err(BackendErr::InvalidTensor);
        }
    }

    #[cfg(target_arch = "riscv64")]
    unsafe {
        gather_bytes_hw(source, offsets, destination, element_size);
    }

    #[cfg(not(target_arch = "riscv64"))]
    for (index, &offset) in offsets.iter().enumerate() {
        let offset = offset as usize;
        let destination_offset = index * element_size;
        destination[destination_offset..destination_offset + element_size]
            .copy_from_slice(&source[offset..offset + element_size]);
    }
    Ok(())
}

/// 把 I32 切片向量转换为 F32。
pub(crate) fn cast_i32_to_f32(input: &[i32], output: &mut [f32]) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        cast_i32_to_f32_hw(input, output);
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = src as f32;
    }
    Ok(())
}

/// 把 F32 切片按向零截断规则向量转换为 I32。
pub(crate) fn cast_f32_to_i32(input: &[f32], output: &mut [i32]) -> Result<(), BackendErr> {
    if input.len() != output.len() {
        return Err(BackendErr::InvalidTensor);
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        cast_f32_to_i32_hw(input, output);
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = src as i32;
    }
    Ok(())
}

/// 对 F32 切片执行向量求和。
pub(crate) fn reduce_sum_f32(input: &[f32]) -> f32 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        return reduce_f32_hw(input, false);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    input.iter().copied().sum()
}

/// 对 F32 切片执行向量最大值归约。
pub(crate) fn reduce_max_f32(input: &[f32]) -> f32 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        return reduce_f32_hw(input, true);
    }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// 软件 fast-exp 镜像，与 RVV 多项式使用相同范围缩减。
#[cfg(any(test, not(any(target_arch = "riscv32", target_arch = "riscv64"))))]
fn fast_exp_scalar(value: f32) -> f32 {
    let x = value.clamp(-80.0, 80.0);
    let n = libm::roundf(x * core::f32::consts::LOG2_E) as i32;
    let r = x - n as f32 * core::f32::consts::LN_2;
    let polynomial =
        ((((((1.0 / 720.0) * r + 1.0 / 120.0) * r + 1.0 / 24.0) * r + 1.0 / 6.0) * r + 0.5) * r
            + 1.0)
            * r
            + 1.0;
    polynomial * f32::from_bits(((n + 127) as u32) << 23)
}

/// RISC-V F32 二元 VLA 循环。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn binary_f32_hw(op: BinaryOp, lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    let mut offset = 0_usize;
    while offset < lhs.len() {
        let remaining = lhs.len() - offset;
        let mut vl: usize;
        let lhs_ptr = unsafe { lhs.as_ptr().add(offset) };
        let rhs_ptr = unsafe { rhs.as_ptr().add(offset) };
        let out_ptr = unsafe { output.as_mut_ptr().add(offset) };
        macro_rules! run {
            ($instruction:literal) => {
                unsafe {
                    core::arch::asm!(
                        ".option push",
                        ".option arch, +v",
                        "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                        "vle32.v v0, ({lhs})",
                        "vle32.v v1, ({rhs})",
                        $instruction,
                        "vse32.v v2, ({output})",
                        ".option pop",
                        vl = lateout(reg) vl,
                        remaining = in(reg) remaining,
                        lhs = in(reg) lhs_ptr,
                        rhs = in(reg) rhs_ptr,
                        output = in(reg) out_ptr,
                        out("v0") _,
                        out("v1") _,
                        out("v2") _,
                        options(nostack),
                    );
                }
            };
        }
        match op {
            BinaryOp::Add => run!("vfadd.vv v2, v0, v1"),
            BinaryOp::Sub => run!("vfsub.vv v2, v0, v1"),
            BinaryOp::Mul => run!("vfmul.vv v2, v0, v1"),
            BinaryOp::Div => run!("vfdiv.vv v2, v0, v1"),
        }
        offset += vl;
    }
}

/// RISC-V F32 affine VLA 循环。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn affine_f32_hw(input: &[f32], output: &mut [f32], alpha: f32, beta: f32) {
    let constants = [alpha, beta];
    let mut offset = 0_usize;
    while offset < input.len() {
        let remaining = input.len() - offset;
        let vl: usize;
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                "vle32.v v0, ({input})",
                "flw ft0, 0({constants})",
                "flw ft1, 4({constants})",
                "vfmul.vf v1, v0, ft0",
                "vfadd.vf v1, v1, ft1",
                "vse32.v v1, ({output})",
                ".option pop",
                vl = lateout(reg) vl,
                remaining = in(reg) remaining,
                input = in(reg) input.as_ptr().add(offset),
                output = in(reg) output.as_mut_ptr().add(offset),
                constants = in(reg) constants.as_ptr(),
                out("ft0") _,
                out("ft1") _,
                out("v0") _,
                out("v1") _,
                options(nostack),
            );
        }
        offset += vl;
    }
}

/// RISC-V fast-exp VLA 循环。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn exp_f32_hw(input: &[f32], output: &mut [f32]) {
    let constants = [
        core::f32::consts::LOG2_E,
        core::f32::consts::LN_2,
        -80.0,
        80.0,
        1.0,
        0.5,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
    ];
    let mut offset = 0_usize;
    while offset < input.len() {
        let remaining = input.len() - offset;
        let vl: usize;
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                "vle32.v v0, ({input})",
                "flw ft0, 8({constants})",
                "vfmax.vf v0, v0, ft0",
                "flw ft0, 12({constants})",
                "vfmin.vf v0, v0, ft0",
                "flw ft0, 0({constants})",
                "vfmul.vf v1, v0, ft0",
                "vfcvt.x.f.v v2, v1",
                "vfcvt.f.x.v v3, v2",
                "flw ft0, 4({constants})",
                "vfmul.vf v3, v3, ft0",
                "vfsub.vv v3, v0, v3",
                "flw ft0, 36({constants})",
                "vfmv.v.f v4, ft0",
                "vfmul.vv v4, v4, v3",
                "flw ft0, 32({constants})",
                "vfadd.vf v4, v4, ft0",
                "vfmul.vv v4, v4, v3",
                "flw ft0, 28({constants})",
                "vfadd.vf v4, v4, ft0",
                "vfmul.vv v4, v4, v3",
                "flw ft0, 24({constants})",
                "vfadd.vf v4, v4, ft0",
                "vfmul.vv v4, v4, v3",
                "flw ft0, 20({constants})",
                "vfadd.vf v4, v4, ft0",
                "vfmul.vv v4, v4, v3",
                "flw ft0, 16({constants})",
                "vfadd.vf v4, v4, ft0",
                "vfmul.vv v4, v4, v3",
                "vfadd.vf v4, v4, ft0",
                "li t1, 127",
                "vadd.vx v2, v2, t1",
                "vsll.vi v2, v2, 23",
                "vfmul.vv v4, v4, v2",
                "vse32.v v4, ({output})",
                ".option pop",
                vl = lateout(reg) vl,
                remaining = in(reg) remaining,
                input = in(reg) input.as_ptr().add(offset),
                output = in(reg) output.as_mut_ptr().add(offset),
                constants = in(reg) constants.as_ptr(),
                out("t1") _,
                out("ft0") _,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                out("v4") _,
                options(nostack),
            );
        }
        offset += vl;
    }
}

/// RISC-V `1 / (1 + x)` VLA 循环。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn reciprocal_one_plus_f32_hw(values: &mut [f32]) {
    let one = 1.0_f32;
    let mut offset = 0_usize;
    while offset < values.len() {
        let remaining = values.len() - offset;
        let vl: usize;
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                "vle32.v v0, ({values})",
                "flw ft0, 0({one})",
                "vfadd.vf v0, v0, ft0",
                "vfrdiv.vf v0, v0, ft0",
                "vse32.v v0, ({values})",
                ".option pop",
                vl = lateout(reg) vl,
                remaining = in(reg) remaining,
                values = in(reg) values.as_mut_ptr().add(offset),
                one = in(reg) &one,
                out("ft0") _,
                out("v0") _,
                options(nostack),
            );
        }
        offset += vl;
    }
}

/// RISC-V byte copy VLA 循环。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn copy_bytes_hw(source: &[u8], destination: &mut [u8]) {
    let mut offset = 0_usize;
    while offset < source.len() {
        let remaining = source.len() - offset;
        let vl: usize;
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {remaining}, e8, m1, ta, ma",
                "vle8.v v0, ({source})",
                "vse8.v v0, ({destination})",
                ".option pop",
                vl = lateout(reg) vl,
                remaining = in(reg) remaining,
                source = in(reg) source.as_ptr().add(offset),
                destination = in(reg) destination.as_mut_ptr().add(offset),
                out("v0") _,
                options(nostack),
            );
        }
        offset += vl;
    }
}

/// RISC-V indexed load，用 64 位 byte offsets gather 元素。
#[cfg(target_arch = "riscv64")]
unsafe fn gather_bytes_hw(
    source: &[u8],
    offsets: &[u64],
    destination: &mut [u8],
    element_size: usize,
) {
    let mut index = 0_usize;
    while index < offsets.len() {
        let remaining = offsets.len() - index;
        let vl: usize;
        macro_rules! run {
            ($sew:literal, $load:literal, $store:literal) => {
                unsafe {
                    core::arch::asm!(
                        ".option push",
                        ".option arch, +v",
                        "vsetvli {vl}, {remaining}, e64, m1, ta, ma",
                        "vle64.v v8, ({offsets})",
                        concat!("vsetvli zero, {vl}, ", $sew, ", m1, ta, ma"),
                        $load,
                        $store,
                        ".option pop",
                        vl = lateout(reg) vl,
                        remaining = in(reg) remaining,
                        source = in(reg) source.as_ptr(),
                        offsets = in(reg) offsets.as_ptr().add(index),
                        destination = in(reg) destination.as_mut_ptr().add(index * element_size),
                        out("v0") _,
                        out("v8") _,
                        out("v9") _,
                        out("v10") _,
                        out("v11") _,
                        out("v12") _,
                        out("v13") _,
                        out("v14") _,
                        out("v15") _,
                        options(nostack),
                    );
                }
            };
        }
        match element_size {
            1 => run!(
                "e8",
                "vloxei64.v v0, ({source}), v8",
                "vse8.v v0, ({destination})"
            ),
            2 => run!(
                "e16",
                "vloxei64.v v0, ({source}), v8",
                "vse16.v v0, ({destination})"
            ),
            4 => run!(
                "e32",
                "vloxei64.v v0, ({source}), v8",
                "vse32.v v0, ({destination})"
            ),
            8 => run!(
                "e64",
                "vloxei64.v v0, ({source}), v8",
                "vse64.v v0, ({destination})"
            ),
            _ => unreachable!("element size validated by caller"),
        }
        index += vl;
    }
}

/// RISC-V I32-to-F32 conversion。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn cast_i32_to_f32_hw(input: &[i32], output: &mut [f32]) {
    let mut offset = 0_usize;
    while offset < input.len() {
        let remaining = input.len() - offset;
        let vl: usize;
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                "vle32.v v0, ({input})",
                "vfcvt.f.x.v v1, v0",
                "vse32.v v1, ({output})",
                ".option pop",
                vl = lateout(reg) vl,
                remaining = in(reg) remaining,
                input = in(reg) input.as_ptr().add(offset),
                output = in(reg) output.as_mut_ptr().add(offset),
                out("v0") _,
                out("v1") _,
                options(nostack),
            );
        }
        offset += vl;
    }
}

/// RISC-V F32-to-I32 conversion。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn cast_f32_to_i32_hw(input: &[f32], output: &mut [i32]) {
    let mut offset = 0_usize;
    while offset < input.len() {
        let remaining = input.len() - offset;
        let vl: usize;
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                "vle32.v v0, ({input})",
                "vfcvt.rtz.x.f.v v1, v0",
                "vse32.v v1, ({output})",
                ".option pop",
                vl = lateout(reg) vl,
                remaining = in(reg) remaining,
                input = in(reg) input.as_ptr().add(offset),
                output = in(reg) output.as_mut_ptr().add(offset),
                out("v0") _,
                out("v1") _,
                options(nostack),
            );
        }
        offset += vl;
    }
}

/// RISC-V F32 sum/max reduction。
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
unsafe fn reduce_f32_hw(input: &[f32], maximum: bool) -> f32 {
    let mut accumulator = if maximum { f32::NEG_INFINITY } else { 0.0 };
    let mut offset = 0_usize;
    while offset < input.len() {
        let remaining = input.len() - offset;
        let vl: usize;
        let seed = accumulator;
        macro_rules! run {
            ($instruction:literal) => {
                unsafe {
                    core::arch::asm!(
                        ".option push",
                        ".option arch, +v",
                        "vsetvli {vl}, {remaining}, e32, m1, ta, ma",
                        "vle32.v v0, ({input})",
                        "vsetivli zero, 1, e32, m1, ta, ma",
                        "vle32.v v1, ({seed})",
                        "vsetvli zero, {remaining}, e32, m1, ta, ma",
                        $instruction,
                        "vsetivli zero, 1, e32, m1, ta, ma",
                        "vse32.v v2, ({output})",
                        ".option pop",
                        vl = lateout(reg) vl,
                        remaining = in(reg) remaining,
                        input = in(reg) input.as_ptr().add(offset),
                        seed = in(reg) &seed,
                        output = in(reg) &mut accumulator,
                        out("v0") _,
                        out("v1") _,
                        out("v2") _,
                        options(nostack),
                    );
                }
            };
        }
        if maximum {
            run!("vfredmax.vs v2, v0, v1");
        } else {
            run!("vfredusum.vs v2, v0, v1");
        }
        offset += vl;
    }
    accumulator
}

/// RVV 原语软件镜像测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// fast-exp 在 Softmax 使用区间内应保持较小相对误差。
    #[test]
    fn fast_exp_matches_libm_on_negative_domain() {
        for step in 0..=160 {
            let x = -(step as f32) * 0.5;
            let expected = libm::expf(x);
            let actual = fast_exp_scalar(x);
            if expected > 1.0e-30 {
                let relative = ((actual - expected) / expected).abs();
                assert!(
                    relative < 5.0e-4,
                    "x={x}, expected={expected}, actual={actual}"
                );
            }
        }
    }

    /// 软件镜像的 binary/affine/sigmoid 应按定义执行。
    #[test]
    fn vector_primitives_match_reference() {
        let input = [-2.0_f32, 0.0, 2.0];
        let rhs = [2.0_f32, 3.0, 4.0];
        let mut output = [0.0_f32; 3];
        binary_f32(BinaryOp::Mul, &input, &rhs, &mut output).unwrap();
        assert_eq!(output, [-4.0, 0.0, 8.0]);
        affine_f32(&input, &mut output, 2.0, 1.0).unwrap();
        assert_eq!(output, [-3.0, 1.0, 5.0]);
        sigmoid_f32(&input, &mut output).unwrap();
        assert!((output[1] - 0.5).abs() < 1.0e-6);
    }
}
