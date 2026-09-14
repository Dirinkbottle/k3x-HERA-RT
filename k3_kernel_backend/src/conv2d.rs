//! FP16 二维卷积算子。

use alloc::vec;

use crate::call::CallContext;
use crate::matmul::{self, F16Matmul};
use crate::{BackendCall, ComputeKernel};
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{
    AiDtype, AiTargetHint, Conv2dAttr, DimSize, ElemStride, KernelOp, MatMulAttr, OpFlags,
};

/// Conv2D 的静态分发标记。
pub(crate) struct Conv2dKernel;

impl ComputeKernel for Conv2dKernel {
    const OP: KernelOp = KernelOp::CONV2D;

    unsafe fn call(call: *const BackendCall) -> Result<(), BackendErr> {
        unsafe { call_conv2d(call) }
    }
}

/// 以 `usize` 表示的卷积几何属性。
struct Conv2dGeom {
    /// batch 数。
    batch: usize,
    /// 输入通道数。
    cin: usize,
    /// 输出通道数。
    cout: usize,
    /// Input height.
    ih: usize,
    /// Input width.
    iw: usize,
    /// Output height.
    oh: usize,
    /// Output width.
    ow: usize,
    /// Kernel height.
    kh: usize,
    /// Kernel width.
    kw: usize,
    /// Vertical stride.
    sh: usize,
    /// Horizontal stride.
    sw: usize,
    /// Vertical padding.
    ph: usize,
    /// Horizontal padding.
    pw: usize,
    /// Vertical dilation.
    dh: usize,
    /// Horizontal dilation.
    dw: usize,
    /// 分组数量。
    groups: usize,
}

impl Conv2dGeom {
    /// 从 ABI 属性构造并校验几何参数。
    fn from_attr(attr: &Conv2dAttr) -> Result<Self, BackendErr> {
        let groups = (attr.groups.get() as usize).max(1);
        let geom = Self {
            batch: (attr.batch.get() as usize).max(1),
            cin: attr.in_channels.get() as usize,
            cout: attr.out_channels.get() as usize,
            ih: attr.input_h.get() as usize,
            iw: attr.input_w.get() as usize,
            oh: attr.output_h.get() as usize,
            ow: attr.output_w.get() as usize,
            kh: attr.kernel_h.get() as usize,
            kw: attr.kernel_w.get() as usize,
            sh: (attr.stride_h.get() as usize).max(1),
            sw: (attr.stride_w.get() as usize).max(1),
            ph: attr.pad_h.get() as usize,
            pw: attr.pad_w.get() as usize,
            dh: (attr.dilation_h.get() as usize).max(1),
            dw: (attr.dilation_w.get() as usize).max(1),
            groups,
        };
        if geom.cin == 0
            || geom.cout == 0
            || geom.ih == 0
            || geom.iw == 0
            || geom.oh == 0
            || geom.ow == 0
            || geom.kh == 0
            || geom.kw == 0
            || !geom.cin.is_multiple_of(groups)
            || !geom.cout.is_multiple_of(groups)
        {
            return Err(BackendErr::InvalidAttr);
        }
        Ok(geom)
    }

    /// 单 group 的输入通道数。
    fn cin_per_group(&self) -> usize {
        self.cin / self.groups
    }

    /// 单 group 的输出通道数。
    fn cout_per_group(&self) -> usize {
        self.cout / self.groups
    }

    /// 一个卷积感受野中的元素数。
    fn patch_size(&self) -> usize {
        self.cin_per_group() * self.kh * self.kw
    }

    /// 单个 batch 的输出空间元素数。
    fn spatial(&self) -> usize {
        self.oh * self.ow
    }
}

/// 解析 ABI 并选择 Conv2D 的 FP16 target 实现。
unsafe fn call_conv2d(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io_range(2..=3, 1..=1)?;
    ctx.reject_input_output_alias()?;
    if ctx.inputs[0].dtype != AiDtype::F16
        || ctx.inputs[1].dtype != AiDtype::F16
        || ctx.outputs[0].dtype != AiDtype::F16
        || ctx
            .inputs
            .get(2)
            .is_some_and(|view| view.dtype != AiDtype::F16)
    {
        return Err(BackendErr::UnsupportedDtype);
    }
    let geom = Conv2dGeom::from_attr(&ctx.read_attr::<Conv2dAttr>()?)?;
    let input = unsafe { ctx.inputs[0].as_slice::<u16>()? };
    let weight = unsafe { ctx.inputs[1].as_slice::<u16>()? };
    let bias = match ctx.inputs.get(2) {
        Some(view) => Some(unsafe { view.as_slice::<u16>()? }),
        None => None,
    };
    let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
    validate_lengths(
        &geom,
        input.len(),
        weight.len(),
        output.len(),
        bias.map(<[u16]>::len),
    )?;
    match ctx.target {
        AiTargetHint::PREFER_CPU => compute_f16_cpu(&geom, input, weight, bias, output),
        AiTargetHint::PREFER_A100 => compute_f16_a100(&geom, input, weight, bias, output),
        AiTargetHint::PREFER_X100 => Err(BackendErr::UnsupportedOp),
        _ => unreachable!("CallContext normalizes and validates target hints"),
    }
}

/// 计算 padding 后感受野元素的输入索引；padding 区域返回 `None`。
#[inline]
fn input_index(
    geom: &Conv2dGeom,
    batch: usize,
    channel: usize,
    out_y: usize,
    out_x: usize,
    kernel_y: usize,
    kernel_x: usize,
) -> Option<usize> {
    let input_y = out_y * geom.sh + kernel_y * geom.dh;
    let input_x = out_x * geom.sw + kernel_x * geom.dw;
    if input_y < geom.ph || input_x < geom.pw {
        return None;
    }
    let input_y = input_y - geom.ph;
    let input_x = input_x - geom.pw;
    (input_y < geom.ih && input_x < geom.iw)
        .then_some(((batch * geom.cin + channel) * geom.ih + input_y) * geom.iw + input_x)
}

/// 校验输入、权重、输出与可选 bias 的最小长度。
fn validate_lengths(
    geom: &Conv2dGeom,
    input_len: usize,
    weight_len: usize,
    output_len: usize,
    bias_len: Option<usize>,
) -> Result<(), BackendErr> {
    let input_need = geom
        .batch
        .checked_mul(geom.cin)
        .and_then(|value| value.checked_mul(geom.ih))
        .and_then(|value| value.checked_mul(geom.iw))
        .ok_or(BackendErr::InvalidTensor)?;
    let weight_need = geom
        .cout
        .checked_mul(geom.patch_size())
        .ok_or(BackendErr::InvalidTensor)?;
    let output_need = geom
        .batch
        .checked_mul(geom.cout)
        .and_then(|value| value.checked_mul(geom.spatial()))
        .ok_or(BackendErr::InvalidTensor)?;
    if input_len < input_need
        || weight_len < weight_need
        || output_len < output_need
        || bias_len.is_some_and(|len| len < geom.cout)
    {
        Err(BackendErr::InvalidTensor)
    } else {
        Ok(())
    }
}

/// FP16 CPU 参考卷积；乘加使用 F32 累加后写回 F16。
fn compute_f16_cpu(
    geom: &Conv2dGeom,
    input: &[u16],
    weight: &[u16],
    bias: Option<&[u16]>,
    output: &mut [u16],
) -> Result<(), BackendErr> {
    let patch = geom.patch_size();
    let cin_group = geom.cin_per_group();
    let cout_group = geom.cout_per_group();
    for batch in 0..geom.batch {
        for out_channel in 0..geom.cout {
            let group = out_channel / cout_group;
            let weight_base = out_channel * patch;
            for out_y in 0..geom.oh {
                for out_x in 0..geom.ow {
                    let mut sum =
                        bias.map_or(0.0, |values| f16::from_bits(values[out_channel]).to_f32());
                    for in_channel in 0..cin_group {
                        for kernel_y in 0..geom.kh {
                            for kernel_x in 0..geom.kw {
                                if let Some(index) = input_index(
                                    geom,
                                    batch,
                                    group * cin_group + in_channel,
                                    out_y,
                                    out_x,
                                    kernel_y,
                                    kernel_x,
                                ) {
                                    let weight_index = weight_base
                                        + (in_channel * geom.kh + kernel_y) * geom.kw
                                        + kernel_x;
                                    sum += f16::from_bits(weight[weight_index]).to_f32()
                                        * f16::from_bits(input[index]).to_f32();
                                }
                            }
                        }
                    }
                    output
                        [((batch * geom.cout + out_channel) * geom.oh + out_y) * geom.ow + out_x] =
                        f16::from_f32(sum).to_bits();
                }
            }
        }
    }
    Ok(())
}

/// FP16 A100 卷积：每个 group 生成临时 im2col，再使用 A100 MatMul 路径。
fn compute_f16_a100(
    geom: &Conv2dGeom,
    input: &[u16],
    weight: &[u16],
    bias: Option<&[u16]>,
    output: &mut [u16],
) -> Result<(), BackendErr> {
    let patch = geom.patch_size();
    let spatial = geom.spatial();
    let out_per_group = geom.cout_per_group();
    let in_per_group = geom.cin_per_group();
    let matmul_attr = compact_matmul_attr(out_per_group, spatial, patch);
    for batch in 0..geom.batch {
        for group in 0..geom.groups {
            let mut col = vec![0_u16; patch * spatial];
            let mut patch_index = 0;
            for in_channel in 0..in_per_group {
                for kernel_y in 0..geom.kh {
                    for kernel_x in 0..geom.kw {
                        for out_y in 0..geom.oh {
                            for out_x in 0..geom.ow {
                                let spatial_index = out_y * geom.ow + out_x;
                                col[patch_index * spatial + spatial_index] = input_index(
                                    geom,
                                    batch,
                                    group * in_per_group + in_channel,
                                    out_y,
                                    out_x,
                                    kernel_y,
                                    kernel_x,
                                )
                                .map_or(0, |index| input[index]);
                            }
                        }
                        patch_index += 1;
                    }
                }
            }
            let output_base = batch * geom.cout * spatial + group * out_per_group * spatial;
            let weight_base = group * out_per_group * patch;
            matmul::compute_f16_a100(F16Matmul {
                lhs: &weight[weight_base..weight_base + out_per_group * patch],
                rhs: &col,
                output: &mut output[output_base..output_base + out_per_group * spatial],
                attr: matmul_attr,
            })?;
            add_bias(output, output_base, out_per_group, spatial, bias, group);
        }
    }
    Ok(())
}

/// 将可选 FP16 bias 加到 group 的卷积输出。
fn add_bias(
    output: &mut [u16],
    output_base: usize,
    out_per_group: usize,
    spatial: usize,
    bias: Option<&[u16]>,
    group: usize,
) {
    if let Some(bias) = bias {
        for out_channel in 0..out_per_group {
            let bias_value = f16::from_bits(bias[group * out_per_group + out_channel]).to_f32();
            for spatial_index in 0..spatial {
                let index = output_base + out_channel * spatial + spatial_index;
                output[index] =
                    f16::from_f32(f16::from_bits(output[index]).to_f32() + bias_value).to_bits();
            }
        }
    }
}

/// 构造 im2col MatMul 使用的紧凑单 batch属性。
fn compact_matmul_attr(m: usize, n: usize, k: usize) -> MatMulAttr {
    MatMulAttr {
        m: DimSize::new(m as u32),
        n: DimSize::new(n as u32),
        k: DimSize::new(k as u32),
        batch: DimSize::new(1),
        lhs_row_stride: ElemStride::new(k as u32),
        lhs_col_stride: ElemStride::new(1),
        lhs_batch_stride: ElemStride::new(0),
        rhs_row_stride: ElemStride::new(n as u32),
        rhs_col_stride: ElemStride::new(1),
        rhs_batch_stride: ElemStride::new(0),
        out_row_stride: ElemStride::new(n as u32),
        out_col_stride: ElemStride::new(1),
        out_batch_stride: ElemStride::new(0),
        flags: OpFlags::new(0),
        accum_dtype: AiDtype::F32,
        reserved: [0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_ai_uabi::KernelStride;

    /// 构造单 batch、groups=1 的紧凑卷积属性。
    fn attr() -> Conv2dAttr {
        Conv2dAttr {
            batch: DimSize::new(1),
            in_channels: DimSize::new(1),
            out_channels: DimSize::new(1),
            input_h: DimSize::new(3),
            input_w: DimSize::new(3),
            output_h: DimSize::new(2),
            output_w: DimSize::new(2),
            kernel_h: DimSize::new(2),
            kernel_w: DimSize::new(2),
            stride_h: KernelStride::new(1),
            stride_w: KernelStride::new(1),
            pad_h: DimSize::new(0),
            pad_w: DimSize::new(0),
            dilation_h: KernelStride::new(1),
            dilation_w: KernelStride::new(1),
            groups: DimSize::new(1),
            flags: OpFlags::new(0),
            reserved: [0; 15],
        }
    }

    /// CPU 和 A100 FP16 路径的卷积结果必须一致。
    #[test]
    fn a100_f16_matches_cpu() {
        let geom = Conv2dGeom::from_attr(&attr()).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .map(|value| f16::from_f32(value).to_bits());
        let weight = [1.0_f32; 4].map(|value| f16::from_f32(value).to_bits());
        let mut cpu = [0_u16; 4];
        let mut a100 = [0_u16; 4];
        compute_f16_cpu(&geom, &input, &weight, None, &mut cpu).unwrap();
        compute_f16_a100(&geom, &input, &weight, None, &mut a100).unwrap();
        assert_eq!(cpu, a100);
    }
}
