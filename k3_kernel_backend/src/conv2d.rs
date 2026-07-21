//! 二维卷积算子。
//!
//! 实现策略：im2col 展开 + matmul。权重视作 `[Cout, Cin*KH*KW]` 矩阵，im2col
//! 把输入每个输出点的感受野展开成 `[Cin*KH*KW, Hout*Wout]` 矩阵，二者相乘得
//! `[Cout, Hout*Wout]` 输出。
//!
//! 计算落点：matmul 部分复用 [`crate::matmul`] 引擎——int8 走 IME vmadot
//! （x100/a100），f32 走软件 matmul。f16 默认解码到 f32 在 cpu 直算；启用
//! `a100-fp16-ime` feature 后，A100 使用流式 im2col + `smt.vfwmadot`。
//!
//! 张量约定：inputs[0]=input(NCHW)、inputs[1]=weight([Cout,Cin/groups,KH,KW] 行主序)，
//! inputs[2]=bias([Cout]) 可选，outputs[0]=output(NCHW)。

use crate::BackendCall;
use crate::call::CallContext;
#[cfg(feature = "a100-fp16-ime")]
use crate::matmul::{Fp16MatmulOutput, Fp16MatmulParameter, ime_f16_f32_matmul};
use crate::matmul::{Int8MatmulParameter, a100_int8_tile, ime_int8_i32_matmul};
use alloc::vec;
use half::f16;
use k3_ai_uabi::error::BackendErr;
use k3_ai_uabi::{AiDtype, AiTargetHint, Conv2dAttr, DimSize, ElemStride, MatMulAttr, OpFlags};
use log::error;

/// 从 `Conv2dAttr` 抽出的以 `usize` 表示的卷积几何参数。
struct Conv2dGeom {
    /// batch 数（0 归一化为 1）。
    batch: usize,
    /// 输入通道数。
    cin: usize,
    /// 输出通道数。
    cout: usize,
    /// 输入高。
    ih: usize,
    /// 输入宽。
    iw: usize,
    /// 输出高。
    oh: usize,
    /// 输出宽。
    ow: usize,
    /// 卷积核高。
    kh: usize,
    /// 卷积核宽。
    kw: usize,
    /// 垂直步长。
    sh: usize,
    /// 水平步长。
    sw: usize,
    /// 垂直 padding。
    ph: usize,
    /// 水平 padding。
    pw: usize,
    /// 垂直膨胀。
    dh: usize,
    /// 水平膨胀。
    dw: usize,
    /// 分组数量。
    groups: usize,
}

impl Conv2dGeom {
    /// 从 attr 构造几何参数，并校验 groups 与输出尺寸自洽。
    fn from_attr(attr: &Conv2dAttr) -> Result<Self, BackendErr> {
        let groups = (attr.groups.get() as usize).max(1);
        let g = Self {
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
        if g.cin == 0
            || g.cout == 0
            || g.ih == 0
            || g.iw == 0
            || g.oh == 0
            || g.ow == 0
            || g.kh == 0
            || g.kw == 0
            || !g.cin.is_multiple_of(groups)
            || !g.cout.is_multiple_of(groups)
        {
            error!(
                "conv2d: invalid geometry cin={}, cout={}, groups={groups}",
                g.cin, g.cout
            );
            return Err(BackendErr::InvalidAttr);
        }
        Ok(g)
    }

    /// 每组输入通道数。
    fn cin_per_group(&self) -> usize {
        self.cin / self.groups
    }

    /// 每组输出通道数。
    fn cout_per_group(&self) -> usize {
        self.cout / self.groups
    }

    /// im2col 矩阵的 K 维（Cin/group*KH*KW）。
    fn patch_size(&self) -> usize {
        self.cin_per_group() * self.kh * self.kw
    }

    /// 每个 batch 的输出空间点数（Hout*Wout）。
    fn spatial(&self) -> usize {
        self.oh * self.ow
    }
}

/// conv2d 算子执行器：解析 `BackendCall`，按 dtype/target 分发。
///
/// # Safety
///
/// `call` 指向有效 `BackendCall`，其 tensor `data` 已映射且生命周期覆盖本次调用。
pub(crate) unsafe fn conv2d_caller(call: *const BackendCall) -> Result<(), BackendErr> {
    let ctx = unsafe { CallContext::from_call(call)? };
    ctx.expect_io_range(2..=3, 1..=1)?;
    ctx.reject_input_output_alias()?;

    let attr = ctx.read_attr::<Conv2dAttr>()?;
    let geom = Conv2dGeom::from_attr(&attr)?;
    let target = ctx.target;
    let in_dtype = ctx.inputs[0].dtype;
    let w_dtype = ctx.inputs[1].dtype;
    let out_dtype = ctx.outputs[0].dtype;
    let bias_dtype = ctx.inputs.get(2).map(|view| view.dtype);

    match (in_dtype, w_dtype, out_dtype) {
        (AiDtype::F32, AiDtype::F32, AiDtype::F32) => {
            if !bias_dtype.is_none_or(|dtype| dtype == AiDtype::F32) {
                return Err(BackendErr::UnsupportedDtype);
            }
            let input = unsafe { ctx.inputs[0].as_slice::<f32>()? };
            let weight = unsafe { ctx.inputs[1].as_slice::<f32>()? };
            let bias = if ctx.inputs.len() == 3 {
                Some(unsafe { ctx.inputs[2].as_slice::<f32>()? })
            } else {
                None
            };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<f32>()? };
            conv2d_f32(&geom, input, weight, bias, output)
        }
        (AiDtype::F16, AiDtype::F16, AiDtype::F16) => {
            if !bias_dtype.is_none_or(|dtype| dtype == AiDtype::F16) {
                return Err(BackendErr::UnsupportedDtype);
            }
            let input = unsafe { ctx.inputs[0].as_slice::<u16>()? };
            let weight = unsafe { ctx.inputs[1].as_slice::<u16>()? };
            let bias = if ctx.inputs.len() == 3 {
                Some(unsafe { ctx.inputs[2].as_slice::<u16>()? })
            } else {
                None
            };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<u16>()? };
            conv2d_f16(&geom, input, weight, bias, output, target)
        }
        (i_dt, w_dt, AiDtype::I32) if is_int8(i_dt) && is_int8(w_dt) => {
            if !bias_dtype.is_none_or(|dtype| dtype == AiDtype::I32) {
                return Err(BackendErr::UnsupportedDtype);
            }
            let input = unsafe { ctx.inputs[0].as_slice::<u8>()? };
            let weight = unsafe { ctx.inputs[1].as_slice::<u8>()? };
            let bias = if ctx.inputs.len() == 3 {
                Some(unsafe { ctx.inputs[2].as_slice::<i32>()? })
            } else {
                None
            };
            let output = unsafe { ctx.outputs[0].as_mut_slice::<i32>()? };
            conv2d_int8(&geom, input, weight, bias, output, i_dt, w_dt)
        }
        _ => {
            error!(
                "conv2d_caller: unsupported dtype in={:?}, w={:?}, out={:?}",
                in_dtype, w_dtype, out_dtype
            );
            Err(BackendErr::UnsupportedDtype)
        }
    }
}

/// 判断 dtype 是否为 8 位整型（I8/U8）。
fn is_int8(dtype: AiDtype) -> bool {
    dtype == AiDtype::I8 || dtype == AiDtype::U8
}

/// 计算单个输出点的一个感受野元素对应的输入线性下标；越界（padding）返回 None。
#[inline]
fn input_index(
    g: &Conv2dGeom,
    b: usize,
    cin: usize,
    oy: usize,
    ox: usize,
    ky: usize,
    kx: usize,
) -> Option<usize> {
    let iy = oy * g.sh + ky * g.dh;
    let ix = ox * g.sw + kx * g.dw;
    if iy < g.ph || ix < g.pw {
        return None;
    }
    let iy = iy - g.ph;
    let ix = ix - g.pw;
    if iy >= g.ih || ix >= g.iw {
        return None;
    }
    Some(((b * g.cin + cin) * g.ih + iy) * g.iw + ix)
}

/// 校验 NCHW 输入、分组权重、可选 bias 与输出长度，避免后续索引 panic。
fn validate_lengths(
    g: &Conv2dGeom,
    input_len: usize,
    weight_len: usize,
    output_len: usize,
    bias_len: Option<usize>,
) -> Result<(), BackendErr> {
    let input_need = g
        .batch
        .checked_mul(g.cin)
        .and_then(|value| value.checked_mul(g.ih))
        .and_then(|value| value.checked_mul(g.iw))
        .ok_or(BackendErr::InvalidTensor)?;
    let weight_need = g
        .cout
        .checked_mul(g.patch_size())
        .ok_or(BackendErr::InvalidTensor)?;
    let output_need = g
        .batch
        .checked_mul(g.cout)
        .and_then(|value| value.checked_mul(g.spatial()))
        .ok_or(BackendErr::InvalidTensor)?;
    if input_len < input_need
        || weight_len < weight_need
        || output_len < output_need
        || bias_len.is_some_and(|len| len < g.cout)
    {
        return Err(BackendErr::InvalidTensor);
    }
    Ok(())
}

/// f32 直接卷积（cpu 软件参考实现）。
fn conv2d_f32(
    g: &Conv2dGeom,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<(), BackendErr> {
    validate_lengths(
        g,
        input.len(),
        weight.len(),
        output.len(),
        bias.map(<[f32]>::len),
    )?;
    let patch = g.patch_size();
    let cin_group = g.cin_per_group();
    let cout_group = g.cout_per_group();
    for b in 0..g.batch {
        for oc in 0..g.cout {
            let group = oc / cout_group;
            let input_channel_base = group * cin_group;
            let w_base = oc * patch;
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let mut acc = bias.map_or(0.0_f32, |bias| bias[oc]);
                    for ic in 0..cin_group {
                        let input_channel = input_channel_base + ic;
                        for ky in 0..g.kh {
                            for kx in 0..g.kw {
                                let w = weight[w_base + (ic * g.kh + ky) * g.kw + kx];
                                if let Some(idx) = input_index(g, b, input_channel, oy, ox, ky, kx)
                                {
                                    acc += w * input[idx];
                                }
                            }
                        }
                    }
                    output[((b * g.cout + oc) * g.oh + oy) * g.ow + ox] = acc;
                }
            }
        }
    }
    Ok(())
}

/// f16 卷积：默认 CPU 直算；A100 feature 打开后走 IME vfwmadot。
fn conv2d_f16(
    g: &Conv2dGeom,
    input: &[u16],
    weight: &[u16],
    bias: Option<&[u16]>,
    output: &mut [u16],
    target: AiTargetHint,
) -> Result<(), BackendErr> {
    match target {
        AiTargetHint::AUTO | AiTargetHint::PREFER_CPU => {
            conv2d_f16_cpu(g, input, weight, bias, output)
        }
        AiTargetHint::PREFER_A100 => {
            #[cfg(feature = "a100-fp16-ime")]
            {
                conv2d_f16_ime(g, input, weight, bias, output)
            }
            #[cfg(not(feature = "a100-fp16-ime"))]
            {
                let _ = (g, input, weight, bias, output);
                error!("conv2d_f16: A100 FP16 IME disabled until MCPM.BF16 is controlled");
                Err(BackendErr::UnsupportedOp)
            }
        }
        AiTargetHint::PREFER_X100 => Err(BackendErr::UnsupportedDtype),
        _ => unreachable!("CallContext rejects unknown targets"),
    }
}

/// f16 直接卷积：解码到 f32 累加后写回 f16（CPU 参考实现）。
fn conv2d_f16_cpu(
    g: &Conv2dGeom,
    input: &[u16],
    weight: &[u16],
    bias: Option<&[u16]>,
    output: &mut [u16],
) -> Result<(), BackendErr> {
    validate_lengths(
        g,
        input.len(),
        weight.len(),
        output.len(),
        bias.map(<[u16]>::len),
    )?;
    let patch = g.patch_size();
    let cin_group = g.cin_per_group();
    let cout_group = g.cout_per_group();
    for b in 0..g.batch {
        for oc in 0..g.cout {
            let group = oc / cout_group;
            let input_channel_base = group * cin_group;
            let w_base = oc * patch;
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let mut acc = bias.map_or(0.0_f32, |bias| f16::from_bits(bias[oc]).to_f32());
                    for ic in 0..cin_group {
                        let input_channel = input_channel_base + ic;
                        for ky in 0..g.kh {
                            for kx in 0..g.kw {
                                let w =
                                    f16::from_bits(weight[w_base + (ic * g.kh + ky) * g.kw + kx])
                                        .to_f32();
                                if let Some(idx) = input_index(g, b, input_channel, oy, ox, ky, kx)
                                {
                                    acc += w * f16::from_bits(input[idx]).to_f32();
                                }
                            }
                        }
                    }
                    output[((b * g.cout + oc) * g.oh + oy) * g.ow + ox] =
                        f16::from_f32(acc).to_bits();
                }
            }
        }
    }
    Ok(())
}

/// f16 卷积：流式 im2col + A100 8×8×8 vfwmadot matmul。
#[cfg(feature = "a100-fp16-ime")]
fn conv2d_f16_ime(
    g: &Conv2dGeom,
    input: &[u16],
    weight: &[u16],
    bias: Option<&[u16]>,
    output: &mut [u16],
) -> Result<(), BackendErr> {
    validate_lengths(
        g,
        input.len(),
        weight.len(),
        output.len(),
        bias.map(<[u16]>::len),
    )?;
    let k = g.patch_size();
    let n = g.spatial();
    let m = g.cout_per_group();
    let cin_group = g.cin_per_group();
    let mm_attr = conv_fp16_matmul_attr(m, n, k);

    for b in 0..g.batch {
        for group in 0..g.groups {
            let mut col = vec![0_u16; k * n];
            let mut p = 0;
            let input_channel_base = group * cin_group;
            for ic in 0..cin_group {
                let input_channel = input_channel_base + ic;
                for ky in 0..g.kh {
                    for kx in 0..g.kw {
                        for oy in 0..g.oh {
                            for ox in 0..g.ow {
                                let s = oy * g.ow + ox;
                                col[p * n + s] =
                                    match input_index(g, b, input_channel, oy, ox, ky, kx) {
                                        Some(idx) => input[idx],
                                        None => 0,
                                    };
                            }
                        }
                        p += 1;
                    }
                }
            }

            let out_base = b * g.cout * n + group * m * n;
            let weight_base = group * m * k;
            let parameter = Fp16MatmulParameter {
                lhs: &weight[weight_base..weight_base + m * k],
                rhs: &col,
                output: Fp16MatmulOutput::F16(&mut output[out_base..out_base + m * n]),
                attr: mm_attr,
            };
            ime_f16_f32_matmul(parameter)?;
            if let Some(bias) = bias {
                for oc in 0..m {
                    let bias_value = f16::from_bits(bias[group * m + oc]).to_f32();
                    for spatial in 0..n {
                        let output_index = out_base + oc * n + spatial;
                        let value = f16::from_bits(output[output_index]).to_f32() + bias_value;
                        output[output_index] = f16::from_f32(value).to_bits();
                    }
                }
            }
        }
    }
    Ok(())
}

/// int8 卷积：im2col 展开为 `[K, N]` col 矩阵，与权重 `[Cout, K]` 相乘复用 IME。
///
/// 复用 [`ime_int8_i32_matmul`]：lhs=weight(M=Cout, K=patch), rhs=col(K, N=spatial)，
/// 行主序无转置，out=[Cout, spatial]。IME tile 逻辑在 x86 测试下走软件参考。
fn conv2d_int8(
    g: &Conv2dGeom,
    input: &[u8],
    weight: &[u8],
    bias: Option<&[i32]>,
    output: &mut [i32],
    in_dtype: AiDtype,
    w_dtype: AiDtype,
) -> Result<(), BackendErr> {
    validate_lengths(
        g,
        input.len(),
        weight.len(),
        output.len(),
        bias.map(<[i32]>::len),
    )?;
    let k = g.patch_size();
    let n = g.spatial();
    let m = g.cout_per_group();
    let cin_group = g.cin_per_group();
    let signedness = crate::matmul::signedness_for(w_dtype, in_dtype);
    let mm_attr = conv_matmul_attr(m, n, k);

    for b in 0..g.batch {
        for group in 0..g.groups {
            // im2col：col[p, s]，p 遍历本 group 内 (ic,ky,kx)，s 遍历 (oy,ox)。
            let mut col = vec![0_u8; k * n];
            let mut p = 0;
            let input_channel_base = group * cin_group;
            for ic in 0..cin_group {
                let input_channel = input_channel_base + ic;
                for ky in 0..g.kh {
                    for kx in 0..g.kw {
                        for oy in 0..g.oh {
                            for ox in 0..g.ow {
                                let s = oy * g.ow + ox;
                                col[p * n + s] =
                                    match input_index(g, b, input_channel, oy, ox, ky, kx) {
                                        Some(idx) => input[idx],
                                        None => 0,
                                    };
                            }
                        }
                        p += 1;
                    }
                }
            }

            let out_base = b * g.cout * n + group * m * n;
            let weight_base = group * m * k;
            let parameter = Int8MatmulParameter {
                lhs: &weight[weight_base..weight_base + m * k],
                rhs: &col,
                out: &mut output[out_base..out_base + m * n],
                attr: mm_attr,
                signedness,
            };
            ime_int8_i32_matmul(parameter, a100_int8_tile())?;
            if let Some(bias) = bias {
                for oc in 0..m {
                    let bias_value = bias[group * m + oc];
                    for spatial in 0..n {
                        output[out_base + oc * n + spatial] += bias_value;
                    }
                }
            }
        }
    }
    let _ = in_dtype;
    Ok(())
}

/// 构造 conv im2col 用的行主序、无转置、单 batch `MatMulAttr`。
fn conv_matmul_attr(m: usize, n: usize, k: usize) -> MatMulAttr {
    MatMulAttr {
        m: DimSize::new(m as u32),
        n: DimSize::new(n as u32),
        k: DimSize::new(k as u32),
        batch: DimSize::new(0),
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
        accum_dtype: AiDtype::I32,
        reserved: [0; 3],
    }
}

/// 构造 conv f16 im2col 用的行主序、无转置、单 batch `MatMulAttr`。
#[cfg(feature = "a100-fp16-ime")]
fn conv_fp16_matmul_attr(m: usize, n: usize, k: usize) -> MatMulAttr {
    let mut attr = conv_matmul_attr(m, n, k);
    attr.accum_dtype = AiDtype::F32;
    attr
}

/// conv2d 各 dtype 路径的正确性单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use k3_ai_uabi::KernelStride;

    /// 构造单 batch、groups=1 的 `Conv2dAttr`。
    #[allow(clippy::too_many_arguments)]
    fn attr(
        cin: u32,
        cout: u32,
        ih: u32,
        iw: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad: u32,
    ) -> Conv2dAttr {
        let oh = (ih + 2 * pad - kh) / stride + 1;
        let ow = (iw + 2 * pad - kw) / stride + 1;
        Conv2dAttr {
            batch: DimSize::new(1),
            in_channels: DimSize::new(cin),
            out_channels: DimSize::new(cout),
            input_h: DimSize::new(ih),
            input_w: DimSize::new(iw),
            output_h: DimSize::new(oh),
            output_w: DimSize::new(ow),
            kernel_h: DimSize::new(kh),
            kernel_w: DimSize::new(kw),
            stride_h: KernelStride::new(stride),
            stride_w: KernelStride::new(stride),
            pad_h: DimSize::new(pad),
            pad_w: DimSize::new(pad),
            dilation_h: KernelStride::new(1),
            dilation_w: KernelStride::new(1),
            groups: DimSize::new(1),
            flags: OpFlags::new(0),
            reserved: [0; 15],
        }
    }

    /// 1ch 3×3 输入、2×2 全 1 卷积核、无 pad、stride1 → 2×2 局部和。
    #[test]
    fn f32_single_channel_sum_kernel() {
        let g = Conv2dGeom::from_attr(&attr(1, 1, 3, 3, 2, 2, 1, 0)).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0_f32; 4];
        let mut out = [0.0_f32; 4];
        conv2d_f32(&g, &input, &weight, None, &mut out).unwrap();
        // 每个 2×2 窗口求和。
        assert_eq!(out, [12.0, 16.0, 24.0, 28.0]);
    }

    /// depthwise(groups=cin=cout) + bias 应只读取各自 channel。
    #[test]
    fn f32_depthwise_with_bias() {
        let mut a = attr(2, 2, 2, 2, 1, 1, 1, 0);
        a.groups = DimSize::new(2);
        let g = Conv2dGeom::from_attr(&a).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = [10.0_f32, 20.0];
        let bias = [1.0_f32, -1.0];
        let mut out = [0.0_f32; 8];
        conv2d_f32(&g, &input, &weight, Some(&bias), &mut out).unwrap();
        assert_eq!(out, [11.0, 21.0, 31.0, 41.0, 99.0, 119.0, 139.0, 159.0]);
    }

    /// 独立的 int8 直接卷积参考实现。
    fn ref_int8(g: &Conv2dGeom, input: &[u8], weight: &[u8], bias: Option<&[i32]>) -> Vec<i32> {
        let patch = g.patch_size();
        let cin_group = g.cin_per_group();
        let cout_group = g.cout_per_group();
        let mut out = vec![0_i32; g.batch * g.cout * g.spatial()];
        for b in 0..g.batch {
            for oc in 0..g.cout {
                let group = oc / cout_group;
                let input_channel_base = group * cin_group;
                for oy in 0..g.oh {
                    for ox in 0..g.ow {
                        let mut acc = bias.map_or(0_i32, |bias| bias[oc]);
                        for ic in 0..cin_group {
                            let input_channel = input_channel_base + ic;
                            for ky in 0..g.kh {
                                for kx in 0..g.kw {
                                    let w = weight[oc * patch + (ic * g.kh + ky) * g.kw + kx] as i8
                                        as i32;
                                    if let Some(idx) =
                                        input_index(g, b, input_channel, oy, ox, ky, kx)
                                    {
                                        acc += w * (input[idx] as i8 as i32);
                                    }
                                }
                            }
                        }
                        out[((b * g.cout + oc) * g.oh + oy) * g.ow + ox] = acc;
                    }
                }
            }
        }
        out
    }

    /// int8 im2col+IME 路径应与独立直接卷积参考一致（含 padding、多通道）。
    #[test]
    fn int8_im2col_matches_reference() {
        let a = attr(3, 4, 5, 5, 3, 3, 2, 1);
        let g = Conv2dGeom::from_attr(&a).unwrap();
        let input: Vec<u8> = (0..3 * 5 * 5).map(|v| (v as i8 - 30) as u8).collect();
        let weight: Vec<u8> = (0..4 * 3 * 3 * 3).map(|v| (v as i8 - 15) as u8).collect();
        let mut out = vec![0_i32; g.cout * g.spatial()];
        conv2d_int8(
            &g,
            &input,
            &weight,
            None,
            &mut out,
            AiDtype::I8,
            AiDtype::I8,
        )
        .unwrap();
        assert_eq!(out, ref_int8(&g, &input, &weight, None));
    }

    /// grouped int8 im2col 只应提交每组自己的输入/输出通道，并在 matmul 后加 bias。
    #[test]
    fn int8_grouped_with_bias_matches_reference() {
        let mut a = attr(4, 4, 3, 3, 1, 1, 1, 0);
        a.groups = DimSize::new(2);
        let g = Conv2dGeom::from_attr(&a).unwrap();
        let input: Vec<u8> = (0..4 * 3 * 3).map(|v| (v as i8 - 18) as u8).collect();
        let weight: Vec<u8> = (0..4 * 2).map(|v| (v as i8 - 3) as u8).collect();
        let bias = [1_i32, -2, 3, -4];
        let mut out = vec![0_i32; g.cout * g.spatial()];
        conv2d_int8(
            &g,
            &input,
            &weight,
            Some(&bias),
            &mut out,
            AiDtype::I8,
            AiDtype::I8,
        )
        .unwrap();
        assert_eq!(out, ref_int8(&g, &input, &weight, Some(&bias)));
    }

    /// feature 打开时，A100 FP16 IME 软件镜像应与 CPU grouped+bias 参考一致。
    #[cfg(feature = "a100-fp16-ime")]
    #[test]
    fn f16_ime_grouped_bias_matches_cpu() {
        let mut a = attr(2, 2, 2, 2, 1, 1, 1, 0);
        a.groups = DimSize::new(2);
        let g = Conv2dGeom::from_attr(&a).unwrap();
        let input: Vec<u16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(|value| f16::from_f32(value).to_bits())
            .collect();
        let weight: Vec<u16> = [10.0_f32, 20.0]
            .into_iter()
            .map(|value| f16::from_f32(value).to_bits())
            .collect();
        let bias: Vec<u16> = [1.0_f32, -1.0]
            .into_iter()
            .map(|value| f16::from_f32(value).to_bits())
            .collect();
        let mut cpu = vec![0_u16; g.cout * g.spatial()];
        let mut ime = vec![0_u16; g.cout * g.spatial()];
        conv2d_f16_cpu(&g, &input, &weight, Some(&bias), &mut cpu).unwrap();
        conv2d_f16_ime(&g, &input, &weight, Some(&bias), &mut ime).unwrap();
        for (&left, &right) in cpu.iter().zip(&ime) {
            let left = f16::from_bits(left).to_f32();
            let right = f16::from_bits(right).to_f32();
            assert!((left - right).abs() < 1.0e-3, "cpu={left}, ime={right}");
        }
    }
}
