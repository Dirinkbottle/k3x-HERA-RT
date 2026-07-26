//! ggml block quant helpers used by K3-native GGML operators.

use crate::BackendTensorView;
use crate::call::QuantTensorMeta;
use half::f16;
use k3_ai_uabi::AiDtype;
use k3_ai_uabi::error::BackendErr;

/// IQ4_NL nonlinear lookup table from ggml-common.h.
const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Read one logical scalar from a ggml quant tensor.
pub(crate) fn read_quant_f32(
    view: &BackendTensorView,
    meta: &QuantTensorMeta,
    coordinates: &[usize; k3_ai_uabi::MAX_DIM],
) -> Result<f32, BackendErr> {
    let (block_offset, intra) = meta.block_offset_for_coordinates(coordinates)?;
    let raw = unsafe { view.as_slice::<u8>()? };
    let block = raw
        .get(block_offset..block_offset + meta.block_bytes)
        .ok_or(BackendErr::InvalidTensor)?;
    match view.dtype {
        AiDtype::Q8_0 => read_q8_0(block, intra),
        AiDtype::IQ4_NL => read_iq4_nl(block, intra),
        AiDtype::Q3_K => read_q3_k(block, intra),
        _ => Err(BackendErr::UnsupportedDtype),
    }
}

/// Dequantize one logical row whose varying dimension is axis 0.
pub(crate) fn dequantize_axis0_row(
    view: &BackendTensorView,
    meta: &QuantTensorMeta,
    suffix_coordinates: &[usize; k3_ai_uabi::MAX_DIM],
    output: &mut [f32],
) -> Result<(), BackendErr> {
    if output.len() != meta.shape[0] {
        return Err(BackendErr::InvalidTensor);
    }
    let mut coordinates = *suffix_coordinates;
    for (index, value) in output.iter_mut().enumerate() {
        coordinates[0] = index;
        *value = read_quant_f32(view, meta, &coordinates)?;
    }
    Ok(())
}

/// Read native ggml half as f32.
fn read_f16(bytes: &[u8]) -> Result<f32, BackendErr> {
    let raw = bytes.get(0..2).ok_or(BackendErr::InvalidTensor)?;
    Ok(f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32())
}

/// Read a Q8_0 scalar.
fn read_q8_0(block: &[u8], intra: usize) -> Result<f32, BackendErr> {
    if intra >= 32 || block.len() < 34 {
        return Err(BackendErr::InvalidTensor);
    }
    let d = read_f16(block)?;
    Ok((block[2 + intra] as i8 as f32) * d)
}

/// Read an IQ4_NL scalar.
fn read_iq4_nl(block: &[u8], intra: usize) -> Result<f32, BackendErr> {
    if intra >= 32 || block.len() < 18 {
        return Err(BackendErr::InvalidTensor);
    }
    let d = read_f16(block)?;
    let packed = block[2 + intra % 16];
    let q = if intra < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(d * IQ4_NL_VALUES[q as usize] as f32)
}

/// Read a Q3_K scalar.
fn read_q3_k(block: &[u8], intra: usize) -> Result<f32, BackendErr> {
    if intra >= 256 || block.len() < 110 {
        return Err(BackendErr::InvalidTensor);
    }
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scale_bytes = &block[96..108];
    let d_all = read_f16(&block[108..110])?;

    let mut aux = [
        read_u32_le(&scale_bytes[0..4])?,
        read_u32_le(&scale_bytes[4..8])?,
        read_u32_le(&scale_bytes[8..12])?,
        0,
    ];
    let kmask1 = 0x0303_0303_u32;
    let kmask2 = 0x0f0f_0f0f_u32;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    let mut scales = [0_i8; 16];
    for (word_index, word) in aux.iter().enumerate() {
        for (byte_index, byte) in word.to_le_bytes().iter().enumerate() {
            scales[word_index * 4 + byte_index] = *byte as i8;
        }
    }

    let half = intra / 128;
    let within = intra % 128;
    let group = within / 32;
    let lane = within % 32;
    let second = lane >= 16;
    let q_index = half * 32 + if second { 16 } else { 0 } + lane % 16;
    let shift = group * 2;
    let mask = 1_u8 << (group + half * 4);
    let scale_index = half * 8 + group * 2 + usize::from(second);
    let q = ((qs[q_index] >> shift) & 3) as i8;
    // hmask 只有 32 字节，half=0 和 half=1 共享同一组 byte，
    // 靠 mask 中的 bit 位区分是上半还是下半。
    let high = hmask[q_index % 32] & mask != 0;
    let centered = q - if high { 0 } else { 4 };
    let scale = scales[scale_index] - 32;
    Ok(d_all * scale as f32 * centered as f32)
}

/// Read a little-endian u32 from a four-byte slice.
fn read_u32_le(bytes: &[u8]) -> Result<u32, BackendErr> {
    if bytes.len() != 4 {
        return Err(BackendErr::InvalidTensor);
    }
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| BackendErr::InvalidTensor)?,
    ))
}
