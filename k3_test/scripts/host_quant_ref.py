#!/usr/bin/env python3
"""Host reference for k3_test/src/test.rs quantized ggml cases."""

import json
import struct


QUANT_K = 32
Q3_K = 256
Q_TEST_M = 2
Q_TEST_N = 3
F16_ONE_LE = struct.pack("<H", 0x3C00)
IQ4_NL_VALUES = [
    -127,
    -104,
    -83,
    -65,
    -49,
    -35,
    -22,
    -10,
    1,
    13,
    25,
    38,
    53,
    69,
    89,
    113,
]


def quant_pattern_q8(row):
    out = []
    for idx in range(QUANT_K):
        if row == 0:
            out.append(float(idx - 16))
        elif row == 1:
            out.append(float(16 - idx))
        else:
            out.append(float(idx % 7 - 3))
    return out


def quant_pattern_iq4(row):
    out = []
    for idx in range(QUANT_K):
        if row == 0:
            table_idx = idx % len(IQ4_NL_VALUES)
        else:
            table_idx = (len(IQ4_NL_VALUES) - 1) - (idx % len(IQ4_NL_VALUES))
        out.append(float(IQ4_NL_VALUES[table_idx]))
    return out


def quant_pattern_q3(row):
    out = []
    for idx in range(Q3_K):
        if row == 0:
            centered = idx % 8 - 4
        else:
            centered = 3 - idx % 8
        out.append(float(-32 * centered))
    return out


def rhs_pattern(k_len, n_col):
    return [
        [float((k + 2 * n) % 11 - 5) for n in range(n_col)]
        for k in range(k_len)
    ]


def pack_q8_0_rows(rows):
    data = bytearray()
    for row in rows:
        data += F16_ONE_LE
        data += bytes((int(v) & 0xFF) for v in row)
    return bytes(data)


def dequant_q8_0_rows(data, rows):
    out = []
    for row in range(rows):
        base = row * 34
        d = half_to_float(data[base : base + 2])
        out.append([i8(data[base + 2 + idx]) * d for idx in range(QUANT_K)])
    return out


def pack_iq4_nl_rows(rows):
    data = bytearray()
    for row in rows:
        data += F16_ONE_LE
        for lane in range(16):
            lo = IQ4_NL_VALUES.index(int(row[lane]))
            hi = IQ4_NL_VALUES.index(int(row[lane + 16]))
            data.append(lo | (hi << 4))
    return bytes(data)


def dequant_iq4_nl_rows(data, rows):
    out = []
    for row in range(rows):
        base = row * 18
        d = half_to_float(data[base : base + 2])
        values = []
        for lane in range(32):
            packed = data[base + 2 + lane % 16]
            q = (packed & 0x0F) if lane < 16 else (packed >> 4)
            values.append(float(IQ4_NL_VALUES[q]) * d)
        out.append(values)
    return out


def pack_q3_k_rows(rows):
    data = bytearray(len(rows) * 110)
    for row_idx, row in enumerate(rows):
        base = row_idx * 110
        data[base + 108 : base + 110] = F16_ONE_LE
        for idx, value in enumerate(row):
            centered = int(-value / 32.0)
            if centered >= 0:
                low, high = centered, True
            else:
                low, high = centered + 4, False
            half = idx // 128
            within = idx % 128
            group = within // 32
            lane = within % 32
            second = lane >= 16
            q_index = half * 32 + (16 if second else 0) + lane % 16
            shift = group * 2
            data[base + 32 + q_index] |= (low & 3) << shift
            if high:
                data[base + q_index % 32] |= 1 << (group + half * 4)
    return bytes(data)


def dequant_q3_k_rows(data, rows):
    out = []
    for row in range(rows):
        base = row * 110
        block = data[base : base + 110]
        hmask = block[0:32]
        qs = block[32:96]
        scale_bytes = block[96:108]
        d_all = half_to_float(block[108:110])
        aux0 = u32_le(scale_bytes[0:4])
        aux1 = u32_le(scale_bytes[4:8])
        aux2 = u32_le(scale_bytes[8:12])
        kmask1 = 0x03030303
        kmask2 = 0x0F0F0F0F
        tmp = aux2
        aux = [0, 0, 0, 0]
        aux[2] = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
        scales = []
        for word in aux:
            for byte in struct.pack("<I", word):
                scales.append(i8(byte))

        values = []
        for idx in range(Q3_K):
            half = idx // 128
            within = idx % 128
            group = within // 32
            lane = within % 32
            second = lane >= 16
            q_index = half * 32 + (16 if second else 0) + lane % 16
            shift = group * 2
            mask = 1 << (group + half * 4)
            scale_index = half * 8 + group * 2 + (1 if second else 0)
            q = (qs[q_index] >> shift) & 3
            high = (hmask[q_index % 32] & mask) != 0
            centered = q - (0 if high else 4)
            scale = scales[scale_index] - 32
            values.append(d_all * float(scale) * float(centered))
        out.append(values)
    return out


def matmul_ref(lhs_rows, rhs):
    out = []
    for m in range(Q_TEST_M):
        for n in range(len(rhs[0])):
            acc = 0.0
            for k in range(len(rhs)):
                acc += lhs_rows[m][k] * rhs[k][n]
            out.append(acc)
    return out


def get_rows_ref(rows, selected):
    out = []
    for col in range(QUANT_K):
        for row in selected:
            out.append(rows[row][col])
    return out


def half_to_float(raw):
    return struct.unpack("<e", raw)[0]


def u32_le(raw):
    return struct.unpack("<I", raw)[0]


def i8(value):
    return value - 256 if value >= 128 else value


def emit(case, values):
    rounded = [round(float(value), 6) for value in values]
    print(json.dumps({"source": "HOST_REF", "case": case, "values": rounded}))


def main():
    q8_get_rows_src = [quant_pattern_q8(0), quant_pattern_q8(1), quant_pattern_q8(2)]
    q8_get_rows = dequant_q8_0_rows(pack_q8_0_rows(q8_get_rows_src), 3)
    emit("quant/q8_0_get_rows", get_rows_ref(q8_get_rows, [2, 0]))

    q8_rows_src = [quant_pattern_q8(0), quant_pattern_q8(1)]
    q8_rows = dequant_q8_0_rows(pack_q8_0_rows(q8_rows_src), Q_TEST_M)
    emit("quant/q8_0_matmul", matmul_ref(q8_rows, rhs_pattern(QUANT_K, Q_TEST_N)))

    iq4_rows_src = [quant_pattern_iq4(0), quant_pattern_iq4(1)]
    iq4_rows = dequant_iq4_nl_rows(pack_iq4_nl_rows(iq4_rows_src), Q_TEST_M)
    emit("quant/iq4_nl_matmul", matmul_ref(iq4_rows, rhs_pattern(QUANT_K, Q_TEST_N)))

    q3_rows_src = [quant_pattern_q3(0), quant_pattern_q3(1)]
    q3_rows = dequant_q3_k_rows(pack_q3_k_rows(q3_rows_src), Q_TEST_M)
    emit("quant/q3_k_matmul", matmul_ref(q3_rows, rhs_pattern(Q3_K, 2)))


if __name__ == "__main__":
    main()
