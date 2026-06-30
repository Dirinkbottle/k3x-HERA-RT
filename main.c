// SpacemiT IME (Intrinsic Matrix Engine) — simple matmul using smt.vmadot i8
//
// C = A × B^T   (Int8 × Int8 → Int32)
//
// The hardware on this A100 silicon uses tile 4×8×4 (same as A60):
//   A[4,8]  ∈ int8   — row-major,       32 elements in lower 1/4 of vreg
//   B[4,8]  ∈ int8   — column-major,     32 elements in lower 1/4 of vreg
//   C[4,4]  ∈ int32  — row-major,        16 elements in lower 1/2 of v16
//
// VLEN = 1024 → vle8.v loads 128 bytes.  Only the first 32 are used by smt.vmadot.
// MUL_C = 2 → vse32.v stores 64 int32.  Only the first 16 hold valid results.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define VREG_BYTES 128   // VLEN=1024 → 128 bytes per vreg
#define M 4
#define K 8
#define N 4              // tile = M×K×N = 4×8×4
#define C_ELEMS (M * N)  // 16 output elements

// Reference implementation matching the hardware pseudocode (Section 5.1.3)
static void ref_matmul(const int8_t *A, const int8_t *B, int32_t *C) {
    for (int p = 0; p < C_ELEMS; p++) {
        int i = (p / M) * K;
        int j = (p % N) * K;
        for (int q = 0; q < K; q++)
            C[p] += (int32_t)A[i + q] * (int32_t)B[j + q];
    }
}

static void matmul_tile(const int8_t *restrict A,
                        const int8_t *restrict B,
                        int32_t *restrict C) {
    __asm__ volatile(
        // Load A, B — each vle8.v loads 128 bytes (VLEN=1024).
        // smt.vmadot only reads the first 32 × int8 from each vreg (tile=4×8×4).
        "vsetvli        t0, zero, e8, m1          \n\t"
        "vle8.v         v0, (%[A])                \n\t"
        "vle8.v         v1, (%[B])                \n\t"

        // Zero accumulator v16/v17 pair (MUL_C=2 → 64 int32)
        "vsetvli        t0, zero, e32, m2         \n\t"
        "vmv.v.i        v16, 0                    \n\t"

        // IME matmul: v16[0..15] += A[4×8] × B[4×8]^T
        // v16[16..31] and v17[0..31] stay zero (outside the 4×4 tile)
        "vsetvli        t0, zero, e8, m1          \n\t"
        "smt.vmadot     v16, v0, v1, i8           \n\t"

        // Store all 64 int32 results (v16+v17)
        "vsetvli        t0, zero, e32, m2         \n\t"
        "vse32.v        v16, (%[C])               \n\t"

        : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
        :
        : "t0", "cc", "memory", "v0", "v1", "v16", "v17"
    );
}

int main(void) {
    printf("=== SpacemiT IME smt.vmadot i8 matmul (tile 4×8×4) ===\n\n");

    // ---- A tile: 4 rows × 8 cols, row-major (32 meaningful + 96 pad) ----
    int8_t A[VREG_BYTES];
    memset(A, 0xAB, sizeof(A));  // poison pad bytes
    // clang-format off
    int8_t a_val[32] = { 0, 1, 2,  3,  4,  5,  6,  7,
                         1, 2, 3,  4,  5,  6,  7,  8,
                         2, 3, 4,  5,  6,  7,  8,  9,
                         4, 5, 6,  7,  8,  9, 10, 11 };
    // clang-format on
    memcpy(A, a_val, sizeof(a_val));

    // ---- B tile: 4 rows × 8 cols, column-major (32 meaningful + 96 pad) ----
    int8_t B[VREG_BYTES];
    memset(B, 0xCD, sizeof(B));
    // clang-format off
    int8_t b_val[32] = { 0, 1, 2,  3,  4,  5,  6,  7,
                         1, 2, 3,  4,  5,  6,  7,  8,
                         2, 3, 4,  5,  6,  7,  8,  9,
                        11, 4, 5,  6,  7,  8,  9, 10 };
    // clang-format on
    memcpy(B, b_val, sizeof(b_val));

    // ---- IME result (64 int32 from vse32.v, only first 16 are meaningful) ----
    int32_t C_ime[C_ELEMS * 4];  // 64 slots
    memset(C_ime, 0xFF, sizeof(C_ime));

    // ---- Reference result ----
    int32_t C_ref[C_ELEMS];
    memset(C_ref, 0, sizeof(C_ref));
    ref_matmul(a_val, b_val, C_ref);

    printf("A (4×8, row-major, first 32 of 128 bytes in vreg):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            printf("%4d ", a_val[i * K + j]);
        printf("\n");
    }

    printf("\nB (4×8, column-major):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            printf("%4d ", b_val[i * K + j]);
        printf("\n");
    }

    // ---- Execute IME matmul ----
    matmul_tile(A, B, C_ime);

    printf("\nC = A × B^T  (4×4 tile from 64-element vreg store):\n");
    printf("  %-10s %-10s %s\n", "IME", "Ref", "Match");

    int ok = 1;
    // Results are row-major in the first 16 slots (v16[0..15])
    int32_t expected[16] = {140, 168, 196, 224, 168, 204, 240, 284,
                            196, 240, 284, 344, 252, 312, 372, 464};
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            int match = (C_ime[idx] == expected[idx]);
            if (!match) ok = 0;
            printf("%-10d %-10d %s", C_ime[idx], expected[idx],
                   match ? "✓" : "✗");
            if (j < N - 1) printf(" | ");
        }
        printf("\n");
    }

    // Also verify ref matches expected
    for (int i = 0; i < C_ELEMS; i++) {
        if (C_ref[i] != expected[i]) {
            printf("  NOTE: ref[%d]=%d != expected[%d]=%d\n",
                   i, C_ref[i], i, expected[i]);
        }
    }

    // Check pad region stayed zero (confirm the 4×4 tile boundary)
    printf("\nPad region check (should all be 0 or 0xFFFFFFFF):\n");
    printf("  C[16..63]:");
    int pad_ok = 1;
    for (int i = C_ELEMS; i < C_ELEMS * 4; i++) {
        if (C_ime[i] != (int32_t)0xFFFFFFFF) {
            printf(" [%d]=%d", i, C_ime[i]);
            pad_ok = 0;
            break;
        }
    }
    if (pad_ok) printf(" all 0xFFFFFFFF (untouched by smt.vmadot)\n");

    printf("\n=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
