// Test OS vector register save/restore during context switches.
//
// VLEN is passed as argv[1] (e.g. 128 for QEMU, 1024 for A100).
// Uses generic RVV 1.0 instructions only (no IME).

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------- runtime VLEN state ----------

static size_t vlenb;    // bytes per vreg  (= VLEN / 8)
static size_t m8_bytes; // bytes per m8 group (= vlenb × 8 = VLEN)

static void set_vlen(unsigned bits) {
    vlenb    = bits / 8;
    m8_bytes = vlenb * 8;
}

// ---------- asm helpers ----------

#define VLE8_M8(start, buf)                                       \
    __asm__ volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"       \
                     "vle8.v v" #start ", (%[p])\n\t"             \
                     : : [p] "r"(buf) : "t0", "memory")

#define VSE8_M8(start, buf)                                       \
    __asm__ volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"       \
                     "vse8.v v" #start ", (%[p])\n\t"             \
                     : : [p] "r"(buf) : "t0", "memory")

static void load32(const uint8_t *g0, const uint8_t *g1,
                   const uint8_t *g2, const uint8_t *g3) {
    VLE8_M8(0, g0);
    VLE8_M8(8, g1);
    VLE8_M8(16, g2);
    VLE8_M8(24, g3);
}

static void store32(uint8_t *g0, uint8_t *g1,
                    uint8_t *g2, uint8_t *g3) {
    VSE8_M8(0, g0);
    VSE8_M8(8, g1);
    VSE8_M8(16, g2);
    VSE8_M8(24, g3);
}

// ---------- buffer helpers ----------

// Allocate and zero an m8-group buffer.
static uint8_t *new_buf(void) {
    return calloc(1, m8_bytes);
}

// Fill each vreg slice in an m8-group buffer with a unique byte.
//   slice r gets base + r repeated vlenb times.
static void fill_buf(uint8_t *buf, uint8_t base) {
    for (int r = 0; r < 8; r++)
        memset(buf + r * vlenb, base + r, vlenb);
}

// Compare two m8-group buffers.  Returns 1 on match.
static int cmp_buf(const uint8_t *got, const uint8_t *exp,
                   const char *label) {
    if (memcmp(got, exp, m8_bytes) == 0) return 1;
    for (size_t i = 0; i < m8_bytes; i++) {
        if (got[i] != exp[i]) {
            printf("  MISMATCH %s[%zu] (v%zu): exp 0x%02x got 0x%02x\n",
                   label, i, i / vlenb, exp[i], got[i]);
            goto done;
        }
    }
done:
    return 0;
}

// ---------- shared state ----------

static volatile int worker_ok = 0;

// ---------- worker thread ----------

static void *worker_thread(void *arg) {
    printf("[worker] started (vlenb=%zu, m8=%zu)\n", vlenb, m8_bytes);

    // ---- Test-A: check initial vector state is zero ----
    uint8_t *g0 = new_buf(), *g1 = new_buf();
    uint8_t *g2 = new_buf(), *g3 = new_buf();
    store32(g0, g1, g2, g3);

    uint8_t *z0 = new_buf(), *z1 = new_buf();
    uint8_t *z2 = new_buf(), *z3 = new_buf();

    int ok_a = 1;
    ok_a &= cmp_buf(g0, z0, "init v0-v7");
    ok_a &= cmp_buf(g1, z1, "init v8-v15");
    ok_a &= cmp_buf(g2, z2, "init v16-v23");
    ok_a &= cmp_buf(g3, z3, "init v24-v31");
    printf("[worker] Test-A %s\n", ok_a ? "PASS" : "FAIL");

    free(g0); free(g1); free(g2); free(g3);
    free(z0); free(z1); free(z2); free(z3);

    // ---- Test-C: fill worker patterns, yield, verify ----
    uint8_t *w0 = new_buf(), *w1 = new_buf();
    uint8_t *w2 = new_buf(), *w3 = new_buf();
    fill_buf(w0, 0xA0); fill_buf(w1, 0xB0);
    fill_buf(w2, 0xC0); fill_buf(w3, 0xD0);

    load32(w0, w1, w2, w3);
    printf("[worker] filled worker patterns, yielding...\n");
    sched_yield();

    g0 = new_buf(); g1 = new_buf();
    g2 = new_buf(); g3 = new_buf();
    store32(g0, g1, g2, g3);

    int ok_c = 1;
    ok_c &= cmp_buf(g0, w0, "yield v0-v7");
    ok_c &= cmp_buf(g1, w1, "yield v8-v15");
    ok_c &= cmp_buf(g2, w2, "yield v16-v23");
    ok_c &= cmp_buf(g3, w3, "yield v24-v31");
    printf("[worker] Test-C %s\n", ok_c ? "PASS" : "FAIL");

    worker_ok = ok_a && ok_c;

    free(w0); free(w1); free(w2); free(w3);
    free(g0); free(g1); free(g2); free(g3);

    printf("[worker] done\n");
    return NULL;
}

// ---------- main ----------

int main(int argc, char **argv) {
    // Read actual hardware VLEN from vlenb CSR
    size_t hw_vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(hw_vlenb));
    unsigned hw_vlen = (unsigned)(hw_vlenb * 8);

    if (argc >= 2) {
        unsigned vlen_arg = (unsigned)atoi(argv[1]);
        if (vlen_arg != hw_vlen) {
            printf("NOTE: arg VLEN=%u but hardware VLEN=%u (vlenb=%zu). "
                   "Using hardware value.\n\n",
                   vlen_arg, hw_vlen, hw_vlenb);
        }
    }

    set_vlen(hw_vlen);

    printf("=== Vector Register Context Switch Test ===\n");
    printf("HW VLEN = %u bits  |  vlenb = %zu  |  m8 group = %zu bytes\n\n",
           hw_vlen, vlenb, m8_bytes);

    // ---- Main patterns ----
    uint8_t *m0 = new_buf(), *m1 = new_buf();
    uint8_t *m2 = new_buf(), *m3 = new_buf();
    fill_buf(m0, 0x00); fill_buf(m1, 0x10);
    fill_buf(m2, 0x20); fill_buf(m3, 0x30);

    // ---- Test-1: same-thread yield ----
    printf("[main] Test-1: same-thread yield\n");
    load32(m0, m1, m2, m3);
    printf("[main] filled main patterns, yielding...\n");
    sched_yield();

    {
        uint8_t *g0 = new_buf(), *g1 = new_buf();
        uint8_t *g2 = new_buf(), *g3 = new_buf();
        store32(g0, g1, g2, g3);

        int ok = 1;
        ok &= cmp_buf(g0, m0, "yield v0-v7");
        ok &= cmp_buf(g1, m1, "yield v8-v15");
        ok &= cmp_buf(g2, m2, "yield v16-v23");
        ok &= cmp_buf(g3, m3, "yield v24-v31");
        printf("[main] Test-1 %s\n\n", ok ? "PASS" : "FAIL");

        free(g0); free(g1); free(g2); free(g3);
    }

    // ---- Test-2+3: cross-thread ----
    printf("[main] Test-2+3: cross-thread\n");

    // Reload main patterns
    load32(m0, m1, m2, m3);

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_thread, NULL) != 0) {
        printf("[main] pthread_create failed\n");
        return 1;
    }
    printf("[main] worker created, waiting...\n");

    pthread_join(worker, NULL);
    printf("[main] worker joined\n");

    // Verify main patterns intact
    uint8_t *g0 = new_buf(), *g1 = new_buf();
    uint8_t *g2 = new_buf(), *g3 = new_buf();
    store32(g0, g1, g2, g3);

    int ok = 1;
    ok &= cmp_buf(g0, m0, "cross v0-v7");
    ok &= cmp_buf(g1, m1, "cross v8-v15");
    ok &= cmp_buf(g2, m2, "cross v16-v23");
    ok &= cmp_buf(g3, m3, "cross v24-v31");
    printf("[main] Test-3 %s\n\n", ok ? "PASS" : "FAIL");

    free(m0); free(m1); free(m2); free(m3);
    free(g0); free(g1); free(g2); free(g3);

    int all_ok = worker_ok && ok;
    printf("=== %s ===\n", all_ok ? "ALL PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
