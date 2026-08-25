/* 1-thread Q8_0 GEMV on GGUF on-disk bytes (binary16 scale + int8 qs).
   Measurement binary only; not linked into the crate. Packing matches
   `demo_gguf` in src/main.rs so y_checksum can match shipped Rust. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#define QK8_0 32
#define Q8_0_BLOCK 34
#define M 4096
#define K 4096
#define NITER 8

static uint16_t f32_to_f16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint16_t sign = (uint16_t)((b >> 16) & 0x8000u);
    int32_t biased = (int32_t)((b >> 23) & 0xffu);
    int32_t exp = biased - 127 + 15;
    uint32_t man = (b >> 13) & 0x3ffu;
    if (exp <= 0) {
        return sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)man);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t man = (uint32_t)(h & 0x3ffu);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int32_t e = -1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ffu;
            bits = sign | ((uint32_t)(e + 113) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static uint64_t seed = 1;

static uint32_t rnd_u32(void) {
    seed = seed * 6364136223846793005ull + 1ull;
    return (uint32_t)(seed >> 33);
}

static int8_t centered_i8(uint32_t u) {
    return (int8_t)((int32_t)(u % 255u) - 128);
}

static void pack_block(uint8_t *dst, float scale, const int8_t qs[QK8_0]) {
    uint16_t h = f32_to_f16(scale);
    memcpy(dst, &h, 2);
    memcpy(dst + 2, qs, (size_t)QK8_0);
}

__attribute__((noinline)) static float add_f32(float a, float b) { return a + b; }

static float vec_dot_q8_row(const uint8_t *row, const uint8_t *x) {
    float sum = 0.0f;
    int nb = K / QK8_0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = row + (size_t)b * Q8_0_BLOCK;
        const uint8_t *xb = x + (size_t)b * Q8_0_BLOCK;
        uint16_t dwb;
        uint16_t dxb;
        memcpy(&dwb, wb, 2);
        memcpy(&dxb, xb, 2);
        float dw = f16_to_f32(dwb);
        float dx = f16_to_f32(dxb);
        int32_t acc = 0;
        for (int j = 0; j < QK8_0; j++) {
            acc += (int32_t)(int8_t)wb[2 + j] * (int32_t)(int8_t)xb[2 + j];
        }
        sum = add_f32(sum, (float)acc * (dw * dx));
    }
    return sum;
}

static uint64_t y_checksum(const float *y, int n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &y[i], 4);
        h ^= (uint64_t)bits;
        h *= 0x100000001b3ull;
    }
    return h;
}

static double now_ns(void) {
    static mach_timebase_info_data_t tb;
    static int init;
    if (!init) {
        mach_timebase_info(&tb);
        init = 1;
    }
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom;
}

int main(void) {
    const int nb = K / QK8_0;
    uint8_t *W = aligned_alloc(64, (size_t)M * (size_t)nb * Q8_0_BLOCK);
    uint8_t *x = aligned_alloc(64, (size_t)nb * Q8_0_BLOCK);
    float *y = aligned_alloc(64, (size_t)M * sizeof(float));
    if (!W || !x || !y) {
        perror("alloc");
        return 1;
    }

    seed = 1;
    for (int r = 0; r < M; r++) {
        for (int b = 0; b < nb; b++) {
            int8_t qs[QK8_0];
            for (int j = 0; j < QK8_0; j++) {
                qs[j] = centered_i8(rnd_u32());
            }
            uint32_t extra = rnd_u32() % 80u;
            float amax = 20.0f / 1000.0f + (float)extra / 1000.0f;
            pack_block(W + ((size_t)r * (size_t)nb + (size_t)b) * Q8_0_BLOCK, amax, qs);
        }
    }
    for (int b = 0; b < nb; b++) {
        int8_t qs[QK8_0];
        for (int j = 0; j < QK8_0; j++) {
            qs[j] = centered_i8(rnd_u32());
        }
        pack_block(x + (size_t)b * Q8_0_BLOCK, 1.0f / 127.0f, qs);
    }

    for (int it = 0; it < NITER; it++) {
        for (int r = 0; r < M; r++) {
            y[r] = vec_dot_q8_row(W + (size_t)r * (size_t)nb * Q8_0_BLOCK, x);
        }
    }

    double t0 = now_ns();
    for (int it = 0; it < NITER; it++) {
        for (int r = 0; r < M; r++) {
            y[r] = vec_dot_q8_row(W + (size_t)r * (size_t)nb * Q8_0_BLOCK, x);
        }
    }
    double t1 = now_ns();
    double sec = (t1 - t0) / 1e9;
    double gemv_s = (sec > 0.0) ? ((double)NITER / sec) : 0.0;
    printf("lang=C kernel=q8_0_gguf M=%d K=%d niter=%d\n", M, K, NITER);
    printf("time_s=%.6f gemv/s=%.2f\n", sec, gemv_s);
    printf("y_checksum=%016llx y0=%.6f\n", (unsigned long long)y_checksum(y, M), y[0]);
    return 0;
}
