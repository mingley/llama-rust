// Extra C cells: Q8_0 GEMV at 1024/2048/8192 and Q4_0 GEMV at 4096.
// 1-thread, same protocol as the frozen q8_gemv.c (8 timed iters).
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#include <math.h>

#define QK8_0 32
#define QK4_0 32

typedef struct {
    float d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    float d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

static double now_ns(void) {
    static mach_timebase_info_data_t tb;
    static int init;
    if (!init) { mach_timebase_info(&tb); init = 1; }
    return (double)mach_absolute_time() * tb.numer / tb.denom;
}

static float vec_dot_q8_0_neon(int n, const block_q8_0 *x, const block_q8_0 *y) {
    const int nb = n / QK8_0;
    int ib = 0;
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0 *x0 = &x[ib + 0];
        const block_q8_0 *x1 = &x[ib + 1];
        const block_q8_0 *y0 = &y[ib + 0];
        const block_q8_0 *y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                    vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                    vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
                x0->d * y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                    vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                    vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
                x1->d * y1->d);
    }
    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) sumi += x[ib].qs[j] * y[ib].qs[j];
        sumf += sumi * (x[ib].d * y[ib].d);
    }
    return sumf;
}

static float vec_dot_q4_q8(int n, const block_q4_0 *w, const block_q8_0 *x) {
    const int nb = n / QK4_0;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        int acc = 0;
        for (int i = 0; i < QK4_0 / 2; i++) {
            unsigned packed = w[ib].qs[i];
            int lo = (int)(packed & 0x0f) - 8;
            int hi = (int)(packed >> 4) - 8;
            acc += lo * (int)x[ib].qs[2 * i];
            acc += hi * (int)x[ib].qs[2 * i + 1];
        }
        sumf += (float)acc * (w[ib].d * x[ib].d);
    }
    return sumf;
}

static void bench_q8(int M, int K, int niter) {
    const int nb = K / QK8_0;
    block_q8_0 *W = aligned_alloc(64, (size_t)M * nb * sizeof(block_q8_0));
    block_q8_0 *x = aligned_alloc(64, (size_t)nb * sizeof(block_q8_0));
    float *y = aligned_alloc(64, (size_t)M * sizeof(float));
    if (!W || !x || !y) { perror("alloc"); exit(1); }
    srand(1);
    for (int i = 0; i < M * nb; i++) {
        float amax = 0.01f + (rand() % 1000) / 1000.0f;
        W[i].d = amax / 127.0f;
        for (int j = 0; j < QK8_0; j++) W[i].qs[j] = (int8_t)((rand() % 255) - 128);
    }
    for (int i = 0; i < nb; i++) {
        x[i].d = 1.0f / 127.0f;
        for (int j = 0; j < QK8_0; j++) x[i].qs[j] = (int8_t)((rand() % 255) - 128);
    }
    for (int r = 0; r < M; r++) y[r] = vec_dot_q8_0_neon(K, W + r * nb, x);
    double t0 = now_ns();
    float sink = 0;
    for (int it = 0; it < niter; it++) {
        for (int r = 0; r < M; r++) {
            float v = vec_dot_q8_0_neon(K, W + r * nb, x);
            y[r] = v;
            sink += v;
        }
    }
    double sec = (now_ns() - t0) / 1e9;
    double wbytes = (double)niter * M * nb * sizeof(block_q8_0);
    printf("lang=C kernel=q8_0_neon M=%d K=%d niter=%d\n", M, K, niter);
    printf("time_s=%.6f gemv/s=%.2f\n", sec, niter / sec);
    printf("weight_GiB/s=%.2f sink=%.4f check=%.4f\n", wbytes / sec / (1 << 30), sink, y[0]);
    free(W); free(x); free(y);
}

static void bench_q4(int M, int K, int niter) {
    const int nb = K / QK4_0;
    block_q4_0 *W = aligned_alloc(64, (size_t)M * nb * sizeof(block_q4_0));
    block_q8_0 *x = aligned_alloc(64, (size_t)nb * sizeof(block_q8_0));
    float *y = aligned_alloc(64, (size_t)M * sizeof(float));
    if (!W || !x || !y) { perror("alloc"); exit(1); }
    srand(3);
    for (int i = 0; i < M * nb; i++) {
        W[i].d = 0.02f + (rand() % 800) / 10000.0f;
        for (int j = 0; j < QK4_0 / 2; j++) W[i].qs[j] = (uint8_t)(rand() % 256);
    }
    for (int i = 0; i < nb; i++) {
        x[i].d = 1.0f / 127.0f;
        for (int j = 0; j < QK8_0; j++) x[i].qs[j] = (int8_t)((rand() % 255) - 128);
    }
    for (int r = 0; r < M; r++) y[r] = vec_dot_q4_q8(K, W + r * nb, x);
    double t0 = now_ns();
    float sink = 0;
    for (int it = 0; it < niter; it++) {
        for (int r = 0; r < M; r++) {
            float v = vec_dot_q4_q8(K, W + r * nb, x);
            y[r] = v;
            sink += v;
        }
    }
    double sec = (now_ns() - t0) / 1e9;
    double wbytes = (double)niter * M * nb * sizeof(block_q4_0);
    printf("lang=C kernel=q4_0 M=%d K=%d niter=%d\n", M, K, niter);
    printf("time_s=%.6f gemv/s=%.2f\n", sec, niter / sec);
    printf("weight_GiB/s=%.2f sink=%.4f check=%.4f\n", wbytes / sec / (1 << 30), sink, y[0]);
    free(W); free(x); free(y);
}

int main(void) {
    const int niter = 8;
    bench_q8(1024, 1024, niter);
    bench_q8(2048, 2048, niter);
    bench_q8(8192, 8192, niter);
    bench_q4(4096, 4096, niter);
    return 0;
}
