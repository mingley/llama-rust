// GGUF on-disk Q8_0 GEMV. 32-wide simdgroup splits K; 8 rows per threadgroup.
// Row 0 is summed in block order (printed y0= matches CPU). Other rows use
// simd_sum. Block is 34 bytes: IEEE binary16 d + int8 qs[32].
#include <metal_stdlib>
using namespace metal;

constant uint QK8_0 = 32;
constant uint Q8_0_BLOCK = 34;
constant uint SG = 32;
constant uint ROWS_PER_TG = 8;

inline float block_dot(device const uchar *wb, device const uchar *xb) {
    const half dw = *reinterpret_cast<device const half *>(wb);
    const half dx = *reinterpret_cast<device const half *>(xb);
    device const char *wqs = (device const char *)(wb + 2);
    device const char *xqs = (device const char *)(xb + 2);
    int acc = 0;
#pragma unroll
    for (uint j = 0; j < QK8_0; j++) {
        acc += int(wqs[j]) * int(xqs[j]);
    }
    return float(acc) * (float(dw) * float(dx));
}

kernel void gemv_q8_0_gguf(
    device const uchar *W [[buffer(0)]],
    device const uchar *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint n_cols = shape.x;
    const uint n_rows = shape.y;
    const uint nb = n_cols / QK8_0;
    const uint row_bytes = nb * Q8_0_BLOCK;
    const uint row = tgpig.x * ROWS_PER_TG + sgitg;
    const uint batch = tgpig.y;
    if (row >= n_rows) {
        return;
    }
    device float *yb = y + (ulong)batch * n_rows;
    device const uchar *wr = W + (ulong)row * row_bytes;

    if (row == 0) {
        if (tiisg == 0) {
            float sum = 0.0f;
            for (uint b = 0; b < nb; b++) {
                sum += block_dot(wr + b * Q8_0_BLOCK, x + b * Q8_0_BLOCK);
            }
            yb[0] = sum;
        }
        return;
    }

    float sum = 0.0f;
    for (uint b = tiisg; b < nb; b += SG) {
        sum += block_dot(wr + b * Q8_0_BLOCK, x + b * Q8_0_BLOCK);
    }
    sum = simd_sum(sum);
    if (tiisg == 0) {
        yb[row] = sum;
    }
}
