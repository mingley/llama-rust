// GGUF on-disk Q8_0 GEMV. One thread per row.
// Block is 34 bytes: IEEE binary16 d + int8 qs[32]. Not a padded C struct.
#include <metal_stdlib>
using namespace metal;

constant uint QK8_0 = 32;
constant uint Q8_0_BLOCK = 34;

kernel void gemv_q8_0_gguf(
    device const uchar *W [[buffer(0)]],
    device const uchar *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &n_cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    const uint nb = n_cols / QK8_0;
    const uint row_bytes = nb * Q8_0_BLOCK;
    device const uchar *wr = W + (ulong)row * row_bytes;
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar *wb = wr + b * Q8_0_BLOCK;
        device const uchar *xb = x + b * Q8_0_BLOCK;
        const half dw = *reinterpret_cast<device const half *>(wb);
        const half dx = *reinterpret_cast<device const half *>(xb);
        int acc = 0;
        for (uint j = 0; j < QK8_0; j++) {
            acc += int(char(wb[2 + j])) * int(char(xb[2 + j]));
        }
        sum += float(acc) * (float(dw) * float(dx));
    }
    y[row] = sum;
}
