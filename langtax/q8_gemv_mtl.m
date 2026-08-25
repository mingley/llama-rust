/* Occupied GPU Q8_0 GEMV on GGUF on-disk bytes. Measurement binary only; not
   linked into the crate. Packing matches `q8_gemv.c` / `demo_gguf`. Kernel
   source is langtax/q8_gemv.metal (runtime-compiled via Metal.framework).
   Timed batch: NITER dispatches, one commit/wait. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#define QK8_0 32
#define Q8_0_BLOCK 34
#define M 4096
#define K 4096
#define NITER 32
#define SG 32
#define ROWS_PER_TG 8

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

static NSString *load_src(const char *path) {
    NSError *err = nil;
    NSString *s = [NSString stringWithContentsOfFile:@(path)
                                            encoding:NSUTF8StringEncoding
                                               error:&err];
    if (!s) {
        fprintf(stderr, "read %s: %s\n", path, err.localizedDescription.UTF8String);
    }
    return s;
}

int main(int argc, char **argv) {
    const char *src_path = argc > 1 ? argv[1] : "langtax/q8_gemv.metal";
    const int nb = K / QK8_0;
    const size_t w_bytes = (size_t)M * (size_t)nb * Q8_0_BLOCK;
    const size_t x_bytes = (size_t)nb * Q8_0_BLOCK;
    uint8_t *W = aligned_alloc(64, w_bytes);
    uint8_t *x = aligned_alloc(64, x_bytes);
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

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        NSString *src = load_src(src_path);
        if (!src) {
            return 3;
        }
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            fprintf(stderr, "metal compile: %s\n", err.localizedDescription.UTF8String);
            return 4;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_q8_0_gguf"];
        if (!fn) {
            fprintf(stderr, "missing kernel gemv_q8_0_gguf\n");
            return 5;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String);
            return 6;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const MTLResourceOptions buf_opt =
            MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> bWcpu = [dev newBufferWithBytes:W length:w_bytes options:buf_opt];
        id<MTLBuffer> bXcpu = [dev newBufferWithBytes:x length:x_bytes options:buf_opt];
        id<MTLBuffer> bW = [dev newBufferWithLength:w_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bX = [dev newBufferWithLength:x_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bY =
            [dev newBufferWithLength:(size_t)M * (size_t)NITER * sizeof(float)
                             options:buf_opt];
        uint32_t shape[2] = {(uint32_t)K, (uint32_t)M};
        id<MTLBuffer> bK = [dev newBufferWithBytes:shape length:sizeof(shape)
                                           options:buf_opt];
        if (!q || !bWcpu || !bXcpu || !bW || !bX || !bY || !bK) {
            fprintf(stderr, "buffer alloc\n");
            return 7;
        }

        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:bWcpu sourceOffset:0 toBuffer:bW destinationOffset:0 size:w_bytes];
            [blit copyFromBuffer:bXcpu sourceOffset:0 toBuffer:bX destinationOffset:0 size:x_bytes];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        const NSUInteger tgs = (NSUInteger)SG * (NSUInteger)ROWS_PER_TG;
        const NSUInteger ntg = (NSUInteger)M / (NSUInteger)ROWS_PER_TG;

        for (int pass = 0; pass < 2; pass++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bW offset:0 atIndex:0];
            [enc setBuffer:bX offset:0 atIndex:1];
            [enc setBuffer:bY offset:0 atIndex:2];
            [enc setBuffer:bK offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(ntg, (NSUInteger)NITER, 1)
                threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
            [enc endEncoding];
            double t0 = 0.0;
            if (pass == 1) {
                t0 = now_ns();
            }
            [cb commit];
            [cb waitUntilCompleted];
            if (pass == 1) {
                double t1 = now_ns();
                memcpy(y, (const char *)bY.contents + (size_t)(NITER - 1) * (size_t)M * sizeof(float),
                       (size_t)M * sizeof(float));
                double sec = (t1 - t0) / 1e9;
                double gemv_s = (sec > 0.0) ? ((double)NITER / sec) : 0.0;
                printf("lang=Metal kernel=q8_0_gguf M=%d K=%d niter=%d\n", M, K, NITER);
                printf("device=%s\n", dev.name.UTF8String);
                printf("time_s=%.6f gemv/s=%.2f\n", sec, gemv_s);
                printf("y_checksum=%016llx y0=%.6f\n",
                       (unsigned long long)y_checksum(y, M), y[0]);
            }
        }
    }
    return 0;
}
