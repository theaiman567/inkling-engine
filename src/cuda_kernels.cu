#include "cuda_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <type_traits>

// =========================================================================
// Read-only Global Device Codebooks accessed via __ldg() (L1 texture cache)
// Eliminates NVIDIA __constant__ 32-way warp divergence serialization!
// =========================================================================
__device__ uint8_t  d_ksigns_iq2xs[128];
__device__ uint64_t d_iq2xxs_grid[256];
__device__ uint64_t d_iq2xs_grid[512];
__device__ uint64_t d_iq2s_grid[1024];
__device__ uint32_t d_iq3xxs_grid[256];
__device__ int8_t   d_kvalues_iq4nl[16];
__device__ int8_t   d_kvalues_fp4[16];
__device__ uint16_t d_fp4_pair[256];   // byte b -> lo byte = fp4[b&0xF], hi byte = fp4[b>>4]
__device__ uint16_t d_iq4nl_pair[256]; // byte b -> lo byte = iq4nl[b&0xF], hi byte = iq4nl[b>>4]
__device__ float    d_ue4m3_half[256];
__device__ uint32_t d_sign_mask4[16]; // 4-bit sign nibble -> 4-byte 0x00/0xFF mask

static float*   g_argmax_bvals = nullptr;
static int32_t* g_argmax_bidx  = nullptr;
static void*    g_batch_qx_buf = nullptr;  // 3.93 MiB global Q8 activation staging (up to 1024 vectors x 3072 dim)
static __half*  g_batch_f16_buf = nullptr; // 6.00 MiB global FP16 activation staging for WMMA Tensor Cores!

void ink_cuda_init_tables() {
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_ksigns_iq2xs,  INK_ksigns_iq2xs,   sizeof(INK_ksigns_iq2xs)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_iq2xxs_grid,   INK_iq2xxs_grid,    sizeof(INK_iq2xxs_grid)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_iq2xs_grid,    INK_iq2xs_grid,     sizeof(INK_iq2xs_grid)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_iq2s_grid,     INK_iq2s_grid,      sizeof(INK_iq2s_grid)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_iq3xxs_grid,   INK_iq3xxs_grid,    sizeof(INK_iq3xxs_grid)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_kvalues_iq4nl, INK_kvalues_iq4nl,  sizeof(INK_kvalues_iq4nl)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_kvalues_fp4,   INK_kvalues_fp4,    sizeof(INK_kvalues_fp4)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_ue4m3_half,    INK_ue4m3_half_lut, sizeof(INK_ue4m3_half_lut)));

    uint16_t fp4p[256], iq4p[256];
    for (int b = 0; b < 256; ++b) {
        uint8_t lo_f = (uint8_t)INK_kvalues_fp4[b & 0xF];
        uint8_t hi_f = (uint8_t)INK_kvalues_fp4[b >> 4];
        fp4p[b] = (uint16_t)lo_f | ((uint16_t)hi_f << 8);
        uint8_t lo_i = (uint8_t)INK_kvalues_iq4nl[b & 0xF];
        uint8_t hi_i = (uint8_t)INK_kvalues_iq4nl[b >> 4];
        iq4p[b] = (uint16_t)lo_i | ((uint16_t)hi_i << 8);
    }
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_fp4_pair,   fp4p, sizeof(fp4p)));
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_iq4nl_pair, iq4p, sizeof(iq4p)));

    uint32_t sm4[16];
    for (int i = 0; i < 16; ++i) {
        uint32_t m = 0;
        if (i & 1) m |= 0x000000FFu;
        if (i & 2) m |= 0x0000FF00u;
        if (i & 4) m |= 0x00FF0000u;
        if (i & 8) m |= 0xFF000000u;
        sm4[i] = m;
    }
    INK_CUDA_CHECK(cudaMemcpyToSymbol(d_sign_mask4, sm4, sizeof(sm4)));
    if (!g_argmax_bvals) {
        INK_CUDA_CHECK(cudaMalloc(&g_argmax_bvals, 256 * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&g_argmax_bidx,  256 * sizeof(int32_t)));
    }
    if (!g_batch_qx_buf) {
        INK_CUDA_CHECK(cudaMalloc(&g_batch_qx_buf, 1024ull * 96ull * 40ull)); // 40 B = sizeof(Q8Block32GPU)
    }
    if (!g_batch_f16_buf) {
        INK_CUDA_CHECK(cudaMalloc(&g_batch_f16_buf, 1024ull * 3072ull * sizeof(__half)));
    }
}

// =========================================================================
// Device helpers
// =========================================================================
__device__ __forceinline__ float d_fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ uint16_t d_f32_to_fp16(float f) {
    return __half_as_ushort(__float2half_rn(f));
}

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}

__device__ __forceinline__ uint16_t load_u16_ua(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

__device__ __forceinline__ uint32_t load_u32_ua(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

template <int G>
__device__ __forceinline__ float group_reduce_sum(float v) {
    #pragma unroll
    for (int offset = G / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset, G);
    }
    return v;
}

__device__ __forceinline__ void unpack_pair4(uint32_t q, const uint16_t* lut, int32_t& lo_out, int32_t& hi_out) {
    uint32_t p0 = __ldg(&lut[(q >>  0) & 0xFFu]);
    uint32_t p1 = __ldg(&lut[(q >>  8) & 0xFFu]);
    uint32_t p2 = __ldg(&lut[(q >> 16) & 0xFFu]);
    uint32_t p3 = __ldg(&lut[(q >> 24) & 0xFFu]);
    lo_out = (int32_t)((p0 & 0xFFu) | ((p1 & 0xFFu) << 8) | ((p2 & 0xFFu) << 16) | ((p3 & 0xFFu) << 24));
    hi_out = (int32_t)((p0 >> 8)    | (p1 & 0xFF00u)      | ((p2 >> 8) << 16)    | ((p3 & 0xFF00u) << 16));
}

__device__ __forceinline__ void d_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ float d_situ(float gate, float up, float beta, float lb) {
    float g = beta * tanhf(gate / beta) * (1.0f / (1.0f + expf(-gate)));
    float u = (lb > 0.0f) ? (lb * tanhf(up / lb)) : up;
    return g * u;
}

// Branchless 4-byte sign apply to packed int8x4 using 4-bit sign nibble
__device__ __forceinline__ int32_t apply_sign4(int32_t qx4, int sign_nib) {
    uint32_t sm = __ldg(&d_sign_mask4[sign_nib]);
    int32_t neg = (int32_t)__vsub4(0u, (uint32_t)qx4);
    return (qx4 & (int32_t)(~sm)) | (neg & (int32_t)sm);
}

// Pack 4 int8 bytes into 1 int32
__device__ __forceinline__ int32_t pack_i8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a)
         | ((int32_t)(uint8_t)b << 8)
         | ((int32_t)(uint8_t)c << 16)
         | ((int32_t)(uint8_t)d << 24);
}

// =========================================================================
// Shared-Memory Q8_1 Activation Block (40 bytes / 32 elements -> 2.88 KB for K=2304!)
// Quantized cooperatively by all warps in the block in < 40 cycles!
// =========================================================================
struct Q8Block32GPU {
    float   d;     // amax / 127.0f
    float   s;     // exact float sum of the 32 elements (for Q4_K / Q5_K min terms!)
    int32_t qs[8]; // 32 packed int8_t in [-127, 127]
};

__device__ __forceinline__ void quantize_x_to_smem(const float* __restrict__ x, Q8Block32GPU* __restrict__ s_qx, int K) {
    int nb = K >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nwarps = blockDim.x >> 5;

    for (int ib = warp; ib < nb; ib += nwarps) {
        float v = x[(ib << 5) + lane];
        float amax = fabsf(v);
        float sum  = v;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
        }
        float d  = amax * (1.0f / 127.0f);
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = __float2int_rn(v * id);
        q = max(-127, min(127, q));

        // Pack 4 consecutive lanes into 1 int32 via warp shuffle!
        int q0 = q;
        int q1 = __shfl_down_sync(0xFFFFFFFFu, q, 1);
        int q2 = __shfl_down_sync(0xFFFFFFFFu, q, 2);
        int q3 = __shfl_down_sync(0xFFFFFFFFu, q, 3);
        if ((lane & 3) == 0) {
            s_qx[ib].qs[lane >> 2] = pack_i8x4((int8_t)q0, (int8_t)q1, (int8_t)q2, (int8_t)q3);
        }
        if (lane == 0) {
            s_qx[ib].d = d;
            s_qx[ib].s = sum;
        }
    }
}

__device__ __forceinline__ void quantize_x_gated_to_smem(const float* __restrict__ x, const float* __restrict__ gate,
                                                         Q8Block32GPU* __restrict__ s_qx, int K) {
    int nb = K >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nwarps = blockDim.x >> 5;

    for (int ib = warp; ib < nb; ib += nwarps) {
        int idx = (ib << 5) + lane;
        float g = gate[idx];
        float v = x[idx] * (1.0f / (1.0f + expf(-g)));
        float amax = fabsf(v);
        float sum  = v;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
        }
        float d  = amax * (1.0f / 127.0f);
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = __float2int_rn(v * id);
        q = max(-127, min(127, q));

        int q0 = q;
        int q1 = __shfl_down_sync(0xFFFFFFFFu, q, 1);
        int q2 = __shfl_down_sync(0xFFFFFFFFu, q, 2);
        int q3 = __shfl_down_sync(0xFFFFFFFFu, q, 3);
        if ((lane & 3) == 0) {
            s_qx[ib].qs[lane >> 2] = pack_i8x4((int8_t)q0, (int8_t)q1, (int8_t)q2, (int8_t)q3);
        }
        if (lane == 0) {
            s_qx[ib].d = d;
            s_qx[ib].s = sum;
        }
    }
}

// =========================================================================
// Ultra-Fast Sub-Warp __dp4a + __ldg Quantized Dot Products!
// Eliminates 25-87.5% warp-tail waste on K=2304 (72 blocks -> G=8 exact; 36 chunks -> G=4 exact!)
// =========================================================================

template <int G>
__device__ __forceinline__ float dot_q8_0_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    const int nb = K >> 5;
    float acc = 0.0f;
    for (int ib = glane; ib < nb; ib += G) {
        const uint8_t* bp = row + ib * 34;
        float dw = d_fp16_to_f32(load_u16_ua(bp));
        const uint8_t* qp = bp + 2;
        const int32_t* xq = s_qx[ib].qs;
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int32_t w4 = (int32_t)load_u32_ua(qp + j * 4);
            sumi = __dp4a(w4, xq[j], sumi);
        }
        acc = fmaf(dw * s_qx[ib].d, (float)sumi, acc);
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_q4_k_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    // 8 sub-blocks of 32 elements per 256-elem Q4_K block -> 72 sub-blocks (K=2304) or 96 exact (K=3072)!
    const int total_sub = K >> 5;
    float acc = 0.0f;
    for (int s = glane; s < total_sub; s += G) {
        int ib = s >> 3;
        int is = s & 7;
        int ic = is >> 1;
        int lshift = (is & 1) << 2;
        const BlockQ4_K* b = (const BlockQ4_K*)(row + ib * 144);
        float d = d_fp16_to_f32(b->d);
        float dmin = d_fp16_to_f32(b->dmin);
        uint8_t sc, m;
        d_get_scale_min_k4(is, b->scales, sc, m);

        const uint32_t* q32 = (const uint32_t*)(b->qs + ic * 32);
        const Q8Block32GPU& xb = s_qx[s];

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int32_t w4 = (int32_t)((q32[j] >> lshift) & 0x0F0F0F0Fu);
            sumi = __dp4a(w4, xb.qs[j], sumi);
        }
        acc += (d * sc) * (xb.d * (float)sumi) - (dmin * m) * xb.s;
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_q5_k_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    // 8 sub-blocks of 32 elements per 256-elem Q5_K block -> 72 sub-blocks (K=2304) or 96 exact (K=3072)!
    const int total_sub = K >> 5;
    float acc = 0.0f;
    for (int s = glane; s < total_sub; s += G) {
        int ib = s >> 3;
        int is = s & 7;
        int ic = is >> 1;
        int lshift = (is & 1) << 2;
        const BlockQ5_K* b = (const BlockQ5_K*)(row + ib * 176);
        float d = d_fp16_to_f32(b->d);
        float dmin = d_fp16_to_f32(b->dmin);
        uint8_t sc, m;
        d_get_scale_min_k4(is, b->scales, sc, m);

        const uint32_t* ql32 = (const uint32_t*)(b->qs + ic * 32);
        const uint32_t* qh32 = (const uint32_t*)(b->qh);
        const Q8Block32GPU& xb = s_qx[s];

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t hbits = ((qh32[j] >> is) & 0x01010101u) << 4;
            int32_t w4 = (int32_t)(((ql32[j] >> lshift) & 0x0F0F0F0Fu) | hbits);
            sumi = __dp4a(w4, xb.qs[j], sumi);
        }
        acc += (d * sc) * (xb.d * (float)sumi) - (dmin * m) * xb.s;
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_q6_k_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    // 8 sub-blocks of 32 elements per 256-elem Q6_K block -> 72 sub-blocks (K=2304) or 96 exact (K=3072)!
    const int total_sub = K >> 5;
    float acc = 0.0f;
    for (int s = glane; s < total_sub; s += G) {
        int ib    = s >> 3;
        int half  = (s >> 2) & 1;
        int qpart = s & 3;
        const uint8_t* bp = row + ib * 210;
        const uint8_t* ql = bp + half * 64 + (qpart & 1) * 32;
        const uint8_t* qh = bp + 128 + half * 32;
        const int8_t*  sc = (const int8_t*)(bp + 192) + half * 8 + qpart * 2;
        float d = d_fp16_to_f32(load_u16_ua(bp + 208));

        int ql_shift = (qpart >> 1) << 2;
        int qh_shift = qpart << 1;
        const Q8Block32GPU& xb = s_qx[s];

        int s_lo = 0, s_hi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t qlv = load_u32_ua(ql + j * 4);
            uint32_t qhv = load_u32_ua(qh + j * 4);
            uint32_t q   = ((qlv >> ql_shift) & 0x0F0F0F0Fu) | (((qhv >> qh_shift) & 0x03030303u) << 4);
            int32_t  w   = (int32_t)__vsub4(q, 0x20202020u);
            s_lo = __dp4a(w, xb.qs[j], s_lo);
        }
        #pragma unroll
        for (int j = 4; j < 8; ++j) {
            uint32_t qlv = load_u32_ua(ql + j * 4);
            uint32_t qhv = load_u32_ua(qh + j * 4);
            uint32_t q   = ((qlv >> ql_shift) & 0x0F0F0F0Fu) | (((qhv >> qh_shift) & 0x03030303u) << 4);
            int32_t  w   = (int32_t)__vsub4(q, 0x20202020u);
            s_hi = __dp4a(w, xb.qs[j], s_hi);
        }
        acc += d * xb.d * (sc[0] * (float)s_lo + sc[1] * (float)s_hi);
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_nvfp4_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    // 72 sub-blocks of 16 elements for K=1152! With G=8 -> 9 exact iterations, 8 __ldg via pair LUT!
    const int n_sub = K >> 4;
    float acc = 0.0f;
    for (int sub = glane; sub < n_sub; sub += G) {
        int ib = sub >> 2;
        int s  = sub & 3;
        const BlockNVFP4* b = (const BlockNVFP4*)(row + ib * 36);
        float dw = __ldg(&d_ue4m3_half[b->d[s]]);
        const uint32_t* qp = (const uint32_t*)(b->qs + s * 8);
        uint32_t q0 = qp[0];
        uint32_t q1 = qp[1];

        const Q8Block32GPU& xblk = s_qx[sub >> 1];
        int xoff = (sub & 1) * 4;

        int32_t lo0, hi0, lo1, hi1;
        unpack_pair4(q0, d_fp4_pair, lo0, hi0);
        unpack_pair4(q1, d_fp4_pair, lo1, hi1);

        int sumi = 0;
        sumi = __dp4a(lo0, xblk.qs[xoff + 0], sumi);
        sumi = __dp4a(lo1, xblk.qs[xoff + 1], sumi);
        sumi = __dp4a(hi0, xblk.qs[xoff + 2], sumi);
        sumi = __dp4a(hi1, xblk.qs[xoff + 3], sumi);
        acc = fmaf(dw * xblk.d, (float)sumi, acc);
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_iq4_nl_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    const int nb = K >> 5;
    float acc = 0.0f;
    for (int ib = glane; ib < nb; ib += G) {
        const uint8_t* bp = row + ib * 18;
        float dw = d_fp16_to_f32(load_u16_ua(bp));
        const uint8_t* qs = bp + 2;
        const Q8Block32GPU& xblk = s_qx[ib];
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t q = load_u32_ua(qs + j * 4);
            int32_t lo, hi;
            unpack_pair4(q, d_iq4nl_pair, lo, hi);
            sumi = __dp4a(lo, xblk.qs[j],     sumi);
            sumi = __dp4a(hi, xblk.qs[j + 4], sumi);
        }
        acc = fmaf(dw * xblk.d, (float)sumi, acc);
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_iq3_xxs_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    const int n_sub32 = K >> 5;
    float acc = 0.0f;
    for (int sub = glane; sub < n_sub32; sub += G) {
        int ib   = sub >> 3;
        int ib32 = sub & 7;
        const uint8_t* bp = row + ib * 98;
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.5f;
        const uint8_t* qs = bp + 2 + ib32 * 8;
        uint32_t aux32 = load_u32_ua(bp + 2 + 64 + ib32 * 4);
        const Q8Block32GPU& xblk = s_qx[sub];
        float db = d * (0.5f + (float)(aux32 >> 28)) * xblk.d;

        int sumi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            uint8_t signs = __ldg(&d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127]);
            int32_t g1 = (int32_t)__ldg(&d_iq3xxs_grid[qs[2 * l + 0]]);
            int32_t g2 = (int32_t)__ldg(&d_iq3xxs_grid[qs[2 * l + 1]]);
            sumi = __dp4a(g1, apply_sign4(xblk.qs[2 * l + 0], signs & 0xF), sumi);
            sumi = __dp4a(g2, apply_sign4(xblk.qs[2 * l + 1], signs >> 4),  sumi);
        }
        acc = fmaf(db, (float)sumi, acc);
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_iq2_s_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    const int n_sub32 = K >> 5;
    float acc = 0.0f;
    for (int sub = glane; sub < n_sub32; sub += G) {
        int ib   = sub >> 3;
        int ib32 = sub & 7;
        const uint8_t* bp = row + ib * 82;
        const Q8Block32GPU& xblk = s_qx[sub];
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.25f * xblk.d;
        const uint8_t* qs = bp + 2 + ib32 * 4;
        const uint8_t* signs = bp + 2 + 32 + ib32 * 4;
        uint32_t qhb = bp[2 + 64 + ib32];
        uint8_t sc = bp[2 + 72 + ib32];
        float db0 = d * (0.5f + (float)(sc & 0xF));
        float db1 = d * (0.5f + (float)(sc >> 4));

        int s0 = 0, s1 = 0;
        {
            uint64_t g0 = __ldg(&d_iq2s_grid[qs[0] | ((qhb << 8) & 0x300)]);
            uint64_t g1 = __ldg(&d_iq2s_grid[qs[1] | ((qhb << 6) & 0x300)]);
            s0 = __dp4a((int32_t)(g0 & 0xFFFFFFFFu), apply_sign4(xblk.qs[0], signs[0] & 0xF), s0);
            s0 = __dp4a((int32_t)(g0 >> 32),         apply_sign4(xblk.qs[1], signs[0] >> 4),  s0);
            s0 = __dp4a((int32_t)(g1 & 0xFFFFFFFFu), apply_sign4(xblk.qs[2], signs[1] & 0xF), s0);
            s0 = __dp4a((int32_t)(g1 >> 32),         apply_sign4(xblk.qs[3], signs[1] >> 4),  s0);
        }
        {
            uint64_t g2 = __ldg(&d_iq2s_grid[qs[2] | ((qhb << 4) & 0x300)]);
            uint64_t g3 = __ldg(&d_iq2s_grid[qs[3] | ((qhb << 2) & 0x300)]);
            s1 = __dp4a((int32_t)(g2 & 0xFFFFFFFFu), apply_sign4(xblk.qs[4], signs[2] & 0xF), s1);
            s1 = __dp4a((int32_t)(g2 >> 32),         apply_sign4(xblk.qs[5], signs[2] >> 4),  s1);
            s1 = __dp4a((int32_t)(g3 & 0xFFFFFFFFu), apply_sign4(xblk.qs[6], signs[3] & 0xF), s1);
            s1 = __dp4a((int32_t)(g3 >> 32),         apply_sign4(xblk.qs[7], signs[3] >> 4),  s1);
        }
        acc = fmaf(db0, (float)s0, fmaf(db1, (float)s1, acc));
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_iq2_xs_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    const int n_sub32 = K >> 5;
    float acc = 0.0f;
    for (int sub = glane; sub < n_sub32; sub += G) {
        int ib   = sub >> 3;
        int ib32 = sub & 7;
        const uint8_t* bp = row + ib * 74;
        const Q8Block32GPU& xblk = s_qx[sub];
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.25f * xblk.d;
        const uint8_t* qp = bp + 2 + ib32 * 8;
        uint8_t sc = bp[2 + 64 + ib32];
        float db0 = d * (0.5f + (float)(sc & 0xF));
        float db1 = d * (0.5f + (float)(sc >> 4));

        int s0 = 0, s1 = 0;
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            uint16_t q = load_u16_ua(qp + l * 2);
            uint64_t g = __ldg(&d_iq2xs_grid[q & 511]);
            uint8_t sgn = __ldg(&d_ksigns_iq2xs[q >> 9]);
            s0 = __dp4a((int32_t)(g & 0xFFFFFFFFu), apply_sign4(xblk.qs[2 * l + 0], sgn & 0xF), s0);
            s0 = __dp4a((int32_t)(g >> 32),         apply_sign4(xblk.qs[2 * l + 1], sgn >> 4),  s0);
        }
        #pragma unroll
        for (int l = 2; l < 4; ++l) {
            uint16_t q = load_u16_ua(qp + l * 2);
            uint64_t g = __ldg(&d_iq2xs_grid[q & 511]);
            uint8_t sgn = __ldg(&d_ksigns_iq2xs[q >> 9]);
            s1 = __dp4a((int32_t)(g & 0xFFFFFFFFu), apply_sign4(xblk.qs[2 * l + 0], sgn & 0xF), s1);
            s1 = __dp4a((int32_t)(g >> 32),         apply_sign4(xblk.qs[2 * l + 1], sgn >> 4),  s1);
        }
        acc = fmaf(db0, (float)s0, fmaf(db1, (float)s1, acc));
    }
    return group_reduce_sum<G>(acc);
}

template <int G>
__device__ __forceinline__ float dot_iq2_xxs_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    const int n_sub32 = K >> 5;
    float acc = 0.0f;
    for (int sub = glane; sub < n_sub32; sub += G) {
        int ib   = sub >> 3;
        int ib32 = sub & 7;
        const uint8_t* bp = row + ib * 66;
        const Q8Block32GPU& xblk = s_qx[sub];
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.25f * xblk.d;
        const uint8_t* qp = bp + 2 + ib32 * 8;
        uint32_t aux1 = load_u32_ua(qp + 4);
        float db = d * (0.5f + (float)(aux1 >> 28));
        int sumi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            uint64_t g = __ldg(&d_iq2xxs_grid[qp[l]]);
            uint8_t sgn = __ldg(&d_ksigns_iq2xs[(aux1 >> (7 * l)) & 127]);
            sumi = __dp4a((int32_t)(g & 0xFFFFFFFFu), apply_sign4(xblk.qs[2 * l + 0], sgn & 0xF), sumi);
            sumi = __dp4a((int32_t)(g >> 32),         apply_sign4(xblk.qs[2 * l + 1], sgn >> 4),  sumi);
        }
        acc = fmaf(db, (float)sumi, acc);
    }
    return group_reduce_sum<G>(acc);
}

// Float/Half/BF16 vectorized dot products against F32 shared memory
__device__ __forceinline__ float dot_f32_vec(const uint8_t* row, const float* s_x, int K, int lane) {
    const float* w = (const float*)row;
    float acc = 0.0f;
    if ((K & 3) == 0) {
        const float4* w4 = (const float4*)w;
        const float4* x4 = (const float4*)s_x;
        int n4 = K >> 2;
        for (int i = lane; i < n4; i += 32) {
            float4 a = w4[i], b = x4[i];
            acc = fmaf(a.x, b.x, acc);
            acc = fmaf(a.y, b.y, acc);
            acc = fmaf(a.z, b.z, acc);
            acc = fmaf(a.w, b.w, acc);
        }
    } else {
        for (int k = lane; k < K; k += 32) acc = fmaf(w[k], s_x[k], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float dot_f16_vec(const uint8_t* row, const float* s_x, int K, int lane) {
    const __half2* w2 = (const __half2*)row;
    const float2*  x2 = (const float2*)s_x;
    int n2 = K >> 1;
    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        float2 wf = __half22float2(w2[i]);
        float2 xf = x2[i];
        acc = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, acc));
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float dot_bf16_vec(const uint8_t* row, const float* s_x, int K, int lane) {
    const uint32_t* w2 = (const uint32_t*)row;
    const float2*   x2 = (const float2*)s_x;
    int n2 = K >> 1;
    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        uint32_t packed = w2[i];
        float w0 = __uint_as_float((packed & 0xFFFFu) << 16);
        float w1 = __uint_as_float(packed & 0xFFFF0000u);
        float2 xf = x2[i];
        acc = fmaf(w0, xf.x, fmaf(w1, xf.y, acc));
    }
    return warp_reduce_sum(acc);
}

template <uint32_t TYPE, int G>
__device__ __forceinline__ float dot_quant_dp4a(const uint8_t* row, const Q8Block32GPU* s_qx, int K, int glane) {
    if constexpr (TYPE == GGML_TYPE_Q8_0)    return dot_q8_0_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_Q4_K)    return dot_q4_k_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_Q5_K)    return dot_q5_k_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_Q6_K)    return dot_q6_k_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_NVFP4)   return dot_nvfp4_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_IQ4_NL)  return dot_iq4_nl_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_IQ3_XXS) return dot_iq3_xxs_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_IQ2_S)   return dot_iq2_s_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_IQ2_XS)  return dot_iq2_xs_dp4a<G>(row, s_qx, K, glane);
    if constexpr (TYPE == GGML_TYPE_IQ2_XXS) return dot_iq2_xxs_dp4a<G>(row, s_qx, K, glane);
    return 0.0f;
}

template <uint32_t TYPE, int G>
__global__ void k_gemv_dp4a(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                            int M, int K, size_t row_bytes) {
    extern __shared__ Q8Block32GPU s_qx[];
    quantize_x_to_smem(x, s_qx, K);
    __syncthreads();

    constexpr int RPB_SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rows_per_block = blockDim.x >> RPB_SHIFT;
    int group_in_block = threadIdx.x >> RPB_SHIFT;
    int glane          = threadIdx.x & (G - 1);
    int row            = blockIdx.x * rows_per_block + group_in_block;
    int safe_row       = (row < M) ? row : (M - 1);

    float val = dot_quant_dp4a<TYPE, G>(W + (size_t)safe_row * row_bytes, s_qx, K, glane);
    if (glane == 0 && row < M) y[row] = val;
}

template <uint32_t TYPE>
__global__ void k_gemv_float(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                             int M, int K, size_t row_bytes) {
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= M) return;

    float val = 0.0f;
    if constexpr (TYPE == GGML_TYPE_F32)  val = dot_f32_vec(W + (size_t)row * row_bytes, s_x, K, lane);
    if constexpr (TYPE == GGML_TYPE_F16)  val = dot_f16_vec(W + (size_t)row * row_bytes, s_x, K, lane);
    if constexpr (TYPE == GGML_TYPE_BF16) val = dot_bf16_vec(W + (size_t)row * row_bytes, s_x, K, lane);
    if (lane == 0) y[row] = val;
}

void ink_cuda_gemv(uint32_t type, const void* W, const float* x, float* y, int M, int K, cudaStream_t stream) {
    size_t row_bytes = (size_t)(K / ggml_type_block_size(type)) * ggml_type_byte_size(type);
    const uint8_t* Wp = (const uint8_t*)W;

    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        int WPB = (M <= 128) ? 2 : 8;
        int blocks = (M + WPB - 1) / WPB;
        size_t smem = (size_t)K * sizeof(float);
        if (type == GGML_TYPE_F32)       k_gemv_float<GGML_TYPE_F32> <<<blocks, WPB*32, smem, stream>>>(Wp, x, y, M, K, row_bytes);
        else if (type == GGML_TYPE_F16)  k_gemv_float<GGML_TYPE_F16> <<<blocks, WPB*32, smem, stream>>>(Wp, x, y, M, K, row_bytes);
        else                             k_gemv_float<GGML_TYPE_BF16><<<blocks, WPB*32, smem, stream>>>(Wp, x, y, M, K, row_bytes);
        return;
    }

    size_t smem = (size_t)(K >> 5) * sizeof(Q8Block32GPU);
    // Coalesced G=32 (1 warp = 1 row) maximizes GDDR6 burst efficiency on RTX 4060!
    // For huge vocab LM Head (M >= 32768), G=4 amortizes activation quantization across 32 rows/block (1.37 ms!).
    auto launch_gemv = [&](auto tag) {
        constexpr uint32_t T = decltype(tag)::value;
        if (M >= 32768 && (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K)) {
            int threads = 128;
            int rpb = threads / 4;
            int blocks = (M + rpb - 1) / rpb;
            k_gemv_dp4a<T, 4><<<blocks, threads, smem, stream>>>(Wp, x, y, M, K, row_bytes);
        } else {
            int WPB = (M <= 128) ? 2 : 8;
            int blocks = (M + WPB - 1) / WPB;
            k_gemv_dp4a<T, 32><<<blocks, WPB * 32, smem, stream>>>(Wp, x, y, M, K, row_bytes);
        }
    };

    switch (type) {
        case GGML_TYPE_Q8_0:    launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_Q8_0>{}); break;
        case GGML_TYPE_Q4_K:    launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K:    launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K:    launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_Q6_K>{}); break;
        case GGML_TYPE_NVFP4:   launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_NVFP4>{}); break;
        case GGML_TYPE_IQ4_NL:  launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_IQ4_NL>{}); break;
        case GGML_TYPE_IQ3_XXS: launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_IQ3_XXS>{}); break;
        case GGML_TYPE_IQ2_S:   launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_IQ2_S>{}); break;
        case GGML_TYPE_IQ2_XS:  launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_IQ2_XS>{}); break;
        case GGML_TYPE_IQ2_XXS: launch_gemv(std::integral_constant<uint32_t, GGML_TYPE_IQ2_XXS>{}); break;
        default:
            std::fprintf(stderr, "[CUDA] Unsupported GEMV type %u\n", type);
            std::abort();
    }
}

// =========================================================================
// InkEngine v3.0 Super-Fused Multi-Projection GEMV & Gated Wo GEMV
// (Only instantiates dense attention quant types Q8_0/Q4_K/Q5_K/Q6_K/BF16 -> low register count, max occupancy!)
// =========================================================================
__global__ void __launch_bounds__(256, 3) k_fused_multi_gemv(FusedProjList list, int K) {
    extern __shared__ Q8Block32GPU s_qx[];
    int bid = blockIdx.x;
    int sidx = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i < list.count && bid >= list.slots[i].block_start && bid < list.slots[i].block_end) {
            sidx = i;
        }
    }
    const FusedProjSlot sl = list.slots[sidx];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = (bid - sl.block_start) * (blockDim.x >> 5) + warp;

    if (sl.type == GGML_TYPE_BF16 || sl.type == GGML_TYPE_F32 || sl.type == GGML_TYPE_F16) {
        if (row >= sl.M) return;
        const uint8_t* wrow = sl.W + (size_t)row * sl.row_bytes;
        float val = (sl.type == GGML_TYPE_BF16)
            ? dot_bf16_vec(wrow, sl.x, K, lane)
            : ((sl.type == GGML_TYPE_F16) ? dot_f16_vec(wrow, sl.x, K, lane) : dot_f32_vec(wrow, sl.x, K, lane));
        if (lane == 0) sl.y[row] = val;
        return;
    }

    quantize_x_to_smem(sl.x, s_qx, K);
    __syncthreads();
    if (row >= sl.M) return;

    const uint8_t* wrow = sl.W + (size_t)row * sl.row_bytes;
    float val = 0.0f;
    if      (sl.type == GGML_TYPE_Q8_0) val = dot_quant_dp4a<GGML_TYPE_Q8_0, 32>(wrow, s_qx, K, lane);
    else if (sl.type == GGML_TYPE_Q6_K) val = dot_quant_dp4a<GGML_TYPE_Q6_K, 32>(wrow, s_qx, K, lane);
    else if (sl.type == GGML_TYPE_Q5_K) val = dot_quant_dp4a<GGML_TYPE_Q5_K, 32>(wrow, s_qx, K, lane);
    else if (sl.type == GGML_TYPE_Q4_K) val = dot_quant_dp4a<GGML_TYPE_Q4_K, 32>(wrow, s_qx, K, lane);
    if (lane == 0) sl.y[row] = val;
}

void ink_cuda_fused_multi_gemv(const FusedProjList& list, int K, cudaStream_t stream) {
    if (list.total_blocks <= 0) return;
    size_t smem = (size_t)(K >> 5) * sizeof(Q8Block32GPU);
    k_fused_multi_gemv<<<list.total_blocks, 256, smem, stream>>>(list, K);
}

template <uint32_t TYPE>
__global__ void k_gemv_gated_dp4a(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                  const float* __restrict__ gate, float* __restrict__ y,
                                  int M, int K, size_t row_bytes) {
    extern __shared__ Q8Block32GPU s_qx[];
    quantize_x_gated_to_smem(x, gate, s_qx, K);
    __syncthreads();

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= M) return;

    float val = dot_quant_dp4a<TYPE, 32>(W + (size_t)row * row_bytes, s_qx, K, lane);
    if (lane == 0) y[row] = val;
}

void ink_cuda_gemv_gated(uint32_t type, const void* W, const float* x, const float* gate,
                         float* y, int M, int K, cudaStream_t stream) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        ink_cuda_gate_sigmoid_mul(const_cast<float*>(x), gate, K, stream);
        ink_cuda_gemv(type, W, x, y, M, K, stream);
        return;
    }
    size_t row_bytes = (size_t)(K / ggml_type_block_size(type)) * ggml_type_byte_size(type);
    const uint8_t* Wp = (const uint8_t*)W;
    size_t smem = (size_t)(K >> 5) * sizeof(Q8Block32GPU);
    constexpr int WPB = 8;
    int blocks = (M + WPB - 1) / WPB;
    switch (type) {
        case GGML_TYPE_Q8_0:    k_gemv_gated_dp4a<GGML_TYPE_Q8_0>   <<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        case GGML_TYPE_Q4_K:    k_gemv_gated_dp4a<GGML_TYPE_Q4_K>   <<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        case GGML_TYPE_Q5_K:    k_gemv_gated_dp4a<GGML_TYPE_Q5_K>   <<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        case GGML_TYPE_Q6_K:    k_gemv_gated_dp4a<GGML_TYPE_Q6_K>   <<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        case GGML_TYPE_NVFP4:   k_gemv_gated_dp4a<GGML_TYPE_NVFP4>  <<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        case GGML_TYPE_IQ4_NL:  k_gemv_gated_dp4a<GGML_TYPE_IQ4_NL> <<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        case GGML_TYPE_IQ3_XXS: k_gemv_gated_dp4a<GGML_TYPE_IQ3_XXS><<<blocks, WPB*32, smem, stream>>>(Wp, x, gate, y, M, K, row_bytes); break;
        default:
            ink_cuda_gate_sigmoid_mul(const_cast<float*>(x), gate, K, stream);
            ink_cuda_gemv(type, W, x, y, M, K, stream);
            break;
    }
}

// =========================================================================
// Vectorized RMSNorm, Elementwise, Block Attention Residuals, Loop Control
// =========================================================================
__global__ void k_f32_to_f16(const float* src, uint16_t* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = d_f32_to_fp16(src[i]);
}
void ink_cuda_f32_to_f16(const float* src, uint16_t* dst, int n, cudaStream_t stream) {
    k_f32_to_f16<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

__global__ void k_rms_norm(const float* __restrict__ x, const float* __restrict__ w,
                           float* __restrict__ out, int n, float eps) {
    __shared__ float s_partial[8];
    int tid = threadIdx.x;
    int n4 = n >> 2;
    const float4* x4 = (const float4*)x;
    float ss = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = x4[i];
        ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = warp_reduce_sum(ss);
    if ((tid & 31) == 0) s_partial[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < (blockDim.x >> 5); ++i) tot += s_partial[i];
        s_partial[0] = rsqrtf(tot / (float)n + eps);
    }
    __syncthreads();
    float inv = s_partial[0];
    float4* o4 = (float4*)out;
    if (w) {
        const float4* w4 = (const float4*)w;
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i], g = w4[i];
            o4[i] = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z, v.w * inv * g.w);
        }
    } else {
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i];
            o4[i] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        }
    }
}
void ink_cuda_rms_norm(const float* x, const float* w, float* out, int n, float eps, cudaStream_t stream) {
    k_rms_norm<<<1, 256, 0, stream>>>(x, w, out, n, eps);
}

__global__ void k_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
void ink_cuda_add(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    k_add<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
}
void ink_cuda_add_inplace(float* a, const float* b, int n, cudaStream_t stream) {
    k_add<<<(n + 255) / 256, 256, 0, stream>>>(a, b, a, n);
}

__global__ void k_gate_sigmoid_mul(float* x, const float* gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * (1.0f / (1.0f + expf(-gate[i])));
    }
}
void ink_cuda_gate_sigmoid_mul(float* x, const float* gate, int n, cudaStream_t stream) {
    k_gate_sigmoid_mul<<<(n + 255) / 256, 256, 0, stream>>>(x, gate, n);
}

// Vectorized float4 Block Attention Residuals (8-warp cooperative coalesced loads)
__global__ void k_attn_res_mix(BlockPtrList states, const float* __restrict__ query,
                               float* __restrict__ out, int n_embd, int n_queries) {
    __shared__ float s_inv[12];
    __shared__ float s_dot[3][12];
    __shared__ float s_w[3][12];
    __shared__ float s_red[8];

    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int M = states.count;
    int n4 = n_embd >> 2;

    for (int m = 0; m < M; ++m) {
        const float4* sp = (const float4*)states.ptrs[m];
        float ss = 0.0f;
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = sp[i];
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
        ss = warp_reduce_sum(ss);
        if ((tid & 31) == 0) s_red[tid >> 5] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < nw; ++w) tot += s_red[w];
            s_inv[m] = rsqrtf(tot / (float)n_embd + 1e-6f);
        }
        __syncthreads();
    }

    float scale = rsqrtf((float)n_embd);
    for (int q = 0; q < n_queries; ++q) {
        const float4* qp = (const float4*)(query + q * n_embd);
        for (int m = 0; m < M; ++m) {
            const float4* sp = (const float4*)states.ptrs[m];
            float inv = s_inv[m];
            float d = 0.0f;
            for (int i = tid; i < n4; i += blockDim.x) {
                float4 s = sp[i], qv = qp[i];
                d = fmaf(s.x * inv, qv.x, fmaf(s.y * inv, qv.y, fmaf(s.z * inv, qv.z, fmaf(s.w * inv, qv.w, d))));
            }
            d = warp_reduce_sum(d);
            if ((tid & 31) == 0) s_red[tid >> 5] = d;
            __syncthreads();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < nw; ++w) tot += s_red[w];
                s_dot[q][m] = tot * scale;
            }
            __syncthreads();
        }
        if (tid == 0) {
            float mx = -FLT_MAX;
            for (int m = 0; m < M; ++m) mx = fmaxf(mx, s_dot[q][m]);
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                float e = expf(s_dot[q][m] - mx);
                s_w[q][m] = e;
                sum += e;
            }
            float isum = 1.0f / sum;
            for (int m = 0; m < M; ++m) s_w[q][m] = s_w[q][m] * isum * s_inv[m];
        }
        __syncthreads();

        float4* op = (float4*)(out + q * n_embd);
        for (int i = tid; i < n4; i += blockDim.x) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
            for (int m = 0; m < M; ++m) {
                float wm = s_w[q][m];
                float4 s = ((const float4*)states.ptrs[m])[i];
                ax = fmaf(wm, s.x, ax);
                ay = fmaf(wm, s.y, ay);
                az = fmaf(wm, s.z, az);
                aw = fmaf(wm, s.w, aw);
            }
            op[i] = make_float4(ax, ay, az, aw);
        }
        __syncthreads();
    }
}

void ink_cuda_attn_res_mix(BlockPtrList states, const float* query, float* out, int n_embd, int n_queries, cudaStream_t stream) {
    k_attn_res_mix<<<1, 256, 0, stream>>>(states, query, out, n_embd, n_queries);
}

// Vectorized Loop Message Controller (half2 / float4)
template <uint32_t WTYPE>
__global__ void k_loop_down(const float* __restrict__ cur, const float* __restrict__ msg,
                            const uint8_t* __restrict__ w_down, float* __restrict__ z_out,
                            int n_embd, int rank, size_t row_bytes) {
    extern __shared__ float s_h[];
    __shared__ float s_red[16];
    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int n4 = n_embd >> 2;
    const float4* c4 = (const float4*)cur;
    const float4* m4 = (const float4*)msg;

    float ss0 = 0.0f, ss1 = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 c = c4[i], m = m4[i];
        ss0 = fmaf(c.x, c.x, fmaf(c.y, c.y, fmaf(c.z, c.z, fmaf(c.w, c.w, ss0))));
        ss1 = fmaf(m.x, m.x, fmaf(m.y, m.y, fmaf(m.z, m.z, fmaf(m.w, m.w, ss1))));
    }
    ss0 = warp_reduce_sum(ss0);
    ss1 = warp_reduce_sum(ss1);
    if ((tid & 31) == 0) { s_red[tid >> 5] = ss0; s_red[8 + (tid >> 5)] = ss1; }
    __syncthreads();
    if (tid == 0) {
        float t0 = 0.0f, t1 = 0.0f;
        for (int w = 0; w < nw; ++w) { t0 += s_red[w]; t1 += s_red[8 + w]; }
        s_red[0] = rsqrtf(t0 / (float)n_embd + 1e-6f);
        s_red[1] = rsqrtf(t1 / (float)n_embd + 1e-6f);
    }
    __syncthreads();
    float inv0 = s_red[0], inv1 = s_red[1];
    float4* sh0 = (float4*)s_h;
    float4* sh1 = (float4*)(s_h + n_embd);
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 c = c4[i], m = m4[i];
        sh0[i] = make_float4(c.x * inv0, c.y * inv0, c.z * inv0, c.w * inv0);
        sh1[i] = make_float4(m.x * inv1, m.y * inv1, m.z * inv1, m.w * inv1);
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int row = blockIdx.x * nw + warp;
    if (row >= rank) return;
    float v = 0.0f;
    if constexpr (WTYPE == GGML_TYPE_F16) v = dot_f16_vec(w_down + (size_t)row * row_bytes, s_h, 2 * n_embd, lane);
    else                                  v = dot_f32_vec(w_down + (size_t)row * row_bytes, s_h, 2 * n_embd, lane);
    if (lane == 0) z_out[row] = v;
}

template <uint32_t WTYPE>
__global__ void k_loop_apply(const float* __restrict__ cur, const float* __restrict__ msg,
                             const float* __restrict__ z, const uint8_t* __restrict__ w_gain,
                             const uint8_t* __restrict__ w_forget, float* __restrict__ out,
                             int n_embd, int rank, size_t row_bytes) {
    __shared__ float s_z[128];
    if (threadIdx.x < rank) s_z[threadIdx.x] = z[threadIdx.x];
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_embd) return;

    const uint8_t* gp = w_gain + (size_t)row * row_bytes;
    const uint8_t* fp = w_forget + (size_t)row * row_bytes;
    float g = 0.0f, f = 0.0f;
    if constexpr (WTYPE == GGML_TYPE_F16) {
        const __half2* g2 = (const __half2*)gp;
        const __half2* f2 = (const __half2*)fp;
        const float2*  z2 = (const float2*)s_z;
        int r2 = rank >> 1;
        #pragma unroll 4
        for (int k = 0; k < r2; ++k) {
            float2 zv = z2[k];
            float2 gv = __half22float2(g2[k]);
            float2 fv = __half22float2(f2[k]);
            g = fmaf(gv.x, zv.x, fmaf(gv.y, zv.y, g));
            f = fmaf(fv.x, zv.x, fmaf(fv.y, zv.y, f));
        }
    } else {
        const float4* g4 = (const float4*)gp;
        const float4* f4 = (const float4*)fp;
        const float4* z4 = (const float4*)s_z;
        int r4 = rank >> 2;
        #pragma unroll 4
        for (int k = 0; k < r4; ++k) {
            float4 zv = z4[k], gv = g4[k], fv = f4[k];
            g = fmaf(gv.x, zv.x, fmaf(gv.y, zv.y, fmaf(gv.z, zv.z, fmaf(gv.w, zv.w, g))));
            f = fmaf(fv.x, zv.x, fmaf(fv.y, zv.y, fmaf(fv.z, zv.z, fmaf(fv.w, zv.w, f))));
        }
    }
    out[row] = (1.0f - tanhf(f)) * cur[row] + tanhf(g) * msg[row];
}

void ink_cuda_loop_control(
    const float* cur, const float* msg, uint32_t w_type,
    const void* w_down, const void* w_gain, const void* w_forget,
    float* z_scratch, float* out, int n_embd, int rank, cudaStream_t stream)
{
    size_t elem_sz = (w_type == GGML_TYPE_F16) ? 2 : 4;
    size_t down_rb = (size_t)(2 * n_embd) * elem_sz;
    size_t gf_rb   = (size_t)rank * elem_sz;
    int down_blocks = (rank + 3) / 4; // 16 blocks across SMs
    size_t down_smem = (size_t)(2 * n_embd) * sizeof(float);
    int apply_blocks = (n_embd + 255) / 256;

    if (w_type == GGML_TYPE_F16) {
        k_loop_down<GGML_TYPE_F16><<<down_blocks, 128, down_smem, stream>>>(cur, msg, (const uint8_t*)w_down, z_scratch, n_embd, rank, down_rb);
        k_loop_apply<GGML_TYPE_F16><<<apply_blocks, 256, 0, stream>>>(cur, msg, z_scratch, (const uint8_t*)w_gain, (const uint8_t*)w_forget, out, n_embd, rank, gf_rb);
    } else {
        k_loop_down<GGML_TYPE_F32><<<down_blocks, 128, down_smem, stream>>>(cur, msg, (const uint8_t*)w_down, z_scratch, n_embd, rank, down_rb);
        k_loop_apply<GGML_TYPE_F32><<<apply_blocks, 256, 0, stream>>>(cur, msg, z_scratch, (const uint8_t*)w_gain, (const uint8_t*)w_forget, out, n_embd, rank, gf_rb);
    }
}

// 3-Way Batched Loop Control & 3-Way RMSNorm for Core Passes 1 & 2 (9 launches -> 3, 3x SM saturation!)
template <uint32_t WTYPE>
__global__ void k_loop_down_3way(
    const float* cur0, const float* cur1, const float* cur2,
    const float* msg0, const float* msg1, const float* msg2,
    const uint8_t* wd0, const uint8_t* wd1, const uint8_t* wd2,
    float* z_out_3x, int n_embd, int rank, size_t row_bytes)
{
    int j = blockIdx.y;
    const float* cur = (j == 0) ? cur0 : ((j == 1) ? cur1 : cur2);
    const float* msg = (j == 0) ? msg0 : ((j == 1) ? msg1 : msg2);
    const uint8_t* w_down = (j == 0) ? wd0 : ((j == 1) ? wd1 : wd2);
    float* z_out = z_out_3x + j * rank;

    extern __shared__ float s_h[];
    __shared__ float s_red[16];
    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int n4 = n_embd >> 2;
    const float4* c4 = (const float4*)cur;
    const float4* m4 = (const float4*)msg;

    float ss0 = 0.0f, ss1 = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 c = c4[i], m = m4[i];
        ss0 = fmaf(c.x, c.x, fmaf(c.y, c.y, fmaf(c.z, c.z, fmaf(c.w, c.w, ss0))));
        ss1 = fmaf(m.x, m.x, fmaf(m.y, m.y, fmaf(m.z, m.z, fmaf(m.w, m.w, ss1))));
    }
    ss0 = warp_reduce_sum(ss0);
    ss1 = warp_reduce_sum(ss1);
    if ((tid & 31) == 0) { s_red[tid >> 5] = ss0; s_red[8 + (tid >> 5)] = ss1; }
    __syncthreads();
    if (tid == 0) {
        float t0 = 0.0f, t1 = 0.0f;
        for (int w = 0; w < nw; ++w) { t0 += s_red[w]; t1 += s_red[8 + w]; }
        s_red[0] = rsqrtf(t0 / (float)n_embd + 1e-6f);
        s_red[1] = rsqrtf(t1 / (float)n_embd + 1e-6f);
    }
    __syncthreads();
    float inv0 = s_red[0], inv1 = s_red[1];
    float4* sh0 = (float4*)s_h;
    float4* sh1 = (float4*)(s_h + n_embd);
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 c = c4[i], m = m4[i];
        sh0[i] = make_float4(c.x * inv0, c.y * inv0, c.z * inv0, c.w * inv0);
        sh1[i] = make_float4(m.x * inv1, m.y * inv1, m.z * inv1, m.w * inv1);
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int row = blockIdx.x * nw + warp;
    if (row >= rank) return;
    float v = 0.0f;
    if constexpr (WTYPE == GGML_TYPE_F16) v = dot_f16_vec(w_down + (size_t)row * row_bytes, s_h, 2 * n_embd, lane);
    else                                  v = dot_f32_vec(w_down + (size_t)row * row_bytes, s_h, 2 * n_embd, lane);
    if (lane == 0) z_out[row] = v;
}

template <uint32_t WTYPE>
__global__ void k_loop_apply_3way(
    const float* cur0, const float* cur1, const float* cur2,
    const float* msg0, const float* msg1, const float* msg2,
    const float* z_3x,
    const uint8_t* wg0, const uint8_t* wf0,
    const uint8_t* wg1, const uint8_t* wf1,
    const uint8_t* wg2, const uint8_t* wf2,
    float* out0, float* out1, float* out2,
    int n_embd, int rank, size_t row_bytes)
{
    int j = blockIdx.y;
    const float* cur = (j == 0) ? cur0 : ((j == 1) ? cur1 : cur2);
    const float* msg = (j == 0) ? msg0 : ((j == 1) ? msg1 : msg2);
    const uint8_t* w_gain   = (j == 0) ? wg0 : ((j == 1) ? wg1 : wg2);
    const uint8_t* w_forget = (j == 0) ? wf0 : ((j == 1) ? wf1 : wf2);
    float* out = (j == 0) ? out0 : ((j == 1) ? out1 : out2);

    __shared__ float s_z[128];
    if (threadIdx.x < rank) s_z[threadIdx.x] = z_3x[j * rank + threadIdx.x];
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_embd) return;

    const uint8_t* gp = w_gain + (size_t)row * row_bytes;
    const uint8_t* fp = w_forget + (size_t)row * row_bytes;
    float g = 0.0f, f = 0.0f;
    if constexpr (WTYPE == GGML_TYPE_F16) {
        const __half2* g2 = (const __half2*)gp;
        const __half2* f2 = (const __half2*)fp;
        const float2*  z2 = (const float2*)s_z;
        int r2 = rank >> 1;
        #pragma unroll 4
        for (int k = 0; k < r2; ++k) {
            float2 zv = z2[k];
            float2 gv = __half22float2(g2[k]);
            float2 fv = __half22float2(f2[k]);
            g = fmaf(gv.x, zv.x, fmaf(gv.y, zv.y, g));
            f = fmaf(fv.x, zv.x, fmaf(fv.y, zv.y, f));
        }
    } else {
        const float4* g4 = (const float4*)gp;
        const float4* f4 = (const float4*)fp;
        const float4* z4 = (const float4*)s_z;
        int r4 = rank >> 2;
        #pragma unroll 4
        for (int k = 0; k < r4; ++k) {
            float4 zv = z4[k], gv = g4[k], fv = f4[k];
            g = fmaf(gv.x, zv.x, fmaf(gv.y, zv.y, fmaf(gv.z, zv.z, fmaf(gv.w, zv.w, g))));
            f = fmaf(fv.x, zv.x, fmaf(fv.y, zv.y, fmaf(fv.z, zv.z, fmaf(fv.w, zv.w, f))));
        }
    }
    out[row] = (1.0f - tanhf(f)) * cur[row] + tanhf(g) * msg[row];
}

void ink_cuda_loop_control_3way(
    const float* cur0, const float* cur1, const float* cur2,
    const float* msg0, const float* msg1, const float* msg2,
    uint32_t w_type,
    const void* w_down0, const void* w_gain0, const void* w_forget0,
    const void* w_down1, const void* w_gain1, const void* w_forget1,
    const void* w_down2, const void* w_gain2, const void* w_forget2,
    float* z_scratch_3x, float* out0, float* out1, float* out2,
    int n_embd, int rank, cudaStream_t stream)
{
    size_t elem_sz = (w_type == GGML_TYPE_F16) ? 2 : 4;
    size_t down_rb = (size_t)(2 * n_embd) * elem_sz;
    size_t gf_rb   = (size_t)rank * elem_sz;
    dim3 g_down((rank + 3) / 4, 3);
    dim3 g_app((n_embd + 255) / 256, 3);
    size_t down_smem = (size_t)(2 * n_embd) * sizeof(float);

    if (w_type == GGML_TYPE_F16) {
        k_loop_down_3way<GGML_TYPE_F16><<<g_down, 128, down_smem, stream>>>(
            cur0, cur1, cur2, msg0, msg1, msg2,
            (const uint8_t*)w_down0, (const uint8_t*)w_down1, (const uint8_t*)w_down2,
            z_scratch_3x, n_embd, rank, down_rb);
        k_loop_apply_3way<GGML_TYPE_F16><<<g_app, 256, 0, stream>>>(
            cur0, cur1, cur2, msg0, msg1, msg2, z_scratch_3x,
            (const uint8_t*)w_gain0, (const uint8_t*)w_forget0,
            (const uint8_t*)w_gain1, (const uint8_t*)w_forget1,
            (const uint8_t*)w_gain2, (const uint8_t*)w_forget2,
            out0, out1, out2, n_embd, rank, gf_rb);
    } else {
        k_loop_down_3way<GGML_TYPE_F32><<<g_down, 128, down_smem, stream>>>(
            cur0, cur1, cur2, msg0, msg1, msg2,
            (const uint8_t*)w_down0, (const uint8_t*)w_down1, (const uint8_t*)w_down2,
            z_scratch_3x, n_embd, rank, down_rb);
        k_loop_apply_3way<GGML_TYPE_F32><<<g_app, 256, 0, stream>>>(
            cur0, cur1, cur2, msg0, msg1, msg2, z_scratch_3x,
            (const uint8_t*)w_gain0, (const uint8_t*)w_forget0,
            (const uint8_t*)w_gain1, (const uint8_t*)w_forget1,
            (const uint8_t*)w_gain2, (const uint8_t*)w_forget2,
            out0, out1, out2, n_embd, rank, gf_rb);
    }
}

__global__ void k_rms_norm_3way(
    const float* x0, const float* x1, const float* x2,
    const float* w0, const float* w1, const float* w2,
    float* out0, float* out1, float* out2, int n, float eps)
{
    int j = blockIdx.x;
    const float* x = (j == 0) ? x0 : ((j == 1) ? x1 : x2);
    const float* w = (j == 0) ? w0 : ((j == 1) ? w1 : w2);
    float* out     = (j == 0) ? out0 : ((j == 1) ? out1 : out2);

    __shared__ float s_partial[8];
    int tid = threadIdx.x;
    int n4 = n >> 2;
    const float4* x4 = (const float4*)x;
    float ss = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = x4[i];
        ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = warp_reduce_sum(ss);
    if ((tid & 31) == 0) s_partial[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < (blockDim.x >> 5); ++i) tot += s_partial[i];
        s_partial[0] = rsqrtf(tot / (float)n + eps);
    }
    __syncthreads();
    float inv = s_partial[0];
    float4* o4 = (float4*)out;
    const float4* w4 = (const float4*)w;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = x4[i], g = w4[i];
        o4[i] = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z, v.w * inv * g.w);
    }
}

void ink_cuda_rms_norm_3way(
    const float* x0, const float* x1, const float* x2,
    const float* w0, const float* w1, const float* w2,
    float* out0, float* out1, float* out2, int n, float eps, cudaStream_t stream)
{
    k_rms_norm_3way<<<3, 256, 0, stream>>>(x0, x1, x2, w0, w1, w2, out0, out1, out2, n, eps);
}

// =========================================================================
// Causal ShortConv, QKV Post, Relative Table
// =========================================================================
__global__ void k_shortconv(const float* __restrict__ x_in, float* __restrict__ state,
                            const float* __restrict__ weight, const float* __restrict__ dynamic,
                            float* __restrict__ out, int n_channels, int n_heads) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float* st = state + c * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2], s3 = x_in[c];
    st[0] = s1; st[1] = s2; st[2] = s3;

    const float4 w = ((const float4*)weight)[c];
    float acc = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;

    if (dynamic != nullptr) {
        int h = c / (n_channels / n_heads);
        const float4 tap = ((const float4*)dynamic)[h];
        acc += tap.x * s3 + tap.y * s2 + tap.z * s1 + tap.w * s0;
        out[c] = (s3 + acc) * (1.0f / 3.0f);
    } else {
        out[c] = (s3 + acc) * 0.4472135955f;
    }
}

void ink_cuda_shortconv(const float* x_in, float* state, const float* weight, const float* dynamic,
                        float* out, int n_channels, int n_heads, cudaStream_t stream) {
    k_shortconv<<<(n_channels + 255) / 256, 256, 0, stream>>>(x_in, state, weight, dynamic, out, n_channels, n_heads);
}

__global__ void k_shortconv_add(const float* __restrict__ x_in, float* __restrict__ state,
                                const float* __restrict__ weight, const float* __restrict__ res,
                                float* __restrict__ out, int n_channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    float* st = state + c * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2], s3 = x_in[c];
    st[0] = s1; st[1] = s2; st[2] = s3;
    const float4 w = ((const float4*)weight)[c];
    float acc = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
    out[c] = res[c] + (s3 + acc) * 0.4472135955f;
}

void ink_cuda_shortconv_add(const float* x_in, float* state, const float* weight, const float* res,
                            float* out, int n_channels, cudaStream_t stream) {
    k_shortconv_add<<<(n_channels + 255) / 256, 256, 0, stream>>>(x_in, state, weight, res, out, n_channels);
}

__global__ void k_shortconv_kv(
    const float* __restrict__ k_in, float* __restrict__ k_state, const float* __restrict__ k_weight, const float* __restrict__ dyn_k, float* __restrict__ k_out,
    const float* __restrict__ v_in, float* __restrict__ v_state, const float* __restrict__ v_weight, const float* __restrict__ dyn_v, float* __restrict__ v_out,
    int n_channels, int n_heads)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    bool is_v = (blockIdx.y != 0);
    const float* x_in    = is_v ? v_in     : k_in;
    float*       state   = is_v ? v_state  : k_state;
    const float* weight  = is_v ? v_weight : k_weight;
    const float* dynamic = is_v ? dyn_v    : dyn_k;
    float*       out     = is_v ? v_out    : k_out;

    float* st = state + c * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2], s3 = x_in[c];
    st[0] = s1; st[1] = s2; st[2] = s3;

    const float4 w = ((const float4*)weight)[c];
    float acc = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
    if (dynamic != nullptr) {
        int h = c / (n_channels / n_heads);
        const float4 tap = ((const float4*)dynamic)[h];
        acc += tap.x * s3 + tap.y * s2 + tap.z * s1 + tap.w * s0;
        out[c] = (s3 + acc) * (1.0f / 3.0f);
    } else {
        out[c] = (s3 + acc) * 0.4472135955f;
    }
}

void ink_cuda_shortconv_kv(
    const float* k_in, float* k_state, const float* k_weight, const float* dyn_k, float* k_out,
    const float* v_in, float* v_state, const float* v_weight, const float* dyn_v, float* v_out,
    int n_channels, int n_heads, cudaStream_t stream)
{
    dim3 grid((n_channels + 255) / 256, 2);
    k_shortconv_kv<<<grid, 256, 0, stream>>>(k_in, k_state, k_weight, dyn_k, k_out, v_in, v_state, v_weight, dyn_v, v_out, n_channels, n_heads);
}

__global__ void k_qkv_post(float* Q, const float* q_clip, float* K, float* V, float* R, const float* r_clip,
                           int n_head, int n_head_kv, int head_dim, int d_rel,
                           float qk_thresh, float rel_thresh, float eps, float tau_val, const float* d_tau_ptr) {
    float tau = d_tau_ptr ? __ldg(d_tau_ptr) : tau_val;
    int b = blockIdx.x;
    int lane = threadIdx.x;

    if (b < n_head) {
        float* q = Q + b * head_dim;
        const float* qc = q_clip + b * head_dim;
        float ss = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            float v = q[d] * qc[d];
            q[d] = v;
            ss = fmaf(v, v, ss);
        }
        ss = warp_reduce_sum(ss);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0)) + 1e-6f)) * tau;
        for (int d = lane; d < head_dim; d += 32) q[d] *= factor;
    } else if (b < n_head + n_head_kv) {
        int h = b - n_head;
        float* k = K + h * head_dim;
        float ss = 0.0f;
        for (int d = lane; d < head_dim; d += 32) ss = fmaf(k[d], k[d], ss);
        ss = warp_reduce_sum(ss);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0)) + 1e-6f));
        for (int d = lane; d < head_dim; d += 32) k[d] *= factor;
    } else if (b < n_head + 2 * n_head_kv) {
        int h = b - n_head - n_head_kv;
        float* v = V + h * head_dim;
        float ss = 0.0f;
        for (int d = lane; d < head_dim; d += 32) ss = fmaf(v[d], v[d], ss);
        ss = warp_reduce_sum(ss);
        float inv = rsqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0) / (float)head_dim + eps);
        for (int d = lane; d < head_dim; d += 32) v[d] *= inv;
    } else {
        int h = b - n_head - 2 * n_head_kv;
        if (h < n_head) {
            float* r = R + h * d_rel;
            const float* rc = r_clip + h * d_rel;
            float val = (lane < d_rel) ? (r[lane] * rc[lane]) : 0.0f;
            float ss = warp_reduce_sum(val * val);
            float factor = fminf(1.0f, rel_thresh / (sqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0)) + 1e-6f)) * tau;
            if (lane < d_rel) r[lane] = val * factor;
        }
    }
}

void ink_cuda_qkv_post(float* Q, const float* q_clip, float* K, float* V, float* R, const float* r_clip,
                       int n_head, int n_head_kv, int head_dim, int d_rel,
                       float qk_clip_thresh, float rel_clip_thresh, float rms_eps, float tau,
                       const float* d_tau_ptr, cudaStream_t stream) {
    k_qkv_post<<<2 * n_head + 2 * n_head_kv, 32, 0, stream>>>(
        Q, q_clip, K, V, R, r_clip, n_head, n_head_kv, head_dim, d_rel, qk_clip_thresh, rel_clip_thresh, rms_eps, tau, d_tau_ptr);
}

__global__ void k_rel_table(const float* __restrict__ rel_bias, const float* __restrict__ R,
                            float* __restrict__ rel_table, int n_head, int d_rel, int rel_extent) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (d >= rel_extent || h >= n_head) return;

    const float4* rb = (const float4*)(rel_bias + d * d_rel);
    const float4* rh = (const float4*)(R + h * d_rel);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 a = rb[i], b = rh[i];
        s = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, s))));
    }
    rel_table[h * rel_extent + d] = s;
}

void ink_cuda_rel_table(const float* rel_bias, const float* R, float* rel_table,
                        int n_head, int d_rel, int rel_extent, cudaStream_t stream) {
    dim3 grid((rel_extent + 127) / 128, n_head);
    k_rel_table<<<grid, 128, 0, stream>>>(rel_bias, R, rel_table, n_head, d_rel, rel_extent);
}

// =========================================================================
// KV Cache Quantization & Storage + Fused pos_slot write
// =========================================================================
__device__ __forceinline__ void quant_block32_q4_0(const float* src, uint8_t* dst, int lane) {
    float v = src[lane];
    float amax = fabsf(v);
    float signed_max = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float oa = __shfl_xor_sync(0xFFFFFFFFu, amax, off);
        float os = __shfl_xor_sync(0xFFFFFFFFu, signed_max, off);
        if (oa > amax) { amax = oa; signed_max = os; }
    }
    float d = signed_max / -8.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    int q = __float2int_rn(v * id + 8.0f);
    q = max(0, min(15, q));
    int q_hi = __shfl_down_sync(0xFFFFFFFFu, q, 16);
    if (lane == 0) {
        uint16_t dh = d_f32_to_fp16(d);
        dst[0] = (uint8_t)(dh & 0xFF);
        dst[1] = (uint8_t)(dh >> 8);
    }
    if (lane < 16) dst[2 + lane] = (uint8_t)(q | (q_hi << 4));
}

__device__ __forceinline__ void quant_block32_q8_0(const float* src, uint8_t* dst, int lane) {
    float v = src[lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    }
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    int q = max(-127, min(127, __float2int_rn(v * id)));
    if (lane == 0) {
        uint16_t dh = d_f32_to_fp16(d);
        dst[0] = (uint8_t)(dh & 0xFF);
        dst[1] = (uint8_t)(dh >> 8);
    }
    dst[2 + lane] = (uint8_t)(int8_t)q;
}

__global__ void k_kv_store(const float* __restrict__ K, const float* __restrict__ V,
                           uint8_t* __restrict__ k_dst, uint8_t* __restrict__ v_dst,
                           float* __restrict__ active_k_min, float* __restrict__ active_k_max,
                           int8_t* __restrict__ page_k_min_q8, int8_t* __restrict__ page_k_max_q8,
                           float* __restrict__ page_k_scale,
                           int is_first_in_page, int32_t* pos_slot_dst, int32_t pos_val,
                           int n_head_kv, int head_dim, int kv_mode) {
    __shared__ float s_amax[4];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid == 0 && tid == 0 && pos_slot_dst != nullptr) {
        *pos_slot_dst = pos_val;
    }

    bool is_v = (bid >= n_head_kv);
    int h = is_v ? (bid - n_head_kv) : bid;
    int warp = tid >> 5;
    int lane = tid & 31;
    const float* src = (is_v ? V : K) + h * head_dim;

    if (!is_v && page_k_min_q8 != nullptr && active_k_min != nullptr) {
        int idx = h * head_dim + tid;
        float val = src[tid];
        float mn = is_first_in_page ? val : fminf(active_k_min[idx], val);
        float mx = is_first_in_page ? val : fmaxf(active_k_max[idx], val);
        active_k_min[idx] = mn;
        active_k_max[idx] = mx;

        // Per-page-head symmetric INT8 quantization with outward (floor/ceil) conservative rounding!
        float am = fmaxf(fabsf(mn), fabsf(mx));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
        if (lane == 0) s_amax[warp] = am;
        __syncthreads();
        if (tid == 0) {
            float m4 = fmaxf(fmaxf(s_amax[0], s_amax[1]), fmaxf(s_amax[2], s_amax[3]));
            float sc = fmaxf(m4, 1e-6f) * (1.0f / 127.0f);
            s_amax[0] = sc;
            page_k_scale[h] = sc;
        }
        __syncthreads();
        float inv_sc = 1.0f / s_amax[0];
        int qmn = max(-127, min(127, (int)floorf(mn * inv_sc)));
        int qmx = max(-127, min(127, (int)ceilf(mx * inv_sc)));
        page_k_min_q8[idx] = (int8_t)qmn;
        page_k_max_q8[idx] = (int8_t)qmx;
    }

    if (kv_mode == KV_QUANT_Q4_0) {
        quant_block32_q4_0(src + warp * 32, (is_v ? v_dst : k_dst) + h * 72 + warp * 18, lane);
    } else if (kv_mode == KV_QUANT_Q8_0) {
        quant_block32_q8_0(src + warp * 32, (is_v ? v_dst : k_dst) + h * 136 + warp * 34, lane);
    } else {
        ((uint16_t*)(is_v ? v_dst : k_dst) + h * head_dim)[tid] = d_f32_to_fp16(src[tid]);
    }
}

void ink_cuda_kv_store(const float* K, const float* V, void* k_dst_token, void* v_dst_token,
                       float* active_k_min, float* active_k_max,
                       int8_t* page_k_min_q8, int8_t* page_k_max_q8, float* page_k_scale,
                       bool is_first_in_page,
                       int32_t* pos_slot_dst, int32_t pos_val,
                       int n_head_kv, int head_dim, KVQuantMode kv_mode, cudaStream_t stream) {
    k_kv_store<<<n_head_kv * 2, head_dim, 0, stream>>>(
        K, V, (uint8_t*)k_dst_token, (uint8_t*)v_dst_token,
        active_k_min, active_k_max, page_k_min_q8, page_k_max_q8, page_k_scale,
        is_first_in_page ? 1 : 0, pos_slot_dst, pos_val,
        n_head_kv, head_dim, (int)kv_mode);
}

// =========================================================================
// GQA-Fused (3x Less GDDR6 Bandwidth!) FlashDecoding SWA & 32-Split Global + Quest Scorer
// =========================================================================
__device__ __forceinline__ float4 load_kv_head_4(const uint8_t* head_ptr, int lane, int kv_mode) {
    float4 out;
    if (kv_mode == KV_QUANT_Q4_0) {
        // Lanes 0..3 load the 4 FP16 block scales once and broadcast across the warp!
        float d_self = 0.0f;
        if (lane < 4) {
            const uint8_t* bp = head_ptr + lane * 18;
            uint16_t h = (uint16_t)__ldg(&bp[0]) | ((uint16_t)__ldg(&bp[1]) << 8);
            d_self = d_fp16_to_f32(h);
        }
        float d0 = __shfl_sync(0xFFFFFFFFu, d_self, 0);
        float d1 = __shfl_sync(0xFFFFFFFFu, d_self, 1);
        float d2 = __shfl_sync(0xFFFFFFFFu, d_self, 2);
        float d3 = __shfl_sync(0xFFFFFFFFu, d_self, 3);
        int bidx = 2 + (lane & 15);
        int sh   = (lane >> 4) << 2;
        out.x = (float)(((__ldg(&head_ptr[ 0 + bidx]) >> sh) & 0xF) - 8) * d0;
        out.y = (float)(((__ldg(&head_ptr[18 + bidx]) >> sh) & 0xF) - 8) * d1;
        out.z = (float)(((__ldg(&head_ptr[36 + bidx]) >> sh) & 0xF) - 8) * d2;
        out.w = (float)(((__ldg(&head_ptr[54 + bidx]) >> sh) & 0xF) - 8) * d3;
    } else if (kv_mode == KV_QUANT_Q8_0) {
        float d_self = 0.0f;
        if (lane < 4) {
            const uint8_t* bp = head_ptr + lane * 34;
            uint16_t h = (uint16_t)__ldg(&bp[0]) | ((uint16_t)__ldg(&bp[1]) << 8);
            d_self = d_fp16_to_f32(h);
        }
        float d0 = __shfl_sync(0xFFFFFFFFu, d_self, 0);
        float d1 = __shfl_sync(0xFFFFFFFFu, d_self, 1);
        float d2 = __shfl_sync(0xFFFFFFFFu, d_self, 2);
        float d3 = __shfl_sync(0xFFFFFFFFu, d_self, 3);
        int bidx = 2 + lane;
        out.x = (float)(int8_t)__ldg(&head_ptr[  0 + bidx]) * d0;
        out.y = (float)(int8_t)__ldg(&head_ptr[ 34 + bidx]) * d1;
        out.z = (float)(int8_t)__ldg(&head_ptr[ 68 + bidx]) * d2;
        out.w = (float)(int8_t)__ldg(&head_ptr[102 + bidx]) * d3;
    } else {
        const uint16_t* hp = (const uint16_t*)head_ptr;
        out.x = d_fp16_to_f32(__ldg(&hp[lane +  0]));
        out.y = d_fp16_to_f32(__ldg(&hp[lane + 32]));
        out.z = d_fp16_to_f32(__ldg(&hp[lane + 64]));
        out.w = d_fp16_to_f32(__ldg(&hp[lane + 96]));
    }
    return out;
}

__device__ __forceinline__ size_t kv_head_bytes_dev(int kv_mode) {
    return (kv_mode == KV_QUANT_Q4_0) ? 72 : ((kv_mode == KV_QUANT_Q8_0) ? 136 : 256);
}

// GQA=3 Fused SWA Split-K Kernel: loads K & V ONCE for all 3 query heads -> 3x LESS GDDR6 traffic!
__global__ void k_attn_swa_gqa_splitk(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ K_ring, const uint8_t* __restrict__ V_ring,
    const int32_t* __restrict__ pos_ring,
    const float* __restrict__ rel_table,
    float* __restrict__ split_out, float* __restrict__ split_ml,
    int n_head, int n_head_kv, int head_dim,
    int swa_len, int n_splits, int pos_q, int rel_extent, int kv_mode)
{
    int h_kv = blockIdx.x;
    int sp   = blockIdx.y;
    if (h_kv >= n_head_kv || sp >= n_splits) return;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int s_start = (swa_len * sp) / n_splits;
    int s_end   = (swa_len * (sp + 1)) / n_splits;

    float q0[3], q1[3], q2[3], q3[3];
    const float* rtab[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h_kv * 3 + g;
        const float* qh = Q + h * head_dim;
        q0[g] = qh[lane + 0]; q1[g] = qh[lane + 32]; q2[g] = qh[lane + 64]; q3[g] = qh[lane + 96];
        rtab[g] = rel_table + h * rel_extent;
    }

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    const size_t hb = kv_head_bytes_dev(kv_mode);
    const size_t tok_stride = (size_t)n_head_kv * hb;

    float m_w[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float l_w[3] = {0.0f, 0.0f, 0.0f};
    float o0[3] = {0.0f, 0.0f, 0.0f}, o1[3] = {0.0f, 0.0f, 0.0f};
    float o2[3] = {0.0f, 0.0f, 0.0f}, o3[3] = {0.0f, 0.0f, 0.0f};

    for (int s = s_start + warp; s < s_end; s += 4) {
        int pos_k = __ldg(&pos_ring[s]);
        int dist = pos_q - pos_k;
        if (pos_k < 0 || dist < 0 || dist >= rel_extent) continue;

        const uint8_t* kp = K_ring + (size_t)s * tok_stride + (size_t)h_kv * hb;
        const uint8_t* vp = V_ring + (size_t)s * tok_stride + (size_t)h_kv * hb;
        float4 kv = load_kv_head_4(kp, lane, kv_mode);
        float4 vv = load_kv_head_4(vp, lane, kv_mode);

        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = warp_reduce_sum(q0[g] * kv.x + q1[g] * kv.y + q2[g] * kv.z + q3[g] * kv.w);
            float score = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d + __ldg(&rtab[g][dist]);
            float m_new = fmaxf(m_w[g], score);
            float alpha = (m_w[g] == -FLT_MAX) ? 0.0f : expf(m_w[g] - m_new);
            float prob  = expf(score - m_new);
            l_w[g] = l_w[g] * alpha + prob;
            o0[g]  = fmaf(o0[g], alpha, prob * vv.x);
            o1[g]  = fmaf(o1[g], alpha, prob * vv.y);
            o2[g]  = fmaf(o2[g], alpha, prob * vv.z);
            o3[g]  = fmaf(o3[g], alpha, prob * vv.w);
            m_w[g] = m_new;
        }
    }

    __shared__ float sm_m[3][4], sm_l[3][4], sm_o[3][4][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[g][warp] = m_w[g]; sm_l[g][warp] = l_w[g]; }
        sm_o[g][warp][lane +  0] = o0[g];
        sm_o[g][warp][lane + 32] = o1[g];
        sm_o[g][warp][lane + 64] = o2[g];
        sm_o[g][warp][lane + 96] = o3[g];
    }
    __syncthreads();

    // Warps 0, 1, 2 reduce query heads g = 0, 1, 2 IN PARALLEL!
    if (warp < 3) {
        int g = warp;
        int h = h_kv * 3 + g;
        float m_all = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < 4; ++w) m_all = fmaxf(m_all, sm_m[g][w]);

        float l_all = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if (m_all > -FLT_MAX * 0.5f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[g][w] > -FLT_MAX * 0.5f) {
                    float sc = expf(sm_m[g][w] - m_all);
                    l_all += sm_l[g][w] * sc;
                    a0 = fmaf(sc, sm_o[g][w][lane +  0], a0);
                    a1 = fmaf(sc, sm_o[g][w][lane + 32], a1);
                    a2 = fmaf(sc, sm_o[g][w][lane + 64], a2);
                    a3 = fmaf(sc, sm_o[g][w][lane + 96], a3);
                }
            }
        }
        float* so = split_out + (sp * n_head + h) * head_dim;
        so[lane + 0] = a0; so[lane + 32] = a1; so[lane + 64] = a2; so[lane + 96] = a3;
        if (lane == 0) {
            split_ml[(sp * n_head + h) * 2 + 0] = m_all;
            split_ml[(sp * n_head + h) * 2 + 1] = l_all;
        }
    }
}

// GQA=3 Fused Global Paged Split-K Kernel: loads K & V ONCE for all 3 query heads -> 3x LESS GDDR6 traffic!
__global__ void k_attn_global_splitk_gqa(
    const float* __restrict__ Q,
    const uint64_t* __restrict__ page_k_ptrs, const uint64_t* __restrict__ page_v_ptrs,
    const int32_t* __restrict__ page_pos0, const int32_t* __restrict__ page_lens,
    int n_pages, int n_splits, const float* __restrict__ rel_table,
    float* __restrict__ split_out, float* __restrict__ split_ml,
    int n_head, int n_head_kv, int head_dim, int pos_q, int rel_extent, int kv_mode)
{
    int h_kv = blockIdx.x, sp = blockIdx.y;
    if (h_kv >= n_head_kv || sp >= n_splits) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    int p_start = (n_pages * sp) / n_splits;
    int p_end   = (n_pages * (sp + 1)) / n_splits;

    float q0[3], q1[3], q2[3], q3[3];
    const float* rtab[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h_kv * 3 + g;
        const float* qh = Q + h * head_dim;
        q0[g] = qh[lane + 0]; q1[g] = qh[lane + 32]; q2[g] = qh[lane + 64]; q3[g] = qh[lane + 96];
        rtab[g] = rel_table + h * rel_extent;
    }

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    const size_t hb = kv_head_bytes_dev(kv_mode);
    const size_t tok_stride = (size_t)n_head_kv * hb;

    float m_w[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float l_w[3] = {0.0f, 0.0f, 0.0f};
    float o0[3] = {0.0f, 0.0f, 0.0f}, o1[3] = {0.0f, 0.0f, 0.0f};
    float o2[3] = {0.0f, 0.0f, 0.0f}, o3[3] = {0.0f, 0.0f, 0.0f};

    for (int p = p_start; p < p_end; ++p) {
        const uint8_t* kp_base = (const uint8_t*)(uintptr_t)__ldg(&page_k_ptrs[p]);
        const uint8_t* vp_base = (const uint8_t*)(uintptr_t)__ldg(&page_v_ptrs[p]);
        int p0 = __ldg(&page_pos0[p]), plen = __ldg(&page_lens[p]);

        for (int t = warp; t < plen; t += 4) {
            int dist = pos_q - (p0 + t);
            if (dist < 0) continue;
            bool in_rel = (dist < rel_extent);

            const uint8_t* kp = kp_base + (size_t)t * tok_stride + (size_t)h_kv * hb;
            const uint8_t* vp = vp_base + (size_t)t * tok_stride + (size_t)h_kv * hb;
            float4 kv = load_kv_head_4(kp, lane, kv_mode);
            float4 vv = load_kv_head_4(vp, lane, kv_mode);

            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                float dot = warp_reduce_sum(q0[g] * kv.x + q1[g] * kv.y + q2[g] * kv.z + q3[g] * kv.w);
                float bias = in_rel ? __ldg(&rtab[g][dist]) : 0.0f;
                float score = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d + bias;
                float m_new = fmaxf(m_w[g], score);
                float alpha = (m_w[g] == -FLT_MAX) ? 0.0f : expf(m_w[g] - m_new);
                float prob  = expf(score - m_new);
                l_w[g] = l_w[g] * alpha + prob;
                o0[g]  = fmaf(o0[g], alpha, prob * vv.x);
                o1[g]  = fmaf(o1[g], alpha, prob * vv.y);
                o2[g]  = fmaf(o2[g], alpha, prob * vv.z);
                o3[g]  = fmaf(o3[g], alpha, prob * vv.w);
                m_w[g] = m_new;
            }
        }
    }

    __shared__ float sm_m[3][4], sm_l[3][4], sm_o[3][4][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[g][warp] = m_w[g]; sm_l[g][warp] = l_w[g]; }
        sm_o[g][warp][lane +  0] = o0[g];
        sm_o[g][warp][lane + 32] = o1[g];
        sm_o[g][warp][lane + 64] = o2[g];
        sm_o[g][warp][lane + 96] = o3[g];
    }
    __syncthreads();

    if (warp < 3) {
        int g = warp;
        int h = h_kv * 3 + g;
        float m_all = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < 4; ++w) m_all = fmaxf(m_all, sm_m[g][w]);

        float l_all = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if (m_all > -FLT_MAX * 0.5f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[g][w] > -FLT_MAX * 0.5f) {
                    float sc = expf(sm_m[g][w] - m_all);
                    l_all += sm_l[g][w] * sc;
                    a0 = fmaf(sc, sm_o[g][w][lane +  0], a0);
                    a1 = fmaf(sc, sm_o[g][w][lane + 32], a1);
                    a2 = fmaf(sc, sm_o[g][w][lane + 64], a2);
                    a3 = fmaf(sc, sm_o[g][w][lane + 96], a3);
                }
            }
        }
        float* so = split_out + (sp * n_head + h) * head_dim;
        so[lane + 0] = a0; so[lane + 32] = a1; so[lane + 64] = a2; so[lane + 96] = a3;
        if (lane == 0) {
            split_ml[(sp * n_head + h) * 2 + 0] = m_all;
            split_ml[(sp * n_head + h) * 2 + 1] = l_all;
        }
    }
}

__global__ void k_attn_global_reduce(const float* __restrict__ split_out, const float* __restrict__ split_ml,
                                     const float* __restrict__ sinks, float* __restrict__ out,
                                     int n_splits, int n_head, int head_dim) {
    int h = blockIdx.x, lane = threadIdx.x;
    if (h >= n_head) return;

    float sink_t = sinks[h];
    float m_all = sink_t;
    for (int sp = 0; sp < n_splits; ++sp) m_all = fmaxf(m_all, split_ml[(sp * n_head + h) * 2 + 0]);

    float denom = expf(sink_t - m_all);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        float ms = split_ml[(sp * n_head + h) * 2 + 0];
        float ls = split_ml[(sp * n_head + h) * 2 + 1];
        if (ms > -FLT_MAX * 0.5f) {
            float sc = expf(ms - m_all);
            denom += ls * sc;
            const float* so = split_out + (sp * n_head + h) * head_dim;
            a0 = fmaf(sc, so[lane +  0], a0);
            a1 = fmaf(sc, so[lane + 32], a1);
            a2 = fmaf(sc, so[lane + 64], a2);
            a3 = fmaf(sc, so[lane + 96], a3);
        }
    }
    float inv_d = 1.0f / denom;
    float* oh = out + h * head_dim;
    oh[lane + 0] = a0 * inv_d; oh[lane + 32] = a1 * inv_d;
    oh[lane + 64] = a2 * inv_d; oh[lane + 96] = a3 * inv_d;
}

void ink_cuda_attn_swa(const float* Q, const void* K_ring, const void* V_ring, const int32_t* pos_ring,
                       const float* rel_table, const float* sinks, float* split_out, float* split_ml, float* out,
                       int n_head, int n_head_kv, int head_dim, int swa_len, int pos_q,
                       int rel_extent, KVQuantMode kv_mode, cudaStream_t stream) {
    int n_splits = (swa_len >= 256) ? 4 : ((swa_len >= 64) ? 2 : 1);
    dim3 grid(n_head_kv, n_splits);
    k_attn_swa_gqa_splitk<<<grid, 128, 0, stream>>>(
        Q, (const uint8_t*)K_ring, (const uint8_t*)V_ring, pos_ring,
        rel_table, split_out, split_ml, n_head, n_head_kv, head_dim, swa_len, n_splits, pos_q, rel_extent, (int)kv_mode);
    k_attn_global_reduce<<<n_head, 32, 0, stream>>>(
        split_out, split_ml, sinks, out, n_splits, n_head, head_dim);
}

void ink_cuda_attn_global_paged(
    const float* Q, const uint64_t* page_k_ptrs, const uint64_t* page_v_ptrs,
    const int32_t* page_pos0, const int32_t* page_lens, int n_pages,
    const float* rel_table, const float* sinks,
    float* split_out, float* split_ml, float* out,
    int n_head, int n_head_kv, int head_dim, int pos_q, int rel_extent,
    KVQuantMode kv_mode, cudaStream_t stream)
{
    int n_splits = 1;
    if (n_pages >= 256)     n_splits = 32; // 8 * 32 = 256 blocks saturating all 24 SMs!
    else if (n_pages >= 64) n_splits = 16; // 8 * 16 = 128 blocks!
    else if (n_pages >= 16) n_splits = 8;
    else if (n_pages >= 4)  n_splits = 4;

    dim3 grid(n_head_kv, n_splits);
    k_attn_global_splitk_gqa<<<grid, 128, 0, stream>>>(
        Q, page_k_ptrs, page_v_ptrs, page_pos0, page_lens,
        n_pages, n_splits, rel_table, split_out, split_ml,
        n_head, n_head_kv, head_dim, pos_q, rel_extent, (int)kv_mode);
    k_attn_global_reduce<<<n_head, 32, 0, stream>>>(
        split_out, split_ml, sinks, out, n_splits, n_head, head_dim);
}

__global__ void k_quest_score_q8(const float* __restrict__ Q,
                                 const int8_t* __restrict__ k_mins_q8, const int8_t* __restrict__ k_maxs_q8,
                                 const float* __restrict__ k_scales,
                                 float* __restrict__ page_scores,
                                 int n_total_pages, int n_head, int n_head_kv, int head_dim) {
    int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (p >= n_total_pages) return;

    int gqa = n_head / n_head_kv;
    const int8_t* pmin = k_mins_q8 + (size_t)p * n_head_kv * head_dim;
    const int8_t* pmax = k_maxs_q8 + (size_t)p * n_head_kv * head_dim;
    const float*  psc  = k_scales  + (size_t)p * n_head_kv;

    float best_page = -FLT_MAX;
    for (int h_kv = 0; h_kv < n_head_kv; ++h_kv) {
        float sc = __ldg(&psc[h_kv]);
        const int8_t* mn = pmin + h_kv * head_dim;
        const int8_t* mx = pmax + h_kv * head_dim;
        float mn0 = (float)__ldg(&mn[lane]),      mn1 = (float)__ldg(&mn[lane + 32]);
        float mn2 = (float)__ldg(&mn[lane + 64]), mn3 = (float)__ldg(&mn[lane + 96]);
        float mx0 = (float)__ldg(&mx[lane]),      mx1 = (float)__ldg(&mx[lane + 32]);
        float mx2 = (float)__ldg(&mx[lane + 64]), mx3 = (float)__ldg(&mx[lane + 96]);
        for (int g = 0; g < gqa; ++g) {
            const float* q = Q + (h_kv * gqa + g) * head_dim;
            float q0 = q[lane], q1 = q[lane + 32], q2 = q[lane + 64], q3 = q[lane + 96];
            float ub = warp_reduce_sum(fmaxf(q0 * mn0, q0 * mx0) + fmaxf(q1 * mn1, q1 * mx1)
                                     + fmaxf(q2 * mn2, q2 * mx2) + fmaxf(q3 * mn3, q3 * mx3)) * sc;
            if (lane == 0) best_page = fmaxf(best_page, ub);
        }
    }
    if (lane == 0) page_scores[p] = best_page;
}

void ink_cuda_quest_score_pages(const float* Q, const int8_t* k_mins_q8, const int8_t* k_maxs_q8,
                                const float* k_scales, float* page_scores, int n_total_pages,
                                int n_head, int n_head_kv, int head_dim, cudaStream_t stream) {
    k_quest_score_q8<<<(n_total_pages + 3) / 4, 128, 0, stream>>>(
        Q, k_mins_q8, k_maxs_q8, k_scales, page_scores, n_total_pages, n_head, n_head_kv, head_dim);
}

__device__ __forceinline__ uint32_t f32_to_sortable_u32(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

// 1-Block Exact O(N) 4-Pass Radix-Select GPU Top-K (~3 us for 8,192 pages, 0 CPU sorting!)
__global__ void k_quest_topk_radix(const float* __restrict__ scores, int hist_start, int hist_end,
                                   int K, int32_t* __restrict__ out_ids) {
    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_rem;
    __shared__ int      s_cnt;

    int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0u; s_rem = K; s_cnt = 0; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - pass * 8;
        uint32_t mask = (pass == 0) ? 0u : (~0u << (shift + 8));
        uint32_t pref = s_prefix;
        s_hist[tid] = 0u;
        __syncthreads();

        for (int i = hist_start + tid; i < hist_end; i += 256) {
            uint32_t k = f32_to_sortable_u32(scores[i]);
            if ((k & mask) == pref) {
                atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int rem = s_rem;
            for (int b = 255; b >= 0; --b) {
                int c = (int)s_hist[b];
                if (c >= rem) {
                    s_prefix = pref | ((uint32_t)b << shift);
                    s_rem = rem;
                    break;
                }
                rem -= c;
            }
        }
        __syncthreads();
    }

    uint32_t kth_key = s_prefix;
    // Pass 1: strictly greater than kth_key
    for (int i = hist_start + tid; i < hist_end; i += 256) {
        if (f32_to_sortable_u32(scores[i]) > kth_key) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < K) out_ids[pos] = i;
        }
    }
    __syncthreads();
    // Pass 2: ties equal to kth_key up to K
    for (int i = hist_start + tid; i < hist_end; i += 256) {
        if (f32_to_sortable_u32(scores[i]) == kth_key) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < K) out_ids[pos] = i;
        }
    }
}

void ink_cuda_quest_topk(const float* page_scores, int hist_start, int hist_end, int K,
                         int32_t* out_ids, cudaStream_t stream) {
    if (K <= 0 || hist_end <= hist_start) return;
    k_quest_topk_radix<<<1, 256, 0, stream>>>(page_scores, hist_start, hist_end, K, out_ids);
}

// =========================================================================
// High-Speed Multi-Block MoE Router (32 blocks across all 24 SMs -> ~0.25 ms total!)
// =========================================================================
__global__ void k_router_gemv(const float* __restrict__ gate_inp, const float* __restrict__ x,
                              float* __restrict__ logits, int n_expert, int n_embd) {
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int e = blockIdx.x * (blockDim.x >> 5) + warp;
    if (e >= n_expert) return;

    float v = dot_f32_vec((const uint8_t*)(gate_inp + (size_t)e * n_embd), s_x, n_embd, lane);
    if (lane == 0) logits[e] = v;
}

__global__ void k_router_topk(const float* __restrict__ logits, const float* __restrict__ bias,
                              int32_t* __restrict__ top_ids, float* __restrict__ top_weights,
                              int n_expert, int k, float w_scale) {
    // 1 warp (32 threads) handles all 128 experts in registers! (4 experts per lane)
    int lane = threadIdx.x;
    float p[4], s[4];
    bool used[4] = {false, false, false, false};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = i * 32 + lane;
        if (e < n_expert) {
            p[i] = 1.0f / (1.0f + expf(-logits[e]));
            s[i] = bias ? (p[i] + bias[e]) : p[i];
        } else {
            p[i] = 0.0f; s[i] = -FLT_MAX;
        }
    }

    float wsum = 0.0f;
    float sel_w[8];
    int32_t sel_i[8];

    for (int j = 0; j < k; ++j) {
        float best_s = -FLT_MAX;
        int best_e = -1;
        float best_p = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (!used[i] && s[i] > best_s) {
                best_s = s[i]; best_e = i * 32 + lane; best_p = p[i];
            }
        }
        // Warp-wide argmax
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float os = __shfl_xor_sync(0xFFFFFFFFu, best_s, off);
            int   oe = __shfl_xor_sync(0xFFFFFFFFu, best_e, off);
            float op = __shfl_xor_sync(0xFFFFFFFFu, best_p, off);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os; best_e = oe; best_p = op;
            }
        }
        sel_i[j] = best_e;
        sel_w[j] = best_p;
        wsum += best_p;
        if ((best_e & 31) == lane) used[best_e >> 5] = true;
    }

    if (lane == 0) {
        float inv = w_scale / (wsum + 1e-20f);
        for (int j = 0; j < k; ++j) {
            top_ids[j] = sel_i[j];
            top_weights[j] = sel_w[j] * inv;
        }
    }
}

void ink_cuda_moe_router(const float* gate_inp, const float* bias, const float* x,
                         float* logits_scratch, int32_t* top_ids, float* top_weights,
                         int n_expert, int n_embd, int k, float weights_scale, cudaStream_t stream) {
    constexpr int WPB = 4;
    int blocks = (n_expert + WPB - 1) / WPB; // 32 blocks across 24 SMs!
    k_router_gemv<<<blocks, WPB * 32, (size_t)n_embd * sizeof(float), stream>>>(gate_inp, x, logits_scratch, n_expert, n_embd);
    k_router_topk<<<1, 32, 0, stream>>>(logits_scratch, bias, top_ids, top_weights, n_expert, k, weights_scale);
}

// Fused FFN RMSNorm + Exact FP32 Router GEMV (100% lossless exact FP32 router weights!)
__global__ void k_fused_ffn_norm_router_gemv(
    const float* __restrict__ ffn_x,
    const float* __restrict__ ffn_norm_w,
    float* __restrict__ d_ffn_normed,
    const float* __restrict__ gate_inp,
    float* __restrict__ logits,
    int n_expert, int n_embd, float eps)
{
    extern __shared__ float s_x[];
    __shared__ float s_red[8];
    int tid = threadIdx.x;
    int nw  = blockDim.x >> 5;
    int n4  = n_embd >> 2;
    const float4* x4 = (const float4*)ffn_x;
    const float4* w4 = (const float4*)ffn_norm_w;

    float ss = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = x4[i];
        ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = warp_reduce_sum(ss);
    if ((tid & 31) == 0) s_red[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < nw; ++w) tot += s_red[w];
        s_red[0] = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    float inv = s_red[0];
    float4* sx4 = (float4*)s_x;
    float4* o4  = (float4*)d_ffn_normed;
    const bool write_out = (blockIdx.x == 0);
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = x4[i], g = w4[i];
        float4 nv = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z, v.w * inv * g.w);
        sx4[i] = nv;
        if (write_out) o4[i] = nv;
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int e = blockIdx.x * nw + warp;
    if (e >= n_expert) return;

    float v = dot_f32_vec((const uint8_t*)(gate_inp + (size_t)e * n_embd), s_x, n_embd, lane);
    if (lane == 0) logits[e] = v;
}

void ink_cuda_fused_ffn_norm_router(
    const float* ffn_x, const float* ffn_norm_w, float* d_ffn_normed,
    const float* gate_inp, const float* bias, float* logits_scratch,
    int32_t* top_ids, float* top_weights,
    int n_expert, int n_embd, int k, float eps, float weights_scale, cudaStream_t stream)
{
    constexpr int WPB = 4;
    int blocks = (n_expert + WPB - 1) / WPB; // 32 blocks across 24 SMs!
    k_fused_ffn_norm_router_gemv<<<blocks, WPB * 32, (size_t)n_embd * sizeof(float), stream>>>(
        ffn_x, ffn_norm_w, d_ffn_normed, gate_inp, logits_scratch, n_expert, n_embd, eps);
    k_router_topk<<<1, 32, 0, stream>>>(logits_scratch, bias, top_ids, top_weights, n_expert, k, weights_scale);
}

// =========================================================================
// Supercharged Sub-Warp __dp4a Fused Multi-Expert GateUp + SITU & Down + Reduce!
// Uses G=8 (72/8=9 exact, 0% warp waste!) and G=4 for IQ4_NL (36/4=9 exact!)
// Supports: (1) host-resolved batch, (2) 100%-resident zero-sync indirect,
//           (3) GPU-first zero-bubble dynamic split + indirect (overlaps GPU & CPU!)
// =========================================================================
template <uint32_t TYPE, int G>
__global__ void k_moe_gate_up_situ_dp4a(GPUExpertBatch batch, const float* __restrict__ x,
                                        float* __restrict__ h_out, int n_embd, int n_ff,
                                        size_t row_bytes, float beta, float lb) {
    int eidx = blockIdx.y;
    if (eidx >= batch.count) return;

    extern __shared__ Q8Block32GPU s_qx[];
    quantize_x_to_smem(x, s_qx, n_embd);
    __syncthreads();

    constexpr int SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rpb   = blockDim.x >> SHIFT;
    int gid   = threadIdx.x >> SHIFT;
    int glane = threadIdx.x & (G - 1);
    int j     = blockIdx.x * rpb + gid;
    int sj    = (j < n_ff) ? j : (n_ff - 1);

    const uint8_t* base = (const uint8_t*)batch.gate_up_ptrs[eidx];
    float sc = batch.gate_up_scales[eidx];
    float g = dot_quant_dp4a<TYPE, G>(base + (size_t)sj * row_bytes, s_qx, n_embd, glane) * sc;
    float u = dot_quant_dp4a<TYPE, G>(base + (size_t)(sj + n_ff) * row_bytes, s_qx, n_embd, glane) * sc;
    if (glane == 0 && j < n_ff) h_out[eidx * n_ff + j] = d_situ(g, u, beta, lb);
}

template <uint32_t TYPE, int G>
__global__ void k_moe_down_reduce_dp4a(GPUExpertBatch batch, const float* __restrict__ h_in,
                                       float* __restrict__ out, int n_embd, int n_ff, size_t row_bytes) {
    extern __shared__ Q8Block32GPU s_qh[];
    int nb_ff = n_ff >> 5;
    for (int k = 0; k < batch.count; ++k) {
        quantize_x_to_smem(h_in + k * n_ff, s_qh + k * nb_ff, n_ff);
    }
    __syncthreads();

    constexpr int SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rpb   = blockDim.x >> SHIFT;
    int gid   = threadIdx.x >> SHIFT;
    int glane = threadIdx.x & (G - 1);
    int row   = blockIdx.x * rpb + gid;
    int srow  = (row < n_embd) ? row : (n_embd - 1);

    float acc = 0.0f;
    for (int k = 0; k < batch.count; ++k) {
        const uint8_t* rp = (const uint8_t*)batch.down_ptrs[k] + (size_t)srow * row_bytes;
        float v = dot_quant_dp4a<TYPE, G>(rp, s_qh + k * nb_ff, n_ff, glane);
        acc = fmaf(v, batch.down_w_scales[k], acc);
    }
    if (glane == 0 && row < n_embd) out[row] = acc;
}

// ZERO-SYNC GPU-INDIRECT KERNELS FOR 100%-RESIDENT LAYERS!
template <uint32_t TYPE, int G>
__global__ void k_moe_gate_up_situ_indirect(
    const uint64_t* __restrict__ d_gu_ptrs, const float* __restrict__ d_gu_scales,
    const int32_t* __restrict__ d_top_ids, int k_cnt,
    const float* __restrict__ x, float* __restrict__ h_out,
    int n_embd, int n_ff, size_t row_bytes, float beta, float lb)
{
    int eidx = blockIdx.y;
    if (eidx >= k_cnt) return;

    extern __shared__ Q8Block32GPU s_qx[];
    quantize_x_to_smem(x, s_qx, n_embd);
    __syncthreads();

    constexpr int SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rpb   = blockDim.x >> SHIFT;
    int gid   = threadIdx.x >> SHIFT;
    int glane = threadIdx.x & (G - 1);
    int j     = blockIdx.x * rpb + gid;
    int sj    = (j < n_ff) ? j : (n_ff - 1);

    int exp_id = d_top_ids[eidx];
    const uint8_t* base = (const uint8_t*)(uintptr_t)d_gu_ptrs[exp_id];
    float sc = d_gu_scales[exp_id];

    float g = dot_quant_dp4a<TYPE, G>(base + (size_t)sj * row_bytes, s_qx, n_embd, glane) * sc;
    float u = dot_quant_dp4a<TYPE, G>(base + (size_t)(sj + n_ff) * row_bytes, s_qx, n_embd, glane) * sc;
    if (glane == 0 && j < n_ff) h_out[eidx * n_ff + j] = d_situ(g, u, beta, lb);
}

template <uint32_t TYPE, int G>
__global__ void k_moe_down_reduce_indirect(
    const uint64_t* __restrict__ d_dn_ptrs, const float* __restrict__ d_dn_scales,
    const int32_t* __restrict__ d_top_ids, const float* __restrict__ d_top_weights, int k_cnt,
    const float* __restrict__ h_in, float* __restrict__ out,
    int n_embd, int n_ff, size_t row_bytes)
{
    extern __shared__ Q8Block32GPU s_qh[];
    int nb_ff = n_ff >> 5;
    for (int k = 0; k < k_cnt; ++k) {
        quantize_x_to_smem(h_in + k * n_ff, s_qh + k * nb_ff, n_ff);
    }
    __syncthreads();

    constexpr int SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rpb   = blockDim.x >> SHIFT;
    int gid   = threadIdx.x >> SHIFT;
    int glane = threadIdx.x & (G - 1);
    int row   = blockIdx.x * rpb + gid;
    int srow  = (row < n_embd) ? row : (n_embd - 1);

    float acc = 0.0f;
    for (int k = 0; k < k_cnt; ++k) {
        int exp_id = d_top_ids[k];
        const uint8_t* rp = (const uint8_t*)(uintptr_t)d_dn_ptrs[exp_id] + (size_t)srow * row_bytes;
        float v = dot_quant_dp4a<TYPE, G>(rp, s_qh + k * nb_ff, n_ff, glane);
        acc = fmaf(v, d_dn_scales[exp_id] * d_top_weights[k], acc);
    }
    if (glane == 0 && row < n_embd) out[row] = acc;
}

// =========================================================================
// GPU-FIRST ZERO-BUBBLE HYBRID SPLIT + DYNAMIC INDIRECT KERNELS!
// Classifies GPU VRAM hits vs CPU RAM/SSD misses on GPU in <1.5us,
// copies d_x -> mapped h_ffn_in ONLY if n_cpu > 0, and lets GPU MoE start
// IMMEDIATELY without waiting for host CPU synchronization!
// =========================================================================
__global__ void k_moe_split_hits(
    const uint64_t* __restrict__ d_gu_ptrs,
    const int32_t* __restrict__ d_top_ids,
    const float* __restrict__ d_top_weights,
    int k_cnt,
    const float* __restrict__ d_x,
    int n_embd,
    int32_t* __restrict__ d_n_gpu,
    int32_t* __restrict__ d_gpu_hit_ids,
    float* __restrict__ d_gpu_hit_weights,
    MoESplitPacket* __restrict__ d_mapped_pkt,
    float* __restrict__ d_mapped_ffn_in)
{
    __shared__ int s_ncpu;
    if (threadIdx.x == 0) {
        int ng = 0, nc = 0;
        float w_all = 0.0f, w_hit = 0.0f;
        for (int i = 0; i < k_cnt; ++i) {
            int e = d_top_ids[i];
            float w = d_top_weights[i];
            w_all += w;
            if (d_mapped_pkt) d_mapped_pkt->all_ids[i] = e;
            if (d_gu_ptrs[e] != 0) {
                d_gpu_hit_ids[ng]     = e;
                d_gpu_hit_weights[ng] = w;
                w_hit += w;
                ng++;
            } else {
                if (d_mapped_pkt) {
                    d_mapped_pkt->miss_ids[nc]     = e;
                    d_mapped_pkt->miss_weights[nc] = w;
                }
                nc++;
            }
        }
        *d_n_gpu = ng;
        if (d_mapped_pkt) {
            d_mapped_pkt->n_gpu = ng;
            d_mapped_pkt->n_cpu = nc;
            s_ncpu = nc;
        } else {
            // GPU-Only 1-Loop Self-Draft mode: zero PCIe writes + renormalize VRAM-hit weights to preserve residual scale!
            if (ng > 0 && nc > 0 && w_hit > 1e-12f) {
                float sc = w_all / w_hit;
                for (int j = 0; j < ng; ++j) d_gpu_hit_weights[j] *= sc;
            }
            s_ncpu = 0;
        }
    }
    __syncthreads();

    if (s_ncpu > 0) {
        int n4 = n_embd >> 2;
        const float4* src = (const float4*)d_x;
        float4*       dst = (float4*)d_mapped_ffn_in;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
    // Every thread that wrote zero-copy pinned host memory MUST issue a system-scope fence!
    __threadfence_system();
    __syncthreads();
}

template <uint32_t TYPE, int G>
__global__ void k_moe_gate_up_situ_dyn(
    const uint64_t* __restrict__ d_gu_ptrs, const float* __restrict__ d_gu_scales,
    const int32_t* __restrict__ d_n_gpu, const int32_t* __restrict__ d_hit_ids,
    const float* __restrict__ x, float* __restrict__ h_out,
    int n_embd, int n_ff, size_t row_bytes, float beta, float lb)
{
    int n_gpu = *d_n_gpu;
    int eidx = blockIdx.y;
    if (eidx >= n_gpu) return;

    extern __shared__ Q8Block32GPU s_qx[];
    quantize_x_to_smem(x, s_qx, n_embd);
    __syncthreads();

    constexpr int SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rpb   = blockDim.x >> SHIFT;
    int gid   = threadIdx.x >> SHIFT;
    int glane = threadIdx.x & (G - 1);
    int j     = blockIdx.x * rpb + gid;
    int sj    = (j < n_ff) ? j : (n_ff - 1);

    int exp_id = d_hit_ids[eidx];
    const uint8_t* base = (const uint8_t*)(uintptr_t)d_gu_ptrs[exp_id];
    float sc = d_gu_scales[exp_id];

    float g = dot_quant_dp4a<TYPE, G>(base + (size_t)sj * row_bytes, s_qx, n_embd, glane) * sc;
    float u = dot_quant_dp4a<TYPE, G>(base + (size_t)(sj + n_ff) * row_bytes, s_qx, n_embd, glane) * sc;
    if (glane == 0 && j < n_ff) h_out[eidx * n_ff + j] = d_situ(g, u, beta, lb);
}

template <uint32_t TYPE, int G>
__global__ void k_moe_down_reduce_dyn(
    const uint64_t* __restrict__ d_dn_ptrs, const float* __restrict__ d_dn_scales,
    const int32_t* __restrict__ d_n_gpu, const int32_t* __restrict__ d_hit_ids, const float* __restrict__ d_hit_weights,
    const float* __restrict__ h_in, float* __restrict__ out,
    int n_embd, int n_ff, size_t row_bytes)
{
    constexpr int SHIFT = (G == 4) ? 2 : ((G == 8) ? 3 : 5);
    int rpb   = blockDim.x >> SHIFT;
    int gid   = threadIdx.x >> SHIFT;
    int glane = threadIdx.x & (G - 1);
    int row   = blockIdx.x * rpb + gid;

    int n_gpu = *d_n_gpu;
    if (n_gpu <= 0) {
        if (glane == 0 && row < n_embd) out[row] = 0.0f;
        return;
    }

    extern __shared__ Q8Block32GPU s_qh[];
    int nb_ff = n_ff >> 5;
    for (int k = 0; k < n_gpu; ++k) {
        quantize_x_to_smem(h_in + k * n_ff, s_qh + k * nb_ff, n_ff);
    }
    __syncthreads();

    int srow = (row < n_embd) ? row : (n_embd - 1);
    float acc = 0.0f;
    for (int k = 0; k < n_gpu; ++k) {
        int exp_id = d_hit_ids[k];
        const uint8_t* rp = (const uint8_t*)(uintptr_t)d_dn_ptrs[exp_id] + (size_t)srow * row_bytes;
        float v = dot_quant_dp4a<TYPE, G>(rp, s_qh + k * nb_ff, n_ff, glane);
        acc = fmaf(v, d_dn_scales[exp_id] * d_hit_weights[k], acc);
    }
    if (glane == 0 && row < n_embd) out[row] = acc;
}

void ink_cuda_moe_gate_up_situ(uint32_t gate_up_type, GPUExpertBatch batch, const float* x,
                               float* h_out, int n_embd, int n_ff, float situ_beta, float situ_linear_beta,
                               cudaStream_t stream) {
    if (batch.count <= 0) return;
    size_t rb = (size_t)(n_embd / ggml_type_block_size(gate_up_type)) * ggml_type_byte_size(gate_up_type);
    size_t smem = (size_t)(n_embd >> 5) * sizeof(Q8Block32GPU);
    constexpr int WPB = 8;
    dim3 grid((n_ff + WPB - 1) / WPB, batch.count);

    switch (gate_up_type) {
        case GGML_TYPE_IQ3_XXS: k_moe_gate_up_situ_dp4a<GGML_TYPE_IQ3_XXS, 32><<<grid, WPB*32, smem, stream>>>(batch, x, h_out, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
        case GGML_TYPE_IQ2_S:   k_moe_gate_up_situ_dp4a<GGML_TYPE_IQ2_S,   32><<<grid, WPB*32, smem, stream>>>(batch, x, h_out, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
        case GGML_TYPE_IQ2_XS:  k_moe_gate_up_situ_dp4a<GGML_TYPE_IQ2_XS,  32><<<grid, WPB*32, smem, stream>>>(batch, x, h_out, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
        case GGML_TYPE_IQ2_XXS: k_moe_gate_up_situ_dp4a<GGML_TYPE_IQ2_XXS, 32><<<grid, WPB*32, smem, stream>>>(batch, x, h_out, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
        case GGML_TYPE_NVFP4:   k_moe_gate_up_situ_dp4a<GGML_TYPE_NVFP4,   32><<<grid, WPB*32, smem, stream>>>(batch, x, h_out, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
        case GGML_TYPE_IQ4_NL:  k_moe_gate_up_situ_dp4a<GGML_TYPE_IQ4_NL,  32><<<grid, WPB*32, smem, stream>>>(batch, x, h_out, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
        default: std::fprintf(stderr, "[CUDA] Unsupported MoE gate_up %u\n", gate_up_type); std::abort();
    }
}

void ink_cuda_moe_down_reduce(uint32_t down_type, GPUExpertBatch batch, const float* h_in,
                              float* out, int n_embd, int n_ff, cudaStream_t stream) {
    if (batch.count <= 0) {
        INK_CUDA_CHECK(cudaMemsetAsync(out, 0, (size_t)n_embd * sizeof(float), stream));
        return;
    }
    size_t rb = (size_t)(n_ff / ggml_type_block_size(down_type)) * ggml_type_byte_size(down_type);
    constexpr int WPB = 8;
    int blocks = (n_embd + WPB - 1) / WPB;
    size_t smem = (size_t)batch.count * (n_ff >> 5) * sizeof(Q8Block32GPU);

    switch (down_type) {
        case GGML_TYPE_NVFP4:   k_moe_down_reduce_dp4a<GGML_TYPE_NVFP4,   32><<<blocks, WPB*32, smem, stream>>>(batch, h_in, out, n_embd, n_ff, rb); break;
        case GGML_TYPE_IQ4_NL:  k_moe_down_reduce_dp4a<GGML_TYPE_IQ4_NL,  32><<<blocks, WPB*32, smem, stream>>>(batch, h_in, out, n_embd, n_ff, rb); break;
        case GGML_TYPE_IQ3_XXS: k_moe_down_reduce_dp4a<GGML_TYPE_IQ3_XXS, 32><<<blocks, WPB*32, smem, stream>>>(batch, h_in, out, n_embd, n_ff, rb); break;
        default: std::fprintf(stderr, "[CUDA] Unsupported MoE down %u\n", down_type); std::abort();
    }
}

void ink_cuda_moe_pinned_indirect(
    uint32_t gate_up_type, uint32_t down_type,
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_top_ids, const float* d_top_weights, int k,
    const float* x, float* h_scratch, float* out,
    int n_embd, int n_ff, float situ_beta, float situ_linear_beta, cudaStream_t stream)
{
    constexpr int WPB = 8;
    {
        size_t rb = (size_t)(n_embd / ggml_type_block_size(gate_up_type)) * ggml_type_byte_size(gate_up_type);
        dim3 grid((n_ff + WPB - 1) / WPB, k);
        size_t smem = (size_t)(n_embd >> 5) * sizeof(Q8Block32GPU);
        switch (gate_up_type) {
            case GGML_TYPE_IQ3_XXS: k_moe_gate_up_situ_indirect<GGML_TYPE_IQ3_XXS, 32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_top_ids, k, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_S:   k_moe_gate_up_situ_indirect<GGML_TYPE_IQ2_S,   32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_top_ids, k, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_XS:  k_moe_gate_up_situ_indirect<GGML_TYPE_IQ2_XS,  32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_top_ids, k, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_XXS: k_moe_gate_up_situ_indirect<GGML_TYPE_IQ2_XXS, 32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_top_ids, k, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            default: std::fprintf(stderr, "[CUDA] Unsupported indirect gate_up %u\n", gate_up_type); std::abort();
        }
    }
    {
        size_t rb = (size_t)(n_ff / ggml_type_block_size(down_type)) * ggml_type_byte_size(down_type);
        int blocks = (n_embd + WPB - 1) / WPB;
        size_t smem = (size_t)k * (n_ff >> 5) * sizeof(Q8Block32GPU);
        switch (down_type) {
            case GGML_TYPE_NVFP4:  k_moe_down_reduce_indirect<GGML_TYPE_NVFP4,  32><<<blocks, WPB*32, smem, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_top_ids, d_top_weights, k, h_scratch, out, n_embd, n_ff, rb); break;
            case GGML_TYPE_IQ4_NL: k_moe_down_reduce_indirect<GGML_TYPE_IQ4_NL, 32><<<blocks, WPB*32, smem, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_top_ids, d_top_weights, k, h_scratch, out, n_embd, n_ff, rb); break;
            default: std::fprintf(stderr, "[CUDA] Unsupported indirect down %u\n", down_type); std::abort();
        }
    }
}

void ink_cuda_moe_split_hits(
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_top_ids, const float* d_top_weights, int k,
    const float* d_x, int n_embd,
    int32_t* d_n_gpu, int32_t* d_gpu_hit_ids, float* d_gpu_hit_weights,
    MoESplitPacket* d_mapped_pkt, float* d_mapped_ffn_in, cudaStream_t stream)
{
    k_moe_split_hits<<<1, 256, 0, stream>>>(
        tbl.d_gu_ptrs, d_top_ids, d_top_weights, k, d_x, n_embd,
        d_n_gpu, d_gpu_hit_ids, d_gpu_hit_weights, d_mapped_pkt, d_mapped_ffn_in);
}

void ink_cuda_moe_dynamic_indirect(
    uint32_t gate_up_type, uint32_t down_type,
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_n_gpu, const int32_t* d_gpu_hit_ids, const float* d_gpu_hit_weights, int max_k,
    const float* x, float* h_scratch, float* out,
    int n_embd, int n_ff, float situ_beta, float situ_linear_beta, cudaStream_t stream)
{
    constexpr int WPB = 8;
    {
        size_t rb = (size_t)(n_embd / ggml_type_block_size(gate_up_type)) * ggml_type_byte_size(gate_up_type);
        dim3 grid((n_ff + WPB - 1) / WPB, max_k);
        size_t smem = (size_t)(n_embd >> 5) * sizeof(Q8Block32GPU);
        switch (gate_up_type) {
            case GGML_TYPE_IQ3_XXS: k_moe_gate_up_situ_dyn<GGML_TYPE_IQ3_XXS, 32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_n_gpu, d_gpu_hit_ids, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_S:   k_moe_gate_up_situ_dyn<GGML_TYPE_IQ2_S,   32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_n_gpu, d_gpu_hit_ids, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_XS:  k_moe_gate_up_situ_dyn<GGML_TYPE_IQ2_XS,  32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_n_gpu, d_gpu_hit_ids, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_XXS: k_moe_gate_up_situ_dyn<GGML_TYPE_IQ2_XXS, 32><<<grid, WPB*32, smem, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_n_gpu, d_gpu_hit_ids, x, h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            default: std::fprintf(stderr, "[CUDA] Unsupported dyn gate_up %u\n", gate_up_type); std::abort();
        }
    }
    {
        size_t rb = (size_t)(n_ff / ggml_type_block_size(down_type)) * ggml_type_byte_size(down_type);
        int blocks = (n_embd + WPB - 1) / WPB;
        size_t smem = (size_t)max_k * (n_ff >> 5) * sizeof(Q8Block32GPU);
        switch (down_type) {
            case GGML_TYPE_NVFP4:  k_moe_down_reduce_dyn<GGML_TYPE_NVFP4,  32><<<blocks, WPB*32, smem, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_n_gpu, d_gpu_hit_ids, d_gpu_hit_weights, h_scratch, out, n_embd, n_ff, rb); break;
            case GGML_TYPE_IQ4_NL: k_moe_down_reduce_dyn<GGML_TYPE_IQ4_NL, 32><<<blocks, WPB*32, smem, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_n_gpu, d_gpu_hit_ids, d_gpu_hit_weights, h_scratch, out, n_embd, n_ff, rb); break;
            default: std::fprintf(stderr, "[CUDA] Unsupported dyn down %u\n", down_type); std::abort();
        }
    }
}

// =========================================================================
// Logit Softcap + Argmax
// =========================================================================
__global__ void k_softcap(float* logits, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && cap > 0.0f) logits[i] = cap * tanhf(logits[i] / cap);
}

__global__ void k_argmax_stage1(const float* logits, int n, float* b_vals, int32_t* b_idx) {
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    int tid = threadIdx.x;
    float best_v = -FLT_MAX;
    int32_t best_i = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFFu, best_v, off);
        int32_t oi = __shfl_down_sync(0xFFFFFFFFu, best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = best_v; si[tid >> 5] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = sv[0]; int32_t bi = si[0];
        for (int w = 1; w < 8; ++w) { if (sv[w] > bv) { bv = sv[w]; bi = si[w]; } }
        b_vals[blockIdx.x] = bv; b_idx[blockIdx.x] = bi;
    }
}

__global__ void k_argmax_stage2(const float* b_vals, const int32_t* b_idx, int nb, int32_t* out_idx) {
    float bv = -FLT_MAX;
    int32_t bi = 0;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        if (b_vals[i] > bv) { bv = b_vals[i]; bi = b_idx[i]; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        int32_t oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
        if (ov > bv) { bv = ov; bi = oi; }
    }
    if (threadIdx.x == 0) *out_idx = bi;
}

void ink_cuda_logit_softcap_argmax(float* logits, int n_unpadded, float softcap, int32_t* d_argmax, cudaStream_t stream) {
    k_softcap<<<(n_unpadded + 255) / 256, 256, 0, stream>>>(logits, n_unpadded, softcap);
    if (d_argmax) {
        k_argmax_stage1<<<256, 256, 0, stream>>>(logits, n_unpadded, g_argmax_bvals, g_argmax_bidx);
        k_argmax_stage2<<<1, 32, 0, stream>>>(g_argmax_bvals, g_argmax_bidx, 256, d_argmax);
    }
}

// =========================================================================
// Stage 4 & Stage 5: Batched GEMM Prefill (B <= 32) & Speculative Verification
// Pre-quantizes all B activations ONCE into 100%-L2-resident Q8Block32GPU buffer
// -> eliminates >1,000,000 redundant quantize_x_to_smem loops & unlocks 100% SM occupancy!
// =========================================================================
// =========================================================================
// InkEngine v3.0 WMMA FP16->FP32 Tensor Core Batched GEMM & Grouped MoE!
// Exact 8-element cooperative dequantizer for all 10 GGUF quant formats!
// =========================================================================
__global__ void k_f32_to_f16_vec(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

template <uint32_t TYPE>
__device__ __forceinline__ void dequant_sub32_8(const uint8_t* row, int sub, int chunk8, __half* dst8) {
    if constexpr (TYPE == GGML_TYPE_Q8_0) {
        const uint8_t* bp = row + sub * 34;
        float dw = d_fp16_to_f32(load_u16_ua(bp));
        const int8_t* qp = (const int8_t*)(bp + 2) + chunk8 * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst8[i] = __float2half_rn(dw * (float)qp[i]);
    } else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        int ib = sub >> 3, is = sub & 7, ic = is >> 1, lshift = (is & 1) << 2;
        const BlockQ4_K* b = (const BlockQ4_K*)(row + ib * 144);
        float d = d_fp16_to_f32(b->d), dmin = d_fp16_to_f32(b->dmin);
        uint8_t sc, m;
        d_get_scale_min_k4(is, b->scales, sc, m);
        float scale = d * (float)sc, minv = dmin * (float)m;
        const uint8_t* qp = b->qs + ic * 32 + chunk8 * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst8[i] = __float2half_rn(fmaf(scale, (float)((qp[i] >> lshift) & 0xF), -minv));
    } else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        int ib = sub >> 3, is = sub & 7, ic = is >> 1, lshift = (is & 1) << 2;
        const BlockQ5_K* b = (const BlockQ5_K*)(row + ib * 176);
        float d = d_fp16_to_f32(b->d), dmin = d_fp16_to_f32(b->dmin);
        uint8_t sc, m;
        d_get_scale_min_k4(is, b->scales, sc, m);
        float scale = d * (float)sc, minv = dmin * (float)m;
        const uint8_t* qp = b->qs + ic * 32 + chunk8 * 8;
        const uint8_t* qh = b->qh + chunk8 * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q = ((qp[i] >> lshift) & 0xF) | (((qh[i] >> is) & 1) << 4);
            dst8[i] = __float2half_rn(fmaf(scale, (float)q, -minv));
        }
    } else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        int ib = sub >> 3, half = (sub >> 2) & 1, qpart = sub & 3;
        const uint8_t* bp = row + ib * 210;
        const uint8_t* ql = bp + half * 64 + (qpart & 1) * 32 + chunk8 * 8;
        const uint8_t* qh = bp + 128 + half * 32 + chunk8 * 8;
        const int8_t* sc_arr = (const int8_t*)(bp + 192) + half * 8 + qpart * 2;
        float d = d_fp16_to_f32(load_u16_ua(bp + 208)) * (float)sc_arr[chunk8 >> 1];
        int ql_shift = (qpart >> 1) << 2, qh_shift = qpart << 1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q = ((ql[i] >> ql_shift) & 0xF) | (((qh[i] >> qh_shift) & 0x3) << 4);
            dst8[i] = __float2half_rn(d * (float)(q - 32));
        }
    } else if constexpr (TYPE == GGML_TYPE_NVFP4) {
        int sub16 = (sub << 1) | (chunk8 >> 1);
        int hi = chunk8 & 1;
        int ib = sub16 >> 2, s = sub16 & 3;
        const BlockNVFP4* b = (const BlockNVFP4*)(row + ib * 36);
        float dw = __ldg(&d_ue4m3_half[b->d[s]]);
        const uint8_t* qp = b->qs + s * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int nib = hi ? (qp[i] >> 4) : (qp[i] & 0xF);
            dst8[i] = __float2half_rn(dw * (float)__ldg(&d_kvalues_fp4[nib]));
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ4_NL) {
        const uint8_t* bp = row + sub * 18;
        float dw = d_fp16_to_f32(load_u16_ua(bp));
        const uint8_t* qs = bp + 2 + (chunk8 & 1) * 8;
        int hi = chunk8 >> 1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int nib = hi ? (qs[i] >> 4) : (qs[i] & 0xF);
            dst8[i] = __float2half_rn(dw * (float)__ldg(&d_kvalues_iq4nl[nib]));
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ3_XXS) {
        int ib = sub >> 3, ib32 = sub & 7, l = chunk8;
        const uint8_t* bp = row + ib * 98;
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.5f;
        const uint8_t* qs = bp + 2 + ib32 * 8;
        uint32_t aux32 = load_u32_ua(bp + 2 + 64 + ib32 * 4);
        float db = d * (0.5f + (float)(aux32 >> 28));
        uint8_t signs = __ldg(&d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127]);
        uint32_t g1 = __ldg(&d_iq3xxs_grid[qs[2 * l + 0]]);
        uint32_t g2 = __ldg(&d_iq3xxs_grid[qs[2 * l + 1]]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float v1 = (float)((g1 >> (8 * i)) & 0xFFu);
            if (signs & (1u << i)) v1 = -v1;
            dst8[i] = __float2half_rn(db * v1);
            float v2 = (float)((g2 >> (8 * i)) & 0xFFu);
            if (signs & (1u << (i + 4))) v2 = -v2;
            dst8[4 + i] = __float2half_rn(db * v2);
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ2_S) {
        int ib = sub >> 3, ib32 = sub & 7, l = chunk8;
        const uint8_t* bp = row + ib * 82;
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.25f;
        const uint8_t* qs = bp + 2 + ib32 * 4;
        const uint8_t* signs = bp + 2 + 32 + ib32 * 4;
        uint32_t qhb = bp[2 + 64 + ib32];
        uint8_t sc = bp[2 + 72 + ib32];
        float db = d * (0.5f + (float)((l < 2) ? (sc & 0xF) : (sc >> 4)));
        uint32_t qidx = qs[l] | (((uint32_t)qhb << (8 - 2 * l)) & 0x300u);
        uint64_t g = __ldg(&d_iq2s_grid[qidx]);
        uint8_t sg = signs[l];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = (float)((g >> (8 * i)) & 0xFFu);
            if (sg & (1u << i)) v = -v;
            dst8[i] = __float2half_rn(db * v);
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ2_XS) {
        int ib = sub >> 3, ib32 = sub & 7, l = chunk8;
        const uint8_t* bp = row + ib * 74;
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.25f;
        const uint8_t* qp = bp + 2 + ib32 * 8;
        uint8_t sc = bp[2 + 64 + ib32];
        float db = d * (0.5f + (float)((l < 2) ? (sc & 0xF) : (sc >> 4)));
        uint16_t q = load_u16_ua(qp + l * 2);
        uint64_t g = __ldg(&d_iq2xs_grid[q & 511]);
        uint8_t sg = __ldg(&d_ksigns_iq2xs[q >> 9]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = (float)((g >> (8 * i)) & 0xFFu);
            if (sg & (1u << i)) v = -v;
            dst8[i] = __float2half_rn(db * v);
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ2_XXS) {
        int ib = sub >> 3, ib32 = sub & 7, l = chunk8;
        const uint8_t* bp = row + ib * 66;
        float d = d_fp16_to_f32(load_u16_ua(bp)) * 0.25f;
        const uint8_t* qp = bp + 2 + ib32 * 8;
        uint32_t aux1 = load_u32_ua(qp + 4);
        float db = d * (0.5f + (float)(aux1 >> 28));
        uint64_t g = __ldg(&d_iq2xxs_grid[qp[l]]);
        uint8_t sg = __ldg(&d_ksigns_iq2xs[(aux1 >> (7 * l)) & 127]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = (float)((g >> (8 * i)) & 0xFFu);
            if (sg & (1u << i)) v = -v;
            dst8[i] = __float2half_rn(db * v);
        }
    }
}

template <uint32_t TYPE>
__global__ void k_gemm_wmma_batch(const uint8_t* __restrict__ W, const __half* __restrict__ X_f16,
                                  float* __restrict__ Y, int B, int M, int K, size_t row_bytes) {
    using namespace nvcuda;
    int row0 = blockIdx.x * 32;
    int b0   = blockIdx.y * 32;
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int wm   = warp >> 1; // 0..1
    int wn   = warp & 1;  // 0..1

    __shared__ __half s_A[32][40];
    __shared__ __half s_B[32][40];
    __shared__ float  s_C[32][32];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int r_t = tid >> 2; // 0..31
    int c8  = tid & 3;  // 0..3
    int n_sub = K >> 5;

    for (int sub = 0; sub < n_sub; ++sub) {
        int k_off = sub * 32 + c8 * 8;
        if (b0 + r_t < B) {
            const __half* src = X_f16 + (size_t)(b0 + r_t) * K + k_off;
            *(uint4*)(&s_A[r_t][c8 * 8]) = *(const uint4*)src;
        } else {
            *(uint4*)(&s_A[r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
        }

        if (row0 + r_t < M) {
            dequant_sub32_8<TYPE>(W + (size_t)(row0 + r_t) * row_bytes, sub, c8, &s_B[r_t][c8 * 8]);
        } else {
            *(uint4*)(&s_B[r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &s_A[wm * 16][ks], 40);
            wmma::load_matrix_sync(b_frag, &s_B[wn * 16][ks], 40);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(&s_C[wm * 16][wn * 16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < 32 * 32; i += 128) {
        int r = i >> 5;
        int c = i & 31;
        if (b0 + r < B && row0 + c < M) {
            Y[(size_t)(b0 + r) * M + (row0 + c)] = s_C[r][c];
        }
    }
}

__global__ void k_quantize_batch_q8(const float* __restrict__ X, Q8Block32GPU* __restrict__ qx_out, int K) {
    int b = blockIdx.x;
    int nb_k = K >> 5;
    quantize_x_to_smem(X + (size_t)b * K, qx_out + (size_t)b * nb_k, K);
}

template <uint32_t TYPE>
__global__ void k_gemm_dp4a_batch(const uint8_t* __restrict__ W, const Q8Block32GPU* __restrict__ qx_all,
                                  float* __restrict__ Y, int B, int M, int K, size_t row_bytes) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= M) return;

    int nb_k = K >> 5;
    const uint8_t* wrow = W + (size_t)row * row_bytes;
    // Weight row (~1 KB) stays hot in SM L1 across ALL B tokens (up to 64x less GDDR6 weight traffic!)
    for (int t = 0; t < B; ++t) {
        float val = dot_quant_dp4a<TYPE, 32>(wrow, qx_all + (size_t)t * nb_k, K, lane);
        if (lane == 0) Y[(size_t)t * M + row] = val;
    }
}

template <uint32_t TYPE>
__global__ void k_gemm_float_batch(const uint8_t* __restrict__ W, const float* __restrict__ X,
                                   float* __restrict__ Y, int B, int M, int K, size_t row_bytes) {
    constexpr int TB = 8;
    constexpr int TK = 256;
    int b0 = blockIdx.y * TB;
    int tb = min(TB, B - b0);
    if (tb <= 0) return;

    __shared__ float s_x[TB][TK];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = blockIdx.x * (blockDim.x >> 5) + warp;
    bool valid = (row < M);
    const uint8_t* wrow = valid ? (W + (size_t)row * row_bytes) : W;

    float acc[TB];
    #pragma unroll
    for (int t = 0; t < TB; ++t) acc[t] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        int tk = min(TK, K - k0);
        for (int idx = threadIdx.x; idx < tb * tk; idx += blockDim.x) {
            int t = idx / tk;
            int k = idx - t * tk;
            s_x[t][k] = X[(size_t)(b0 + t) * K + k0 + k];
        }
        __syncthreads();

        if (valid) {
            for (int t = 0; t < tb; ++t) {
                float v = 0.0f;
                if constexpr (TYPE == GGML_TYPE_F32) {
                    const float* wf = (const float*)wrow + k0;
                    for (int k = lane; k < tk; k += 32) v = fmaf(wf[k], s_x[t][k], v);
                } else if constexpr (TYPE == GGML_TYPE_F16) {
                    const uint16_t* wh = (const uint16_t*)wrow + k0;
                    for (int k = lane; k < tk; k += 32) v = fmaf(d_fp16_to_f32(wh[k]), s_x[t][k], v);
                } else {
                    const uint16_t* wb = (const uint16_t*)wrow + k0;
                    for (int k = lane; k < tk; k += 32) v = fmaf(d_bf16_to_f32(wb[k]), s_x[t][k], v);
                }
                acc[t] += v;
            }
        }
        __syncthreads();
    }

    if (valid) {
        for (int t = 0; t < tb; ++t) {
            float s = warp_reduce_sum(acc[t]);
            if (lane == 0) Y[(size_t)(b0 + t) * M + row] = s;
        }
    }
}

void ink_cuda_gemm_batch(uint32_t type, const void* W, const float* X, float* Y,
                         int B, int M, int K, cudaStream_t stream) {
    if (B <= 0) return;
    if (B == 1) {
        ink_cuda_gemv(type, W, X, Y, M, K, stream);
        return;
    }
    int bs = ggml_type_block_size(type);
    size_t row_bytes = (size_t)(K / bs) * ggml_type_byte_size(type);
    const uint8_t* Wp = (const uint8_t*)W;
    constexpr int WPB = 8;
    dim3 grid_f((M + WPB - 1) / WPB, (B + 7) / 8);

    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        if (type == GGML_TYPE_F32)       k_gemm_float_batch<GGML_TYPE_F32> <<<grid_f, WPB * 32, 0, stream>>>(Wp, X, Y, B, M, K, row_bytes);
        else if (type == GGML_TYPE_F16)  k_gemm_float_batch<GGML_TYPE_F16> <<<grid_f, WPB * 32, 0, stream>>>(Wp, X, Y, B, M, K, row_bytes);
        else                             k_gemm_float_batch<GGML_TYPE_BF16><<<grid_f, WPB * 32, 0, stream>>>(Wp, X, Y, B, M, K, row_bytes);
        return;
    }

    if (B >= 8) {
        int tot_x = B * K;
        k_f32_to_f16_vec<<<(tot_x + 255) / 256, 256, 0, stream>>>(X, g_batch_f16_buf, tot_x);
        dim3 grid_w((M + 31) / 32, (B + 31) / 32);
        switch (type) {
            case GGML_TYPE_Q8_0:    k_gemm_wmma_batch<GGML_TYPE_Q8_0>   <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_Q4_K:    k_gemm_wmma_batch<GGML_TYPE_Q4_K>   <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_Q5_K:    k_gemm_wmma_batch<GGML_TYPE_Q5_K>   <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_Q6_K:    k_gemm_wmma_batch<GGML_TYPE_Q6_K>   <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_NVFP4:   k_gemm_wmma_batch<GGML_TYPE_NVFP4>  <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_IQ4_NL:  k_gemm_wmma_batch<GGML_TYPE_IQ4_NL> <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_IQ3_XXS: k_gemm_wmma_batch<GGML_TYPE_IQ3_XXS><<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_IQ2_S:   k_gemm_wmma_batch<GGML_TYPE_IQ2_S>  <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_IQ2_XS:  k_gemm_wmma_batch<GGML_TYPE_IQ2_XS> <<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            case GGML_TYPE_IQ2_XXS: k_gemm_wmma_batch<GGML_TYPE_IQ2_XXS><<<grid_w, 128, 0, stream>>>(Wp, g_batch_f16_buf, Y, B, M, K, row_bytes); return;
            default: break;
        }
    }

    int blocks_q = (M + WPB - 1) / WPB;
    Q8Block32GPU* d_qx = (Q8Block32GPU*)g_batch_qx_buf;
    k_quantize_batch_q8<<<B, 256, 0, stream>>>(X, d_qx, K);
    switch (type) {
        case GGML_TYPE_Q8_0:    k_gemm_dp4a_batch<GGML_TYPE_Q8_0>   <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_Q4_K:    k_gemm_dp4a_batch<GGML_TYPE_Q4_K>   <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_Q5_K:    k_gemm_dp4a_batch<GGML_TYPE_Q5_K>   <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_Q6_K:    k_gemm_dp4a_batch<GGML_TYPE_Q6_K>   <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_NVFP4:   k_gemm_dp4a_batch<GGML_TYPE_NVFP4>  <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_IQ4_NL:  k_gemm_dp4a_batch<GGML_TYPE_IQ4_NL> <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_IQ3_XXS: k_gemm_dp4a_batch<GGML_TYPE_IQ3_XXS><<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_IQ2_S:   k_gemm_dp4a_batch<GGML_TYPE_IQ2_S>  <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_IQ2_XS:  k_gemm_dp4a_batch<GGML_TYPE_IQ2_XS> <<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        case GGML_TYPE_IQ2_XXS: k_gemm_dp4a_batch<GGML_TYPE_IQ2_XXS><<<blocks_q, WPB * 32, 0, stream>>>(Wp, d_qx, Y, B, M, K, row_bytes); break;
        default: std::fprintf(stderr, "[CUDA] Unsupported batch GEMM type %u\n", type); std::abort();
    }
}

__global__ void k_rms_norm_batch(const float* __restrict__ x, const float* __restrict__ w,
                                 float* __restrict__ out, int n, float eps) {
    int b = blockIdx.x;
    const float4* x4 = (const float4*)(x + (size_t)b * n);
    float4* o4 = (float4*)(out + (size_t)b * n);
    __shared__ float s_partial[8];
    int tid = threadIdx.x;
    int n4 = n >> 2;
    float ss = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = x4[i];
        ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = warp_reduce_sum(ss);
    if ((tid & 31) == 0) s_partial[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < (blockDim.x >> 5); ++i) tot += s_partial[i];
        s_partial[0] = rsqrtf(tot / (float)n + eps);
    }
    __syncthreads();
    float inv = s_partial[0];
    if (w) {
        const float4* w4 = (const float4*)w;
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i], g = w4[i];
            o4[i] = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z, v.w * inv * g.w);
        }
    } else {
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i];
            o4[i] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        }
    }
}
void ink_cuda_rms_norm_batch(const float* x, const float* w, float* out, int B, int n, float eps, cudaStream_t stream) {
    k_rms_norm_batch<<<B, 256, 0, stream>>>(x, w, out, n, eps);
}

__global__ void k_attn_res_mix_batch(BlockPtrList states, const float* __restrict__ query,
                                     float* __restrict__ out, int B, int n_embd, int n_queries) {
    int b = blockIdx.x;
    __shared__ float s_inv[12];
    __shared__ float s_dot[3][12];
    __shared__ float s_w[3][12];
    __shared__ float s_red[8];

    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int M = states.count;
    int n4 = n_embd >> 2;

    for (int m = 0; m < M; ++m) {
        const float4* sp = (const float4*)(states.ptrs[m] + (size_t)b * n_embd);
        float ss = 0.0f;
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = sp[i];
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
        ss = warp_reduce_sum(ss);
        if ((tid & 31) == 0) s_red[tid >> 5] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < nw; ++w) tot += s_red[w];
            s_inv[m] = rsqrtf(tot / (float)n_embd + 1e-6f);
        }
        __syncthreads();
    }

    float scale = rsqrtf((float)n_embd);
    for (int q = 0; q < n_queries; ++q) {
        const float4* qp = (const float4*)(query + (size_t)q * n_embd);
        for (int m = 0; m < M; ++m) {
            const float4* sp = (const float4*)(states.ptrs[m] + (size_t)b * n_embd);
            float inv = s_inv[m];
            float d = 0.0f;
            for (int i = tid; i < n4; i += blockDim.x) {
                float4 s = sp[i], qv = qp[i];
                d = fmaf(s.x * inv, qv.x, fmaf(s.y * inv, qv.y, fmaf(s.z * inv, qv.z, fmaf(s.w * inv, qv.w, d))));
            }
            d = warp_reduce_sum(d);
            if ((tid & 31) == 0) s_red[tid >> 5] = d;
            __syncthreads();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < nw; ++w) tot += s_red[w];
                s_dot[q][m] = tot * scale;
            }
            __syncthreads();
        }
        if (tid == 0) {
            float mx = -FLT_MAX;
            for (int m = 0; m < M; ++m) mx = fmaxf(mx, s_dot[q][m]);
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                float e = expf(s_dot[q][m] - mx);
                s_w[q][m] = e;
                sum += e;
            }
            float isum = 1.0f / sum;
            for (int m = 0; m < M; ++m) s_w[q][m] = s_w[q][m] * isum * s_inv[m];
        }
        __syncthreads();

        // Query-major output: out[(q * B + b) * n_embd] -> each query is contiguous [B, D]!
        float4* op = (float4*)(out + ((size_t)q * B + (size_t)b) * n_embd);
        for (int i = tid; i < n4; i += blockDim.x) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
            for (int m = 0; m < M; ++m) {
                float wm = s_w[q][m];
                float4 s = ((const float4*)(states.ptrs[m] + (size_t)b * n_embd))[i];
                ax = fmaf(wm, s.x, ax);
                ay = fmaf(wm, s.y, ay);
                az = fmaf(wm, s.z, az);
                aw = fmaf(wm, s.w, aw);
            }
            op[i] = make_float4(ax, ay, az, aw);
        }
        __syncthreads();
    }
}
void ink_cuda_attn_res_mix_batch(BlockPtrList states, const float* query, float* out,
                                 int B, int n_embd, int n_queries, cudaStream_t stream) {
    k_attn_res_mix_batch<<<B, 256, 0, stream>>>(states, query, out, B, n_embd, n_queries);
}

template <uint32_t WTYPE>
__global__ void k_loop_down_batch(const float* __restrict__ cur, const float* __restrict__ msg,
                                  const uint8_t* __restrict__ w_down, float* __restrict__ z_out,
                                  int n_embd, int rank, size_t row_bytes) {
    int b = blockIdx.y;
    const float* cb = cur + (size_t)b * n_embd;
    const float* mb = msg + (size_t)b * n_embd;
    float* zb = z_out + (size_t)b * rank;

    extern __shared__ float s_h[];
    __shared__ float s_red[16];
    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int n4 = n_embd >> 2;
    const float4* c4 = (const float4*)cb;
    const float4* m4 = (const float4*)mb;

    float ss0 = 0.0f, ss1 = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 c = c4[i], m = m4[i];
        ss0 = fmaf(c.x, c.x, fmaf(c.y, c.y, fmaf(c.z, c.z, fmaf(c.w, c.w, ss0))));
        ss1 = fmaf(m.x, m.x, fmaf(m.y, m.y, fmaf(m.z, m.z, fmaf(m.w, m.w, ss1))));
    }
    ss0 = warp_reduce_sum(ss0);
    ss1 = warp_reduce_sum(ss1);
    if ((tid & 31) == 0) { s_red[tid >> 5] = ss0; s_red[8 + (tid >> 5)] = ss1; }
    __syncthreads();
    if (tid == 0) {
        float t0 = 0.0f, t1 = 0.0f;
        for (int w = 0; w < nw; ++w) { t0 += s_red[w]; t1 += s_red[8 + w]; }
        s_red[0] = rsqrtf(t0 / (float)n_embd + 1e-6f);
        s_red[1] = rsqrtf(t1 / (float)n_embd + 1e-6f);
    }
    __syncthreads();
    float inv0 = s_red[0], inv1 = s_red[1];
    float4* sh0 = (float4*)s_h;
    float4* sh1 = (float4*)(s_h + n_embd);
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 c = c4[i], m = m4[i];
        sh0[i] = make_float4(c.x * inv0, c.y * inv0, c.z * inv0, c.w * inv0);
        sh1[i] = make_float4(m.x * inv1, m.y * inv1, m.z * inv1, m.w * inv1);
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int row = blockIdx.x * nw + warp;
    if (row >= rank) return;
    float v = 0.0f;
    if constexpr (WTYPE == GGML_TYPE_F16) v = dot_f16_vec(w_down + (size_t)row * row_bytes, s_h, 2 * n_embd, lane);
    else                                  v = dot_f32_vec(w_down + (size_t)row * row_bytes, s_h, 2 * n_embd, lane);
    if (lane == 0) zb[row] = v;
}

template <uint32_t WTYPE>
__global__ void k_loop_apply_batch(const float* __restrict__ cur, const float* __restrict__ msg,
                                   const float* __restrict__ z, const uint8_t* __restrict__ w_gain,
                                   const uint8_t* __restrict__ w_forget, float* __restrict__ out,
                                   int n_embd, int rank, size_t row_bytes) {
    int b = blockIdx.y;
    __shared__ float s_z[128];
    if (threadIdx.x < rank) s_z[threadIdx.x] = z[(size_t)b * rank + threadIdx.x];
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_embd) return;

    const uint8_t* gp = w_gain + (size_t)row * row_bytes;
    const uint8_t* fp = w_forget + (size_t)row * row_bytes;
    float g = 0.0f, f = 0.0f;
    if constexpr (WTYPE == GGML_TYPE_F16) {
        const __half2* g2 = (const __half2*)gp;
        const __half2* f2 = (const __half2*)fp;
        const float2*  z2 = (const float2*)s_z;
        int r2 = rank >> 1;
        #pragma unroll 4
        for (int k = 0; k < r2; ++k) {
            float2 zv = z2[k];
            float2 gv = __half22float2(g2[k]);
            float2 fv = __half22float2(f2[k]);
            g = fmaf(gv.x, zv.x, fmaf(gv.y, zv.y, g));
            f = fmaf(fv.x, zv.x, fmaf(fv.y, zv.y, f));
        }
    } else {
        const float4* g4 = (const float4*)gp;
        const float4* f4 = (const float4*)fp;
        const float4* z4 = (const float4*)s_z;
        int r4 = rank >> 2;
        #pragma unroll 4
        for (int k = 0; k < r4; ++k) {
            float4 zv = z4[k], gv = g4[k], fv = f4[k];
            g = fmaf(gv.x, zv.x, fmaf(gv.y, zv.y, fmaf(gv.z, zv.z, fmaf(gv.w, zv.w, g))));
            f = fmaf(fv.x, zv.x, fmaf(fv.y, zv.y, fmaf(fv.z, zv.z, fmaf(fv.w, zv.w, f))));
        }
    }
    size_t idx = (size_t)b * n_embd + row;
    out[idx] = (1.0f - tanhf(f)) * cur[idx] + tanhf(g) * msg[idx];
}

void ink_cuda_loop_control_batch(const float* cur, const float* msg, uint32_t w_type,
                                 const void* w_down, const void* w_gain, const void* w_forget,
                                 float* z_scratch, float* out, int B, int n_embd, int rank, cudaStream_t stream) {
    size_t elem_sz = (w_type == GGML_TYPE_F16) ? 2 : 4;
    size_t down_rb = (size_t)(2 * n_embd) * elem_sz;
    size_t gf_rb   = (size_t)rank * elem_sz;
    dim3 g_down((rank + 3) / 4, B);
    dim3 g_app((n_embd + 255) / 256, B);
    size_t down_smem = (size_t)(2 * n_embd) * sizeof(float);

    if (w_type == GGML_TYPE_F16) {
        k_loop_down_batch<GGML_TYPE_F16> <<<g_down, 128, down_smem, stream>>>(cur, msg, (const uint8_t*)w_down, z_scratch, n_embd, rank, down_rb);
        k_loop_apply_batch<GGML_TYPE_F16><<<g_app,  256, 0, stream>>>(cur, msg, z_scratch, (const uint8_t*)w_gain, (const uint8_t*)w_forget, out, n_embd, rank, gf_rb);
    } else {
        k_loop_down_batch<GGML_TYPE_F32> <<<g_down, 128, down_smem, stream>>>(cur, msg, (const uint8_t*)w_down, z_scratch, n_embd, rank, down_rb);
        k_loop_apply_batch<GGML_TYPE_F32><<<g_app,  256, 0, stream>>>(cur, msg, z_scratch, (const uint8_t*)w_gain, (const uint8_t*)w_forget, out, n_embd, rank, gf_rb);
    }
}

__global__ void k_shortconv_batch(const float* __restrict__ x_in, float* __restrict__ state,
                                  const float* __restrict__ weight, const float* __restrict__ dynamic,
                                  float* __restrict__ out, int B, int n_channels, int n_heads,
                                  float* __restrict__ spec_states, size_t spec_stride) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float* st = state + c * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const float4 w = ((const float4*)weight)[c];
    int h = (dynamic != nullptr && n_heads > 0) ? (c / (n_channels / n_heads)) : 0;

    for (int t = 0; t < B; ++t) {
        float s3 = x_in[(size_t)t * n_channels + c];
        float acc = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
        if (dynamic != nullptr) {
            const float4 tap = ((const float4*)(dynamic + (size_t)t * n_heads * 4))[h];
            acc += tap.x * s3 + tap.y * s2 + tap.z * s1 + tap.w * s0;
            out[(size_t)t * n_channels + c] = (s3 + acc) * (1.0f / 3.0f);
        } else {
            out[(size_t)t * n_channels + c] = (s3 + acc) * 0.4472135955f;
        }
        s0 = s1; s1 = s2; s2 = s3;
        if (spec_states != nullptr) {
            float* sst = spec_states + (size_t)t * spec_stride + c * 3;
            sst[0] = s0; sst[1] = s1; sst[2] = s2;
        }
    }
    st[0] = s0; st[1] = s1; st[2] = s2;
}

void ink_cuda_shortconv_batch(const float* x_in, float* state, const float* weight, const float* dynamic,
                              float* out, int B, int n_channels, int n_heads,
                              float* spec_states, size_t spec_stride, cudaStream_t stream) {
    k_shortconv_batch<<<(n_channels + 255) / 256, 256, 0, stream>>>(
        x_in, state, weight, dynamic, out, B, n_channels, n_heads, spec_states, spec_stride);
}

__global__ void k_shortconv_add_batch(const float* __restrict__ x_in, float* __restrict__ state,
                                      const float* __restrict__ weight, const float* __restrict__ res,
                                      float* __restrict__ out, int B, int n_channels,
                                      float* __restrict__ spec_states, size_t spec_stride) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float* st = state + c * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const float4 w = ((const float4*)weight)[c];

    for (int t = 0; t < B; ++t) {
        size_t idx = (size_t)t * n_channels + c;
        float s3 = x_in[idx];
        float acc = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
        out[idx] = res[idx] + (s3 + acc) * 0.4472135955f;
        s0 = s1; s1 = s2; s2 = s3;
        if (spec_states != nullptr) {
            float* sst = spec_states + (size_t)t * spec_stride + c * 3;
            sst[0] = s0; sst[1] = s1; sst[2] = s2;
        }
    }
    st[0] = s0; st[1] = s1; st[2] = s2;
}

void ink_cuda_shortconv_add_batch(const float* x_in, float* state, const float* weight, const float* res,
                                  float* out, int B, int n_channels,
                                  float* spec_states, size_t spec_stride, cudaStream_t stream) {
    k_shortconv_add_batch<<<(n_channels + 255) / 256, 256, 0, stream>>>(
        x_in, state, weight, res, out, B, n_channels, spec_states, spec_stride);
}

__global__ void k_qkv_post_batch(float* Q, const float* q_clip, float* K, float* V, float* R, const float* r_clip,
                                 int n_head, int n_head_kv, int head_dim, int d_rel,
                                 float qk_thresh, float rel_thresh, float eps,
                                 int is_swa, int pos0, int log_scale_floor, float log_scale_alpha) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int lane = threadIdx.x;

    float tau = 1.0f;
    if (!is_swa) {
        float eff = (float)(pos0 + t + 1) / (float)log_scale_floor;
        tau = 1.0f + log_scale_alpha * logf(fmaxf(eff, 1.0f));
    }

    float* Qt = Q + (size_t)t * n_head * head_dim;
    float* Kt = K + (size_t)t * n_head_kv * head_dim;
    float* Vt = V + (size_t)t * n_head_kv * head_dim;
    float* Rt = R + (size_t)t * n_head * d_rel;

    if (b < n_head) {
        float* q = Qt + b * head_dim;
        const float* qc = q_clip + b * head_dim;
        float ss = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            float v = q[d] * qc[d];
            q[d] = v;
            ss = fmaf(v, v, ss);
        }
        ss = warp_reduce_sum(ss);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0)) + 1e-6f)) * tau;
        for (int d = lane; d < head_dim; d += 32) q[d] *= factor;
    } else if (b < n_head + n_head_kv) {
        int h = b - n_head;
        float* k = Kt + h * head_dim;
        float ss = 0.0f;
        for (int d = lane; d < head_dim; d += 32) ss = fmaf(k[d], k[d], ss);
        ss = warp_reduce_sum(ss);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0)) + 1e-6f));
        for (int d = lane; d < head_dim; d += 32) k[d] *= factor;
    } else if (b < n_head + 2 * n_head_kv) {
        int h = b - n_head - n_head_kv;
        float* v = Vt + h * head_dim;
        float ss = 0.0f;
        for (int d = lane; d < head_dim; d += 32) ss = fmaf(v[d], v[d], ss);
        ss = warp_reduce_sum(ss);
        float inv = rsqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0) / (float)head_dim + eps);
        for (int d = lane; d < head_dim; d += 32) v[d] *= inv;
    } else {
        int h = b - n_head - 2 * n_head_kv;
        if (h < n_head) {
            float* r = Rt + h * d_rel;
            const float* rc = r_clip + h * d_rel;
            float val = (lane < d_rel) ? (r[lane] * rc[lane]) : 0.0f;
            float ss = warp_reduce_sum(val * val);
            float factor = fminf(1.0f, rel_thresh / (sqrtf(__shfl_sync(0xFFFFFFFFu, ss, 0)) + 1e-6f)) * tau;
            if (lane < d_rel) r[lane] = val * factor;
        }
    }
}

void ink_cuda_qkv_post_batch(float* Q, const float* q_clip, float* K, float* V, float* R, const float* r_clip,
                             int B, int n_head, int n_head_kv, int head_dim, int d_rel,
                             float qk_clip_thresh, float rel_clip_thresh, float rms_eps,
                             bool is_swa, int pos0, int log_scale_floor, float log_scale_alpha, cudaStream_t stream) {
    dim3 grid(2 * n_head + 2 * n_head_kv, B);
    k_qkv_post_batch<<<grid, 32, 0, stream>>>(
        Q, q_clip, K, V, R, r_clip, n_head, n_head_kv, head_dim, d_rel,
        qk_clip_thresh, rel_clip_thresh, rms_eps, is_swa ? 1 : 0, pos0, log_scale_floor, log_scale_alpha);
}

__global__ void k_rel_table_batch(const float* __restrict__ rel_bias, const float* __restrict__ R,
                                  float* __restrict__ rel_table, int n_head, int d_rel, int rel_extent) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    if (d >= rel_extent || h >= n_head) return;

    const float4* rb = (const float4*)(rel_bias + d * d_rel);
    const float4* rh = (const float4*)(R + ((size_t)t * n_head + h) * d_rel);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 a = rb[i], b = rh[i];
        s = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, s))));
    }
    rel_table[((size_t)t * n_head + h) * rel_extent + d] = s;
}

void ink_cuda_rel_table_batch(const float* rel_bias, const float* R, float* rel_table,
                              int B, int n_head, int d_rel, int rel_extent, cudaStream_t stream) {
    dim3 grid((rel_extent + 127) / 128, n_head, B);
    k_rel_table_batch<<<grid, 128, 0, stream>>>(rel_bias, R, rel_table, n_head, d_rel, rel_extent);
}

__global__ void k_kv_store_swa_batch(const float* __restrict__ K, const float* __restrict__ V,
                                     uint8_t* __restrict__ k_ring, uint8_t* __restrict__ v_ring,
                                     int32_t* __restrict__ pos_ring, int swa_window, int pos0,
                                     int n_head_kv, int head_dim, int kv_mode) {
    int bid = blockIdx.x;
    int t   = blockIdx.y;
    int tid = threadIdx.x;
    int pos = pos0 + t;
    int slot = pos % swa_window;
    if (bid == 0 && tid == 0) pos_ring[slot] = pos;

    bool is_v = (bid >= n_head_kv);
    int h = is_v ? (bid - n_head_kv) : bid;
    int warp = tid >> 5;
    int lane = tid & 31;
    const float* src = (is_v ? V : K) + ((size_t)t * n_head_kv + h) * head_dim;
    size_t hb = kv_head_bytes_dev(kv_mode);
    size_t tok_stride = (size_t)n_head_kv * hb;
    uint8_t* dst = (is_v ? v_ring : k_ring) + (size_t)slot * tok_stride + (size_t)h * hb;

    if (kv_mode == KV_QUANT_Q4_0)      quant_block32_q4_0(src + warp * 32, dst + warp * 18, lane);
    else if (kv_mode == KV_QUANT_Q8_0) quant_block32_q8_0(src + warp * 32, dst + warp * 34, lane);
    else                               ((uint16_t*)dst)[tid] = d_f32_to_fp16(src[tid]);
}

void ink_cuda_kv_store_swa_batch(const float* K, const float* V, void* k_ring, void* v_ring, int32_t* pos_ring,
                                 int swa_window, int pos0, int B,
                                 int n_head_kv, int head_dim, KVQuantMode kv_mode, cudaStream_t stream) {
    dim3 grid(n_head_kv * 2, B);
    k_kv_store_swa_batch<<<grid, head_dim, 0, stream>>>(
        K, V, (uint8_t*)k_ring, (uint8_t*)v_ring, pos_ring, swa_window, pos0, n_head_kv, head_dim, (int)kv_mode);
}

__global__ void k_swa_spec_backup_restore(uint8_t* swa_k, uint8_t* swa_v, int32_t* swa_pos,
                                          uint8_t* bak_k, uint8_t* bak_v, int32_t* bak_pos,
                                          int swa_window, int pos0, int t_start, int max_bak_slots,
                                          size_t tok_k_bytes, int is_restore) {
    int si = blockIdx.x;
    int t  = t_start + blockIdx.y;
    int slot = (pos0 + t) % swa_window;
    size_t swa_layer_bytes = (size_t)swa_window * tok_k_bytes;
    size_t bak_layer_bytes = (size_t)max_bak_slots * tok_k_bytes;

    uint8_t* sk = swa_k + (size_t)si * swa_layer_bytes + (size_t)slot * tok_k_bytes;
    uint8_t* sv = swa_v + (size_t)si * swa_layer_bytes + (size_t)slot * tok_k_bytes;
    uint8_t* bk = bak_k + (size_t)si * bak_layer_bytes + (size_t)t * tok_k_bytes;
    uint8_t* bv = bak_v + (size_t)si * bak_layer_bytes + (size_t)t * tok_k_bytes;

    if (threadIdx.x == 0) {
        if (is_restore) swa_pos[(size_t)si * swa_window + slot] = bak_pos[(size_t)si * max_bak_slots + t];
        else            bak_pos[(size_t)si * max_bak_slots + t] = swa_pos[(size_t)si * swa_window + slot];
    }
    int n4 = (int)(tok_k_bytes >> 2);
    uint32_t* sk4 = (uint32_t*)sk; uint32_t* sv4 = (uint32_t*)sv;
    uint32_t* bk4 = (uint32_t*)bk; uint32_t* bv4 = (uint32_t*)bv;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        if (is_restore) { sk4[i] = bk4[i]; sv4[i] = bv4[i]; }
        else            { bk4[i] = sk4[i]; bv4[i] = sv4[i]; }
    }
}

void ink_cuda_swa_spec_backup_restore(void* d_swa_k, void* d_swa_v, int32_t* d_swa_pos,
                                      void* d_bak_k, void* d_bak_v, int32_t* d_bak_pos,
                                      int n_swa_layers, int swa_window, int pos0, int t_start, int t_end, int max_bak_slots,
                                      size_t tok_k_bytes, bool is_restore, cudaStream_t stream) {
    int count = t_end - t_start;
    if (count <= 0) return;
    dim3 grid(n_swa_layers, count);
    k_swa_spec_backup_restore<<<grid, 256, 0, stream>>>(
        (uint8_t*)d_swa_k, (uint8_t*)d_swa_v, d_swa_pos,
        (uint8_t*)d_bak_k, (uint8_t*)d_bak_v, d_bak_pos,
        swa_window, pos0, t_start, max_bak_slots, tok_k_bytes, is_restore ? 1 : 0);
}

__global__ void k_attn_swa_gqa_splitk_batch(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ K_ring, const uint8_t* __restrict__ V_ring,
    const int32_t* __restrict__ pos_ring,
    const float* __restrict__ rel_table,
    float* __restrict__ split_out, float* __restrict__ split_ml,
    int n_head, int n_head_kv, int head_dim,
    int swa_len, int n_splits, int pos0, int rel_extent, int kv_mode)
{
    int h_kv = blockIdx.x;
    int sp   = blockIdx.y;
    int tb   = blockIdx.z;
    int pos_q = pos0 + tb;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int s_start = (swa_len * sp) / n_splits;
    int s_end   = (swa_len * (sp + 1)) / n_splits;

    const float* Qt = Q + (size_t)tb * n_head * head_dim;
    const float* Rt = rel_table + (size_t)tb * n_head * rel_extent;
    float q0[3], q1[3], q2[3], q3[3];
    const float* rtab[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h_kv * 3 + g;
        const float* qh = Qt + h * head_dim;
        q0[g] = qh[lane + 0]; q1[g] = qh[lane + 32]; q2[g] = qh[lane + 64]; q3[g] = qh[lane + 96];
        rtab[g] = Rt + h * rel_extent;
    }

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    const size_t hb = kv_head_bytes_dev(kv_mode);
    const size_t tok_stride = (size_t)n_head_kv * hb;

    float m_w[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float l_w[3] = {0.0f, 0.0f, 0.0f};
    float o0[3] = {0.0f, 0.0f, 0.0f}, o1[3] = {0.0f, 0.0f, 0.0f};
    float o2[3] = {0.0f, 0.0f, 0.0f}, o3[3] = {0.0f, 0.0f, 0.0f};

    for (int s = s_start + warp; s < s_end; s += 4) {
        int pos_k = __ldg(&pos_ring[s]);
        int dist = pos_q - pos_k;
        if (pos_k < 0 || dist < 0 || dist >= rel_extent) continue;

        const uint8_t* kp = K_ring + (size_t)s * tok_stride + (size_t)h_kv * hb;
        const uint8_t* vp = V_ring + (size_t)s * tok_stride + (size_t)h_kv * hb;
        float4 kv = load_kv_head_4(kp, lane, kv_mode);
        float4 vv = load_kv_head_4(vp, lane, kv_mode);

        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = warp_reduce_sum(q0[g] * kv.x + q1[g] * kv.y + q2[g] * kv.z + q3[g] * kv.w);
            float score = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d + __ldg(&rtab[g][dist]);
            float m_new = fmaxf(m_w[g], score);
            float alpha = (m_w[g] == -FLT_MAX) ? 0.0f : expf(m_w[g] - m_new);
            float prob  = expf(score - m_new);
            l_w[g] = l_w[g] * alpha + prob;
            o0[g]  = fmaf(o0[g], alpha, prob * vv.x);
            o1[g]  = fmaf(o1[g], alpha, prob * vv.y);
            o2[g]  = fmaf(o2[g], alpha, prob * vv.z);
            o3[g]  = fmaf(o3[g], alpha, prob * vv.w);
            m_w[g] = m_new;
        }
    }

    __shared__ float sm_m[3][4], sm_l[3][4], sm_o[3][4][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[g][warp] = m_w[g]; sm_l[g][warp] = l_w[g]; }
        sm_o[g][warp][lane +  0] = o0[g];
        sm_o[g][warp][lane + 32] = o1[g];
        sm_o[g][warp][lane + 64] = o2[g];
        sm_o[g][warp][lane + 96] = o3[g];
    }
    __syncthreads();

    if (warp < 3) {
        int g = warp;
        int h = h_kv * 3 + g;
        float m_all = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < 4; ++w) m_all = fmaxf(m_all, sm_m[g][w]);
        float l_all = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if (m_all > -FLT_MAX * 0.5f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[g][w] > -FLT_MAX * 0.5f) {
                    float sc = expf(sm_m[g][w] - m_all);
                    l_all += sm_l[g][w] * sc;
                    a0 = fmaf(sc, sm_o[g][w][lane +  0], a0);
                    a1 = fmaf(sc, sm_o[g][w][lane + 32], a1);
                    a2 = fmaf(sc, sm_o[g][w][lane + 64], a2);
                    a3 = fmaf(sc, sm_o[g][w][lane + 96], a3);
                }
            }
        }
        size_t slot = ((size_t)tb * n_splits + sp) * n_head + h;
        float* so = split_out + slot * head_dim;
        so[lane + 0] = a0; so[lane + 32] = a1; so[lane + 64] = a2; so[lane + 96] = a3;
        if (lane == 0) {
            split_ml[slot * 2 + 0] = m_all;
            split_ml[slot * 2 + 1] = l_all;
        }
    }
}

__global__ void k_attn_global_splitk_gqa_batch(
    const float* __restrict__ Q,
    const uint64_t* __restrict__ page_k_ptrs, const uint64_t* __restrict__ page_v_ptrs,
    const int32_t* __restrict__ page_pos0, const int32_t* __restrict__ page_lens,
    int n_pages, int n_splits, const float* __restrict__ rel_table,
    float* __restrict__ split_out, float* __restrict__ split_ml,
    int n_head, int n_head_kv, int head_dim, int pos0, int rel_extent, int kv_mode)
{
    int h_kv = blockIdx.x, sp = blockIdx.y, tb = blockIdx.z;
    int pos_q = pos0 + tb;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    int p_start = (n_pages * sp) / n_splits;
    int p_end   = (n_pages * (sp + 1)) / n_splits;

    const float* Qt = Q + (size_t)tb * n_head * head_dim;
    const float* Rt = rel_table + (size_t)tb * n_head * rel_extent;
    float q0[3], q1[3], q2[3], q3[3];
    const float* rtab[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h_kv * 3 + g;
        const float* qh = Qt + h * head_dim;
        q0[g] = qh[lane + 0]; q1[g] = qh[lane + 32]; q2[g] = qh[lane + 64]; q3[g] = qh[lane + 96];
        rtab[g] = Rt + h * rel_extent;
    }

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    const size_t hb = kv_head_bytes_dev(kv_mode);
    const size_t tok_stride = (size_t)n_head_kv * hb;

    float m_w[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float l_w[3] = {0.0f, 0.0f, 0.0f};
    float o0[3] = {0.0f, 0.0f, 0.0f}, o1[3] = {0.0f, 0.0f, 0.0f};
    float o2[3] = {0.0f, 0.0f, 0.0f}, o3[3] = {0.0f, 0.0f, 0.0f};

    for (int p = p_start; p < p_end; ++p) {
        int p0 = __ldg(&page_pos0[p]);
        if (p0 > pos_q) continue; // strict causal skip!
        const uint8_t* kp_base = (const uint8_t*)(uintptr_t)__ldg(&page_k_ptrs[p]);
        const uint8_t* vp_base = (const uint8_t*)(uintptr_t)__ldg(&page_v_ptrs[p]);
        int plen = __ldg(&page_lens[p]);

        for (int t = warp; t < plen; t += 4) {
            int dist = pos_q - (p0 + t);
            if (dist < 0) continue;
            bool in_rel = (dist < rel_extent);

            const uint8_t* kp = kp_base + (size_t)t * tok_stride + (size_t)h_kv * hb;
            const uint8_t* vp = vp_base + (size_t)t * tok_stride + (size_t)h_kv * hb;
            float4 kv = load_kv_head_4(kp, lane, kv_mode);
            float4 vv = load_kv_head_4(vp, lane, kv_mode);

            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                float dot = warp_reduce_sum(q0[g] * kv.x + q1[g] * kv.y + q2[g] * kv.z + q3[g] * kv.w);
                float bias = in_rel ? __ldg(&rtab[g][dist]) : 0.0f;
                float score = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d + bias;
                float m_new = fmaxf(m_w[g], score);
                float alpha = (m_w[g] == -FLT_MAX) ? 0.0f : expf(m_w[g] - m_new);
                float prob  = expf(score - m_new);
                l_w[g] = l_w[g] * alpha + prob;
                o0[g]  = fmaf(o0[g], alpha, prob * vv.x);
                o1[g]  = fmaf(o1[g], alpha, prob * vv.y);
                o2[g]  = fmaf(o2[g], alpha, prob * vv.z);
                o3[g]  = fmaf(o3[g], alpha, prob * vv.w);
                m_w[g] = m_new;
            }
        }
    }

    __shared__ float sm_m[3][4], sm_l[3][4], sm_o[3][4][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[g][warp] = m_w[g]; sm_l[g][warp] = l_w[g]; }
        sm_o[g][warp][lane +  0] = o0[g];
        sm_o[g][warp][lane + 32] = o1[g];
        sm_o[g][warp][lane + 64] = o2[g];
        sm_o[g][warp][lane + 96] = o3[g];
    }
    __syncthreads();

    if (warp < 3) {
        int g = warp;
        int h = h_kv * 3 + g;
        float m_all = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < 4; ++w) m_all = fmaxf(m_all, sm_m[g][w]);
        float l_all = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if (m_all > -FLT_MAX * 0.5f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[g][w] > -FLT_MAX * 0.5f) {
                    float sc = expf(sm_m[g][w] - m_all);
                    l_all += sm_l[g][w] * sc;
                    a0 = fmaf(sc, sm_o[g][w][lane +  0], a0);
                    a1 = fmaf(sc, sm_o[g][w][lane + 32], a1);
                    a2 = fmaf(sc, sm_o[g][w][lane + 64], a2);
                    a3 = fmaf(sc, sm_o[g][w][lane + 96], a3);
                }
            }
        }
        size_t slot = ((size_t)tb * n_splits + sp) * n_head + h;
        float* so = split_out + slot * head_dim;
        so[lane + 0] = a0; so[lane + 32] = a1; so[lane + 64] = a2; so[lane + 96] = a3;
        if (lane == 0) {
            split_ml[slot * 2 + 0] = m_all;
            split_ml[slot * 2 + 1] = l_all;
        }
    }
}

__global__ void k_attn_global_reduce_batch(const float* __restrict__ split_out, const float* __restrict__ split_ml,
                                           const float* __restrict__ sinks, float* __restrict__ out,
                                           int n_splits, int n_head, int head_dim) {
    int h = blockIdx.x, tb = blockIdx.y, lane = threadIdx.x;
    float sink_t = sinks[(size_t)tb * n_head + h];
    float m_all = sink_t;
    for (int sp = 0; sp < n_splits; ++sp) {
        size_t slot = ((size_t)tb * n_splits + sp) * n_head + h;
        m_all = fmaxf(m_all, split_ml[slot * 2 + 0]);
    }
    float denom = expf(sink_t - m_all);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        size_t slot = ((size_t)tb * n_splits + sp) * n_head + h;
        float ms = split_ml[slot * 2 + 0];
        float ls = split_ml[slot * 2 + 1];
        if (ms > -FLT_MAX * 0.5f) {
            float sc = expf(ms - m_all);
            denom += ls * sc;
            const float* so = split_out + slot * head_dim;
            a0 = fmaf(sc, so[lane +  0], a0);
            a1 = fmaf(sc, so[lane + 32], a1);
            a2 = fmaf(sc, so[lane + 64], a2);
            a3 = fmaf(sc, so[lane + 96], a3);
        }
    }
    float inv_d = 1.0f / denom;
    float* oh = out + ((size_t)tb * n_head + h) * head_dim;
    oh[lane + 0] = a0 * inv_d; oh[lane + 32] = a1 * inv_d;
    oh[lane + 64] = a2 * inv_d; oh[lane + 96] = a3 * inv_d;
}

void ink_cuda_attn_swa_batch(const float* Q, const void* K_ring, const void* V_ring, const int32_t* pos_ring,
                             const float* rel_table, const float* sinks, float* split_out, float* split_ml, float* out,
                             int B, int n_head, int n_head_kv, int head_dim, int swa_len, int pos0,
                             int rel_extent, KVQuantMode kv_mode, cudaStream_t stream) {
    int n_splits = (B >= 64) ? 2 : ((B >= 8) ? 2 : ((swa_len >= 256) ? 4 : 2));
    int cap_sp = 256 / B;
    if (cap_sp < 1) cap_sp = 1;
    if (n_splits > cap_sp) n_splits = cap_sp;
    dim3 grid(n_head_kv, n_splits, B);
    k_attn_swa_gqa_splitk_batch<<<grid, 128, 0, stream>>>(
        Q, (const uint8_t*)K_ring, (const uint8_t*)V_ring, pos_ring, rel_table,
        split_out, split_ml, n_head, n_head_kv, head_dim, swa_len, n_splits, pos0, rel_extent, (int)kv_mode);
    k_attn_global_reduce_batch<<<dim3(n_head, B), 32, 0, stream>>>(
        split_out, split_ml, sinks, out, n_splits, n_head, head_dim);
}

void ink_cuda_attn_global_paged_batch(const float* Q, const uint64_t* page_k_ptrs, const uint64_t* page_v_ptrs,
                                      const int32_t* page_pos0, const int32_t* page_lens, int n_pages,
                                      const float* rel_table, const float* sinks,
                                      float* split_out, float* split_ml, float* out,
                                      int B, int n_head, int n_head_kv, int head_dim, int pos0, int rel_extent,
                                      KVQuantMode kv_mode, cudaStream_t stream) {
    int max_sp = (B <= 8) ? 16 : ((B <= 64) ? 4 : 2);
    int cap_sp = 256 / B;
    if (cap_sp < 1) cap_sp = 1;
    if (max_sp > cap_sp) max_sp = cap_sp;
    int n_splits = 1;
    if (n_pages >= 64)      n_splits = max_sp;
    else if (n_pages >= 16) n_splits = min(max_sp, 8);
    else if (n_pages >= 4)  n_splits = min(max_sp, 4);
    dim3 grid(n_head_kv, n_splits, B);
    k_attn_global_splitk_gqa_batch<<<grid, 128, 0, stream>>>(
        Q, page_k_ptrs, page_v_ptrs, page_pos0, page_lens, n_pages, n_splits, rel_table,
        split_out, split_ml, n_head, n_head_kv, head_dim, pos0, rel_extent, (int)kv_mode);
    k_attn_global_reduce_batch<<<dim3(n_head, B), 32, 0, stream>>>(
        split_out, split_ml, sinks, out, n_splits, n_head, head_dim);
}

__global__ void k_router_topk_batch(const float* __restrict__ logits, const float* __restrict__ bias,
                                    int32_t* __restrict__ top_ids, float* __restrict__ top_weights,
                                    int n_expert, int k, float w_scale) {
    int b = blockIdx.x;
    int lane = threadIdx.x;
    const float* lb = logits + (size_t)b * n_expert;
    int32_t* id_out = top_ids + (size_t)b * k;
    float*   w_out  = top_weights + (size_t)b * k;

    float p[4], s[4];
    bool used[4] = {false, false, false, false};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = i * 32 + lane;
        float lg = (e < n_expert) ? lb[e] : -FLT_MAX;
        float pr = (e < n_expert) ? (1.0f / (1.0f + expf(-lg))) : 0.0f;
        p[i] = pr;
        s[i] = (e < n_expert) ? (pr + (bias ? bias[e] : 0.0f)) : -FLT_MAX;
    }

    int sel_i[8];
    float sel_w[8];
    float wsum = 0.0f;
    for (int j = 0; j < k; ++j) {
        float best_s = -FLT_MAX;
        int best_e = 0;
        float best_p = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (!used[i] && s[i] > best_s) { best_s = s[i]; best_e = i * 32 + lane; best_p = p[i]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float os = __shfl_xor_sync(0xFFFFFFFFu, best_s, off);
            int   oe = __shfl_xor_sync(0xFFFFFFFFu, best_e, off);
            float op = __shfl_xor_sync(0xFFFFFFFFu, best_p, off);
            if (os > best_s || (os == best_s && oe < best_e)) { best_s = os; best_e = oe; best_p = op; }
        }
        sel_i[j] = best_e; sel_w[j] = best_p; wsum += best_p;
        if ((best_e & 31) == lane) used[best_e >> 5] = true;
    }
    if (lane == 0) {
        float inv = w_scale / (wsum + 1e-20f);
        for (int j = 0; j < k; ++j) { id_out[j] = sel_i[j]; w_out[j] = sel_w[j] * inv; }
    }
}

void ink_cuda_moe_router_batch(const float* gate_inp, const float* bias, const float* x,
                               float* logits_scratch, int32_t* top_ids, float* top_weights,
                               int B, int n_expert, int n_embd, int k, float weights_scale, cudaStream_t stream) {
    ink_cuda_gemm_batch(GGML_TYPE_F32, gate_inp, x, logits_scratch, B, n_expert, n_embd, stream);
    k_router_topk_batch<<<B, 32, 0, stream>>>(logits_scratch, bias, top_ids, top_weights, n_expert, k, weights_scale);
}

// Grouped Batched GPU MoE: pre-quantizes activations ONCE into 100%-L2-resident Q8 buffer (0 smem -> max occupancy!)
template <uint32_t TYPE>
__global__ void k_moe_grouped_gate_up(
    const uint64_t* __restrict__ d_gu_ptrs, const float* __restrict__ d_gu_scales,
    const int32_t* __restrict__ grp_exp_ids, const int32_t* __restrict__ grp_tok_off,
    const int32_t* __restrict__ grp_tok_ids,
    const Q8Block32GPU* __restrict__ qx_all, float* __restrict__ h_out,
    int n_embd, int n_ff, size_t row_bytes, float beta, float lb)
{
    int g = blockIdx.y;
    int exp_id = grp_exp_ids[g];
    int p_start = grp_tok_off[g];
    int p_end   = grp_tok_off[g + 1];
    int cnt = p_end - p_start;
    if (cnt <= 0) return;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_ff) return;

    const uint8_t* base = (const uint8_t*)(uintptr_t)d_gu_ptrs[exp_id];
    float sc = d_gu_scales[exp_id];
    const uint8_t* g_row = base + (size_t)j * row_bytes;
    const uint8_t* u_row = base + (size_t)(j + n_ff) * row_bytes;
    int nb_k = n_embd >> 5;

    for (int t = 0; t < cnt; ++t) {
        int tok = grp_tok_ids[p_start + t];
        const Q8Block32GPU* qxt = qx_all + (size_t)tok * nb_k;
        float gv = dot_quant_dp4a<TYPE, 32>(g_row, qxt, n_embd, lane) * sc;
        float uv = dot_quant_dp4a<TYPE, 32>(u_row, qxt, n_embd, lane) * sc;
        if (lane == 0) {
            h_out[(size_t)(p_start + t) * n_ff + j] = d_situ(gv, uv, beta, lb);
        }
    }
}

template <uint32_t TYPE>
__global__ void k_moe_grouped_down(
    const uint64_t* __restrict__ d_dn_ptrs, const float* __restrict__ d_dn_scales,
    const int32_t* __restrict__ grp_exp_ids, const int32_t* __restrict__ grp_tok_off,
    const int32_t* __restrict__ grp_tok_ids, const float* __restrict__ grp_tok_w,
    const Q8Block32GPU* __restrict__ qh_all, float* __restrict__ out_batch,
    int n_embd, int n_ff, size_t row_bytes)
{
    int g = blockIdx.y;
    int exp_id = grp_exp_ids[g];
    int p_start = grp_tok_off[g];
    int p_end   = grp_tok_off[g + 1];
    int cnt = p_end - p_start;
    if (cnt <= 0) return;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= n_embd) return;

    const uint8_t* base = (const uint8_t*)(uintptr_t)d_dn_ptrs[exp_id];
    float dsc = d_dn_scales[exp_id];
    const uint8_t* d_row = base + (size_t)row * row_bytes;
    int nb_ff = n_ff >> 5;

    for (int t = 0; t < cnt; ++t) {
        float v = dot_quant_dp4a<TYPE, 32>(d_row, qh_all + (size_t)(p_start + t) * nb_ff, n_ff, lane);
        if (lane == 0) {
            int tok = grp_tok_ids[p_start + t];
            float w = grp_tok_w[p_start + t];
            atomicAdd(&out_batch[(size_t)tok * n_embd + row], v * (dsc * w));
        }
    }
}

// WMMA FP16->FP32 Tensor Core Grouped MoE GateUp + SITU (unpacks codebooks ONCE per 16 tokens!)
template <uint32_t TYPE>
__global__ void k_moe_grouped_wmma_gate_up(
    const uint64_t* __restrict__ d_gu_ptrs, const float* __restrict__ d_gu_scales,
    const int32_t* __restrict__ grp_exp_ids, const int32_t* __restrict__ grp_tok_off,
    const int32_t* __restrict__ grp_tok_ids,
    const __half* __restrict__ x_f16_all, float* __restrict__ h_out,
    int n_embd, int n_ff, size_t row_bytes, float beta, float lb)
{
    using namespace nvcuda;
    int g = blockIdx.y;
    int exp_id  = grp_exp_ids[g];
    int p_start = grp_tok_off[g];
    int p_end   = grp_tok_off[g + 1];
    int cnt     = p_end - p_start;
    if (cnt <= 0) return;

    int j0   = blockIdx.x * 32;
    int tid  = threadIdx.x;
    int warp = tid >> 5; // 0,1 = gate rows [j0..j0+31]; 2,3 = up rows [j0+n_ff..j0+n_ff+31]
    int r_t  = tid >> 2; // 0..31
    int c8   = tid & 3;  // 0..3

    const uint8_t* base = (const uint8_t*)(uintptr_t)d_gu_ptrs[exp_id];
    float sc = d_gu_scales[exp_id];
    int n_sub = n_embd >> 5;

    __shared__ __half s_A[16][40];
    __shared__ __half s_B[64][40];
    __shared__ float  s_gate[16][32];
    __shared__ float  s_up[16][32];

    for (int t0 = 0; t0 < cnt; t0 += 16) {
        int tb = min(16, cnt - t0);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int sub = 0; sub < n_sub; ++sub) {
            int k_off = sub * 32 + c8 * 8;
            if (r_t < 16) {
                if (r_t < tb) {
                    int tok = grp_tok_ids[p_start + t0 + r_t];
                    *(uint4*)(&s_A[r_t][c8 * 8]) = *(const uint4*)(x_f16_all + (size_t)tok * n_embd + k_off);
                } else {
                    *(uint4*)(&s_A[r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
                }
            }
            if (j0 + r_t < n_ff) {
                dequant_sub32_8<TYPE>(base + (size_t)(j0 + r_t) * row_bytes,        sub, c8, &s_B[r_t][c8 * 8]);
                dequant_sub32_8<TYPE>(base + (size_t)(j0 + n_ff + r_t) * row_bytes, sub, c8, &s_B[32 + r_t][c8 * 8]);
            } else {
                *(uint4*)(&s_B[r_t][c8 * 8])      = make_uint4(0, 0, 0, 0);
                *(uint4*)(&s_B[32 + r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
            }
            __syncthreads();

            #pragma unroll
            for (int ks = 0; ks < 32; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, &s_A[0][ks], 40);
                wmma::load_matrix_sync(b_frag, &s_B[warp * 16][ks], 40);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        if (warp < 2) wmma::store_matrix_sync(&s_gate[0][warp * 16],       c_frag, 32, wmma::mem_row_major);
        else          wmma::store_matrix_sync(&s_up[0][(warp - 2) * 16],   c_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int i = tid; i < tb * 32; i += 128) {
            int t = i >> 5;
            int c = i & 31;
            if (j0 + c < n_ff) {
                h_out[(size_t)(p_start + t0 + t) * n_ff + (j0 + c)] =
                    d_situ(s_gate[t][c] * sc, s_up[t][c] * sc, beta, lb);
            }
        }
        __syncthreads();
    }
}

template <uint32_t TYPE>
__global__ void k_moe_grouped_wmma_down(
    const uint64_t* __restrict__ d_dn_ptrs, const float* __restrict__ d_dn_scales,
    const int32_t* __restrict__ grp_exp_ids, const int32_t* __restrict__ grp_tok_off,
    const int32_t* __restrict__ grp_tok_ids, const float* __restrict__ grp_tok_w,
    const __half* __restrict__ h_f16_all, float* __restrict__ out_batch,
    int n_embd, int n_ff, size_t row_bytes)
{
    using namespace nvcuda;
    int g = blockIdx.y;
    int exp_id  = grp_exp_ids[g];
    int p_start = grp_tok_off[g];
    int p_end   = grp_tok_off[g + 1];
    int cnt     = p_end - p_start;
    if (cnt <= 0) return;

    int row0 = blockIdx.x * 64;
    int tid  = threadIdx.x;
    int warp = tid >> 5; // 0..3 -> rows [row0 + warp*16 .. +15]
    int r_t  = tid >> 2; // 0..31
    int c8   = tid & 3;  // 0..3

    const uint8_t* base = (const uint8_t*)(uintptr_t)d_dn_ptrs[exp_id];
    float dsc = d_dn_scales[exp_id];
    int n_sub = n_ff >> 5;

    __shared__ __half s_A[16][40];
    __shared__ __half s_B[64][40];
    __shared__ float  s_out[16][64];

    for (int t0 = 0; t0 < cnt; t0 += 16) {
        int tb = min(16, cnt - t0);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int sub = 0; sub < n_sub; ++sub) {
            int k_off = sub * 32 + c8 * 8;
            if (r_t < 16) {
                if (r_t < tb) {
                    *(uint4*)(&s_A[r_t][c8 * 8]) = *(const uint4*)(h_f16_all + (size_t)(p_start + t0 + r_t) * n_ff + k_off);
                } else {
                    *(uint4*)(&s_A[r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
                }
            }
            if (row0 + r_t < n_embd) {
                dequant_sub32_8<TYPE>(base + (size_t)(row0 + r_t) * row_bytes, sub, c8, &s_B[r_t][c8 * 8]);
            } else {
                *(uint4*)(&s_B[r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
            }
            if (row0 + 32 + r_t < n_embd) {
                dequant_sub32_8<TYPE>(base + (size_t)(row0 + 32 + r_t) * row_bytes, sub, c8, &s_B[32 + r_t][c8 * 8]);
            } else {
                *(uint4*)(&s_B[32 + r_t][c8 * 8]) = make_uint4(0, 0, 0, 0);
            }
            __syncthreads();

            #pragma unroll
            for (int ks = 0; ks < 32; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, &s_A[0][ks], 40);
                wmma::load_matrix_sync(b_frag, &s_B[warp * 16][ks], 40);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        wmma::store_matrix_sync(&s_out[0][warp * 16], c_frag, 64, wmma::mem_row_major);
        __syncthreads();

        for (int i = tid; i < tb * 64; i += 128) {
            int t = i >> 6;
            int c = i & 63;
            if (row0 + c < n_embd) {
                int tok = grp_tok_ids[p_start + t0 + t];
                float w = grp_tok_w[p_start + t0 + t];
                atomicAdd(&out_batch[(size_t)tok * n_embd + (row0 + c)], s_out[t][c] * (dsc * w));
            }
        }
        __syncthreads();
    }
}

void ink_cuda_moe_grouped_gpu(
    uint32_t gate_up_type, uint32_t down_type,
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_grp_exp_ids, const int32_t* d_grp_tok_off,
    const int32_t* d_grp_tok_ids, const float* d_grp_tok_w,
    int n_groups, int total_pairs, int B,
    const float* d_x_batch, float* d_h_scratch, float* d_out_batch,
    int n_embd, int n_ff, float situ_beta, float situ_linear_beta,
    bool zero_out, bool preconvert_x, cudaStream_t stream)
{
    if (zero_out) {
        INK_CUDA_CHECK(cudaMemsetAsync(d_out_batch, 0, (size_t)B * (size_t)n_embd * sizeof(float), stream));
    }
    if (n_groups <= 0 || total_pairs <= 0) return;

    // Use WMMA FP16->FP32 Tensor Cores for prompt prefill (B >= 16); ultra-low-latency __dp4a for small batches (spec verify!)
    const bool use_wmma = (B >= 16);
    if (use_wmma) {
        if (preconvert_x) {
            int tot_x = B * n_embd;
            k_f32_to_f16_vec<<<(tot_x + 255) / 256, 256, 0, stream>>>(d_x_batch, g_batch_f16_buf, tot_x);
        }
        {
            size_t rb = (size_t)(n_embd / ggml_type_block_size(gate_up_type)) * ggml_type_byte_size(gate_up_type);
            dim3 grid((n_ff + 31) / 32, n_groups);
            switch (gate_up_type) {
                case GGML_TYPE_IQ3_XXS: k_moe_grouped_wmma_gate_up<GGML_TYPE_IQ3_XXS><<<grid, 128, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, g_batch_f16_buf, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
                case GGML_TYPE_IQ2_S:   k_moe_grouped_wmma_gate_up<GGML_TYPE_IQ2_S>  <<<grid, 128, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, g_batch_f16_buf, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
                case GGML_TYPE_IQ2_XS:  k_moe_grouped_wmma_gate_up<GGML_TYPE_IQ2_XS> <<<grid, 128, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, g_batch_f16_buf, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
                case GGML_TYPE_IQ2_XXS: k_moe_grouped_wmma_gate_up<GGML_TYPE_IQ2_XXS><<<grid, 128, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, g_batch_f16_buf, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
                case GGML_TYPE_NVFP4:   k_moe_grouped_wmma_gate_up<GGML_TYPE_NVFP4>  <<<grid, 128, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, g_batch_f16_buf, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
                case GGML_TYPE_IQ4_NL:  k_moe_grouped_wmma_gate_up<GGML_TYPE_IQ4_NL> <<<grid, 128, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, g_batch_f16_buf, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
                default: std::fprintf(stderr, "[CUDA] Unsupported grouped wmma gate_up %u\n", gate_up_type); std::abort();
            }
        }
        // Convert only active `total_pairs` hidden vectors to FP16 right after x_f16 (offset B * n_embd)!
        __half* h_f16 = g_batch_f16_buf + (size_t)B * n_embd;
        int tot_h = total_pairs * n_ff;
        k_f32_to_f16_vec<<<(tot_h + 255) / 256, 256, 0, stream>>>(d_h_scratch, h_f16, tot_h);
        {
            size_t rb = (size_t)(n_ff / ggml_type_block_size(down_type)) * ggml_type_byte_size(down_type);
            dim3 grid((n_embd + 63) / 64, n_groups);
            switch (down_type) {
                case GGML_TYPE_NVFP4:   k_moe_grouped_wmma_down<GGML_TYPE_NVFP4>  <<<grid, 128, 0, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_grp_tok_w, h_f16, d_out_batch, n_embd, n_ff, rb); break;
                case GGML_TYPE_IQ4_NL:  k_moe_grouped_wmma_down<GGML_TYPE_IQ4_NL> <<<grid, 128, 0, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_grp_tok_w, h_f16, d_out_batch, n_embd, n_ff, rb); break;
                case GGML_TYPE_IQ3_XXS: k_moe_grouped_wmma_down<GGML_TYPE_IQ3_XXS><<<grid, 128, 0, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_grp_tok_w, h_f16, d_out_batch, n_embd, n_ff, rb); break;
                default: std::fprintf(stderr, "[CUDA] Unsupported grouped wmma down %u\n", down_type); std::abort();
            }
        }
        return;
    }

    Q8Block32GPU* d_qx = (Q8Block32GPU*)g_batch_qx_buf;
    if (preconvert_x) {
        k_quantize_batch_q8<<<B, 256, 0, stream>>>(d_x_batch, d_qx, n_embd);
    }

    constexpr int WPB = 8;
    {
        size_t rb = (size_t)(n_embd / ggml_type_block_size(gate_up_type)) * ggml_type_byte_size(gate_up_type);
        dim3 grid((n_ff + WPB - 1) / WPB, n_groups);
        switch (gate_up_type) {
            case GGML_TYPE_IQ3_XXS: k_moe_grouped_gate_up<GGML_TYPE_IQ3_XXS><<<grid, WPB*32, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_qx, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_S:   k_moe_grouped_gate_up<GGML_TYPE_IQ2_S>  <<<grid, WPB*32, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_qx, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_XS:  k_moe_grouped_gate_up<GGML_TYPE_IQ2_XS> <<<grid, WPB*32, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_qx, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ2_XXS: k_moe_grouped_gate_up<GGML_TYPE_IQ2_XXS><<<grid, WPB*32, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_qx, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_NVFP4:   k_moe_grouped_gate_up<GGML_TYPE_NVFP4>  <<<grid, WPB*32, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_qx, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            case GGML_TYPE_IQ4_NL:  k_moe_grouped_gate_up<GGML_TYPE_IQ4_NL> <<<grid, WPB*32, 0, stream>>>(tbl.d_gu_ptrs, tbl.d_gu_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_qx, d_h_scratch, n_embd, n_ff, rb, situ_beta, situ_linear_beta); break;
            default: std::fprintf(stderr, "[CUDA] Unsupported grouped gate_up %u\n", gate_up_type); std::abort();
        }
    }
    Q8Block32GPU* d_qh = d_qx + (size_t)B * (n_embd >> 5);
    k_quantize_batch_q8<<<total_pairs, 256, 0, stream>>>(d_h_scratch, d_qh, n_ff);
    {
        size_t rb = (size_t)(n_ff / ggml_type_block_size(down_type)) * ggml_type_byte_size(down_type);
        dim3 grid((n_embd + WPB - 1) / WPB, n_groups);
        switch (down_type) {
            case GGML_TYPE_NVFP4:   k_moe_grouped_down<GGML_TYPE_NVFP4>  <<<grid, WPB*32, 0, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_grp_tok_w, d_qh, d_out_batch, n_embd, n_ff, rb); break;
            case GGML_TYPE_IQ4_NL:  k_moe_grouped_down<GGML_TYPE_IQ4_NL> <<<grid, WPB*32, 0, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_grp_tok_w, d_qh, d_out_batch, n_embd, n_ff, rb); break;
            case GGML_TYPE_IQ3_XXS: k_moe_grouped_down<GGML_TYPE_IQ3_XXS><<<grid, WPB*32, 0, stream>>>(tbl.d_dn_ptrs, tbl.d_dn_scales, d_grp_exp_ids, d_grp_tok_off, d_grp_tok_ids, d_grp_tok_w, d_qh, d_out_batch, n_embd, n_ff, rb); break;
            default: std::fprintf(stderr, "[CUDA] Unsupported grouped down %u\n", down_type); std::abort();
        }
    }
}

void ink_cuda_logit_softcap_argmax_batch(float* logits, int B, int n_unpadded, float softcap,
                                         int32_t* d_argmax, cudaStream_t stream) {
    for (int b = 0; b < B; ++b) {
        ink_cuda_logit_softcap_argmax(logits + (size_t)b * n_unpadded, n_unpadded, softcap,
                                      d_argmax ? (d_argmax + b) : nullptr, stream);
    }
}

// =========================================================================
// InkEngine v3.2: Streamed Multimodal Vision Kernels (inkling2v ViT + Projector)
// =========================================================================

__global__ void k_vision_layernorm(const float* __restrict__ x, const float* __restrict__ w,
                                   const float* __restrict__ b, float* __restrict__ out,
                                   int D, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * D;
    float* orow = out + (size_t)row * D;
    __shared__ float s_sum[8];
    __shared__ float s_sq[8];

    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) sum += xr[i];
    sum = warp_reduce_sum(sum);
    if ((tid & 31) == 0) s_sum[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < 8) ? s_sum[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sum[0] = v / (float)D;
    }
    __syncthreads();
    float mean = s_sum[0];

    float sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float d = xr[i] - mean;
        sq = fmaf(d, d, sq);
    }
    sq = warp_reduce_sum(sq);
    if ((tid & 31) == 0) s_sq[tid >> 5] = sq;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < 8) ? s_sq[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sq[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    float inv = s_sq[0];

    for (int i = tid; i < D; i += blockDim.x) {
        float norm_v = (xr[i] - mean) * inv;
        if (w) norm_v *= w[i];
        if (b) norm_v += b[i];
        orow[i] = norm_v;
    }
}

void ink_cuda_vision_layernorm(const float* x, const float* w, const float* b, float* out,
                               int N, int D, float eps, cudaStream_t stream) {
    if (N <= 0) return;
    k_vision_layernorm<<<N, 256, 0, stream>>>(x, w, b, out, D, eps);
}

// Fast WMMA FP16->FP32 Tensor Core GEMM for F16 / BF16 / F32 weights: Y[N, M] = X[N, K] @ W[M, K]^T + bias[M]
template <uint32_t W_TYPE>
__global__ void k_vision_gemm_wmma(const uint8_t* __restrict__ W, const float* __restrict__ bias,
                                   const __half* __restrict__ X_f16, float* __restrict__ Y,
                                   int N, int M, int K, int ld_out, size_t w_row_bytes) {
    using namespace nvcuda;
    int m_base = blockIdx.x * 32;
    int n_base = blockIdx.y * 32;
    int warp   = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int wm     = (warp & 1) * 16;
    int wn     = (warp >> 1) * 16;

    __shared__ __half s_W[32][16];
    __shared__ __half s_X[32][16];
    __shared__ float  s_C[4][256];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        // 128 threads cooperatively load 32x16 = 512 elements of W and 32x16 = 512 elements of X (4 elems/thread)
        #pragma unroll
        for (int idx = threadIdx.x; idx < 512; idx += 128) {
            int r = idx >> 4;
            int c = idx & 15;
            int gm = m_base + r;
            int gk = k0 + c;
            __half wv = __float2half(0.0f);
            if (gm < M && gk < K) {
                const uint8_t* rp = W + (size_t)gm * w_row_bytes;
                if constexpr (W_TYPE == GGML_TYPE_F16) {
                    wv = ((const __half*)rp)[gk];
                } else if constexpr (W_TYPE == GGML_TYPE_BF16) {
                    wv = __float2half(d_bf16_to_f32(((const uint16_t*)rp)[gk]));
                } else {
                    wv = __float2half(((const float*)rp)[gk]);
                }
            }
            s_W[r][c] = wv;

            int gn = n_base + r;
            __half xv = (gn < N && gk < K) ? X_f16[(size_t)gn * K + gk] : __float2half(0.0f);
            s_X[r][c] = xv;
        }
        __syncthreads();

        wmma::load_matrix_sync(frag_a, &s_X[wn][0], 16);
        wmma::load_matrix_sync(frag_b, &s_W[wm][0], 16);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncthreads();
    }

    wmma::store_matrix_sync(s_C[warp], frag_c, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        int r = i >> 4;
        int c = i & 15;
        int gn = n_base + wn + r;
        int gm = m_base + wm + c;
        if (gn < N && gm < M) {
            float val = s_C[warp][i];
            if (bias) val += bias[gm];
            Y[(size_t)gn * ld_out + gm] = val;
        }
    }
}

void ink_cuda_vision_gemm(uint32_t w_type, const void* W, const float* bias,
                          const float* X, float* Y,
                          int N, int M, int K, int ld_out, cudaStream_t stream) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (ld_out < M) ld_out = M;
    size_t elem_bytes = (w_type == GGML_TYPE_F32) ? 4 : 2;
    size_t w_row_bytes = (size_t)K * elem_bytes;
    const uint8_t* Wp = (const uint8_t*)W;

    // g_batch_f16_buf holds 1024 * 3072 = 3,145,728 halfs (6 MiB, zero new VRAM!)
    int max_chunk = std::max(32, (int)((1024ull * 3072ull) / (size_t)K));
    max_chunk = (max_chunk / 32) * 32;
    if (max_chunk < 32) max_chunk = 32;

    for (int n0 = 0; n0 < N; n0 += max_chunk) {
        int cn = std::min(max_chunk, N - n0);
        int tot_x = cn * K;
        k_f32_to_f16_vec<<<(tot_x + 255) / 256, 256, 0, stream>>>(X + (size_t)n0 * K, g_batch_f16_buf, tot_x);
        dim3 grid((M + 31) / 32, (cn + 31) / 32);
        float* Yc = Y + (size_t)n0 * ld_out;
        if (w_type == GGML_TYPE_F16) {
            k_vision_gemm_wmma<GGML_TYPE_F16> <<<grid, 128, 0, stream>>>(Wp, bias, g_batch_f16_buf, Yc, cn, M, K, ld_out, w_row_bytes);
        } else if (w_type == GGML_TYPE_BF16) {
            k_vision_gemm_wmma<GGML_TYPE_BF16><<<grid, 128, 0, stream>>>(Wp, bias, g_batch_f16_buf, Yc, cn, M, K, ld_out, w_row_bytes);
        } else {
            k_vision_gemm_wmma<GGML_TYPE_F32> <<<grid, 128, 0, stream>>>(Wp, bias, g_batch_f16_buf, Yc, cn, M, K, ld_out, w_row_bytes);
        }
    }
}

// 2D Vision RoPE: dims [0..d/2) rotated by pos_w, dims [d/2..d) rotated by pos_h, adjacent pairs (mode=0)
__global__ void k_vision_rope_2d(float* __restrict__ x, const int32_t* __restrict__ pos_w,
                                 const int32_t* __restrict__ pos_h, int N, int n_head, int d_head, float theta) {
    int half_d = d_head >> 1;
    int n_pairs = half_d >> 1; // 16 pairs per half for d_head=64
    int total = N * n_head * n_pairs;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int pair = idx % n_pairs;
    int tmp  = idx / n_pairs;
    int h    = tmp % n_head;
    int t    = tmp / n_head;

    float inv_freq = powf(theta, -(float)(2 * pair) / (float)half_d);
    float* ptr = x + ((size_t)t * n_head + h) * d_head;

    // First half: pos_w
    float ang_w = (float)pos_w[t] * inv_freq;
    float cw = cosf(ang_w), sw = sinf(ang_w);
    float x0 = ptr[2 * pair];
    float x1 = ptr[2 * pair + 1];
    ptr[2 * pair]     = x0 * cw - x1 * sw;
    ptr[2 * pair + 1] = x0 * sw + x1 * cw;

    // Second half: pos_h
    float ang_h = (float)pos_h[t] * inv_freq;
    float ch = cosf(ang_h), sh = sinf(ang_h);
    float y0 = ptr[half_d + 2 * pair];
    float y1 = ptr[half_d + 2 * pair + 1];
    ptr[half_d + 2 * pair]     = y0 * ch - y1 * sh;
    ptr[half_d + 2 * pair + 1] = y0 * sh + y1 * ch;
}

void ink_cuda_vision_rope_2d(float* x, const int32_t* pos_w, const int32_t* pos_h,
                             int N, int n_head, int d_head, float theta, cudaStream_t stream) {
    int n_pairs = d_head >> 2;
    int total = N * n_head * n_pairs;
    if (total <= 0) return;
    k_vision_rope_2d<<<(total + 255) / 256, 256, 0, stream>>>(x, pos_w, pos_h, N, n_head, d_head, theta);
}

// Tiled Bidirectional FlashAttention for ViT (d_head = 64, 4 warps/block = 4 queries/block, O(1) VRAM!)
__global__ void k_vision_flash_attn_d64(const float* __restrict__ Q, const float* __restrict__ K,
                                        const float* __restrict__ V, float* __restrict__ out,
                                        int N, int n_head, float scale) {
    constexpr int TILE_K = 32;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int q_idx = blockIdx.x * 4 + warp;
    int h = blockIdx.y;
    int stride = n_head * 64;

    __shared__ float s_K[TILE_K][64];
    __shared__ float s_V[TILE_K][64];

    float q0 = 0.0f, q1 = 0.0f;
    if (q_idx < N) {
        const float* qp = Q + (size_t)q_idx * stride + h * 64;
        q0 = qp[lane];
        q1 = qp[lane + 32];
    }

    float m_max = -FLT_MAX;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f;

    for (int k0 = 0; k0 < N; k0 += TILE_K) {
        int tile_len = min(TILE_K, N - k0);
        // 128 threads cooperatively load TILE_K x 64 = 2048 floats (16 floats/thread)
        for (int idx = threadIdx.x; idx < tile_len * 64; idx += 128) {
            int r = idx >> 6;
            int c = idx & 63;
            size_t off = (size_t)(k0 + r) * stride + h * 64 + c;
            s_K[r][c] = K[off];
            s_V[r][c] = V[off];
        }
        __syncthreads();

        if (q_idx < N) {
            for (int t = 0; t < tile_len; ++t) {
                float dot = q0 * s_K[t][lane] + q1 * s_K[t][lane + 32];
                dot = warp_reduce_sum(dot);
                float score = __shfl_sync(0xFFFFFFFFu, dot, 0) * scale;
                float m_new = fmaxf(m_max, score);
                float alpha = (m_max == -FLT_MAX) ? 0.0f : expf(m_max - m_new);
                float p     = expf(score - m_new);
                l_sum = l_sum * alpha + p;
                o0 = fmaf(o0, alpha, p * s_V[t][lane]);
                o1 = fmaf(o1, alpha, p * s_V[t][lane + 32]);
                m_max = m_new;
            }
        }
        __syncthreads();
    }

    if (q_idx < N) {
        float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* op = out + (size_t)q_idx * stride + h * 64;
        op[lane]      = o0 * inv_l;
        op[lane + 32] = o1 * inv_l;
    }
}

void ink_cuda_vision_flash_attn(const float* Q, const float* K, const float* V, float* out,
                                int N, int n_head, int d_head, float scale, cudaStream_t stream) {
    if (N <= 0 || d_head != 64) return;
    dim3 grid((N + 3) / 4, n_head);
    k_vision_flash_attn_d64<<<grid, 128, 0, stream>>>(Q, K, V, out, N, n_head, scale);
}

__global__ void k_vision_gelu_erf(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v * 0.5f * (1.0f + erff(v * 0.7071067811865475f));
    }
}

void ink_cuda_vision_gelu_erf(float* x, int total_elems, cudaStream_t stream) {
    if (total_elems <= 0) return;
    k_vision_gelu_erf<<<(total_elems + 255) / 256, 256, 0, stream>>>(x, total_elems);
}

__global__ void k_vision_layerscale_add(float* __restrict__ residual, const float* __restrict__ delta,
                                        const float* __restrict__ ls, int N, int D) {
    int total = N * D;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        residual[i] += delta[i] * ls[i % D];
    }
}

void ink_cuda_vision_layerscale_add(float* residual, const float* delta, const float* ls_weight,
                                    int N, int D, cudaStream_t stream) {
    int total = N * D;
    if (total <= 0) return;
    k_vision_layerscale_add<<<(total + 255) / 256, 256, 0, stream>>>(residual, delta, ls_weight, N, D);
}
