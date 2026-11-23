#include "cpu_kernels.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {

// 256-entry sign table packed directly as uint64_t (8 int8_t lanes of +1 or -1)
alignas(64) uint64_t g_sign_u64[256];
bool g_tables_init = false;

void init_avx_tables() {
    if (g_tables_init) return;
    for (int s = 0; s < 256; ++s) {
        int8_t tmp[8];
        for (int j = 0; j < 8; ++j) {
            tmp[j] = (s & (1 << j)) ? (int8_t)-1 : (int8_t)1;
        }
        std::memcpy(&g_sign_u64[s], tmp, 8);
    }
    g_tables_init = true;
}

inline float fp16_fast(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Zero-stack-spill 4x64-bit -> __m256i composition (eliminates MSVC store-forwarding stalls!)
inline __m256i pack_4x64_reg(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    __m128i lo = _mm_insert_epi64(_mm_cvtsi64_si128((long long)a), (long long)b, 1);
    __m128i hi = _mm_insert_epi64(_mm_cvtsi64_si128((long long)c), (long long)d, 1);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// Per-32-element quantized F32 activation (cache-line aligned)
struct alignas(64) Q8Block32 {
    int8_t qs[32];
    float  d;
    float  pad[7];
};

inline void quantize_f32_to_q8_32(const float* __restrict x, Q8Block32* __restrict out, int n) {
    const int nb = n / 32;
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (int i = 0; i < nb; ++i) {
        const float* xb = x + i * 32;
        __m256 v0 = _mm256_loadu_ps(xb +  0);
        __m256 v1 = _mm256_loadu_ps(xb +  8);
        __m256 v2 = _mm256_loadu_ps(xb + 16);
        __m256 v3 = _mm256_loadu_ps(xb + 24);
        __m256 m = _mm256_max_ps(
            _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask)),
            _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask)));
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
        float amax = _mm_cvtss_f32(m128);
        float d = amax * (1.0f / 127.0f);
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        out[i].d = d;
        __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
        __m128i p01_lo = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
        __m128i p01_hi = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
        __m128i p23_lo = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
        __m128i p23_hi = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));
        _mm_store_si128((__m128i*)(out[i].qs +  0), _mm_packs_epi16(p01_lo, p01_hi));
        _mm_store_si128((__m128i*)(out[i].qs + 16), _mm_packs_epi16(p23_lo, p23_hi));
    }
}

inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehdup_ps(lo));
    lo = _mm_add_ss(lo, _mm_movehl_ps(lo, lo));
    return _mm_cvtss_f32(lo);
}

// =========================================================================
// Zero-Stall AVX2 Integer Dot Products (vectorized FMA accumulation -> 1 hsum/row!)
// =========================================================================

float vec_dot_iq3_xxs_q8(const BlockIQ3_XXS* __restrict blocks, const Q8Block32* __restrict qx, int n) {
    const int nb = n / 256;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_fast(blocks[i].d) * 0.5f;
        const uint8_t* qs = blocks[i].qs;
        const uint8_t* ss = qs + 64;
        const Q8Block32* xb = qx + i * 8;

        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, ss + 4 * ib32, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * xb[ib32].d;

            uint64_t g0 = (uint64_t)INK_iq3xxs_grid[qs[0]] | ((uint64_t)INK_iq3xxs_grid[qs[1]] << 32);
            uint64_t g1 = (uint64_t)INK_iq3xxs_grid[qs[2]] | ((uint64_t)INK_iq3xxs_grid[qs[3]] << 32);
            uint64_t g2 = (uint64_t)INK_iq3xxs_grid[qs[4]] | ((uint64_t)INK_iq3xxs_grid[qs[5]] << 32);
            uint64_t g3 = (uint64_t)INK_iq3xxs_grid[qs[6]] | ((uint64_t)INK_iq3xxs_grid[qs[7]] << 32);
            __m256i w32 = pack_4x64_reg(g0, g1, g2, g3);

            __m256i sgn = pack_4x64_reg(
                g_sign_u64[INK_ksigns_iq2xs[(aux32 >>  0) & 127]],
                g_sign_u64[INK_ksigns_iq2xs[(aux32 >>  7) & 127]],
                g_sign_u64[INK_ksigns_iq2xs[(aux32 >> 14) & 127]],
                g_sign_u64[INK_ksigns_iq2xs[(aux32 >> 21) & 127]]);

            __m256i x32 = _mm256_load_si256((const __m256i*)xb[ib32].qs);
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w32, _mm256_sign_epi8(x32, sgn)), ones);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(db), _mm256_cvtepi32_ps(p32), acc);
            qs += 8;
        }
    }
    return hsum256_ps(acc);
}

float vec_dot_iq2_s_q8(const BlockIQ2_S* __restrict blocks, const Q8Block32* __restrict qx, int n) {
    const int nb = n / 256;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_fast(blocks[i].d) * 0.25f;
        const uint8_t* qs = blocks[i].qs;
        const uint8_t* qh = blocks[i].qh;
        const uint8_t* signs = qs + 32;
        const Q8Block32* xb = qx + i * 8;

        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float dx = d * xb[ib32].d;
            const float db0 = dx * (0.5f + (float)(blocks[i].scales[ib32] & 0xF));
            const float db1 = dx * (0.5f + (float)(blocks[i].scales[ib32] >> 4));
            const uint32_t qhb = qh[ib32];

            __m256i w32 = pack_4x64_reg(
                INK_iq2s_grid[qs[0] | ((qhb << 8) & 0x300)],
                INK_iq2s_grid[qs[1] | ((qhb << 6) & 0x300)],
                INK_iq2s_grid[qs[2] | ((qhb << 4) & 0x300)],
                INK_iq2s_grid[qs[3] | ((qhb << 2) & 0x300)]);
            __m256i sgn = pack_4x64_reg(g_sign_u64[signs[0]], g_sign_u64[signs[1]], g_sign_u64[signs[2]], g_sign_u64[signs[3]]);
            __m256i x32 = _mm256_load_si256((const __m256i*)xb[ib32].qs);
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w32, _mm256_sign_epi8(x32, sgn)), ones);
            __m256 vdb = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(db0)), _mm_set1_ps(db1), 1);
            acc = _mm256_fmadd_ps(vdb, _mm256_cvtepi32_ps(p32), acc);
            qs += 4;
            signs += 4;
        }
    }
    return hsum256_ps(acc);
}

float vec_dot_iq2_xs_q8(const BlockIQ2_XS* __restrict blocks, const Q8Block32* __restrict qx, int n) {
    const int nb = n / 256;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_fast(blocks[i].d) * 0.25f;
        const Q8Block32* xb = qx + i * 8;

        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float dx = d * xb[ib32].d;
            const float db0 = dx * (0.5f + (float)(blocks[i].scales[ib32] & 0xF));
            const float db1 = dx * (0.5f + (float)(blocks[i].scales[ib32] >> 4));
            const uint16_t* q = blocks[i].qs + 4 * ib32;

            __m256i w32 = pack_4x64_reg(
                INK_iq2xs_grid[q[0] & 511], INK_iq2xs_grid[q[1] & 511],
                INK_iq2xs_grid[q[2] & 511], INK_iq2xs_grid[q[3] & 511]);
            __m256i sgn = pack_4x64_reg(
                g_sign_u64[INK_ksigns_iq2xs[q[0] >> 9]], g_sign_u64[INK_ksigns_iq2xs[q[1] >> 9]],
                g_sign_u64[INK_ksigns_iq2xs[q[2] >> 9]], g_sign_u64[INK_ksigns_iq2xs[q[3] >> 9]]);
            __m256i x32 = _mm256_load_si256((const __m256i*)xb[ib32].qs);
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w32, _mm256_sign_epi8(x32, sgn)), ones);
            __m256 vdb = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(db0)), _mm_set1_ps(db1), 1);
            acc = _mm256_fmadd_ps(vdb, _mm256_cvtepi32_ps(p32), acc);
        }
    }
    return hsum256_ps(acc);
}

float vec_dot_iq2_xxs_q8(const BlockIQ2_XXS* __restrict blocks, const Q8Block32* __restrict qx, int n) {
    const int nb = n / 256;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_fast(blocks[i].d) * 0.25f;
        const Q8Block32* xb = qx + i * 8;

        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32[2];
            std::memcpy(aux32, blocks[i].qs + 4 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * xb[ib32].d;

            __m256i w32 = pack_4x64_reg(
                INK_iq2xxs_grid[aux8[0]], INK_iq2xxs_grid[aux8[1]],
                INK_iq2xxs_grid[aux8[2]], INK_iq2xxs_grid[aux8[3]]);
            __m256i sgn = pack_4x64_reg(
                g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >>  0) & 127]],
                g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >>  7) & 127]],
                g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >> 14) & 127]],
                g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >> 21) & 127]]);
            __m256i x32 = _mm256_load_si256((const __m256i*)xb[ib32].qs);
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w32, _mm256_sign_epi8(x32, sgn)), ones);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(db), _mm256_cvtepi32_ps(p32), acc);
        }
    }
    return hsum256_ps(acc);
}

float vec_dot_nvfp4_q8(const BlockNVFP4* __restrict blocks, const Q8Block32* __restrict qx, int n) {
    const int nb = n / 64;
    const __m128i lut = _mm_loadu_si128((const __m128i*)INK_kvalues_fp4);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const BlockNVFP4& b = blocks[i];
        const Q8Block32* xb = qx + i * 2;

        // Half 0 (sub-blocks 0, 1)
        {
            __m128i q0 = _mm_loadl_epi64((const __m128i*)(b.qs + 0));
            __m128i q1 = _mm_loadl_epi64((const __m128i*)(b.qs + 8));
            __m128i w0 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q0, mask0f), _mm_and_si128(_mm_srli_epi16(q0, 4), mask0f)));
            __m128i w1 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q1, mask0f), _mm_and_si128(_mm_srli_epi16(q1, 4), mask0f)));
            __m256i w32 = _mm256_inserti128_si256(_mm256_castsi128_si256(w0), w1, 1);
            __m256i x32 = _mm256_load_si256((const __m256i*)xb[0].qs);
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(x32), _mm256_sign_epi8(w32, x32)), ones);
            float dx = xb[0].d;
            __m256 vd = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_set1_ps(INK_ue4m3_half_lut[b.d[0]] * dx)),
                _mm_set1_ps(INK_ue4m3_half_lut[b.d[1]] * dx), 1);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32), acc);
        }
        // Half 1 (sub-blocks 2, 3)
        {
            __m128i q2 = _mm_loadl_epi64((const __m128i*)(b.qs + 16));
            __m128i q3 = _mm_loadl_epi64((const __m128i*)(b.qs + 24));
            __m128i w2 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q2, mask0f), _mm_and_si128(_mm_srli_epi16(q2, 4), mask0f)));
            __m128i w3 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q3, mask0f), _mm_and_si128(_mm_srli_epi16(q3, 4), mask0f)));
            __m256i w32 = _mm256_inserti128_si256(_mm256_castsi128_si256(w2), w3, 1);
            __m256i x32 = _mm256_load_si256((const __m256i*)xb[1].qs);
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(x32), _mm256_sign_epi8(w32, x32)), ones);
            float dx = xb[1].d;
            __m256 vd = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_set1_ps(INK_ue4m3_half_lut[b.d[2]] * dx)),
                _mm_set1_ps(INK_ue4m3_half_lut[b.d[3]] * dx), 1);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32), acc);
        }
    }
    return hsum256_ps(acc);
}

float vec_dot_iq4_nl_q8(const BlockIQ4_NL* __restrict blocks, const Q8Block32* __restrict qx, int n) {
    const int nb = n / 32;
    const __m128i lut = _mm_loadu_si128((const __m128i*)INK_kvalues_iq4nl);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_fast(blocks[i].d) * qx[i].d;
        __m128i q16 = _mm_loadu_si128((const __m128i*)blocks[i].qs);
        __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q16, mask0f));
        __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q16, 4), mask0f));
        __m256i w32 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        __m256i x32 = _mm256_load_si256((const __m256i*)qx[i].qs);
        __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(x32), _mm256_sign_epi8(w32, x32)), ones);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
    }
    return hsum256_ps(acc);
}

inline float vec_dot_row_q8(uint32_t type, const void* row_data, const Q8Block32* qx, int n) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8((const BlockIQ3_XXS*)row_data, qx, n);
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8  ((const BlockIQ2_S*)row_data,   qx, n);
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8 ((const BlockIQ2_XS*)row_data,  qx, n);
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8((const BlockIQ2_XXS*)row_data, qx, n);
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8  ((const BlockNVFP4*)row_data,   qx, n);
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8 ((const BlockIQ4_NL*)row_data,  qx, n);
        default: return 0.0f;
    }
}

// =========================================================================
// Unpack-Once, Dot-N AVX2 Kernel for Grouped Batched MoE (Prefill & Verify!)
// Unpacks IQ3/IQ2/NVFP4/IQ4_NL codebooks ONCE per weight block in YMM registers,
// then dots all `cnt` tokens routed to this expert with only 4 AVX2 ops/block/token!
// =========================================================================
#define DOT_N_BLOCK(w_abs, w_sgn, vdb_expr, blk_idx) do { \
    for (int t = 0; t < cnt; ++t) { \
        const Q8Block32& xb = qx_ptrs[t][(blk_idx)]; \
        __m256i x32 = _mm256_load_si256((const __m256i*)xb.qs); \
        __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16((w_abs), _mm256_sign_epi8(x32, (w_sgn))), ones); \
        acc[t] = _mm256_fmadd_ps(_mm256_mul_ps((vdb_expr), _mm256_set1_ps(xb.d)), _mm256_cvtepi32_ps(p32), acc[t]); \
    } \
} while (0)

void vec_dot_row_q8_multi(uint32_t type, const void* __restrict row_data,
                          const Q8Block32* const* __restrict qx_ptrs, int cnt, int n, float* __restrict out) {
    if (cnt == 1) {
        out[0] = vec_dot_row_q8(type, row_data, qx_ptrs[0], n);
        return;
    }
    const __m256i ones = _mm256_set1_epi16(1);
    alignas(32) __m256 acc[64];
    for (int t = 0; t < cnt; ++t) acc[t] = _mm256_setzero_ps();

    if (type == GGML_TYPE_IQ3_XXS) {
        const BlockIQ3_XXS* blocks = (const BlockIQ3_XXS*)row_data;
        const int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d) * 0.5f;
            const uint8_t* qs = blocks[i].qs;
            const uint8_t* ss = qs + 64;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t aux32;
                std::memcpy(&aux32, ss + 4 * ib32, 4);
                __m256 vdb = _mm256_set1_ps(d * (0.5f + (float)(aux32 >> 28)));
                uint64_t g0 = (uint64_t)INK_iq3xxs_grid[qs[0]] | ((uint64_t)INK_iq3xxs_grid[qs[1]] << 32);
                uint64_t g1 = (uint64_t)INK_iq3xxs_grid[qs[2]] | ((uint64_t)INK_iq3xxs_grid[qs[3]] << 32);
                uint64_t g2 = (uint64_t)INK_iq3xxs_grid[qs[4]] | ((uint64_t)INK_iq3xxs_grid[qs[5]] << 32);
                uint64_t g3 = (uint64_t)INK_iq3xxs_grid[qs[6]] | ((uint64_t)INK_iq3xxs_grid[qs[7]] << 32);
                __m256i w32 = pack_4x64_reg(g0, g1, g2, g3);
                __m256i sgn = pack_4x64_reg(
                    g_sign_u64[INK_ksigns_iq2xs[(aux32 >>  0) & 127]],
                    g_sign_u64[INK_ksigns_iq2xs[(aux32 >>  7) & 127]],
                    g_sign_u64[INK_ksigns_iq2xs[(aux32 >> 14) & 127]],
                    g_sign_u64[INK_ksigns_iq2xs[(aux32 >> 21) & 127]]);
                DOT_N_BLOCK(w32, sgn, vdb, i * 8 + ib32);
                qs += 8;
            }
        }
    } else if (type == GGML_TYPE_IQ2_S) {
        const BlockIQ2_S* blocks = (const BlockIQ2_S*)row_data;
        const int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d) * 0.25f;
            const uint8_t* qs = blocks[i].qs;
            const uint8_t* qh = blocks[i].qh;
            const uint8_t* signs = qs + 32;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db0 = d * (0.5f + (float)(blocks[i].scales[ib32] & 0xF));
                const float db1 = d * (0.5f + (float)(blocks[i].scales[ib32] >> 4));
                __m256 vdb = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(db0)), _mm_set1_ps(db1), 1);
                const uint32_t qhb = qh[ib32];
                __m256i w32 = pack_4x64_reg(
                    INK_iq2s_grid[qs[0] | ((qhb << 8) & 0x300)],
                    INK_iq2s_grid[qs[1] | ((qhb << 6) & 0x300)],
                    INK_iq2s_grid[qs[2] | ((qhb << 4) & 0x300)],
                    INK_iq2s_grid[qs[3] | ((qhb << 2) & 0x300)]);
                __m256i sgn = pack_4x64_reg(g_sign_u64[signs[0]], g_sign_u64[signs[1]], g_sign_u64[signs[2]], g_sign_u64[signs[3]]);
                DOT_N_BLOCK(w32, sgn, vdb, i * 8 + ib32);
                qs += 4; signs += 4;
            }
        }
    } else if (type == GGML_TYPE_IQ2_XS) {
        const BlockIQ2_XS* blocks = (const BlockIQ2_XS*)row_data;
        const int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d) * 0.25f;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db0 = d * (0.5f + (float)(blocks[i].scales[ib32] & 0xF));
                const float db1 = d * (0.5f + (float)(blocks[i].scales[ib32] >> 4));
                __m256 vdb = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(db0)), _mm_set1_ps(db1), 1);
                const uint16_t* q = blocks[i].qs + 4 * ib32;
                __m256i w32 = pack_4x64_reg(
                    INK_iq2xs_grid[q[0] & 511], INK_iq2xs_grid[q[1] & 511],
                    INK_iq2xs_grid[q[2] & 511], INK_iq2xs_grid[q[3] & 511]);
                __m256i sgn = pack_4x64_reg(
                    g_sign_u64[INK_ksigns_iq2xs[q[0] >> 9]], g_sign_u64[INK_ksigns_iq2xs[q[1] >> 9]],
                    g_sign_u64[INK_ksigns_iq2xs[q[2] >> 9]], g_sign_u64[INK_ksigns_iq2xs[q[3] >> 9]]);
                DOT_N_BLOCK(w32, sgn, vdb, i * 8 + ib32);
            }
        }
    } else if (type == GGML_TYPE_IQ2_XXS) {
        const BlockIQ2_XXS* blocks = (const BlockIQ2_XXS*)row_data;
        const int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d) * 0.25f;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t aux32[2];
                std::memcpy(aux32, blocks[i].qs + 4 * ib32, 8);
                const uint8_t* aux8 = (const uint8_t*)aux32;
                __m256 vdb = _mm256_set1_ps(d * (0.5f + (float)(aux32[1] >> 28)));
                __m256i w32 = pack_4x64_reg(
                    INK_iq2xxs_grid[aux8[0]], INK_iq2xxs_grid[aux8[1]],
                    INK_iq2xxs_grid[aux8[2]], INK_iq2xxs_grid[aux8[3]]);
                __m256i sgn = pack_4x64_reg(
                    g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >>  0) & 127]],
                    g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >>  7) & 127]],
                    g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >> 14) & 127]],
                    g_sign_u64[INK_ksigns_iq2xs[(aux32[1] >> 21) & 127]]);
                DOT_N_BLOCK(w32, sgn, vdb, i * 8 + ib32);
            }
        }
    } else if (type == GGML_TYPE_NVFP4) {
        const BlockNVFP4* blocks = (const BlockNVFP4*)row_data;
        const int nb = n / 64;
        const __m128i lut = _mm_loadu_si128((const __m128i*)INK_kvalues_fp4);
        const __m128i mask0f = _mm_set1_epi8(0x0F);
        for (int i = 0; i < nb; ++i) {
            const BlockNVFP4& b = blocks[i];
            {
                __m128i q0 = _mm_loadl_epi64((const __m128i*)(b.qs + 0));
                __m128i q1 = _mm_loadl_epi64((const __m128i*)(b.qs + 8));
                __m128i w0 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q0, mask0f), _mm_and_si128(_mm_srli_epi16(q0, 4), mask0f)));
                __m128i w1 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q1, mask0f), _mm_and_si128(_mm_srli_epi16(q1, 4), mask0f)));
                __m256i w32 = _mm256_inserti128_si256(_mm256_castsi128_si256(w0), w1, 1);
                __m256i wa = _mm256_abs_epi8(w32);
                __m256 vd = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_set1_ps(INK_ue4m3_half_lut[b.d[0]])),
                    _mm_set1_ps(INK_ue4m3_half_lut[b.d[1]]), 1);
                DOT_N_BLOCK(wa, w32, vd, i * 2 + 0);
            }
            {
                __m128i q2 = _mm_loadl_epi64((const __m128i*)(b.qs + 16));
                __m128i q3 = _mm_loadl_epi64((const __m128i*)(b.qs + 24));
                __m128i w2 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q2, mask0f), _mm_and_si128(_mm_srli_epi16(q2, 4), mask0f)));
                __m128i w3 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(_mm_and_si128(q3, mask0f), _mm_and_si128(_mm_srli_epi16(q3, 4), mask0f)));
                __m256i w32 = _mm256_inserti128_si256(_mm256_castsi128_si256(w2), w3, 1);
                __m256i wa = _mm256_abs_epi8(w32);
                __m256 vd = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_set1_ps(INK_ue4m3_half_lut[b.d[2]])),
                    _mm_set1_ps(INK_ue4m3_half_lut[b.d[3]]), 1);
                DOT_N_BLOCK(wa, w32, vd, i * 2 + 1);
            }
        }
    } else if (type == GGML_TYPE_IQ4_NL) {
        const BlockIQ4_NL* blocks = (const BlockIQ4_NL*)row_data;
        const int nb = n / 32;
        const __m128i lut = _mm_loadu_si128((const __m128i*)INK_kvalues_iq4nl);
        const __m128i mask0f = _mm_set1_epi8(0x0F);
        for (int i = 0; i < nb; ++i) {
            __m128i q16 = _mm_loadu_si128((const __m128i*)blocks[i].qs);
            __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q16, mask0f));
            __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q16, 4), mask0f));
            __m256i w32 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
            __m256i wa = _mm256_abs_epi8(w32);
            __m256 vd = _mm256_set1_ps(fp16_fast(blocks[i].d));
            DOT_N_BLOCK(wa, w32, vd, i);
        }
    }
    for (int t = 0; t < cnt; ++t) out[t] = hsum256_ps(acc[t]);
}
#undef DOT_N_BLOCK

inline float situ_act(float gate, float up, float beta, float lb) {
    float g = beta * std::tanh(gate / beta) * (1.0f / (1.0f + std::exp(-gate)));
    float u = (lb > 0.0f) ? (lb * std::tanh(up / lb)) : up;
    return g * u;
}

} // namespace

CPUThreadPool::CPUThreadPool(int n_threads) : n_threads_(std::max(1, n_threads)) {
    init_avx_tables();
    h_buf_.resize(16 * 1152, 0.0f);
    for (int t = 1; t < n_threads_; ++t) {
        workers_.emplace_back([this, t]() {
            worker_loop(t);
        });
    }
}

CPUThreadPool::~CPUThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_.store(true, std::memory_order_seq_cst);
        gen_.fetch_add(1, std::memory_order_seq_cst);
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void CPUThreadPool::worker_loop(int tid) {
    uint64_t seen = 0;
    while (true) {
        uint64_t g = gen_.load(std::memory_order_acquire);
        if (g == seen) {
            for (int spins = 0; spins < 8000; ++spins) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen) break;
                _mm_pause();
            }
            if (g == seen) {
                std::unique_lock<std::mutex> lk(mtx_);
                sleeping_.fetch_add(1, std::memory_order_seq_cst);
                cv_.wait(lk, [&]() {
                    return gen_.load(std::memory_order_acquire) != seen ||
                           stop_.load(std::memory_order_acquire);
                });
                sleeping_.fetch_sub(1, std::memory_order_seq_cst);
                g = gen_.load(std::memory_order_acquire);
            }
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = g;
        cur_fn_(cur_ctx_, tid, n_threads_);
        done_count_.fetch_add(1, std::memory_order_release);
    }
}

void CPUThreadPool::dispatch(TaskFn fn, void* ctx) {
    cur_fn_ = fn;
    cur_ctx_ = ctx;
    done_count_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lk(mtx_);
        cv_.notify_all();
    }
    fn(ctx, 0, n_threads_);
    const int expected = n_threads_ - 1;
    while (done_count_.load(std::memory_order_acquire) < expected) {
        _mm_pause();
    }
}

float CPUThreadPool::vec_dot_row(uint32_t type, const void* row_data, const float* x, int n) {
    init_avx_tables();
    alignas(64) Q8Block32 qx[128];
    quantize_f32_to_q8_32(x, qx, n);
    return vec_dot_row_q8(type, row_data, qx, n);
}

void CPUThreadPool::dequantize_row(uint32_t type, const void* src, float* dst, int n) {
    if (type == GGML_TYPE_F32) { std::memcpy(dst, src, (size_t)n * 4); return; }
    if (type == GGML_TYPE_BF16) {
        const uint16_t* s = (const uint16_t*)src;
        for (int i = 0; i < n; ++i) dst[i] = bf16_to_fp32_host(s[i]);
        return;
    }
    if (type == GGML_TYPE_F16) {
        const uint16_t* s = (const uint16_t*)src;
        for (int i = 0; i < n; ++i) dst[i] = fp16_fast(s[i]);
        return;
    }
    if (type == GGML_TYPE_Q8_0) {
        const BlockQ8_0* b = (const BlockQ8_0*)src;
        int nb = n / 32;
        for (int i = 0; i < nb; ++i) {
            float d = fp16_fast(b[i].d);
            for (int j = 0; j < 32; ++j) dst[i * 32 + j] = b[i].qs[j] * d;
        }
        return;
    }
    if (type == GGML_TYPE_Q4_K) {
        const BlockQ4_K* blocks = (const BlockQ4_K*)src;
        int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d);
            const float dmin = fp16_fast(blocks[i].dmin);
            const uint8_t* q = blocks[i].qs;
            float* y = dst + i * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, blocks[i].scales, sc, m);
                const float d1 = d * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, blocks[i].scales, sc, m);
                const float d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32; ++l) y[l + 0]  = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) y[l + 32] = d2 * (q[l] >> 4)  - m2;
                y += 64; q += 32; is += 2;
            }
        }
        return;
    }
    if (type == GGML_TYPE_Q5_K) {
        const BlockQ5_K* blocks = (const BlockQ5_K*)src;
        int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d);
            const float dmin = fp16_fast(blocks[i].dmin);
            const uint8_t* ql = blocks[i].qs;
            const uint8_t* qh = blocks[i].qh;
            float* y = dst + i * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, blocks[i].scales, sc, m);
                const float d1 = d * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, blocks[i].scales, sc, m);
                const float d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32; ++l) y[l + 0]  = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) y[l + 32] = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
                y += 64; ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
        return;
    }
    if (type == GGML_TYPE_Q6_K) {
        const BlockQ6_K* blocks = (const BlockQ6_K*)src;
        int nb = n / 256;
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_fast(blocks[i].d);
            const uint8_t* ql = blocks[i].ql;
            const uint8_t* qh = blocks[i].qh;
            const int8_t* sc = blocks[i].scales;
            float* y = dst + i * 256;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l +  0] = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        return;
    }
    std::memset(dst, 0, (size_t)n * 4);
}

void CPUThreadPool::run_moe_experts(
    const float* x,
    float* out,
    const CPUExpertTask* tasks,
    int n_tasks,
    int n_embd,
    int n_ff,
    float situ_beta,
    float situ_linear_beta)
{
    if (n_tasks <= 0) {
        std::memset(out, 0, (size_t)n_embd * sizeof(float));
        return;
    }
    if ((int)h_buf_.size() < n_tasks * n_ff) {
        h_buf_.resize((size_t)n_tasks * n_ff);
    }

    alignas(64) Q8Block32 qx[128];
    quantize_f32_to_q8_32(x, qx, n_embd);

    size_t gu_rb = (size_t)(n_embd / ggml_type_block_size(tasks[0].gate_up_type)) * ggml_type_byte_size(tasks[0].gate_up_type);
    size_t dn_rb = (size_t)(n_ff   / ggml_type_block_size(tasks[0].down_type))    * ggml_type_byte_size(tasks[0].down_type);
    uint32_t gu_type = tasks[0].gate_up_type;
    uint32_t dn_type = tasks[0].down_type;

    // Pass 1: Sequential gate stream THEN sequential up stream per expert -> 100% hardware prefetcher hit!
    parallel_for_threads([&](int tid, int nth) {
        int j0 = (n_ff * tid) / nth;
        int j1 = (n_ff * (tid + 1)) / nth;
        float g_tmp[1152];
        for (int t = 0; t < n_tasks; ++t) {
            const CPUExpertTask& tk = tasks[t];
            const uint8_t* base = (const uint8_t*)tk.gate_up_data;
            float sc = tk.gate_up_scale;
            for (int j = j0; j < j1; ++j) {
                g_tmp[j - j0] = vec_dot_row_q8(gu_type, base + (size_t)j * gu_rb, qx, n_embd) * sc;
            }
            float* hout = h_buf_.data() + (size_t)t * n_ff;
            for (int j = j0; j < j1; ++j) {
                float u = vec_dot_row_q8(gu_type, base + (size_t)(j + n_ff) * gu_rb, qx, n_embd) * sc;
                hout[j] = situ_act(g_tmp[j - j0], u, situ_beta, situ_linear_beta);
            }
        }
    });

    alignas(64) Q8Block32 qh[8][40];
    for (int t = 0; t < n_tasks; ++t) {
        quantize_f32_to_q8_32(h_buf_.data() + (size_t)t * n_ff, qh[t], n_ff);
    }

    // Pass 2: Expert-outer sequential streaming with 64-byte (16-float) cache-line aligned slices (zero false sharing!)
    parallel_for_threads([&](int tid, int nth) {
        int n_lines = (n_embd + 15) / 16;
        int r0 = std::min(n_embd, ((n_lines * tid) / nth) * 16);
        int r1 = std::min(n_embd, ((n_lines * (tid + 1)) / nth) * 16);
        if (r0 >= r1) return;
        std::memset(out + r0, 0, (size_t)(r1 - r0) * sizeof(float));
        for (int t = 0; t < n_tasks; ++t) {
            const CPUExpertTask& tk = tasks[t];
            const uint8_t* base = (const uint8_t*)tk.down_data;
            float sc = tk.down_scale * tk.router_weight;
            const Q8Block32* qht = qh[t];
            for (int r = r0; r < r1; ++r) {
                out[r] += vec_dot_row_q8(dn_type, base + (size_t)r * dn_rb, qht, n_ff) * sc;
            }
        }
    });
}

void CPUThreadPool::run_moe_experts_grouped(
    const float* x_batch,
    float* out_batch,
    int B,
    const CPUExpertGroupTask* groups,
    int n_groups,
    const int32_t* tok_ids,
    const float* tok_weights,
    int total_pairs,
    int n_embd,
    int n_ff,
    float situ_beta,
    float situ_linear_beta)
{
    if (n_groups <= 0 || total_pairs <= 0 || B <= 0) {
        std::memset(out_batch, 0, (size_t)B * (size_t)n_embd * sizeof(float));
        return;
    }

    const int nb_embd = n_embd / 32;
    const int nb_ff   = n_ff / 32;
    std::vector<Q8Block32> qx_all((size_t)B * nb_embd);
    parallel_for_threads([&](int tid, int nth) {
        for (int b = tid; b < B; b += nth) {
            quantize_f32_to_q8_32(x_batch + (size_t)b * n_embd, qx_all.data() + (size_t)b * nb_embd, n_embd);
        }
    });

    if ((int)h_buf_.size() < total_pairs * n_ff) {
        h_buf_.resize((size_t)total_pairs * n_ff);
    }

    const uint32_t gu_type = groups[0].gate_up_type;
    const uint32_t dn_type = groups[0].down_type;
    const size_t gu_rb = (size_t)(n_embd / ggml_type_block_size(gu_type)) * ggml_type_byte_size(gu_type);
    const size_t dn_rb = (size_t)(n_ff   / ggml_type_block_size(dn_type)) * ggml_type_byte_size(dn_type);

    // Pass 1: Weight-outer Gate + Up + SITU with UNPACK-ONCE-DOT-N AVX2!
    parallel_for_threads([&](int tid, int nth) {
        int j0 = (n_ff * tid) / nth;
        int j1 = (n_ff * (tid + 1)) / nth;
        float g_vals[64], u_vals[64];
        const Q8Block32* ptrs[64];
        for (int g = 0; g < n_groups; ++g) {
            const auto& grp = groups[g];
            const uint8_t* base = (const uint8_t*)grp.gate_up_data;
            const float sc = grp.gate_up_scale;
            const int off = grp.tok_offset;
            const int cnt = grp.tok_count;
            for (int t = 0; t < cnt; ++t) ptrs[t] = qx_all.data() + (size_t)tok_ids[off + t] * nb_embd;

            for (int j = j0; j < j1; ++j) {
                vec_dot_row_q8_multi(gu_type, base + (size_t)j * gu_rb,          ptrs, cnt, n_embd, g_vals);
                vec_dot_row_q8_multi(gu_type, base + (size_t)(j + n_ff) * gu_rb, ptrs, cnt, n_embd, u_vals);
                for (int t = 0; t < cnt; ++t) {
                    h_buf_[(size_t)(off + t) * n_ff + j] = situ_act(g_vals[t] * sc, u_vals[t] * sc, situ_beta, situ_linear_beta);
                }
            }
        }
    });

    // Quantize all intermediate hidden vectors in parallel
    std::vector<Q8Block32> qh_all((size_t)total_pairs * nb_ff);
    parallel_for_threads([&](int tid, int nth) {
        for (int p = tid; p < total_pairs; p += nth) {
            quantize_f32_to_q8_32(h_buf_.data() + (size_t)p * n_ff, qh_all.data() + (size_t)p * nb_ff, n_ff);
        }
    });

    // Pass 2: Weight-outer Down + Weighted Scatter-Add with UNPACK-ONCE-DOT-N AVX2 (ZERO atomics, 64B-aligned slices!)
    parallel_for_threads([&](int tid, int nth) {
        int n_lines = (n_embd + 15) / 16;
        int r0 = std::min(n_embd, ((n_lines * tid) / nth) * 16);
        int r1 = std::min(n_embd, ((n_lines * (tid + 1)) / nth) * 16);
        if (r0 >= r1) return;
        for (int b = 0; b < B; ++b) {
            std::memset(out_batch + (size_t)b * n_embd + r0, 0, (size_t)(r1 - r0) * sizeof(float));
        }
        float d_vals[64];
        const Q8Block32* ptrs[64];
        for (int g = 0; g < n_groups; ++g) {
            const auto& grp = groups[g];
            const uint8_t* base = (const uint8_t*)grp.down_data;
            const float dsc = grp.down_scale;
            const int off = grp.tok_offset;
            const int cnt = grp.tok_count;
            for (int t = 0; t < cnt; ++t) ptrs[t] = qh_all.data() + (size_t)(off + t) * nb_ff;

            for (int r = r0; r < r1; ++r) {
                vec_dot_row_q8_multi(dn_type, base + (size_t)r * dn_rb, ptrs, cnt, n_ff, d_vals);
                for (int t = 0; t < cnt; ++t) {
                    int b = tok_ids[off + t];
                    out_batch[(size_t)b * n_embd + r] += d_vals[t] * (dsc * tok_weights[off + t]);
                }
            }
        }
    });
}
