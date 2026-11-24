#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "quant_tables.h"

#define INK_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e); \
        std::abort(); \
    } \
} while (0)

enum KVQuantMode : int {
    KV_QUANT_Q4_0 = 0, // 72 bytes / head (1152 B / token / layer -> 9 KB/tok across 8 global layers)
    KV_QUANT_Q8_0 = 1, // 136 bytes / head (2176 B / token / layer -> 17 KB/tok across 8 global layers)
    KV_QUANT_F16  = 2, // 256 bytes / head (4096 B / token / layer -> 32 KB/tok across 8 global layers)
};

constexpr int INK_PAGE_TOKENS = 64;
constexpr int INK_MAX_GPU_EXPERTS = 8;

struct GPUExpertBatch {
    const void* gate_up_ptrs[INK_MAX_GPU_EXPERTS];
    const void* down_ptrs[INK_MAX_GPU_EXPERTS];
    float gate_up_scales[INK_MAX_GPU_EXPERTS];
    float down_w_scales[INK_MAX_GPU_EXPERTS]; // down_scale * router_weight
    int count;
};

struct PinnedLayerDeviceTable {
    uint64_t* d_gu_ptrs = nullptr;  // [128] (0 if expert is NOT in VRAM)
    uint64_t* d_dn_ptrs = nullptr;  // [128] (0 if expert is NOT in VRAM)
    float*    d_gu_scales = nullptr; // [128]
    float*    d_dn_scales = nullptr; // [128]
    bool      valid = false;
};

struct alignas(64) MoESplitPacket {
    int32_t n_gpu;
    int32_t n_cpu;
    int32_t all_ids[8];
    int32_t miss_ids[8];
    float   miss_weights[8];
};

struct BlockPtrList {
    const float* ptrs[12];
    int count;
};

struct FusedProjSlot {
    const uint8_t* W;
    const float*   x;
    float*         y;
    uint32_t       type;
    int            M;
    size_t         row_bytes;
    int            block_start;
    int            block_end;
};

struct FusedProjList {
    FusedProjSlot slots[8];
    int count;
    int total_blocks;
};

void ink_cuda_init_tables();

void ink_cuda_f32_to_f16(const float* src, uint16_t* dst, int n, cudaStream_t stream);

void ink_cuda_gemv(
    uint32_t type,
    const void* W,
    const float* x,
    float* y,
    int M,
    int K,
    cudaStream_t stream);

// Super-fused multi-projection GEMV (Q+Gate+K+V+Rel+Sink[+DynK+DynV] in 1 launch!)
void ink_cuda_fused_multi_gemv(
    const FusedProjList& list,
    int K,
    cudaStream_t stream);

// Fused gate_sigmoid_mul(x, gate) + GEMV (eliminates 48 gate_mul launches + VRAM roundtrips!)
void ink_cuda_gemv_gated(
    uint32_t type,
    const void* W,
    const float* x,
    const float* gate,
    float* y,
    int M,
    int K,
    cudaStream_t stream);

// Fused 3-way Loop Control + 3-way RMSNorm for Core Passes 1 & 2 (9 launches -> 3, 3x SM occupancy!)
void ink_cuda_loop_control_3way(
    const float* cur0, const float* cur1, const float* cur2,
    const float* msg0, const float* msg1, const float* msg2,
    uint32_t w_type,
    const void* w_down0, const void* w_gain0, const void* w_forget0,
    const void* w_down1, const void* w_gain1, const void* w_forget1,
    const void* w_down2, const void* w_gain2, const void* w_forget2,
    float* z_scratch_3x, // [3 * rank]
    float* out0, float* out1, float* out2,
    int n_embd, int rank, cudaStream_t stream);

void ink_cuda_rms_norm_3way(
    const float* x0, const float* x1, const float* x2,
    const float* w0, const float* w1, const float* w2,
    float* out0, float* out1, float* out2,
    int n, float eps, cudaStream_t stream);

// Fused K + V ShortConv in 1 launch
void ink_cuda_shortconv_kv(
    const float* k_in, float* k_state, const float* k_weight, const float* dyn_k, float* k_out,
    const float* v_in, float* v_state, const float* v_weight, const float* dyn_v, float* v_out,
    int n_channels, int n_heads, cudaStream_t stream);

// Fused FFN RMSNorm + Exact FP32 MoE Router + Top-K (eliminates 48 norm launches & global reload!)
void ink_cuda_fused_ffn_norm_router(
    const float* ffn_x,
    const float* ffn_norm_w,
    float* d_ffn_normed,
    const float* gate_inp,
    const float* bias,
    float* logits_scratch,
    int32_t* top_ids,
    float* top_weights,
    int n_expert, int n_embd, int k,
    float eps, float weights_scale,
    cudaStream_t stream);

void ink_cuda_rms_norm(
    const float* x,
    const float* w,
    float* out,
    int n,
    float eps,
    cudaStream_t stream);

void ink_cuda_add(const float* a, const float* b, float* out, int n, cudaStream_t stream);
void ink_cuda_add_inplace(float* a, const float* b, int n, cudaStream_t stream);
void ink_cuda_gate_sigmoid_mul(float* x, const float* gate, int n, cudaStream_t stream);

void ink_cuda_attn_res_mix(
    BlockPtrList states,
    const float* query,
    float* out,
    int n_embd,
    int n_queries,
    cudaStream_t stream);

void ink_cuda_loop_control(
    const float* cur,
    const float* msg,
    uint32_t w_type,
    const void* w_down,
    const void* w_gain,
    const void* w_forget,
    float* z_scratch,
    float* out,
    int n_embd,
    int rank,
    cudaStream_t stream);

void ink_cuda_shortconv(
    const float* x_in,
    float* state,
    const float* weight,
    const float* dynamic,
    float* out,
    int n_channels,
    int n_heads,
    cudaStream_t stream);

// Fused ShortConv + Residual Add: out = res + shortconv(x_in)
void ink_cuda_shortconv_add(
    const float* x_in,
    float* state,
    const float* weight,
    const float* res,
    float* out,
    int n_channels,
    cudaStream_t stream);

void ink_cuda_qkv_post(
    float* Q, const float* q_clip,
    float* K,
    float* V,
    float* R, const float* r_clip,
    int n_head, int n_head_kv, int head_dim, int d_rel,
    float qk_clip_thresh, float rel_clip_thresh,
    float rms_eps, float tau,
    const float* d_tau_ptr,
    cudaStream_t stream);

void ink_cuda_rel_table(
    const float* rel_bias,
    const float* R,
    float* rel_table,
    int n_head, int d_rel, int rel_extent,
    cudaStream_t stream);

// Fused KV quantize + store + optional INT8 Quest min/max summary + optional pos_slot write
// Optional `save_act_min_dst` / `save_act_max_dst` snapshot the post-token FP32 extrema for R8 exact Quest rollback!
void ink_cuda_kv_store(
    const float* K,
    const float* V,
    void* k_dst_token,
    void* v_dst_token,
    float* active_k_min,     // [n_head_kv * head_dim] FP32 active-page accumulator (optional)
    float* active_k_max,     // [n_head_kv * head_dim]
    int8_t* page_k_min_q8,   // [n_head_kv * head_dim] INT8 compressed Quest summary (optional)
    int8_t* page_k_max_q8,   // [n_head_kv * head_dim]
    float* page_k_scale,     // [n_head_kv] per-head INT8 scale
    bool is_first_in_page,
    int32_t* pos_slot_dst,   // optional: writes pos_val into *pos_slot_dst directly on GPU!
    int32_t  pos_val,
    int n_head_kv, int head_dim,
    KVQuantMode kv_mode,
    cudaStream_t stream,
    float* save_act_min_dst = nullptr,
    float* save_act_max_dst = nullptr);

// Re-quantize restored FP32 active page extrema into INT8 Quest summary (R8 speculative rollback!)
void ink_cuda_quest_quantize_summary(
    const float* active_k_min, // [n_glob_layers, n_head_kv * head_dim]
    const float* active_k_max, // [n_glob_layers, n_head_kv * head_dim]
    int8_t* k_mins_q8,         // [n_glob_layers, max_pages_per_layer, n_head_kv * head_dim]
    int8_t* k_maxs_q8,         // [n_glob_layers, max_pages_per_layer, n_head_kv * head_dim]
    float*  k_scales,          // [n_glob_layers, max_pages_per_layer, n_head_kv]
    int n_glob_layers,
    int max_pages_per_layer,
    int page_idx,
    int n_head_kv,
    int head_dim,
    cudaStream_t stream);

void ink_cuda_attn_swa(
    const float* Q,
    const void* K_ring,
    const void* V_ring,
    const int32_t* pos_ring,
    const float* rel_table,
    const float* sinks,
    float* split_out,
    float* split_ml,
    float* out,
    int n_head, int n_head_kv, int head_dim,
    int swa_len,
    int pos_q,
    int rel_extent,
    KVQuantMode kv_mode,
    cudaStream_t stream);

void ink_cuda_attn_global_paged(
    const float* Q,
    const uint64_t* page_k_ptrs,
    const uint64_t* page_v_ptrs,
    const int32_t* page_pos0,
    const int32_t* page_lens,
    int n_pages,
    const float* rel_table,
    const float* sinks,
    float* split_out,
    float* split_ml,
    float* out,
    int n_head, int n_head_kv, int head_dim,
    int pos_q,
    int rel_extent,
    KVQuantMode kv_mode,
    cudaStream_t stream);

void ink_cuda_quest_score_pages(
    const float* Q,
    const int8_t* k_mins_q8,
    const int8_t* k_maxs_q8,
    const float* k_scales,
    float* page_scores,
    int n_total_pages,
    int n_head, int n_head_kv, int head_dim,
    cudaStream_t stream);

// Exact O(N) GPU Radix-Select Top-K for Quest pages (returns K winning page IDs -> 512 B D2H instead of 32 KB!)
void ink_cuda_quest_topk(
    const float* page_scores,
    int hist_start,
    int hist_end,
    int K,
    int32_t* out_ids,
    cudaStream_t stream);

// High-speed multi-block MoE Router (uses all 24 SMs!)
void ink_cuda_moe_router(
    const float* gate_inp,
    const float* bias,
    const float* x,
    float* logits_scratch,   // [n_expert]
    int32_t* top_ids,
    float* top_weights,
    int n_expert, int n_embd, int k,
    float weights_scale,
    cudaStream_t stream);

// Fused Multi-Expert GateUp + SITU (host-resolved GPU hit batch)
void ink_cuda_moe_gate_up_situ(
    uint32_t gate_up_type,
    GPUExpertBatch batch,
    const float* x,
    float* h_out,
    int n_embd, int n_ff,
    float situ_beta, float situ_linear_beta,
    cudaStream_t stream);

// Fused Multi-Expert Down + Weighted Accumulation (host-resolved GPU hit batch)
void ink_cuda_moe_down_reduce(
    uint32_t down_type,
    GPUExpertBatch batch,
    const float* h_in,
    float* out,
    int n_embd, int n_ff,
    cudaStream_t stream);

// ZERO-SYNC GPU-Indirect Multi-Expert GateUp + SITU & Down + Reduce for 100%-resident layers!
void ink_cuda_moe_pinned_indirect(
    uint32_t gate_up_type,
    uint32_t down_type,
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_top_ids,
    const float* d_top_weights,
    int k,
    const float* x,
    float* h_scratch,        // [k * n_ff]
    float* out,              // [n_embd]
    int n_embd, int n_ff,
    float situ_beta, float situ_linear_beta,
    cudaStream_t stream);

// GPU-First Hybrid MoE Split: classifies top-k into GPU hits vs CPU misses in <1.5us on GPU,
// writes zero-copy mapped MoESplitPacket + copies x->h_ffn_in ONLY if n_cpu > 0!
void ink_cuda_moe_split_hits(
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_top_ids,
    const float* d_top_weights,
    int k,
    const float* d_x,
    int n_embd,
    int32_t* d_n_gpu,
    int32_t* d_gpu_hit_ids,
    float* d_gpu_hit_weights,
    MoESplitPacket* d_mapped_pkt,
    float* d_mapped_ffn_in,
    cudaStream_t stream);

// GPU-First Dynamic Indirect MoE: launches IMMEDIATELY after ink_cuda_moe_split_hits with ZERO host sync!
void ink_cuda_moe_dynamic_indirect(
    uint32_t gate_up_type,
    uint32_t down_type,
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_n_gpu,
    const int32_t* d_gpu_hit_ids,
    const float* d_gpu_hit_weights,
    int max_k,
    const float* x,
    float* h_scratch,        // [max_k * n_ff]
    float* out,              // [n_embd]
    int n_embd, int n_ff,
    float situ_beta, float situ_linear_beta,
    cudaStream_t stream);

void ink_cuda_logit_softcap_argmax(
    float* logits,
    int n_unpadded,
    float softcap,
    int32_t* d_argmax,
    cudaStream_t stream);

// =========================================================================
// Stage 4 & Stage 5: Batched GEMM Prefill (B <= 32) & Speculative Verification (B <= 6)
// =========================================================================
void ink_cuda_gemm_batch(
    uint32_t type,
    const void* W,
    const float* X,  // [B, K]
    float* Y,        // [B, M]
    int B, int M, int K,
    cudaStream_t stream);

void ink_cuda_rms_norm_batch(
    const float* x, const float* w, float* out,
    int B, int n, float eps, cudaStream_t stream);

void ink_cuda_attn_res_mix_batch(
    BlockPtrList states, // each ptr is [B, n_embd]
    const float* query,  // [n_queries, n_embd]
    float* out,          // [n_queries, B, n_embd] (query-major so inputs[j] is contiguous [B, D]!)
    int B, int n_embd, int n_queries,
    cudaStream_t stream);

void ink_cuda_loop_control_batch(
    const float* cur,    // [B, n_embd]
    const float* msg,    // [B, n_embd]
    uint32_t w_type,
    const void* w_down,
    const void* w_gain,
    const void* w_forget,
    float* z_scratch,    // [B, rank]
    float* out,          // [B, n_embd]
    int B, int n_embd, int rank,
    cudaStream_t stream);

// 1-launch sequential ShortConv across B tokens in registers (+ optional per-token spec rollback states!)
void ink_cuda_shortconv_batch(
    const float* x_in,   // [B, n_channels]
    float* state,        // [n_channels * 3]
    const float* weight, // [n_channels * 4]
    const float* dynamic,// optional [B, n_heads * 4]
    float* out,          // [B, n_channels]
    int B, int n_channels, int n_heads,
    float* spec_states,  // optional: writes post-token-t state to spec_states + t * spec_stride
    size_t spec_stride,
    cudaStream_t stream);

void ink_cuda_shortconv_add_batch(
    const float* x_in,   // [B, n_channels]
    float* state,        // [n_channels * 3]
    const float* weight, // [n_channels * 4]
    const float* res,    // [B, n_channels]
    float* out,          // [B, n_channels]
    int B, int n_channels,
    float* spec_states,
    size_t spec_stride,
    cudaStream_t stream);

void ink_cuda_qkv_post_batch(
    float* Q, const float* q_clip,
    float* K, float* V,
    float* R, const float* r_clip,
    int B, int n_head, int n_head_kv, int head_dim, int d_rel,
    float qk_clip_thresh, float rel_clip_thresh, float rms_eps,
    bool is_swa, int pos0, int log_scale_floor, float log_scale_alpha,
    cudaStream_t stream);

void ink_cuda_rel_table_batch(
    const float* rel_bias, const float* R, float* rel_table,
    int B, int n_head, int d_rel, int rel_extent, cudaStream_t stream);

void ink_cuda_kv_store_swa_batch(
    const float* K, const float* V,
    void* k_ring, void* v_ring, int32_t* pos_ring,
    int swa_window, int pos0, int B,
    int n_head_kv, int head_dim, KVQuantMode kv_mode, cudaStream_t stream);

void ink_cuda_swa_spec_backup_restore(
    void* d_swa_k, void* d_swa_v, int32_t* d_swa_pos,
    void* d_bak_k, void* d_bak_v, int32_t* d_bak_pos,
    int n_swa_layers, int swa_window, int pos0, int t_start, int t_end, int max_bak_slots,
    size_t tok_k_bytes, bool is_restore, cudaStream_t stream);

void ink_cuda_attn_swa_batch(
    const float* Q, const void* K_ring, const void* V_ring, const int32_t* pos_ring,
    const float* rel_table, const float* sinks,
    float* split_out, float* split_ml, float* out,
    int B, int n_head, int n_head_kv, int head_dim,
    int swa_len, int pos0, int rel_extent, KVQuantMode kv_mode, cudaStream_t stream);

void ink_cuda_attn_global_paged_batch(
    const float* Q,
    const uint64_t* page_k_ptrs, const uint64_t* page_v_ptrs,
    const int32_t* page_pos0, const int32_t* page_lens, int n_pages,
    const float* rel_table, const float* sinks,
    float* split_out, float* split_ml, float* out,
    int B, int n_head, int n_head_kv, int head_dim,
    int pos0, int rel_extent, KVQuantMode kv_mode, cudaStream_t stream);

// R4 (SR3-02): Chunked online-softmax exact global attention (single-sink finalize, zero frame reuse!)
void ink_cuda_attn_global_paged_chunk(
    const float* Q,
    const uint64_t* page_k_ptrs, const uint64_t* page_v_ptrs,
    const int32_t* page_pos0, const int32_t* page_lens, int n_pages,
    const float* rel_table,
    float* split_out, float* split_ml,
    float* acc_out, float* acc_ml,
    int B, int n_head, int n_head_kv, int head_dim,
    int pos0, int rel_extent, KVQuantMode kv_mode,
    bool is_first_chunk, cudaStream_t stream);

void ink_cuda_attn_global_finalize(
    const float* acc_out, const float* acc_ml,
    const float* sinks, float* out,
    int B, int n_head, int head_dim, cudaStream_t stream);

// R14 (SR1-03): Bit-exact GPU superset candidate pre-filter for sampling (eliminates 800 KB D2H copy!)
constexpr int INK_SAMPLE_INLINE_CAP   = 512;
constexpr int INK_SAMPLE_OVERFLOW_CAP = 8192;

struct alignas(64) SamplePrefilterPacket {
    int32_t count;
    int32_t overflow;
    float   raw_max;
    int32_t _pad;
    int32_t ids[INK_SAMPLE_INLINE_CAP];
    float   vals[INK_SAMPLE_INLINE_CAP];
};

void ink_cuda_sample_prefilter(
    const float* logits,
    int n_vocab,
    int32_t argmax_val,
    float inv_t,
    float log_min_p,
    SamplePrefilterPacket* d_pkt,
    int32_t* d_ext_ids,
    float* d_ext_vals,
    cudaStream_t stream);

void ink_cuda_moe_router_batch(
    const float* gate_inp, const float* bias, const float* x,
    float* logits_scratch, int32_t* top_ids, float* top_weights,
    int B, int n_expert, int n_embd, int k, float weights_scale,
    cudaStream_t stream);

void ink_cuda_moe_grouped_gpu(
    uint32_t gate_up_type, uint32_t down_type,
    const PinnedLayerDeviceTable& tbl,
    const int32_t* d_grp_exp_ids,   // [n_groups]
    const int32_t* d_grp_tok_off,   // [n_groups + 1]
    const int32_t* d_grp_tok_ids,   // [total_pairs]
    const float*   d_grp_tok_w,     // [total_pairs]
    int n_groups, int total_pairs, int B,
    const float* d_x_batch,         // [B, n_embd]
    float* d_h_scratch,             // [total_pairs, n_ff]
    float* d_out_batch,             // [B, n_embd]
    int n_embd, int n_ff,
    float situ_beta, float situ_linear_beta,
    bool zero_out,
    bool preconvert_x,
    cudaStream_t stream);

void ink_cuda_logit_softcap_argmax_batch(
    float* logits,       // [B, n_unpadded]
    int B, int n_unpadded, float softcap,
    int32_t* d_argmax,   // [B]
    cudaStream_t stream);

// =========================================================================
// InkEngine v3.2: Streamed Multimodal Vision Kernels (inkling2v ViT + Projector)
// =========================================================================
void ink_cuda_vision_layernorm(
    const float* x, const float* w, const float* b, float* out,
    int N, int D, float eps, cudaStream_t stream);

// WMMA FP16->FP32 Tensor Core GEMM for vision weights (F16 / BF16 / F32) + optional bias.
// Supports arbitrary N (internally chunked via g_batch_f16_buf -> 0 extra VRAM) and custom ld_out >= M.
void ink_cuda_vision_gemm(
    uint32_t w_type, const void* W, const float* bias,
    const float* X, float* Y,
    int N, int M, int K, int ld_out,
    cudaStream_t stream);

// 2D RoPE on Q or K [N, n_head * d_head]: dims [0..d/2) use pos_w, [d/2..d) use pos_h (adjacent pairs)
void ink_cuda_vision_rope_2d(
    float* x, const int32_t* pos_w, const int32_t* pos_h,
    int N, int n_head, int d_head, float theta,
    cudaStream_t stream);

// Zero-VRAM-matrix Bidirectional Multi-Head FlashAttention (d_head == 64)
void ink_cuda_vision_flash_attn(
    const float* Q, const float* K, const float* V, float* out,
    int N, int n_head, int d_head, float scale,
    cudaStream_t stream);

void ink_cuda_vision_gelu_erf(float* x, int total_elems, cudaStream_t stream);

void ink_cuda_vision_layerscale_add(
    float* residual, const float* delta, const float* ls_weight,
    int N, int D, cudaStream_t stream);
