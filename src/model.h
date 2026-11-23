#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include "gguf.h"
#include "quant_tables.h"
#include "cuda_kernels.cuh"
#include "cpu_kernels.h"
#include "expert_cache.h"
#include "paged_kv.h"
#include "tokenizer.h"
#include "moe_predictor.h"
#include "vision.h"

struct MultimodalEmbeddings {
    // Negative synthetic token ID -> raw [n_embd] FP32 vision embedding (bypasses tok_norm!)
    std::unordered_map<int32_t, const float*> id_to_embd;
};

struct EngineConfig {
    std::string model_path;
    int max_ctx = 65536;
    KVQuantMode kv_quant = KV_QUANT_Q4_0;
    KVAttentionMode kv_mode = KV_ATTN_AUTO;
    int quest_top_pages = 128; // 128 pages = 8192 active history tokens + 1024 recent + 128 sink
    ExpertCachePolicy cache_policy = CACHE_LOOP_LFU;
    size_t vram_kv_mb = 0;     // 0 = auto
    size_t vram_exp_mb = 0;    // 0 = auto (use all remaining free VRAM minus headroom)
    size_t ram_exp_mb = 0;     // 0 = auto
    size_t ram_kv_mb = 1024;   // Tier-1 RAM KV spillover pool when ctx > VRAM KV
    int n_threads = 10;
    bool preload_ram = true;   // sequential preload of non-VRAM experts into RAM
    bool async_ram_preload = false; // background thread RAM preload (~14s boot time in server mode!)
    bool fp16_loop = true;     // store loop controller weights in FP16 (saves 42 MiB & 2x faster)
    std::string kv_swap_dir;   // defaults to %TEMP% (C: NVMe SSD!)
    bool use_predictor = true; // auto-load Lanilen/Inkling-2-Mini-Expert-Predictor (.moepred) if found
    std::string predictor_path;
    std::string predictor_start_path;
    float predictor_answer_gain = 8.0f;
    bool use_cuda_graphs = true;    // CUDA Graphs for all 48 layer preludes/postludes + LM Head!
    bool use_batched_prefill = true; // WMMA Tensor-Core + PCIe-Streamed Batched Prefill (B=128!)
    bool use_speculative = true;    // Zero-VRAM lossless O(1) Suffix-Hash speculative decoding (K <= 6)!
    int  self_draft_k = 0;          // Opt-in 1-Loop (24L) GPU-Only lossless self-draft (--self-draft-k 2|3, up to 37.9 tok/s on code!)
    bool use_disk_kv_cache = true;  // Persistent NVMe OpenCode KV prefix cache (<100 ms cold Turn 1!)
    std::string kv_cache_dir;       // defaults to %LOCALAPPDATA%\InkEngine\kv_cache
    bool use_vision = true;         // Auto-load mmproj-F16.gguf (0 MiB permanent VRAM/RAM, GPU-streamed!)
    std::string mmproj_path;        // Optional explicit mmproj GGUF path (auto-detected beside model GGUF)
};

struct SamplingParams {
    float temperature = 0.0f;  // 0.0 = greedy argmax (CLI); HTTP server defaults to official 1.0
    int   top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    int   max_tokens = 256;
    uint64_t seed = 42;
};

struct InkHParams {
    int n_layer_logical = 48;
    int n_layer_phys = 24;
    int n_embd = 2304;
    int n_ff_exp = 1152;
    int n_head = 24;
    int n_head_kv = 8;
    int head_dim = 128;
    int n_expert = 128;
    int n_expert_used = 8;
    int vocab_padded = 201024;
    int vocab_unpadded = 200058;
    int swa_window = 1024;
    int d_rel = 16;
    int rel_extent = 1024;
    int sconv_k = 4;
    int attn_res_block_size = 6;
    int core_start = 6;
    int core_end = 18;
    int num_loops = 3;
    int loop_gate_rank = 64;
    int log_scale_floor = 8192;
    float log_scale_alpha = 0.1f;
    float rms_eps = 1e-6f;
    float qk_clip = 22.6274166f;
    float rel_clip = 8.0f;
    float expert_weights_scale = 8.0f;
    float situ_beta = 4.0f;
    float situ_linear_beta = 25.0f;
    float final_logit_softcap = 30.0f;
    std::vector<uint8_t> is_swa; // [48]

    int loop_layer(int il) const {
        if (num_loops < 2 || il < core_start) return il;
        int n_core = core_end - core_start;
        if (il < core_start + num_loops * n_core) {
            return core_start + (il - core_start) % n_core;
        }
        return il - (num_loops - 1) * n_core;
    }
};

struct GPUTensor {
    void*    ptr = nullptr;
    uint32_t type = 0;
    int64_t  ne[4] = {1, 1, 1, 1};
    size_t   nbytes = 0;
};

struct PhysLayerWeights {
    bool is_swa = true;
    bool is_core = false;

    GPUTensor wq, wk, wv, wo, wqkv_gate;
    GPUTensor attn_q_clip, attn_sink_proj;
    GPUTensor attn_rel, attn_rel_bias, attn_rel_clip;
    GPUTensor attn_k_conv, attn_v_conv, attn_post_conv, ffn_post_conv;

    GPUTensor attn_norm;
    GPUTensor attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm;
    GPUTensor attn_k_conv_proj, attn_v_conv_proj;

    GPUTensor attn_res_query, ffn_res_query;
    GPUTensor ffn_norm, ffn_gate_inp, ffn_exp_probs_b;

    // Loop message controllers (only for ip in [core_start, core_end))
    // m = 0: attn Q, 1: attn K, 2: attn V, 3: mlp
    GPUTensor loop_down[4];
    GPUTensor loop_gain[4];
    GPUTensor loop_forget[4];
};

struct TokenProfile {
    double embed_ms = 0;
    double mix_loop_ms = 0;
    double attn_ms = 0;
    double router_ms = 0;
    double moe_core_ms = 0;
    double moe_outer_ms = 0;
    double lm_head_ms = 0;
    double total_ms = 0;
    int    tokens = 0;
};

class InklingEngine {
public:
    InklingEngine() = default;
    ~InklingEngine();
    InklingEngine(const InklingEngine&) = delete;
    InklingEngine& operator=(const InklingEngine&) = delete;

    bool load(const EngineConfig& cfg);
    void unload();
    void reset_context();
    void wait_for_ram_preload() { experts_.wait_for_ram_preload(); }

    // Fast synthetic KV population for long-context (32K-512K) decode & NVMe SSD paging benchmarks
    bool benchmark_populate_kv(int target_pos);

    // Forward a single token at position `pos`. If `need_logits` is true, computes logits on GPU.
    // Returns greedy argmax token when need_logits == true (or -1 when need_logits == false).
    int32_t forward_token(int32_t token_id, int pos, bool need_logits, bool enable_profile = false,
                          bool is_answer = false, bool capture_pred_hidden = false);

    // Ultra-fast 1-Loop (24L) GPU-Only Self-Draft forward pass (zero CPU syncs, ~7-8 ms/tok, 100% lossless when verified!)
    int32_t forward_token_1loop_draft(int32_t token_id, int pos);

    // Batched forward pass for B tokens [pos0 .. pos0 + B - 1] (B <= BATCH_MAX = 128):
    // Powers both WMMA Tensor-Core PCIe-Streamed Prefill AND Zero-VRAM Lossless Speculative Verification!
    static constexpr int BATCH_MAX = 128;
    static constexpr int STAGE_BUFS = 2;
    static constexpr int STAGE_EXPS = 4;
    static constexpr size_t STAGE_SLOT_BYTES = 3670016; // 3.50 MiB (>= max expert combined_bytes 3.36 MiB)
    int32_t forward_batch(
        const int32_t* tokens, int B, int pos0,
        bool need_last_logits, bool need_all_logits,
        bool is_answer, bool capture_last_pred_hidden,
        bool save_spec_states,
        int32_t* out_argmax_batch = nullptr,
        const float* raw_embd = nullptr);

    // Multimodal Vision helpers (zero permanent VRAM/RAM — streams mmproj through prefill staging!)
    bool has_vision() const { return vision_.loaded(); }
    bool encode_image_input(const std::string& url_or_path, EncodedImage& out);
    static void expand_multimodal_tokens(
        const std::vector<int32_t>& raw_tokens,
        int32_t image_placeholder_id,
        const std::vector<EncodedImage>& images,
        std::vector<int32_t>& out_tokens,
        MultimodalEmbeddings& out_mm);

    // Run learned expert predictor hooks (turn-start mean embedding & main hidden-state update)
    void run_start_predictor(const int32_t* turn_tokens, int n_tokens);
    void run_main_predictor(int64_t answer_pos, bool is_answer_start);
    bool predictor_active() const { return pred_main_.loaded() || pred_start_.loaded(); }

    // Prefill a token sequence (uses Batched GEMM Prefill B=32 when enabled!)
    int32_t prefill(const std::vector<int32_t>& tokens, int start_pos = 0, double* out_ms = nullptr);

    // Sample next token from current GPU logits buffer (or from a specific logits buffer)
    int32_t sample_next(const SamplingParams& sp, uint64_t& rng_state);
    int32_t sample_from_logits(const float* d_logits_ptr, int32_t argmax_val, const SamplingParams& sp,
                               uint64_t& rng_state, int32_t draft_tok = -1, bool* out_accepted = nullptr);

    // Generate tokens with streaming callback: cb(token_id, piece_text) -> return false to stop early
    struct GenResult {
        std::vector<int32_t> tokens;
        std::string text;
        int prompt_tokens = 0;
        int cached_tokens = 0;
        int gen_tokens = 0;
        int spec_drafted = 0;
        int spec_accepted = 0;
        int spec_steps = 0;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
    };
    GenResult generate(
        const std::vector<int32_t>& prompt_tokens,
        const SamplingParams& sp,
        const std::function<bool(int32_t, const std::string&)>& on_token = nullptr,
        bool enable_profile = false,
        bool reuse_kv_prefix = false,
        int static_prefix_tokens = 0,
        const std::function<bool(int, int)>& on_prefill = nullptr,
        int tools_prefix_tokens = 0,
        int full_sys_prefix_tokens = 0,
        const MultimodalEmbeddings* mm_embd = nullptr);

    // Copy current logits to host (for verification / top-K inspection)
    void copy_logits_to_host(std::vector<float>& dst) const;

    // Empirical probe: measures exact 1-Loop Early-Exit (24L) vs Full 3-Loop (48L) token agreement rate!
    void probe_early_exit(const std::string& prompt_text, int n_tokens = 48);

    const InkHParams& hparams() const { return hp_; }
    const InkTokenizer& tokenizer() const { return tok_; }
    ExpertCacheStats expert_stats() const { return experts_.stats(); }
    PagedKVStats kv_stats() const { return kv_.stats(); }
    TokenProfile profile() const { return prof_; }
    void reset_profile() { prof_ = TokenProfile{}; experts_.reset_stats(); kv_.reset_counters(); }

private:
    GPUTensor upload_tensor_gpu(const std::string& name, bool convert_f32_to_f16 = false);
    GPUTensor upload_tensor_gpu_opt(const std::string& name);
    void capture_cuda_graphs();
    void destroy_cuda_graphs();
    void emit_layer_pre(int il, int& n_blocks, int& prelude_n_blocks, int& n_bank, float tau_global);
    void emit_layer_post(int il, int& n_blocks, int n_bank);
    void emit_lm_head(int n_blocks);

    EngineConfig cfg_;
    GGUFFile gguf_;
    InkHParams hp_;
    InkTokenizer tok_;
    std::unique_ptr<CPUThreadPool> cpu_pool_;
    ExpertCache experts_;
    HybridPagedKV kv_;
    VisionEncoder vision_;

    cudaStream_t stream_ = nullptr;

    // Host-resident token embedding table (248 MiB Q4_K -> only 1.3 KB/token read!)
    std::vector<uint8_t> h_tok_embd_;
    uint32_t tok_embd_type_ = GGML_TYPE_Q4_K;
    size_t   tok_embd_row_bytes_ = 0;

    // Dense GPU weights (all pinned in VRAM ~999 MiB)
    std::vector<void*> gpu_allocs_;
    GPUTensor tok_norm_, output_, output_norm_, output_res_query_;
    GPUTensor loop_res_query_, loop_kv_res_query_;
    std::vector<PhysLayerWeights> layers_;

    // GPU Scratch activation buffers (~24 MiB total — vs llama.cpp's 3,475 MiB!)
    float* d_cur_ = nullptr;
    float* d_inpSA_ = nullptr;
    float* d_ffn_inp_ = nullptr;
    float* d_blocks_[8] = {};
    float* d_prelude_cur_ = nullptr;
    float* d_bank_[4] = {};
    float* d_bank_last_ = nullptr;
    float* d_mixed_ = nullptr;       // [3 * n_embd]
    float* d_msg_attn_ = nullptr;    // [2 * n_embd]
    float* d_msg_kv_ = nullptr;      // [2 * n_embd]
    float* d_ctrl_out_ = nullptr;    // [3 * n_embd]
    float* d_normed_[3] = {};        // [n_embd] each
    float* d_loop_z_ = nullptr;      // [loop_gate_rank]

    float* d_Q_ = nullptr;           // [3072]
    float* d_K_ = nullptr;           // [1024]
    float* d_K_conv_ = nullptr;      // [1024]
    float* d_V_ = nullptr;           // [1024]
    float* d_V_conv_ = nullptr;      // [1024]
    float* d_dyn_k_ = nullptr;       // [32]
    float* d_dyn_v_ = nullptr;       // [32]
    float* d_R_ = nullptr;           // [384]
    float* d_sinks_ = nullptr;       // [24]
    float* d_rel_table_ = nullptr;   // [24 * 1024] = 96 KB!
    float* d_attn_out_ = nullptr;    // [3072]
    float* d_attn_gate_ = nullptr;   // [3072]
    float* d_wo_out_ = nullptr;      // [2304]
    float* d_post_conv_ = nullptr;   // [2304]

    float* d_ffn_normed_ = nullptr;  // [2304]
    float* d_moe_h_ = nullptr;       // [8 * 1152]
    float* d_moe_out_ = nullptr;     // [2304]
    float* d_moe_cpu_out_ = nullptr; // [2304]
    float* d_router_logits_ = nullptr; // [128]
    int32_t* d_top_ids_ = nullptr;   // [8]
    float*   d_top_weights_ = nullptr; // [8]
    int32_t* d_n_gpu_ = nullptr;       // [1]
    int32_t* d_gpu_hit_ids_ = nullptr; // [8]
    float*   d_gpu_hit_weights_ = nullptr; // [8]
    float*   d_logits_ = nullptr;    // [vocab_padded]
    int32_t* d_argmax_ = nullptr;    // [1]

    // Pinned host staging & zero-copy mapped buffers for GPU-first zero-bubble MoE dispatch
    float*   h_embd_row_ = nullptr;    // [2304]
    int32_t* h_top_ids_ = nullptr;     // [8]
    float*   h_top_weights_ = nullptr; // [8]
    float*   h_ffn_in_ = nullptr;      // [2304] (cudaHostAllocMapped)
    float*   d_mapped_ffn_in_ = nullptr;
    MoESplitPacket* h_split_pkt_ = nullptr; // (cudaHostAllocMapped)
    MoESplitPacket* d_split_pkt_ = nullptr;
    int32_t* h_zero_sync_ids_ = nullptr; // [48 * 8] async D2H ids for 100%-VRAM dynamic layers
    float*   h_moe_cpu_out_ = nullptr; // [2304]
    int32_t* h_argmax_ = nullptr;      // [1]
    cudaEvent_t ev_split_ = nullptr;

    TokenProfile prof_{};
    std::vector<int32_t> kv_tokens_; // exact token history currently resident in kv_
    KVCheckpoint full_sys_ckpt_;     // tier-1 RAM checkpoint at end of ALL system msgs (~99% prompt hit!)
    KVCheckpoint prefix_ckpt_;       // tier-2 RAM checkpoint before dynamic <env> (~80% fallback!)
    KVCheckpoint tools_ckpt_;        // tier-3 RAM checkpoint at tool_declare boundary (survives agent switches!)
    TurnRewindSnapshot turn_snap_;   // ~49 MiB zero-Global-copy in-place rewind snapshot (~4 ms retry recovery!)
    uint64_t     model_sig_ = 0;     // 64-bit model+hparams fingerprint for persistent NVMe KV cache
    std::string  kv_cache_dir_;      // NVMe directory for persistent KV checkpoints

    // Native MoE Expert Predictor (Lanilen/Inkling-2-Mini-Expert-Predictor pred-v3 + start-v3)
    MoePredictorModel   pred_main_;
    MoePredictorModel   pred_start_;
    MoePredictorTracker pred_tracker_;
    float* h_pred_hidden_ = nullptr; // pinned [4 * n_embd] for async D2H of pass-0 layer 8 & 16 states
    int    pred_hidden_slots_ = 0;

    // CUDA Graphs for fixed layer preludes/postludes + LM Head (eliminates ~82% of WDDM launches!)
    float* d_tau_global_ = nullptr;
    float  last_tau_global_ = -1.0f;
    cudaGraphExec_t graph_pre_[48] = {};
    cudaGraphExec_t graph_post_[48] = {};
    cudaGraphExec_t graph_lm_head_ = nullptr;
    bool graphs_ready_ = false;

    // Stage 4 & Stage 5 Batched Prefill (B <= 32) + Speculative Verify (B <= 6) Scratch (~21 MiB VRAM)
    void alloc_batch_buffers();
    void free_batch_buffers();
    bool batch_ready_ = false;
    float* d_b_cur_ = nullptr;
    float* d_b_inpSA_ = nullptr;
    float* d_b_ffn_inp_ = nullptr;
    float* d_b_blocks_[8] = {};
    float* d_b_prelude_cur_ = nullptr;
    float* d_b_bank_[4] = {};
    float* d_b_bank_last_ = nullptr;
    float* d_b_mixed_ = nullptr;        // [3 * BATCH_MAX * D]
    float* d_b_msg_attn_ = nullptr;     // [2 * BATCH_MAX * D]
    float* d_b_msg_kv_ = nullptr;       // [2 * BATCH_MAX * D]
    float* d_b_ctrl_out_ = nullptr;     // [3 * BATCH_MAX * D]
    float* d_b_normed_[3] = {};         // [BATCH_MAX * D] each
    float* d_b_loop_z_ = nullptr;       // [BATCH_MAX * rank]
    float* d_b_Q_ = nullptr;            // [BATCH_MAX * Qdim]
    float* d_b_K_ = nullptr;            // [BATCH_MAX * KVdim]
    float* d_b_K_conv_ = nullptr;       // [BATCH_MAX * KVdim]
    float* d_b_V_ = nullptr;            // [BATCH_MAX * KVdim]
    float* d_b_V_conv_ = nullptr;       // [BATCH_MAX * KVdim]
    float* d_b_dyn_k_ = nullptr;        // [BATCH_MAX * 32]
    float* d_b_dyn_v_ = nullptr;        // [BATCH_MAX * 32]
    float* d_b_R_ = nullptr;            // [BATCH_MAX * 384]
    float* d_b_sinks_ = nullptr;        // [BATCH_MAX * 24]
    float* d_b_rel_table_ = nullptr;    // [BATCH_MAX * 24 * 1024]
    float* d_b_attn_out_ = nullptr;     // [BATCH_MAX * Qdim]
    float* d_b_attn_gate_ = nullptr;    // [BATCH_MAX * Qdim]
    float* d_b_wo_out_ = nullptr;       // [BATCH_MAX * D]
    float* d_b_ffn_normed_ = nullptr;   // [BATCH_MAX * D]
    float* d_b_moe_h_ = nullptr;        // [BATCH_MAX * 8 * n_ff_exp]
    float* d_b_moe_out_ = nullptr;      // [BATCH_MAX * D]
    float* d_b_moe_cpu_out_ = nullptr;  // [BATCH_MAX * D]
    float* d_b_router_logits_ = nullptr;// [BATCH_MAX * 128]
    int32_t* d_b_top_ids_ = nullptr;    // [BATCH_MAX * 8]
    float*   d_b_top_weights_ = nullptr;// [BATCH_MAX * 8]
    int32_t* d_grp_exp_ids_ = nullptr;  // [128]
    int32_t* d_grp_tok_off_ = nullptr;  // [129]
    int32_t* d_grp_tok_ids_ = nullptr;  // [BATCH_MAX * 8]
    float*   d_grp_tok_w_ = nullptr;    // [BATCH_MAX * 8]
    float*   d_b_logits_ = nullptr;     // [SPEC_MAX_B * vocab_unpadded]
    int32_t* d_b_argmax_ = nullptr;     // [SPEC_MAX_B]

    // Pinned host staging for batched prefill / verify
    float*   h_b_embd_ = nullptr;       // [BATCH_MAX * D]
    float*   h_b_ffn_in_ = nullptr;     // [BATCH_MAX * D]
    float*   h_b_moe_cpu_out_ = nullptr;// [BATCH_MAX * D]
    int32_t* h_b_top_ids_ = nullptr;    // [BATCH_MAX * 8]
    float*   h_b_top_weights_ = nullptr;// [BATCH_MAX * 8]
    int32_t* h_grp_exp_ids_ = nullptr;  // [128]
    int32_t* h_grp_tok_off_ = nullptr;  // [129]
    int32_t* h_grp_tok_ids_ = nullptr;  // [BATCH_MAX * 8]
    float*   h_grp_tok_w_ = nullptr;    // [BATCH_MAX * 8]
    int32_t* h_b_argmax_ = nullptr;     // [SPEC_MAX_B]

    // Double-buffered PCIe cold-expert streaming ring for Tensor-Core Prefill (28 MiB VRAM + 28 MiB pinned RAM)
    uint8_t* d_prefill_stage_[STAGE_BUFS] = {};
    uint8_t* h_prefill_stage_[STAGE_BUFS] = {};
    PinnedLayerDeviceTable stage_tbl_[STAGE_BUFS] = {};
    uint64_t* h_stage_gu_ptrs_[STAGE_BUFS] = {};
    uint64_t* h_stage_dn_ptrs_[STAGE_BUFS] = {};
    float*    h_stage_gu_sc_[STAGE_BUFS] = {};
    float*    h_stage_dn_sc_[STAGE_BUFS] = {};
    int32_t*  d_stage_exp_ids_[STAGE_BUFS] = {};
    int32_t*  d_stage_tok_off_[STAGE_BUFS] = {};
    int32_t*  d_stage_tok_ids_[STAGE_BUFS] = {};
    float*    d_stage_tok_w_[STAGE_BUFS] = {};
    int32_t*  h_stage_tok_off_[STAGE_BUFS] = {};
    int32_t*  h_stage_tok_ids_[STAGE_BUFS] = {};
    float*    h_stage_tok_w_[STAGE_BUFS] = {};
    cudaEvent_t ev_stage_dma_[STAGE_BUFS] = {};
    cudaEvent_t ev_stage_gpu_[STAGE_BUFS] = {};
};
