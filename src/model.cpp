#include "model.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <immintrin.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}
}

InklingEngine::~InklingEngine() { unload(); }

void InklingEngine::destroy_cuda_graphs() {
    for (int i = 0; i < 48; ++i) {
        if (graph_pre_[i])  { cudaGraphExecDestroy(graph_pre_[i]);  graph_pre_[i] = nullptr; }
        if (graph_post_[i]) { cudaGraphExecDestroy(graph_post_[i]); graph_post_[i] = nullptr; }
    }
    if (graph_lm_head_) { cudaGraphExecDestroy(graph_lm_head_); graph_lm_head_ = nullptr; }
    graphs_ready_ = false;
}

void InklingEngine::unload() {
    destroy_cuda_graphs();
    free_batch_buffers();
    vision_.unload();
    experts_.shutdown();
    kv_.shutdown();
    for (void* p : gpu_allocs_) {
        if (p) cudaFree(p);
    }
    gpu_allocs_.clear();

    auto free_d = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto free_h = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };

    free_d(d_cur_); free_d(d_inpSA_); free_d(d_ffn_inp_);
    for (auto& p : d_blocks_) free_d(p);
    free_d(d_prelude_cur_);
    for (auto& p : d_bank_) free_d(p);
    free_d(d_bank_last_);
    free_d(d_mixed_); free_d(d_msg_attn_); free_d(d_msg_kv_); free_d(d_ctrl_out_);
    for (auto& p : d_normed_) free_d(p);
    free_d(d_loop_z_);
    free_d(d_Q_); free_d(d_K_); free_d(d_K_conv_); free_d(d_V_); free_d(d_V_conv_);
    free_d(d_dyn_k_); free_d(d_dyn_v_); free_d(d_R_); free_d(d_sinks_); free_d(d_rel_table_);
    free_d(d_attn_out_); free_d(d_attn_gate_); free_d(d_wo_out_); free_d(d_post_conv_);
    free_d(d_ffn_normed_); free_d(d_moe_h_); free_d(d_moe_out_); free_d(d_moe_cpu_out_);
    free_d(d_router_logits_); free_d(d_top_ids_); free_d(d_top_weights_);
    free_d(d_n_gpu_); free_d(d_gpu_hit_ids_); free_d(d_gpu_hit_weights_);
    free_d(d_logits_); free_d(d_argmax_); free_d(d_tau_global_);
    free_d(d_sample_pkt_); free_d(d_sample_ext_ids_); free_d(d_sample_ext_vals_);
    last_tau_global_ = -1.0f;

    free_h(h_embd_row_); free_h(h_top_ids_); free_h(h_top_weights_);
    free_h(h_ffn_in_); free_h(h_split_pkt_); free_h(h_zero_sync_ids_);
    free_h(h_moe_cpu_out_); free_h(h_argmax_); free_h(h_pred_hidden_);
    free_h(h_sample_pkt_); free_h(h_sample_ext_ids_); free_h(h_sample_ext_vals_);
    h_samp_cand_.clear();
    batch_pred_ids_.clear();
    batch_pred_B_ = 0;
    pred_hidden_slots_ = 0;
    d_mapped_ffn_in_ = nullptr;
    d_split_pkt_ = nullptr;
    io_error_ = false;
    last_error_.clear();

    if (ev_split_) { cudaEventDestroy(ev_split_); ev_split_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    kv_tokens_.clear();
    full_sys_ckpt_ = KVCheckpoint{};
    prefix_ckpt_ = KVCheckpoint{};
    tools_ckpt_ = KVCheckpoint{};
    turn_snap_ = TurnRewindSnapshot{};
    gguf_.close();
}

GPUTensor InklingEngine::upload_tensor_gpu(const std::string& name, bool convert_f32_to_f16) {
    const auto& ti = gguf_.require_tensor(name);
    GPUTensor gt;
    gt.type = ti.type;
    for (int i = 0; i < 4; ++i) gt.ne[i] = (int64_t)ti.ne[i];

    std::vector<uint8_t> host_buf(ti.nbytes);
    if (!gguf_.pread(ti.abs_offset, host_buf.data(), ti.nbytes)) {
        throw std::runtime_error("Failed reading tensor from GGUF: " + name);
    }

    if (convert_f32_to_f16 && ti.type == GGML_TYPE_F32) {
        size_t nelem = ti.nbytes / 4;
        std::vector<uint16_t> f16_buf(nelem);
        const float* src = (const float*)host_buf.data();
        for (size_t i = 0; i < nelem; ++i) f16_buf[i] = fp32_to_fp16_host(src[i]);
        gt.type = GGML_TYPE_F16;
        gt.nbytes = nelem * 2;
        INK_CUDA_CHECK(cudaMalloc(&gt.ptr, gt.nbytes));
        INK_CUDA_CHECK(cudaMemcpy(gt.ptr, f16_buf.data(), gt.nbytes, cudaMemcpyHostToDevice));
    } else {
        gt.nbytes = ti.nbytes;
        INK_CUDA_CHECK(cudaMalloc(&gt.ptr, gt.nbytes));
        INK_CUDA_CHECK(cudaMemcpy(gt.ptr, host_buf.data(), gt.nbytes, cudaMemcpyHostToDevice));
    }
    gpu_allocs_.push_back(gt.ptr);
    return gt;
}

GPUTensor InklingEngine::upload_tensor_gpu_opt(const std::string& name) {
    if (!gguf_.find_tensor(name)) return GPUTensor{};
    return upload_tensor_gpu(name, false);
}

bool InklingEngine::load(const EngineConfig& cfg) {
    unload();
    cfg_ = cfg;

    auto t_start = now_ms();
    std::printf("================================================================\n");
    std::printf("  InkEngine v4.0 - Custom Looped-MoE 3-Tier Serving Engine\n");
    std::printf("================================================================\n");

    if (!gguf_.open(cfg.model_path)) return false;

    std::string arch = gguf_.get_str("general.architecture");
    if (arch != "inkling2") {
        std::fprintf(stderr, "[Engine] Expected architecture 'inkling2', got '%s'\n", arch.c_str());
        return false;
    }

    // Read hyperparameters directly from GGUF metadata
    hp_.n_layer_logical      = (int)gguf_.get_u32("inkling2.block_count", 48);
    hp_.n_embd               = (int)gguf_.get_u32("inkling2.embedding_length", 2304);
    hp_.n_ff_exp             = (int)gguf_.get_u32("inkling2.expert_feed_forward_length", 1152);
    hp_.n_expert             = (int)gguf_.get_u32("inkling2.expert_count", 128);
    hp_.n_expert_used        = (int)gguf_.get_u32("inkling2.expert_used_count", 8);
    hp_.head_dim             = (int)gguf_.get_u32("inkling2.attention.key_length", 128);
    hp_.vocab_padded         = (int)gguf_.get_u32("inkling2.vocab_size", 201024);
    hp_.vocab_unpadded       = (int)gguf_.get_u32("inkling2.vocab_size_unpadded", 200058);
    hp_.swa_window           = (int)gguf_.get_u32("inkling2.attention.sliding_window", 1024);
    hp_.d_rel                = (int)gguf_.get_u32("inkling2.attention.relative_length", 16);
    hp_.rel_extent           = (int)gguf_.get_u32("inkling2.attention.relative_extent", 1024);
    hp_.sconv_k              = (int)gguf_.get_u32("inkling2.shortconv.l_cache", 4);
    hp_.attn_res_block_size  = (int)gguf_.get_u32("inkling2.attn_res.block_size", 6);
    hp_.num_loops            = (int)gguf_.get_u32("inkling2.num_loops", 3);
    hp_.core_start           = (int)gguf_.get_u32("inkling2.loop.core_start", 6);
    hp_.core_end             = (int)gguf_.get_u32("inkling2.loop.core_end", 18);
    hp_.loop_gate_rank       = (int)gguf_.get_u32("inkling2.loop.gate_rank", 64);
    hp_.log_scale_floor      = (int)gguf_.get_u32("inkling2.attention.log_scale_floor", 8192);
    hp_.log_scale_alpha      = gguf_.get_f32("inkling2.attention.log_scale_alpha", 0.1f);
    hp_.rms_eps              = gguf_.get_f32("inkling2.attention.layer_norm_rms_epsilon", 1e-6f);
    hp_.qk_clip              = gguf_.get_f32("inkling2.attention.qk_clip", 22.6274166f);
    hp_.rel_clip             = gguf_.get_f32("inkling2.attention.relative_clip", 8.0f);
    hp_.expert_weights_scale = gguf_.get_f32("inkling2.expert_weights_scale", 8.0f);
    hp_.situ_beta            = gguf_.get_f32("inkling2.activation.situ_beta", 4.0f);
    hp_.situ_linear_beta     = gguf_.get_f32("inkling2.activation.situ_linear_beta", 25.0f);
    hp_.final_logit_softcap  = gguf_.get_f32("inkling2.final_logit_softcapping", 30.0f);

    const auto& hc = gguf_.get_i32_arr("inkling2.attention.head_count");
    const auto& hk = gguf_.get_i32_arr("inkling2.attention.head_count_kv");
    if (!hc.empty()) hp_.n_head = hc[0];
    if (!hk.empty()) hp_.n_head_kv = hk[0];

    hp_.is_swa = gguf_.get_bool_arr("inkling2.attention.sliding_window_pattern");
    int n_extra = (hp_.num_loops > 1) ? (hp_.num_loops - 1) * (hp_.core_end - hp_.core_start) : 0;
    hp_.n_layer_phys = hp_.n_layer_logical - n_extra;

    std::printf("[Model] %s | %d physical -> %d logical layers (core [%d..%d) x%d) | d=%d | MoE %d/%d (ff=%d) | SWA=%d\n",
                gguf_.get_str("general.name", "Inkling-2-Mini").c_str(),
                hp_.n_layer_phys, hp_.n_layer_logical, hp_.core_start, hp_.core_end, hp_.num_loops,
                hp_.n_embd, hp_.n_expert_used, hp_.n_expert, hp_.n_ff_exp, hp_.swa_window);

    // Load tokenizer
    if (!tok_.load(gguf_)) return false;

    // Init CUDA device & constant tables (Spin schedule eliminates WDDM OS-sleep sync jitter!)
    int dev = 0;
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    cudaGetLastError(); // ignore cudaErrorSetOnActiveProcess if context already active
    cudaDeviceProp prop{};
    INK_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t vram_free0 = 0, vram_tot = 0;
    INK_CUDA_CHECK(cudaMemGetInfo(&vram_free0, &vram_tot));
    std::printf("[GPU] %s (sm_%d%d, %d SMs) | VRAM: %.0f MiB free / %.0f MiB total\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                vram_free0 / (1024.0 * 1024.0), vram_tot / (1024.0 * 1024.0));

    ink_cuda_init_tables();
    INK_CUDA_CHECK(cudaStreamCreate(&stream_));
    cpu_pool_ = std::make_unique<CPUThreadPool>(cfg.n_threads);

    // 1. Load token_embd.weight into Host RAM (248 MiB -> zero VRAM wasted!)
    {
        const auto& te = gguf_.require_tensor("token_embd.weight");
        tok_embd_type_ = te.type;
        tok_embd_row_bytes_ = (size_t)(hp_.n_embd / ggml_type_block_size(te.type)) * ggml_type_byte_size(te.type);
        h_tok_embd_.resize(te.nbytes);
        if (!gguf_.pread(te.abs_offset, h_tok_embd_.data(), te.nbytes)) return false;
    }

    // 2. Upload ALL dense non-expert weights to VRAM (~999 MiB total!)
    std::printf("[Model] Uploading all dense weights (attention, loop controllers, routers, lm_head) to VRAM...\n");
    std::fflush(stdout);
    size_t dense_bytes = 0;
    auto track = [&](const GPUTensor& gt) { dense_bytes += gt.nbytes; return gt; };

    tok_norm_          = track(upload_tensor_gpu("token_embd_norm.weight"));
    output_            = track(upload_tensor_gpu("output.weight"));
    output_norm_       = track(upload_tensor_gpu("output_norm.weight"));
    output_res_query_  = track(upload_tensor_gpu("output_res_query.weight"));
    loop_res_query_    = track(upload_tensor_gpu("loop_res_query.weight"));
    loop_kv_res_query_ = track(upload_tensor_gpu("loop_kv_res_query.weight"));

    layers_.resize(hp_.n_layer_phys);
    for (int ip = 0; ip < hp_.n_layer_phys; ++ip) {
        auto& L = layers_[ip];
        int iv = (ip < hp_.core_end) ? ip : (ip + n_extra);
        L.is_swa  = (hp_.is_swa[iv] != 0);
        L.is_core = (ip >= hp_.core_start && ip < hp_.core_end);

        auto tname = [&](const char* suffix) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "blk.%d.%s", ip, suffix);
            return std::string(buf);
        };

        L.wq             = track(upload_tensor_gpu(tname("attn_q.weight")));
        L.wk             = track(upload_tensor_gpu(tname("attn_k.weight")));
        L.wv             = track(upload_tensor_gpu(tname("attn_v.weight")));
        L.wo             = track(upload_tensor_gpu(tname("attn_output.weight")));
        L.wqkv_gate      = track(upload_tensor_gpu(tname("attn_gate.weight")));
        L.attn_q_clip    = track(upload_tensor_gpu(tname("attn_q_clip.weight")));
        L.attn_sink_proj = track(upload_tensor_gpu(tname("attn_sink_proj.weight")));
        L.attn_rel       = track(upload_tensor_gpu(tname("attn_rel.weight")));
        L.attn_rel_bias  = track(upload_tensor_gpu(tname("attn_rel_bias.weight")));
        L.attn_rel_clip  = track(upload_tensor_gpu(tname("attn_rel_clip.weight")));

        L.attn_k_conv    = track(upload_tensor_gpu(tname("attn_k_conv.weight")));
        L.attn_v_conv    = track(upload_tensor_gpu(tname("attn_v_conv.weight")));
        L.attn_post_conv = track(upload_tensor_gpu(tname("attn_post_conv.weight")));
        L.ffn_post_conv  = track(upload_tensor_gpu(tname("ffn_post_conv.weight")));

        if (L.is_swa) {
            L.attn_norm = track(upload_tensor_gpu(tname("attn_norm.weight")));
        } else {
            L.attn_q_inp_norm  = track(upload_tensor_gpu(tname("attn_q_inp_norm.weight")));
            L.attn_k_inp_norm  = track(upload_tensor_gpu(tname("attn_k_inp_norm.weight")));
            L.attn_v_inp_norm  = track(upload_tensor_gpu(tname("attn_v_inp_norm.weight")));
            L.attn_k_conv_proj = track(upload_tensor_gpu(tname("attn_k_conv_proj.weight")));
            L.attn_v_conv_proj = track(upload_tensor_gpu(tname("attn_v_conv_proj.weight")));
        }

        if (ip > 0) L.attn_res_query = track(upload_tensor_gpu(tname("attn_res_query.weight")));
        L.ffn_res_query   = track(upload_tensor_gpu(tname("ffn_res_query.weight")));
        L.ffn_norm        = track(upload_tensor_gpu(tname("ffn_norm.weight")));
        L.ffn_gate_inp    = track(upload_tensor_gpu(tname("ffn_gate_inp.weight")));
        L.ffn_exp_probs_b = track(upload_tensor_gpu_opt(tname("exp_probs_b.bias")));

        if (L.is_core) {
            static const char* mods[4] = {"attn", "k", "v", "mlp"};
            for (int m = 0; m < 4; ++m) {
                char b1[64], b2[64], b3[64];
                std::snprintf(b1, sizeof(b1), "loop_%s_down.weight",   mods[m]);
                std::snprintf(b2, sizeof(b2), "loop_%s_gain.weight",   mods[m]);
                std::snprintf(b3, sizeof(b3), "loop_%s_forget.weight", mods[m]);
                L.loop_down[m]   = track(upload_tensor_gpu(tname(b1), cfg.fp16_loop));
                L.loop_gain[m]   = track(upload_tensor_gpu(tname(b2), cfg.fp16_loop));
                L.loop_forget[m] = track(upload_tensor_gpu(tname(b3), cfg.fp16_loop));
            }
        }
    }
    std::printf("[Model] Dense weights pinned in VRAM: %.1f MiB (vs 3,475 MiB scratch bloat eliminated!)\n",
                dense_bytes / (1024.0 * 1024.0));

    // 3. Allocate tiny GPU activation scratch buffers (~2.5 MiB total!)
    auto alloc_f32 = [](float*& ptr, size_t n) { INK_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float))); };
    int D = hp_.n_embd;
    int Qdim = hp_.n_head * hp_.head_dim;
    int KVdim = hp_.n_head_kv * hp_.head_dim;

    alloc_f32(d_cur_, D); alloc_f32(d_inpSA_, D); alloc_f32(d_ffn_inp_, D);
    for (auto& p : d_blocks_) alloc_f32(p, D);
    alloc_f32(d_prelude_cur_, D);
    for (auto& p : d_bank_) alloc_f32(p, D);
    alloc_f32(d_bank_last_, D);
    alloc_f32(d_mixed_, 3 * D);
    alloc_f32(d_msg_attn_, 2 * D);
    alloc_f32(d_msg_kv_, 2 * D);
    alloc_f32(d_ctrl_out_, 3 * D);
    for (auto& p : d_normed_) alloc_f32(p, D);
    alloc_f32(d_loop_z_, 3 * hp_.loop_gate_rank);

    alloc_f32(d_Q_, Qdim); alloc_f32(d_K_, KVdim); alloc_f32(d_K_conv_, KVdim);
    alloc_f32(d_V_, KVdim); alloc_f32(d_V_conv_, KVdim);
    alloc_f32(d_dyn_k_, hp_.sconv_k * hp_.n_head_kv);
    alloc_f32(d_dyn_v_, hp_.sconv_k * hp_.n_head_kv);
    alloc_f32(d_R_, hp_.n_head * hp_.d_rel);
    alloc_f32(d_sinks_, hp_.n_head);
    alloc_f32(d_rel_table_, hp_.n_head * hp_.rel_extent);
    alloc_f32(d_attn_out_, Qdim);
    alloc_f32(d_attn_gate_, Qdim);
    alloc_f32(d_wo_out_, D);
    alloc_f32(d_post_conv_, D);

    alloc_f32(d_ffn_normed_, D);
    alloc_f32(d_moe_h_, hp_.n_expert_used * hp_.n_ff_exp);
    alloc_f32(d_moe_out_, D);
    alloc_f32(d_moe_cpu_out_, D);
    alloc_f32(d_router_logits_, hp_.n_expert);
    INK_CUDA_CHECK(cudaMalloc(&d_top_ids_, hp_.n_expert_used * sizeof(int32_t)));
    alloc_f32(d_top_weights_, hp_.n_expert_used);
    INK_CUDA_CHECK(cudaMalloc(&d_n_gpu_, sizeof(int32_t)));
    INK_CUDA_CHECK(cudaMalloc(&d_gpu_hit_ids_, hp_.n_expert_used * sizeof(int32_t)));
    alloc_f32(d_gpu_hit_weights_, hp_.n_expert_used);
    alloc_f32(d_logits_, hp_.vocab_padded);
    INK_CUDA_CHECK(cudaMalloc(&d_argmax_, sizeof(int32_t)));

    INK_CUDA_CHECK(cudaHostAlloc(&h_embd_row_,      D * sizeof(float), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_top_ids_,       hp_.n_expert_used * sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_top_weights_,   hp_.n_expert_used * sizeof(float), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_ffn_in_,        D * sizeof(float), cudaHostAllocMapped));
    INK_CUDA_CHECK(cudaHostGetDevicePointer(&d_mapped_ffn_in_, h_ffn_in_, 0));
    INK_CUDA_CHECK(cudaHostAlloc(&h_split_pkt_,     sizeof(MoESplitPacket), cudaHostAllocMapped));
    std::memset(h_split_pkt_, 0, sizeof(MoESplitPacket));
    split_pkt_seq_ = 0;
    INK_CUDA_CHECK(cudaHostGetDevicePointer(&d_split_pkt_, h_split_pkt_, 0));
    INK_CUDA_CHECK(cudaHostAlloc(&h_zero_sync_ids_, (size_t)hp_.n_layer_logical * hp_.n_expert_used * sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_moe_cpu_out_,   D * sizeof(float), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_argmax_,        sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_split_, cudaEventDisableTiming));

    // R14 (SR1-03): Bit-exact GPU candidate pre-filter scratch (~64 KB VRAM + ~64 KB pinned RAM)
    constexpr int ext_cap = INK_SAMPLE_OVERFLOW_CAP - INK_SAMPLE_INLINE_CAP;
    INK_CUDA_CHECK(cudaMalloc(&d_sample_pkt_, sizeof(SamplePrefilterPacket)));
    INK_CUDA_CHECK(cudaMalloc(&d_sample_ext_ids_,  (size_t)ext_cap * sizeof(int32_t)));
    INK_CUDA_CHECK(cudaMalloc(&d_sample_ext_vals_, (size_t)ext_cap * sizeof(float)));
    INK_CUDA_CHECK(cudaHostAlloc(&h_sample_pkt_, sizeof(SamplePrefilterPacket), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_sample_ext_ids_,  (size_t)ext_cap * sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_sample_ext_vals_, (size_t)ext_cap * sizeof(float),   cudaHostAllocDefault));
    h_samp_cand_.reserve(512);

    // 4. Initialize HybridPagedKV (40 SWA in VRAM + 8 Global 3-Tier VRAM->RAM->NVMe SSD!)
    size_t vram_free1 = 0;
    INK_CUDA_CHECK(cudaMemGetInfo(&vram_free1, &vram_tot));
    const size_t vram_safety = 260ull * 1024 * 1024; // Windows WDDM safety headroom

    // Compute full Global KV size for max_ctx
    size_t hb = (cfg.kv_quant == KV_QUANT_Q4_0) ? 72 : ((cfg.kv_quant == KV_QUANT_Q8_0) ? 136 : 256);
    size_t full_glob_kv = 8ull * ((size_t)(cfg.max_ctx + 63) / 64) * 64 * 2 * hp_.n_head_kv * hb;
    size_t vram_kv_budget = cfg.vram_kv_mb > 0
        ? (cfg.vram_kv_mb * 1024ull * 1024ull)
        : std::min<size_t>(full_glob_kv, 640ull * 1024 * 1024); // default cap 640 MiB -> keeps >= 5.05 GiB for Core experts!

    // If user asks for 128K in Q4_0 (1152 MiB) and explicitly wants all in VRAM, respect vram_kv_mb;
    // otherwise at >64K, 640 MiB VRAM holds 71K hot tokens + Quest/NVMe SSD handles up to 1M tokens!
    std::string swap_path = cfg.kv_swap_dir;
    if (swap_path.empty()) {
        char tmpdir[MAX_PATH];
        GetTempPathA(MAX_PATH, tmpdir);
        swap_path = std::string(tmpdir) + "inkengine_kv_swap.bin";
    } else {
        if (swap_path.back() != '\\' && swap_path.back() != '/') swap_path.push_back('\\');
        swap_path += "inkengine_kv_swap.bin";
    }

    if (!kv_.init(hp_.n_layer_logical, hp_.is_swa, hp_.n_head, hp_.n_head_kv, hp_.head_dim,
                  hp_.n_embd, hp_.swa_window, cfg.max_ctx, cfg.kv_quant, cfg.kv_mode,
                  vram_kv_budget, cfg.ram_kv_mb * 1024ull * 1024ull, swap_path, cfg.quest_top_pages)) {
        return false;
    }

    // 4b. Allocate Batched Prefill (B <= 32) & Speculative Verify (B <= 6) scratch (~21 MiB VRAM) BEFORE ExpertCache!
    if (cfg.use_batched_prefill || cfg.use_speculative) {
        alloc_batch_buffers();
    }

    // 5. Initialize 3-Tier Loop-Aware ExpertCache using ALL remaining free VRAM!
    size_t vram_free2 = 0;
    INK_CUDA_CHECK(cudaMemGetInfo(&vram_free2, &vram_tot));
    size_t exp_vram_budget = (vram_free2 > vram_safety) ? (vram_free2 - vram_safety) : (256ull * 1024 * 1024);
    if (cfg.vram_exp_mb > 0) {
        exp_vram_budget = std::min(exp_vram_budget, cfg.vram_exp_mb * 1024ull * 1024ull);
    }

    MEMORYSTATUSEX ms{}; ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    size_t avail_ram = (size_t)ms.ullAvailPhys;
    size_t ram_safety = 1100ull * 1024 * 1024;
    size_t exp_ram_budget = cfg.ram_exp_mb > 0
        ? (cfg.ram_exp_mb * 1024ull * 1024ull)
        : (avail_ram > ram_safety ? std::min<size_t>(avail_ram - ram_safety, 5800ull * 1024 * 1024) : 1024ull * 1024 * 1024);

    std::printf("[Memory] Free VRAM after dense+KV: %.0f MiB -> Expert VRAM budget: %.0f MiB | Free RAM: %.0f MiB -> Expert RAM budget: %.0f MiB\n",
                vram_free2 / (1024.0 * 1024.0), exp_vram_budget / (1024.0 * 1024.0),
                avail_ram / (1024.0 * 1024.0), exp_ram_budget / (1024.0 * 1024.0));

    if (!experts_.init(gguf_, hp_.n_layer_phys, hp_.n_expert, hp_.core_start, hp_.core_end,
                       exp_vram_budget, exp_ram_budget, cfg.cache_policy, cfg.preload_ram, cfg.async_ram_preload)) {
        return false;
    }

    // Allocate device scalar for dynamic global log-N tau (so CUDA Graphs work at ALL context lengths!)
    INK_CUDA_CHECK(cudaMalloc(&d_tau_global_, sizeof(float)));
    float init_tau = 1.0f;
    INK_CUDA_CHECK(cudaMemcpy(d_tau_global_, &init_tau, sizeof(float), cudaMemcpyHostToDevice));
    last_tau_global_ = 1.0f;
    if (cfg_.use_cuda_graphs) {
        capture_cuda_graphs();
    }

    // 6. Optional Native MoE Expert Predictor (Lanilen/Inkling-2-Mini-Expert-Predictor pred-v3 + start-v3)
    // Compute 64-bit model signature & initialize NVMe persistent KV cache directory (%LOCALAPPDATA%\InkEngine\kv_cache)
    {
        uint64_t sig = 14695981039346656037ULL;
        auto mix = [&](uint64_t v) { sig ^= v; sig *= 1099511628211ULL; };
        mix((uint64_t)hp_.n_layer_logical);
        mix((uint64_t)hp_.n_embd);
        mix((uint64_t)hp_.vocab_unpadded);
        mix((uint64_t)cfg.kv_quant);
        mix((uint64_t)cfg.fp16_loop);
        WIN32_FILE_ATTRIBUTE_DATA fad{};
        if (GetFileAttributesExA(cfg.model_path.c_str(), GetFileExInfoStandard, &fad)) {
            uint64_t fsz = ((uint64_t)fad.nFileSizeHigh << 32) | (uint64_t)fad.nFileSizeLow;
            mix(fsz);
        }
        size_t sl = cfg.model_path.find_last_of("\\/");
        std::string bname = (sl == std::string::npos) ? cfg.model_path : cfg.model_path.substr(sl + 1);
        for (unsigned char c : bname) mix((uint64_t)c);
        model_sig_ = sig;

        kv_cache_dir_ = cfg.kv_cache_dir;
        if (kv_cache_dir_.empty()) {
            char env_buf[MAX_PATH] = {};
            if (GetEnvironmentVariableA("LOCALAPPDATA", env_buf, MAX_PATH) > 0) {
                std::string base = std::string(env_buf) + "\\InkEngine";
                CreateDirectoryA(base.c_str(), nullptr);
                kv_cache_dir_ = base + "\\kv_cache";
                CreateDirectoryA(kv_cache_dir_.c_str(), nullptr);
            }
        } else {
            CreateDirectoryA(kv_cache_dir_.c_str(), nullptr);
        }
    }
    pred_tracker_.init(hp_.n_layer_phys, hp_.n_expert, hp_.core_start, hp_.core_end, hp_.num_loops);
    if (cfg.use_predictor) {
        auto resolve_model_file = [](const std::string& explicit_path, const char* rel_name) -> std::string {
            auto exists = [](const std::string& p) {
                DWORD a = GetFileAttributesA(p.c_str());
                return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
            };
            if (!explicit_path.empty()) return exists(explicit_path) ? explicit_path : "";
            // Try beside executable (../models/...) and current working dir (models/...)
            char exe_path[MAX_PATH] = {};
            if (GetModuleFileNameA(nullptr, exe_path, MAX_PATH) > 0) {
                std::string dir(exe_path);
                size_t sl = dir.find_last_of("\\/");
                if (sl != std::string::npos) {
                    dir.resize(sl);
                    std::string p1 = dir + "\\..\\models\\" + rel_name;
                    if (exists(p1)) return p1;
                    std::string p2 = dir + "\\models\\" + rel_name;
                    if (exists(p2)) return p2;
                }
            }
            std::string p3 = std::string("models\\") + rel_name;
            if (exists(p3)) return p3;
            return "";
        };

        std::string p_main  = resolve_model_file(cfg.predictor_path, "inkling2-pred-v3.moepred");
        std::string p_start = resolve_model_file(cfg.predictor_start_path, "inkling2-start-v3.moepred");
        if (!p_main.empty()) {
            if (pred_main_.load(p_main, hp_.n_layer_phys, hp_.n_expert, hp_.n_embd)) {
                pred_hidden_slots_ = (int)pred_main_.hidden_layers().size();
                if (pred_hidden_slots_ > 0) {
                    INK_CUDA_CHECK(cudaHostAlloc(&h_pred_hidden_, (size_t)pred_hidden_slots_ * hp_.n_embd * sizeof(float), cudaHostAllocDefault));
                }
            }
        }
        if (!p_start.empty()) {
            pred_start_.load(p_start, hp_.n_layer_phys, hp_.n_expert, hp_.n_embd);
        }
    }

    if (cfg.use_vision) {
        auto file_exists = [](const std::string& p) {
            DWORD a = GetFileAttributesA(p.c_str());
            return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
        };
        std::string mm_path = cfg.mmproj_path;
        if (mm_path.empty()) {
            // Auto-detect mmproj beside model GGUF (F16 preferred -> BF16 -> F32)
            std::string mdir = cfg.model_path;
            size_t sl = mdir.find_last_of("\\/");
            mdir = (sl != std::string::npos) ? mdir.substr(0, sl + 1) : "";
            static const char* kCandidates[] = {
                "mmproj-F16.gguf",
                "mmproj-Inkling-2-Mini-F16.gguf",
                "mmproj-BF16.gguf",
                "mmproj-Inkling-2-Mini-BF16.gguf",
                "mmproj-F32.gguf"
            };
            for (const char* cand : kCandidates) {
                if (file_exists(mdir + cand)) { mm_path = mdir + cand; break; }
            }
        }
        if (!mm_path.empty() && file_exists(mm_path)) {
            vision_.load(mm_path);
        }
    }

    size_t vram_free_final = 0;
    INK_CUDA_CHECK(cudaMemGetInfo(&vram_free_final, &vram_tot));
    std::printf("[Engine] Loaded in %.2fs | Final free VRAM: %.0f MiB / %.0f MiB\n",
                (now_ms() - t_start) / 1000.0,
                vram_free_final / (1024.0 * 1024.0), vram_tot / (1024.0 * 1024.0));
    std::printf("================================================================\n");
    std::fflush(stdout);
    return true;
}

void InklingEngine::reset_context() {
    kv_.reset();
    kv_tokens_.clear();
    pred_tracker_.reset_conversation();
    experts_.clear_predictor_scores();
}

void InklingEngine::run_start_predictor(const int32_t* turn_tokens, int n_tokens) {
    if (!pred_start_.loaded() || n_tokens <= 0) return;
    const int D = hp_.n_embd;
    std::vector<double> acc(D, 0.0);
    std::vector<float> row(D);
    for (int i = 0; i < n_tokens; ++i) {
        int32_t tid = turn_tokens[i];
        if (tid < 0 || tid >= hp_.vocab_padded) continue;
        const uint8_t* src = h_tok_embd_.data() + (size_t)tid * tok_embd_row_bytes_;
        CPUThreadPool::dequantize_row(tok_embd_type_, src, row.data(), D);
        for (int d = 0; d < D; ++d) acc[d] += row[d];
    }
    float inv = 1.0f / (float)n_tokens;
    for (int d = 0; d < D; ++d) row[d] = (float)(acc[d] * inv);

    std::vector<float> scores;
    if (pred_tracker_.compute_start_scores(pred_start_, row.data(), scores)) {
        // Bulk-preload up to 224 hot predicted experts (especially outer layers!) before prefill starts!
        experts_.apply_predictor_scores(scores, /*min_bulk_gain=*/4.0f, /*max_bulk_swaps=*/224, /*min_bg_gain=*/3.0f, stream_);
    }
}

void InklingEngine::run_main_predictor(int64_t answer_pos, bool is_answer_start) {
    if (!pred_main_.loaded()) return;
    const int D = hp_.n_embd;
    std::vector<const float*> hptrs(pred_hidden_slots_);
    for (int i = 0; i < pred_hidden_slots_; ++i) {
        hptrs[i] = h_pred_hidden_ + (size_t)i * D;
    }
    std::vector<float> scores;
    if (pred_tracker_.compute_main_scores(pred_main_, hptrs, answer_pos, scores)) {
        if (is_answer_start) {
            experts_.apply_predictor_scores(scores, cfg_.predictor_answer_gain, /*max_bulk_swaps=*/96, /*min_bg_gain=*/3.0f, stream_);
        } else {
            // Every 16 generated tokens: refresh phased & LM-Head background swap queues (zero stall!)
            experts_.apply_predictor_scores(scores, /*min_bulk_gain=*/1000.0f, /*max_bulk_swaps=*/0, /*min_bg_gain=*/3.0f, stream_);
        }
    }
}

bool InklingEngine::benchmark_populate_kv(int target_pos) {
    kv_tokens_.clear();
    return kv_.benchmark_populate(target_pos, stream_);
}

void InklingEngine::emit_layer_pre(int il, int& n_blocks, int& prelude_n_blocks, int& n_bank, float tau_global) {
    const int D = hp_.n_embd;
    const int Qdim = hp_.n_head * hp_.head_dim;
    const int KVdim = hp_.n_head_kv * hp_.head_dim;
    const int n_loop_core = (hp_.num_loops > 1) ? (hp_.core_end - hp_.core_start) : 0;
    const int ip = hp_.loop_layer(il);
    const auto& L = layers_[ip];

    float* msg_ptr[4] = {nullptr, nullptr, nullptr, nullptr};

    if (n_loop_core > 0 && ip >= hp_.core_start && ip < hp_.core_end) {
        const int pass = (il - hp_.core_start) / n_loop_core;
        if (ip == hp_.core_start) {
            if (pass == 0) {
                prelude_n_blocks = n_blocks;
                INK_CUDA_CHECK(cudaMemcpyAsync(d_prelude_cur_, d_cur_, (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            } else {
                n_bank = 0;
                for (int b = prelude_n_blocks + 1; b < n_blocks; ++b) {
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_bank_[n_bank], d_blocks_[b], (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                    n_bank++;
                }
                INK_CUDA_CHECK(cudaMemcpyAsync(d_bank_last_, d_cur_, (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                n_blocks = prelude_n_blocks;
                INK_CUDA_CHECK(cudaMemcpyAsync(d_cur_, d_prelude_cur_, (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            }
        }

        if (pass > 0) {
            BlockPtrList bstates{};
            for (int b = 0; b < n_bank; ++b) bstates.ptrs[bstates.count++] = d_bank_[b];
            bstates.ptrs[bstates.count++] = d_bank_last_;

            const int row = 2 * (ip - hp_.core_start);
            const float* q_attn = (const float*)loop_res_query_.ptr    + (size_t)row * D;
            const float* q_kv   = (const float*)loop_kv_res_query_.ptr + (size_t)row * D;

            ink_cuda_attn_res_mix(bstates, q_attn, d_msg_attn_, D, 2, stream_);
            ink_cuda_attn_res_mix(bstates, q_kv,   d_msg_kv_,   D, 2, stream_);

            msg_ptr[0] = d_msg_attn_;       // attn Q message
            msg_ptr[3] = d_msg_attn_ + D;   // mlp message (preserved until emit_layer_post!)
            msg_ptr[1] = d_msg_kv_;         // attn K message
            msg_ptr[2] = d_msg_kv_ + D;     // attn V message
        }
    }

    INK_CUDA_CHECK(cudaMemcpyAsync(d_inpSA_, d_cur_, (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

    float* inputs[3] = {d_cur_, d_cur_, d_cur_};
    if (L.attn_res_query.ptr != nullptr) {
        BlockPtrList states{};
        for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_blocks_[b];
        states.ptrs[states.count++] = d_cur_;
        int nq = L.is_swa ? 1 : 3;
        ink_cuda_attn_res_mix(states, (const float*)L.attn_res_query.ptr, d_mixed_, D, nq, stream_);
        if (L.is_swa) {
            inputs[0] = inputs[1] = inputs[2] = d_mixed_;
        } else {
            inputs[0] = d_mixed_;
            inputs[1] = d_mixed_ + D;
            inputs[2] = d_mixed_ + 2 * D;
        }
    }

    if (msg_ptr[0] != nullptr) {
        ink_cuda_loop_control_3way(
            inputs[0], inputs[1], inputs[2],
            msg_ptr[0], msg_ptr[1], msg_ptr[2],
            L.loop_down[0].type,
            L.loop_down[0].ptr, L.loop_gain[0].ptr, L.loop_forget[0].ptr,
            L.loop_down[1].ptr, L.loop_gain[1].ptr, L.loop_forget[1].ptr,
            L.loop_down[2].ptr, L.loop_gain[2].ptr, L.loop_forget[2].ptr,
            d_loop_z_, d_ctrl_out_, d_ctrl_out_ + D, d_ctrl_out_ + 2 * D,
            D, hp_.loop_gate_rank, stream_);
        inputs[0] = d_ctrl_out_;
        inputs[1] = d_ctrl_out_ + D;
        inputs[2] = d_ctrl_out_ + 2 * D;
    }

    if (L.is_swa && msg_ptr[0] != nullptr) {
        ink_cuda_rms_norm_3way(inputs[0], inputs[1], inputs[2],
                               (const float*)L.attn_norm.ptr, (const float*)L.attn_norm.ptr, (const float*)L.attn_norm.ptr,
                               d_normed_[0], d_normed_[1], d_normed_[2], D, hp_.rms_eps, stream_);
        inputs[0] = d_normed_[0];
        inputs[1] = d_normed_[1];
        inputs[2] = d_normed_[2];
    } else if (L.is_swa) {
        ink_cuda_rms_norm(inputs[0], (const float*)L.attn_norm.ptr, d_normed_[0], D, hp_.rms_eps, stream_);
        inputs[0] = inputs[1] = inputs[2] = d_normed_[0];
    } else {
        ink_cuda_rms_norm_3way(inputs[0], inputs[1], inputs[2],
                               (const float*)L.attn_q_inp_norm.ptr, (const float*)L.attn_k_inp_norm.ptr, (const float*)L.attn_v_inp_norm.ptr,
                               d_normed_[0], d_normed_[1], d_normed_[2], D, hp_.rms_eps, stream_);
        inputs[0] = d_normed_[0];
        inputs[1] = d_normed_[1];
        inputs[2] = d_normed_[2];
    }

    // Super-Fused 1-Launch Attention Projections (Q + Gate + K + V + Rel + Sink [+ DynK + DynV]!)
    FusedProjList pl{};
    auto add_proj = [&](const GPUTensor& wt, const float* x_in, float* y_out, int M) {
        FusedProjSlot& sl = pl.slots[pl.count++];
        sl.W = (const uint8_t*)wt.ptr;
        sl.x = x_in;
        sl.y = y_out;
        sl.type = wt.type;
        sl.M = M;
        sl.row_bytes = (size_t)(D / ggml_type_block_size(wt.type)) * ggml_type_byte_size(wt.type);
        sl.block_start = pl.total_blocks;
        int nb = (M + 7) / 8; // WPB = 8 warps/block
        pl.total_blocks += nb;
        sl.block_end = pl.total_blocks;
    };
    add_proj(L.wq,             inputs[0], d_Q_,         Qdim);
    add_proj(L.wqkv_gate,      inputs[0], d_attn_gate_, Qdim);
    add_proj(L.wk,             inputs[1], d_K_,         KVdim);
    add_proj(L.wv,             inputs[2], d_V_,         KVdim);
    add_proj(L.attn_rel,       inputs[0], d_R_,         hp_.n_head * hp_.d_rel);
    add_proj(L.attn_sink_proj, inputs[0], d_sinks_,     hp_.n_head);

    float* dyn_k = nullptr;
    float* dyn_v = nullptr;
    if (!L.is_swa) {
        add_proj(L.attn_k_conv_proj, inputs[1], d_dyn_k_, hp_.sconv_k * hp_.n_head_kv);
        add_proj(L.attn_v_conv_proj, inputs[2], d_dyn_v_, hp_.sconv_k * hp_.n_head_kv);
        dyn_k = d_dyn_k_;
        dyn_v = d_dyn_v_;
    }
    ink_cuda_fused_multi_gemv(pl, D, stream_);

    ink_cuda_shortconv_kv(d_K_, kv_.conv_state_ptr(il, 0),     (const float*)L.attn_k_conv.ptr, dyn_k, d_K_conv_,
                          d_V_, kv_.conv_state_ptr(il, KVdim), (const float*)L.attn_v_conv.ptr, dyn_v, d_V_conv_,
                          KVdim, hp_.n_head_kv, stream_);

    ink_cuda_qkv_post(d_Q_, (const float*)L.attn_q_clip.ptr,
                      d_K_conv_, d_V_conv_,
                      d_R_, (const float*)L.attn_rel_clip.ptr,
                      hp_.n_head, hp_.n_head_kv, hp_.head_dim, hp_.d_rel,
                      hp_.qk_clip, hp_.rel_clip, hp_.rms_eps,
                      L.is_swa ? 1.0f : tau_global,
                      L.is_swa ? nullptr : d_tau_global_, stream_);

    int rel_ext = L.is_swa ? hp_.swa_window : hp_.rel_extent;
    ink_cuda_rel_table((const float*)L.attn_rel_bias.ptr, d_R_, d_rel_table_, hp_.n_head, hp_.d_rel, rel_ext, stream_);
}

void InklingEngine::emit_layer_post(int il, int& n_blocks, int n_bank) {
    (void)n_bank;
    const int D = hp_.n_embd;
    const int Qdim = hp_.n_head * hp_.head_dim;
    const int KVdim = hp_.n_head_kv * hp_.head_dim;
    const int n_loop_core = (hp_.num_loops > 1) ? (hp_.core_end - hp_.core_start) : 0;
    const int ip = hp_.loop_layer(il);
    const auto& L = layers_[ip];

    // Fused Gate-Sigmoid-Mul + Wo GEMV (zero d_attn_out_ roundtrip!)
    ink_cuda_gemv_gated(L.wo.type, L.wo.ptr, d_attn_out_, d_attn_gate_, d_wo_out_, D, Qdim, stream_);

    if (ip % hp_.attn_res_block_size == 0) {
        ink_cuda_shortconv(d_wo_out_, kv_.conv_state_ptr(il, 2 * KVdim), (const float*)L.attn_post_conv.ptr, nullptr, d_cur_, D, 0, stream_);
        INK_CUDA_CHECK(cudaMemcpyAsync(d_blocks_[n_blocks], d_inpSA_, (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        n_blocks++;
    } else {
        // Fused AttnPostConv + Residual Add (1 kernel instead of 2 for 40/48 logical layers!)
        ink_cuda_shortconv_add(d_wo_out_, kv_.conv_state_ptr(il, 2 * KVdim), (const float*)L.attn_post_conv.ptr, d_inpSA_, d_cur_, D, stream_);
    }

    INK_CUDA_CHECK(cudaMemcpyAsync(d_ffn_inp_, d_cur_, (size_t)D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

    {
        BlockPtrList states{};
        for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_blocks_[b];
        states.ptrs[states.count++] = d_cur_;
        ink_cuda_attn_res_mix(states, (const float*)L.ffn_res_query.ptr, d_mixed_, D, 1, stream_);
        float* ffn_x = d_mixed_;
        bool has_mlp_msg = (n_loop_core > 0 && ip >= hp_.core_start && ip < hp_.core_end &&
                            ((il - hp_.core_start) / n_loop_core) > 0);
        if (has_mlp_msg) {
            ink_cuda_loop_control(ffn_x, d_msg_attn_ + D, L.loop_down[3].type,
                                  L.loop_down[3].ptr, L.loop_gain[3].ptr, L.loop_forget[3].ptr,
                                  d_loop_z_, d_ctrl_out_, D, hp_.loop_gate_rank, stream_);
            ffn_x = d_ctrl_out_;
        }
        // Fused FFN RMSNorm + Exact FP32 MoE Router + Top-K (100% exact FP32 router weights!)
        ink_cuda_fused_ffn_norm_router(ffn_x, (const float*)L.ffn_norm.ptr, d_ffn_normed_,
                                       (const float*)L.ffn_gate_inp.ptr,
                                       (const float*)L.ffn_exp_probs_b.ptr,
                                       d_router_logits_, d_top_ids_, d_top_weights_,
                                       hp_.n_expert, D, hp_.n_expert_used,
                                       hp_.rms_eps, hp_.expert_weights_scale, stream_);
    }
}

void InklingEngine::emit_lm_head(int n_blocks) {
    const int D = hp_.n_embd;
    BlockPtrList states{};
    for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_blocks_[b];
    states.ptrs[states.count++] = d_cur_;
    ink_cuda_attn_res_mix(states, (const float*)output_res_query_.ptr, d_mixed_, D, 1, stream_);
    ink_cuda_rms_norm(d_mixed_, (const float*)output_norm_.ptr, d_cur_, D, hp_.rms_eps, stream_);
    ink_cuda_gemv(output_.type, output_.ptr, d_cur_, d_logits_, hp_.vocab_unpadded, D, stream_);
    ink_cuda_logit_softcap_argmax(d_logits_, hp_.vocab_unpadded, hp_.final_logit_softcap, d_argmax_, stream_);
    INK_CUDA_CHECK(cudaMemcpyAsync(h_argmax_, d_argmax_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
}

void InklingEngine::capture_cuda_graphs() {
    destroy_cuda_graphs();
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    double tg0 = now_ms();

    int n_blocks = 0, prelude_n_blocks = 0, n_bank = 0;
    for (int il = 0; il < hp_.n_layer_logical; ++il) {
        cudaGraph_t g_pre = nullptr;
        INK_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        emit_layer_pre(il, n_blocks, prelude_n_blocks, n_bank, 1.0f);
        INK_CUDA_CHECK(cudaStreamEndCapture(stream_, &g_pre));
        INK_CUDA_CHECK(cudaGraphInstantiate(&graph_pre_[il], g_pre, nullptr, nullptr, 0));
        INK_CUDA_CHECK(cudaGraphDestroy(g_pre));

        cudaGraph_t g_post = nullptr;
        INK_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        emit_layer_post(il, n_blocks, n_bank);
        INK_CUDA_CHECK(cudaStreamEndCapture(stream_, &g_post));
        INK_CUDA_CHECK(cudaGraphInstantiate(&graph_post_[il], g_post, nullptr, nullptr, 0));
        INK_CUDA_CHECK(cudaGraphDestroy(g_post));
    }

    cudaGraph_t g_lm = nullptr;
    INK_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    emit_lm_head(n_blocks);
    INK_CUDA_CHECK(cudaStreamEndCapture(stream_, &g_lm));
    INK_CUDA_CHECK(cudaGraphInstantiate(&graph_lm_head_, g_lm, nullptr, nullptr, 0));
    INK_CUDA_CHECK(cudaGraphDestroy(g_lm));

    graphs_ready_ = true;
    std::printf("[CUDA] Captured 97 CUDA Graphs (48x pre + 48x post + LM Head) in %.1f ms (~82%% launch reduction!)\n",
                now_ms() - tg0);
}

static std::vector<float>* g_l0_attn_capture = nullptr;

// R19: Spin-poll zero-copy mapped MoESplitPacket::seq written after __threadfence_system()
// Avoids WDDM cudaEventRecord + cudaEventSynchronize driver overhead (~15-40 us per hybrid layer!).
void InklingEngine::wait_split_packet(uint32_t target_seq) {
    cudaError_t q = cudaStreamQuery(stream_);
    if (q != cudaSuccess && q != cudaErrorNotReady) INK_CUDA_CHECK(q);
    const volatile uint32_t* pseq = &h_split_pkt_->seq;
    uint32_t spins = 0;
    while (*pseq != target_seq) {
        _mm_pause();
        ++spins;
        if ((spins & 0xFFFFu) == 0) {
            cudaError_t qe = cudaStreamQuery(stream_);
            if (qe != cudaSuccess && qe != cudaErrorNotReady) INK_CUDA_CHECK(qe);
            if (spins >= 0x100000u) std::this_thread::yield();
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

void InklingEngine::wait_batch_split_packet(uint32_t target_seq) {
    cudaError_t q = cudaStreamQuery(stream_);
    if (q != cudaSuccess && q != cudaErrorNotReady) INK_CUDA_CHECK(q);
    const volatile uint32_t* pseq = &h_b_split_pkt_->seq;
    uint32_t spins = 0;
    while (*pseq != target_seq) {
        _mm_pause();
        ++spins;
        if ((spins & 0xFFFFu) == 0) {
            cudaError_t qe = cudaStreamQuery(stream_);
            if (qe != cudaSuccess && qe != cudaErrorNotReady) INK_CUDA_CHECK(qe);
            if (spins >= 0x100000u) std::this_thread::yield();
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

int32_t InklingEngine::forward_token(int32_t token_id, int pos, bool need_logits, bool enable_profile,
                                     bool is_answer, bool capture_pred_hidden) {
    double t0 = 0, t_step = 0;
    auto mark = [&](double& accum) {
        if (!enable_profile) return;
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        double now = now_ms();
        accum += (now - t_step);
        t_step = now;
    };
    if (enable_profile) {
        t0 = t_step = now_ms();
    }

    const bool track_pred = predictor_active();
    if (track_pred) {
        if (is_answer) pred_tracker_.step_answer_token();
        else           pred_tracker_.step_prompt_token();
    }
    const auto& pred_hid_layers = pred_main_.hidden_layers();

    const int D = hp_.n_embd;
    const int KVdim = hp_.n_head_kv * hp_.head_dim;

    // 1. Token Embedding Lookup on CPU (9 KB) -> GPU + RMSNorm
    const uint8_t* emb_row = h_tok_embd_.data() + (size_t)token_id * tok_embd_row_bytes_;
    CPUThreadPool::dequantize_row(tok_embd_type_, emb_row, h_embd_row_, D);
    INK_CUDA_CHECK(cudaMemcpyAsync(d_cur_, h_embd_row_, (size_t)D * sizeof(float), cudaMemcpyHostToDevice, stream_));
    ink_cuda_rms_norm(d_cur_, (const float*)tok_norm_.ptr, d_cur_, D, hp_.rms_eps, stream_);
    mark(prof_.embed_ms);

    // Log-N tau for global attention layers (updated on GPU only when changed -> 0 copies for pos < 8192!)
    float eff = (float)(pos + 1) / (float)hp_.log_scale_floor;
    float tau_global = 1.0f + hp_.log_scale_alpha * std::log(std::max(eff, 1.0f));
    if (tau_global != last_tau_global_) {
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        INK_CUDA_CHECK(cudaMemcpy(d_tau_global_, &tau_global, sizeof(float), cudaMemcpyHostToDevice));
        last_tau_global_ = tau_global;
    }

    const bool use_graphs = graphs_ready_ && !enable_profile;
    const int coda_start_il = (hp_.num_loops > 1)
        ? (hp_.core_start + hp_.num_loops * (hp_.core_end - hp_.core_start))
        : hp_.core_end;
    int n_blocks = 0;
    int prelude_n_blocks = 0;
    int n_bank = 0;
    int zero_sync_layers[48];
    int n_zero_sync = 0;

    for (int il = 0; il < hp_.n_layer_logical; ++il) {
        const int ip = hp_.loop_layer(il);
        const auto& L = layers_[ip];

        // Loop-Phased Continuous Background Expert DMA (InkEngine v3.0 — ZERO race, ZERO stall!):
        //  - Entering 36 Core layers (il=6): swap up to 2 high-gain Outer experts into Outer victims while Core runs!
        //  - Entering Outer Coda (il=42): finalize completed Outer DMAs (now immediately live for this token's Coda!)
        if (il == hp_.core_start) {
            experts_.begin_phased_swaps(stream_, /*phase_filter=*/1, /*max_swaps=*/2);
        } else if (il == coda_start_il) {
            experts_.finish_phased_swaps(stream_);
        }

        // Capture pass-0 residual stream entering physical layers 8 & 16 for MoE predictor (9 KB async D2H)
        if (capture_pred_hidden && h_pred_hidden_ && il == ip) {
            for (int hi = 0; hi < pred_hidden_slots_; ++hi) {
                if (pred_hid_layers[hi] == ip) {
                    INK_CUDA_CHECK(cudaMemcpyAsync(h_pred_hidden_ + (size_t)hi * D, d_cur_,
                                                   (size_t)D * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                }
            }
        }

        // Subgraph A: Block AttnRes + Loop Controllers + RMSNorm + QKV/Gate/Rel/Sink GEMVs + K/V ShortConv + QKV Post + RelTable
        if (use_graphs) {
            INK_CUDA_CHECK(cudaGraphLaunch(graph_pre_[il], stream_));
        } else {
            emit_layer_pre(il, n_blocks, prelude_n_blocks, n_bank, tau_global);
            mark(prof_.mix_loop_ms);
        }

        // Dynamic Paged Attention (SWA ring or 3-tier Global + Quest)
        int rel_ext = L.is_swa ? hp_.swa_window : hp_.rel_extent;
        if (!kv_.append_and_attend(il, pos, d_Q_, d_K_conv_, d_V_conv_, d_rel_table_, d_sinks_, d_attn_out_, rel_ext, stream_)) {
            record_io_error(kv_.last_error());
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            return -1;
        }
        if (il == 0 && g_l0_attn_capture) {
            int qdim = hp_.n_head * hp_.head_dim;
            g_l0_attn_capture->resize(qdim);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            INK_CUDA_CHECK(cudaMemcpy(g_l0_attn_capture->data(), d_attn_out_, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Subgraph B: GateMul + Wo + AttnPostConv + Residual + FFN AttnRes + LoopCtrl + FFN RMSNorm + MoE Router
        if (use_graphs) {
            INK_CUDA_CHECK(cudaGraphLaunch(graph_post_[il], stream_));
        } else {
            emit_layer_post(il, n_blocks, n_bank);
            mark(prof_.attn_ms);
        }

        const auto& lm = experts_.layer_meta(ip);
        const auto& tbl = experts_.layer_device_table(ip);

        if (experts_.is_layer_all_vram(ip)) {
            // ZERO-SYNC GPU-INDIRECT FAST PATH
            ink_cuda_moe_pinned_indirect(lm.gate_up_type, lm.down_type, tbl,
                                         d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                         d_ffn_normed_, d_moe_h_, d_moe_out_,
                                         D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
            if (!experts_.is_layer_pinned(ip) || track_pred) {
                INK_CUDA_CHECK(cudaMemcpyAsync(h_zero_sync_ids_ + n_zero_sync * hp_.n_expert_used,
                                               d_top_ids_, hp_.n_expert_used * sizeof(int32_t),
                                               cudaMemcpyDeviceToHost, stream_));
                zero_sync_layers[n_zero_sync++] = ip;
            }
            experts_.record_pinned_layer_hits(ip, hp_.n_expert_used);
        } else {
            // GPU-FIRST ZERO-BUBBLE HYBRID PATH (R19: zero-copy spin-poll, no WDDM event wait!)
            uint32_t seq = (++split_pkt_seq_ == 0) ? ++split_pkt_seq_ : split_pkt_seq_;
            ink_cuda_moe_split_hits(tbl, d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                    d_ffn_normed_, D,
                                    d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_,
                                    d_split_pkt_, d_mapped_ffn_in_, seq, stream_);

            ink_cuda_moe_dynamic_indirect(lm.gate_up_type, lm.down_type, tbl,
                                          d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_, hp_.n_expert_used,
                                          d_ffn_normed_, d_moe_h_, d_moe_out_,
                                          D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);

            wait_split_packet(seq);

            CPUExpertTask cpu_tasks[8];
            int n_cpu = 0;
            if (!experts_.resolve_gpu_split(ip, *h_split_pkt_, cpu_tasks, n_cpu)) {
                record_io_error(experts_.last_error());
                INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                return -1;
            }
            if (track_pred) {
                pred_tracker_.record_selections(ip, h_split_pkt_->all_ids, hp_.n_expert_used, is_answer);
            }

            if (n_cpu > 0) {
                cpu_pool_->run_moe_experts(h_ffn_in_, h_moe_cpu_out_, cpu_tasks, n_cpu,
                                           D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta);
                INK_CUDA_CHECK(cudaMemcpyAsync(d_moe_cpu_out_, h_moe_cpu_out_, (size_t)D * sizeof(float), cudaMemcpyHostToDevice, stream_));
                ink_cuda_add_inplace(d_moe_out_, d_moe_cpu_out_, D, stream_);
            }
        }

        // Fused FFN Post-Conv + Residual Add (1 kernel instead of 2, zero intermediate GDDR6 roundtrip!)
        ink_cuda_shortconv_add(d_moe_out_, kv_.conv_state_ptr(il, 2 * KVdim + D),
                               (const float*)L.ffn_post_conv.ptr, d_ffn_inp_, d_cur_, D, stream_);

        mark(L.is_core ? prof_.moe_core_ms : prof_.moe_outer_ms);
    }

    // Begin cross-tier/general exclusive VRAM <-> RAM swaps overlapped with LM Head (ZERO stall!)
    const int max_swaps = need_logits ? 1 : 8;
    if (n_zero_sync == 0 || predictor_active()) {
        experts_.begin_token_end_swaps(stream_, max_swaps);
    }

    int32_t greedy = -1;
    if (need_logits) {
        if (use_graphs) {
            INK_CUDA_CHECK(cudaGraphLaunch(graph_lm_head_, stream_));
        } else {
            emit_lm_head(n_blocks);
        }
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        greedy = *h_argmax_;
        mark(prof_.lm_head_ms);
    } else {
        // Always sync before returning so host staging buffers (h_embd_row_, per-layer page tables) are safe for next token
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    if (n_zero_sync > 0) {
        for (int z = 0; z < n_zero_sync; ++z) {
            const int32_t* zids = h_zero_sync_ids_ + z * hp_.n_expert_used;
            experts_.record_zero_sync_layer_ids(zero_sync_layers[z], zids, hp_.n_expert_used);
            if (track_pred) {
                pred_tracker_.record_selections(zero_sync_layers[z], zids, hp_.n_expert_used, is_answer);
            }
        }
        if (!predictor_active()) {
            experts_.begin_token_end_swaps(stream_, max_swaps);
        }
    }
    experts_.finish_token_end_swaps(stream_, true);

    if (enable_profile) {
        prof_.total_ms += (now_ms() - t0);
        prof_.tokens++;
    }
    return greedy;
}

int32_t InklingEngine::forward_token_1loop_draft(int32_t token_id, int pos) {
    const int D = hp_.n_embd;
    const int KVdim = hp_.n_head_kv * hp_.head_dim;

    const uint8_t* emb_row = h_tok_embd_.data() + (size_t)token_id * tok_embd_row_bytes_;
    CPUThreadPool::dequantize_row(tok_embd_type_, emb_row, h_embd_row_, D);
    INK_CUDA_CHECK(cudaMemcpyAsync(d_cur_, h_embd_row_, (size_t)D * sizeof(float), cudaMemcpyHostToDevice, stream_));
    ink_cuda_rms_norm(d_cur_, (const float*)tok_norm_.ptr, d_cur_, D, hp_.rms_eps, stream_);

    float eff = (float)(pos + 1) / (float)hp_.log_scale_floor;
    float tau_global = 1.0f + hp_.log_scale_alpha * std::log(std::max(eff, 1.0f));
    if (tau_global != last_tau_global_) {
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        INK_CUDA_CHECK(cudaMemcpy(d_tau_global_, &tau_global, sizeof(float), cudaMemcpyHostToDevice));
        last_tau_global_ = tau_global;
    }

    const bool use_graphs = graphs_ready_;
    int n_blocks = 0, prelude_n_blocks = 0, n_bank = 0;

    // 1-Loop Early-Exit (24 logical layers: Prelude [0..5] + Pass-0 Core [6..17] + Coda [42..47]), 100% GPU-Only!
    for (int step = 0; step < 24; ++step) {
        const int il = (step < 18) ? step : (step + 24);
        const int ip = hp_.loop_layer(il);
        const auto& L = layers_[ip];

        if (use_graphs) {
            INK_CUDA_CHECK(cudaGraphLaunch(graph_pre_[il], stream_));
        } else {
            emit_layer_pre(il, n_blocks, prelude_n_blocks, n_bank, tau_global);
        }

        int rel_ext = L.is_swa ? hp_.swa_window : hp_.rel_extent;
        if (!kv_.append_and_attend(il, pos, d_Q_, d_K_conv_, d_V_conv_, d_rel_table_, d_sinks_, d_attn_out_, rel_ext, stream_)) {
            record_io_error(kv_.last_error());
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            return -1;
        }

        if (use_graphs) {
            INK_CUDA_CHECK(cudaGraphLaunch(graph_post_[il], stream_));
        } else {
            emit_layer_post(il, n_blocks, n_bank);
        }

        const auto& lm = experts_.layer_meta(ip);
        const auto& tbl = experts_.layer_device_table(ip);
        if (experts_.is_layer_all_vram(ip)) {
            ink_cuda_moe_pinned_indirect(lm.gate_up_type, lm.down_type, tbl,
                                         d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                         d_ffn_normed_, d_moe_h_, d_moe_out_,
                                         D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
        } else {
            // GPU-Only VRAM-hit MoE (zero CPU sync, zero PCIe traffic, renormalized hit weights!)
            ink_cuda_moe_split_hits(tbl, d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                    d_ffn_normed_, D,
                                    d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_,
                                    nullptr, nullptr, 0, stream_);
            ink_cuda_moe_dynamic_indirect(lm.gate_up_type, lm.down_type, tbl,
                                          d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_, hp_.n_expert_used,
                                          d_ffn_normed_, d_moe_h_, d_moe_out_,
                                          D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
        }
        ink_cuda_shortconv_add(d_moe_out_, kv_.conv_state_ptr(il, 2 * KVdim + D),
                               (const float*)L.ffn_post_conv.ptr, d_ffn_inp_, d_cur_, D, stream_);
    }

    if (use_graphs) {
        INK_CUDA_CHECK(cudaGraphLaunch(graph_lm_head_, stream_));
    } else {
        emit_lm_head(n_blocks);
    }
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return *h_argmax_;
}

void InklingEngine::alloc_batch_buffers() {
    free_batch_buffers();
    const size_t B = (size_t)BATCH_MAX;
    const size_t D = (size_t)hp_.n_embd;
    const size_t Qdim = (size_t)hp_.n_head * hp_.head_dim;
    const size_t KVdim = (size_t)hp_.n_head_kv * hp_.head_dim;
    const size_t rel_ext = (size_t)std::max(hp_.swa_window, hp_.rel_extent);

    auto ad = [](auto*& p, size_t n) { INK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(*p))); };
    auto ah = [](auto*& p, size_t n) { INK_CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(*p), cudaHostAllocDefault)); };

    ad(d_b_cur_, B * D); ad(d_b_inpSA_, B * D); ad(d_b_ffn_inp_, B * D);
    for (int i = 0; i < 8; ++i) ad(d_b_blocks_[i], B * D);
    ad(d_b_prelude_cur_, B * D);
    for (int i = 0; i < 4; ++i) ad(d_b_bank_[i], B * D);
    ad(d_b_bank_last_, B * D);
    ad(d_b_mixed_, 3 * B * D);
    ad(d_b_msg_attn_, 2 * B * D);
    ad(d_b_msg_kv_, 2 * B * D);
    ad(d_b_ctrl_out_, 3 * B * D);
    for (int i = 0; i < 3; ++i) ad(d_b_normed_[i], B * D);
    ad(d_b_loop_z_, B * (size_t)hp_.loop_gate_rank);
    ad(d_b_Q_, B * Qdim); ad(d_b_K_, B * KVdim); ad(d_b_K_conv_, B * KVdim);
    ad(d_b_V_, B * KVdim); ad(d_b_V_conv_, B * KVdim);
    ad(d_b_dyn_k_, B * (size_t)hp_.sconv_k * hp_.n_head_kv);
    ad(d_b_dyn_v_, B * (size_t)hp_.sconv_k * hp_.n_head_kv);
    ad(d_b_R_, B * (size_t)hp_.n_head * hp_.d_rel);
    ad(d_b_sinks_, B * (size_t)hp_.n_head);
    ad(d_b_rel_table_, B * (size_t)hp_.n_head * rel_ext);
    ad(d_b_attn_out_, B * Qdim); ad(d_b_attn_gate_, B * Qdim);
    ad(d_b_wo_out_, B * D);
    ad(d_b_ffn_normed_, B * D);
    ad(d_b_moe_h_, B * (size_t)hp_.n_expert_used * (size_t)hp_.n_ff_exp);
    ad(d_b_moe_out_, B * D); ad(d_b_moe_cpu_out_, B * D);
    ad(d_b_router_logits_, B * (size_t)hp_.n_expert);
    ad(d_b_top_ids_, B * (size_t)hp_.n_expert_used);
    ad(d_b_top_weights_, B * (size_t)hp_.n_expert_used);
    ad(d_grp_exp_ids_, (size_t)hp_.n_expert);
    ad(d_grp_tok_off_, (size_t)hp_.n_expert + 1);
    ad(d_grp_tok_ids_, B * (size_t)hp_.n_expert_used);
    ad(d_grp_tok_w_, B * (size_t)hp_.n_expert_used);
    ad(d_b_logits_, (size_t)HybridPagedKV::SPEC_MAX_B * (size_t)hp_.vocab_unpadded);
    ad(d_b_argmax_, (size_t)HybridPagedKV::SPEC_MAX_B);

    ah(h_b_embd_, B * D);
    ah(h_b_ffn_in_, B * D);
    ah(h_b_moe_cpu_out_, B * D);
    ah(h_b_top_ids_, B * (size_t)hp_.n_expert_used);
    ah(h_b_top_weights_, B * (size_t)hp_.n_expert_used);
    ah(h_grp_exp_ids_, (size_t)hp_.n_expert);
    ah(h_grp_tok_off_, (size_t)hp_.n_expert + 1);
    ah(h_grp_tok_ids_, B * (size_t)hp_.n_expert_used);
    ah(h_grp_tok_w_, B * (size_t)hp_.n_expert_used);
    ah(h_b_argmax_, (size_t)HybridPagedKV::SPEC_MAX_B);

    // R13: Zero-Sync & Spin-Poll Batch Split buffers for Speculative Verification (<100 KB total!)
    const size_t SB = (size_t)HybridPagedKV::SPEC_MAX_B;
    ad(d_b_n_gpu_, SB);
    ad(d_b_hit_ids_, SB * (size_t)hp_.n_expert_used);
    ad(d_b_hit_weights_, SB * (size_t)hp_.n_expert_used);
    INK_CUDA_CHECK(cudaHostAlloc(&h_b_mapped_ffn_in_, SB * D * sizeof(float), cudaHostAllocMapped));
    INK_CUDA_CHECK(cudaHostGetDevicePointer(&d_b_mapped_ffn_in_, h_b_mapped_ffn_in_, 0));
    INK_CUDA_CHECK(cudaHostAlloc(&h_b_split_pkt_, sizeof(MoEBatchSplitPacket), cudaHostAllocMapped));
    std::memset(h_b_split_pkt_, 0, sizeof(MoEBatchSplitPacket));
    batch_split_seq_ = 0;
    INK_CUDA_CHECK(cudaHostGetDevicePointer(&d_b_split_pkt_, h_b_split_pkt_, 0));
    ah(h_b_zero_sync_ids_, (size_t)hp_.n_layer_logical * SB * (size_t)hp_.n_expert_used);

    // Double-buffered PCIe staging ring for Layer-Wise Tensor-Core Prefill (28 MiB VRAM + 28 MiB pinned RAM)
    int32_t seq_ids[STAGE_EXPS];
    for (int i = 0; i < STAGE_EXPS; ++i) seq_ids[i] = i;
    for (int s = 0; s < STAGE_BUFS; ++s) {
        INK_CUDA_CHECK(cudaMalloc(&d_prefill_stage_[s], (size_t)STAGE_EXPS * STAGE_SLOT_BYTES));
        INK_CUDA_CHECK(cudaHostAlloc(&h_prefill_stage_[s], (size_t)STAGE_EXPS * STAGE_SLOT_BYTES, cudaHostAllocDefault));
        INK_CUDA_CHECK(cudaMalloc(&stage_tbl_[s].d_gu_ptrs,   STAGE_EXPS * sizeof(uint64_t)));
        INK_CUDA_CHECK(cudaMalloc(&stage_tbl_[s].d_dn_ptrs,   STAGE_EXPS * sizeof(uint64_t)));
        INK_CUDA_CHECK(cudaMalloc(&stage_tbl_[s].d_gu_scales, STAGE_EXPS * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&stage_tbl_[s].d_dn_scales, STAGE_EXPS * sizeof(float)));
        stage_tbl_[s].valid = true;
        ah(h_stage_gu_ptrs_[s], STAGE_EXPS);
        ah(h_stage_dn_ptrs_[s], STAGE_EXPS);
        ah(h_stage_gu_sc_[s],   STAGE_EXPS);
        ah(h_stage_dn_sc_[s],   STAGE_EXPS);
        ad(d_stage_exp_ids_[s], STAGE_EXPS);
        INK_CUDA_CHECK(cudaMemcpy(d_stage_exp_ids_[s], seq_ids, STAGE_EXPS * sizeof(int32_t), cudaMemcpyHostToDevice));
        ad(d_stage_tok_off_[s], STAGE_EXPS + 1);
        ad(d_stage_tok_ids_[s], B * (size_t)hp_.n_expert_used);
        ad(d_stage_tok_w_[s],   B * (size_t)hp_.n_expert_used);
        ah(h_stage_tok_off_[s], STAGE_EXPS + 1);
        ah(h_stage_tok_ids_[s], B * (size_t)hp_.n_expert_used);
        ah(h_stage_tok_w_[s],   B * (size_t)hp_.n_expert_used);
        INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_stage_dma_[s], cudaEventDisableTiming));
        INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_stage_gpu_[s], cudaEventDisableTiming));
    }
    batch_ready_ = true;
}

void InklingEngine::free_batch_buffers() {
    auto fd = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto fh = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
    for (int s = 0; s < STAGE_BUFS; ++s) {
        fd(d_prefill_stage_[s]); fh(h_prefill_stage_[s]);
        fd(stage_tbl_[s].d_gu_ptrs); fd(stage_tbl_[s].d_dn_ptrs);
        fd(stage_tbl_[s].d_gu_scales); fd(stage_tbl_[s].d_dn_scales);
        stage_tbl_[s].valid = false;
        fh(h_stage_gu_ptrs_[s]); fh(h_stage_dn_ptrs_[s]);
        fh(h_stage_gu_sc_[s]);   fh(h_stage_dn_sc_[s]);
        fd(d_stage_exp_ids_[s]); fd(d_stage_tok_off_[s]);
        fd(d_stage_tok_ids_[s]); fd(d_stage_tok_w_[s]);
        fh(h_stage_tok_off_[s]); fh(h_stage_tok_ids_[s]); fh(h_stage_tok_w_[s]);
        if (ev_stage_dma_[s]) { cudaEventDestroy(ev_stage_dma_[s]); ev_stage_dma_[s] = nullptr; }
        if (ev_stage_gpu_[s]) { cudaEventDestroy(ev_stage_gpu_[s]); ev_stage_gpu_[s] = nullptr; }
    }
    fd(d_b_cur_); fd(d_b_inpSA_); fd(d_b_ffn_inp_);
    for (auto& p : d_b_blocks_) fd(p);
    fd(d_b_prelude_cur_);
    for (auto& p : d_b_bank_) fd(p);
    fd(d_b_bank_last_);
    fd(d_b_mixed_); fd(d_b_msg_attn_); fd(d_b_msg_kv_); fd(d_b_ctrl_out_);
    for (auto& p : d_b_normed_) fd(p);
    fd(d_b_loop_z_);
    fd(d_b_Q_); fd(d_b_K_); fd(d_b_K_conv_); fd(d_b_V_); fd(d_b_V_conv_);
    fd(d_b_dyn_k_); fd(d_b_dyn_v_); fd(d_b_R_); fd(d_b_sinks_); fd(d_b_rel_table_);
    fd(d_b_attn_out_); fd(d_b_attn_gate_); fd(d_b_wo_out_);
    fd(d_b_ffn_normed_); fd(d_b_moe_h_); fd(d_b_moe_out_); fd(d_b_moe_cpu_out_);
    fd(d_b_router_logits_); fd(d_b_top_ids_); fd(d_b_top_weights_);
    fd(d_grp_exp_ids_); fd(d_grp_tok_off_); fd(d_grp_tok_ids_); fd(d_grp_tok_w_);
    fd(d_b_logits_); fd(d_b_argmax_);

    fh(h_b_embd_); fh(h_b_ffn_in_); fh(h_b_moe_cpu_out_);
    fh(h_b_top_ids_); fh(h_b_top_weights_);
    fh(h_grp_exp_ids_); fh(h_grp_tok_off_); fh(h_grp_tok_ids_); fh(h_grp_tok_w_);
    fh(h_b_argmax_);
    fd(d_b_n_gpu_); fd(d_b_hit_ids_); fd(d_b_hit_weights_);
    fh(h_b_mapped_ffn_in_); d_b_mapped_ffn_in_ = nullptr;
    fh(h_b_split_pkt_);     d_b_split_pkt_ = nullptr;
    fh(h_b_zero_sync_ids_);
    batch_ready_ = false;
}

int32_t InklingEngine::forward_batch(
    const int32_t* tokens, int B, int pos0,
    bool need_last_logits, bool need_all_logits,
    bool is_answer, bool capture_last_pred_hidden,
    bool save_spec_states,
    int32_t* out_argmax_batch,
    const float* raw_embd)
{
    if (B <= 0) return -1;
    if (B == 1 && !save_spec_states && !need_all_logits && raw_embd == nullptr) {
        int32_t g = forward_token(tokens[0], pos0, need_last_logits, false, is_answer, capture_last_pred_hidden);
        if (out_argmax_batch) out_argmax_batch[0] = g;
        return g;
    }
    if (!batch_ready_) alloc_batch_buffers();
    if (B > BATCH_MAX) B = BATCH_MAX;

    const int D = hp_.n_embd;
    const int Qdim = hp_.n_head * hp_.head_dim;
    const int KVdim = hp_.n_head_kv * hp_.head_dim;
    const int K_exp = hp_.n_expert_used;
    const bool track_pred = predictor_active() && !save_spec_states;
    const bool record_batch_routes = predictor_active();
    const auto& pred_hid_layers = pred_main_.hidden_layers();
    const size_t spec_stride = kv_.conv_spec_stride();

    // R10 (SR3-05): Capture per-token routed IDs across all 48 layers so we replay decay+record
    // in exact scalar token order at batch end (and on speculative commit for accepted tokens!).
    batch_pred_B_ = 0;
    if (record_batch_routes) {
        batch_pred_ids_.resize((size_t)hp_.n_layer_logical * (size_t)B * (size_t)K_exp);
    }

    // 1. Input embeddings:
    //    - Text tokens: parallel CPU dequant -> H2D -> tok_norm (RMSNorm)
    //    - Raw multimodal vision embeddings: H2D straight into d_b_cur_, BYPASSING tok_norm! (matches inkling2 spec)
    if (raw_embd != nullptr) {
        INK_CUDA_CHECK(cudaMemcpyAsync(d_b_cur_, raw_embd, (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice, stream_));
    } else {
        cpu_pool_->parallel_for_threads([&](int tid, int nth) {
            for (int b = tid; b < B; b += nth) {
                const uint8_t* src = h_tok_embd_.data() + (size_t)tokens[b] * tok_embd_row_bytes_;
                CPUThreadPool::dequantize_row(tok_embd_type_, src, h_b_embd_ + (size_t)b * D, D);
            }
        });
        INK_CUDA_CHECK(cudaMemcpyAsync(d_b_cur_, h_b_embd_, (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice, stream_));
        ink_cuda_rms_norm_batch(d_b_cur_, (const float*)tok_norm_.ptr, d_b_cur_, B, D, hp_.rms_eps, stream_);
    }

    const int n_loop_core = (hp_.num_loops > 1) ? (hp_.core_end - hp_.core_start) : 0;
    int n_blocks = 0, prelude_n_blocks = 0, n_bank = 0;

    // Reusable per-expert bucket vectors for grouped hybrid MoE
    int bucket_cnt[128];
    int bucket_tok[128][BATCH_MAX];
    float bucket_w[128][BATCH_MAX];
    CPUExpertGroupTask cpu_groups[128];
    int32_t cpu_tok_ids[BATCH_MAX * 8];
    float   cpu_tok_w[BATCH_MAX * 8];
    bool    zero_sync_route_mask[48] = {};

    for (int il = 0; il < hp_.n_layer_logical; ++il) {
        const int ip = hp_.loop_layer(il);
        const auto& L = layers_[ip];

        if (capture_last_pred_hidden && h_pred_hidden_ && il == ip) {
            for (int hi = 0; hi < pred_hidden_slots_; ++hi) {
                if (pred_hid_layers[hi] == ip) {
                    INK_CUDA_CHECK(cudaMemcpyAsync(h_pred_hidden_ + (size_t)hi * D,
                                                   d_b_cur_ + (size_t)(B - 1) * D,
                                                   (size_t)D * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                }
            }
        }

        float* msg_ptr[4] = {nullptr, nullptr, nullptr, nullptr};
        if (n_loop_core > 0 && ip >= hp_.core_start && ip < hp_.core_end) {
            const int pass = (il - hp_.core_start) / n_loop_core;
            if (ip == hp_.core_start) {
                if (pass == 0) {
                    prelude_n_blocks = n_blocks;
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_b_prelude_cur_, d_b_cur_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                } else {
                    n_bank = 0;
                    for (int b = prelude_n_blocks + 1; b < n_blocks; ++b) {
                        INK_CUDA_CHECK(cudaMemcpyAsync(d_b_bank_[n_bank], d_b_blocks_[b], (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                        n_bank++;
                    }
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_b_bank_last_, d_b_cur_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                    n_blocks = prelude_n_blocks;
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_b_cur_, d_b_prelude_cur_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                }
            }
            if (pass > 0) {
                BlockPtrList bstates{};
                for (int b = 0; b < n_bank; ++b) bstates.ptrs[bstates.count++] = d_b_bank_[b];
                bstates.ptrs[bstates.count++] = d_b_bank_last_;
                const int row = 2 * (ip - hp_.core_start);
                ink_cuda_attn_res_mix_batch(bstates, (const float*)loop_res_query_.ptr    + (size_t)row * D, d_b_msg_attn_, B, D, 2, stream_);
                ink_cuda_attn_res_mix_batch(bstates, (const float*)loop_kv_res_query_.ptr + (size_t)row * D, d_b_msg_kv_,   B, D, 2, stream_);
                msg_ptr[0] = d_b_msg_attn_;
                msg_ptr[3] = d_b_msg_attn_ + (size_t)B * D;
                msg_ptr[1] = d_b_msg_kv_;
                msg_ptr[2] = d_b_msg_kv_ + (size_t)B * D;
            }
        }

        INK_CUDA_CHECK(cudaMemcpyAsync(d_b_inpSA_, d_b_cur_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

        float* inputs[3] = {d_b_cur_, d_b_cur_, d_b_cur_};
        if (L.attn_res_query.ptr != nullptr) {
            BlockPtrList states{};
            for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_b_blocks_[b];
            states.ptrs[states.count++] = d_b_cur_;
            int nq = L.is_swa ? 1 : 3;
            ink_cuda_attn_res_mix_batch(states, (const float*)L.attn_res_query.ptr, d_b_mixed_, B, D, nq, stream_);
            if (L.is_swa) {
                inputs[0] = inputs[1] = inputs[2] = d_b_mixed_;
            } else {
                inputs[0] = d_b_mixed_;
                inputs[1] = d_b_mixed_ + (size_t)B * D;
                inputs[2] = d_b_mixed_ + (size_t)2 * B * D;
            }
        }

        if (msg_ptr[0] != nullptr) {
            for (int j = 0; j < 3; ++j) {
                float* dst = d_b_ctrl_out_ + (size_t)j * B * D;
                ink_cuda_loop_control_batch(inputs[j], msg_ptr[j], L.loop_down[j].type,
                                            L.loop_down[j].ptr, L.loop_gain[j].ptr, L.loop_forget[j].ptr,
                                            d_b_loop_z_, dst, B, D, hp_.loop_gate_rank, stream_);
                inputs[j] = dst;
            }
        }

        if (L.is_swa && msg_ptr[0] != nullptr) {
            for (int j = 0; j < 3; ++j) {
                ink_cuda_rms_norm_batch(inputs[j], (const float*)L.attn_norm.ptr, d_b_normed_[j], B, D, hp_.rms_eps, stream_);
                inputs[j] = d_b_normed_[j];
            }
        } else if (L.is_swa) {
            ink_cuda_rms_norm_batch(inputs[0], (const float*)L.attn_norm.ptr, d_b_normed_[0], B, D, hp_.rms_eps, stream_);
            inputs[0] = inputs[1] = inputs[2] = d_b_normed_[0];
        } else {
            ink_cuda_rms_norm_batch(inputs[0], (const float*)L.attn_q_inp_norm.ptr, d_b_normed_[0], B, D, hp_.rms_eps, stream_);
            ink_cuda_rms_norm_batch(inputs[1], (const float*)L.attn_k_inp_norm.ptr, d_b_normed_[1], B, D, hp_.rms_eps, stream_);
            ink_cuda_rms_norm_batch(inputs[2], (const float*)L.attn_v_inp_norm.ptr, d_b_normed_[2], B, D, hp_.rms_eps, stream_);
            inputs[0] = d_b_normed_[0]; inputs[1] = d_b_normed_[1]; inputs[2] = d_b_normed_[2];
        }

        // Batched GEMM dense attention projections (8x less GDDR6 weight traffic!)
        ink_cuda_gemm_batch(L.wq.type, L.wq.ptr, inputs[0], d_b_Q_, B, Qdim, D, stream_);
        ink_cuda_gemm_batch(L.wqkv_gate.type, L.wqkv_gate.ptr, inputs[0], d_b_attn_gate_, B, Qdim, D, stream_);
        ink_cuda_gemm_batch(L.wk.type, L.wk.ptr, inputs[1], d_b_K_, B, KVdim, D, stream_);
        ink_cuda_gemm_batch(L.wv.type, L.wv.ptr, inputs[2], d_b_V_, B, KVdim, D, stream_);
        ink_cuda_gemm_batch(L.attn_rel.type, L.attn_rel.ptr, inputs[0], d_b_R_, B, hp_.n_head * hp_.d_rel, D, stream_);
        ink_cuda_gemm_batch(L.attn_sink_proj.type, L.attn_sink_proj.ptr, inputs[0], d_b_sinks_, B, hp_.n_head, D, stream_);

        float* dyn_k = nullptr;
        float* dyn_v = nullptr;
        if (!L.is_swa) {
            ink_cuda_gemm_batch(L.attn_k_conv_proj.type, L.attn_k_conv_proj.ptr, inputs[1], d_b_dyn_k_, B, hp_.sconv_k * hp_.n_head_kv, D, stream_);
            ink_cuda_gemm_batch(L.attn_v_conv_proj.type, L.attn_v_conv_proj.ptr, inputs[2], d_b_dyn_v_, B, hp_.sconv_k * hp_.n_head_kv, D, stream_);
            dyn_k = d_b_dyn_k_;
            dyn_v = d_b_dyn_v_;
        }
        ink_cuda_shortconv_batch(d_b_K_, kv_.conv_state_ptr(il, 0),     (const float*)L.attn_k_conv.ptr, dyn_k, d_b_K_conv_, B, KVdim, hp_.n_head_kv,
                                 save_spec_states ? kv_.conv_spec_ptr(il, 0) : nullptr, spec_stride, stream_);
        ink_cuda_shortconv_batch(d_b_V_, kv_.conv_state_ptr(il, KVdim), (const float*)L.attn_v_conv.ptr, dyn_v, d_b_V_conv_, B, KVdim, hp_.n_head_kv,
                                 save_spec_states ? kv_.conv_spec_ptr(il, KVdim) : nullptr, spec_stride, stream_);

        ink_cuda_qkv_post_batch(d_b_Q_, (const float*)L.attn_q_clip.ptr, d_b_K_conv_, d_b_V_conv_,
                                d_b_R_, (const float*)L.attn_rel_clip.ptr,
                                B, hp_.n_head, hp_.n_head_kv, hp_.head_dim, hp_.d_rel,
                                hp_.qk_clip, hp_.rel_clip, hp_.rms_eps,
                                L.is_swa, pos0, hp_.log_scale_floor, hp_.log_scale_alpha, stream_);

        int rel_ext = L.is_swa ? hp_.swa_window : hp_.rel_extent;
        ink_cuda_rel_table_batch((const float*)L.attn_rel_bias.ptr, d_b_R_, d_b_rel_table_, B, hp_.n_head, hp_.d_rel, rel_ext, stream_);

        if (!kv_.append_and_attend_batch(il, pos0, B, d_b_Q_, d_b_K_conv_, d_b_V_conv_,
                                         d_b_rel_table_, d_b_sinks_, d_b_attn_out_, rel_ext, stream_)) {
            record_io_error(kv_.last_error());
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            return -1;
        }
        if (il == 0 && g_l0_attn_capture) {
            g_l0_attn_capture->resize(Qdim);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            INK_CUDA_CHECK(cudaMemcpy(g_l0_attn_capture->data(), d_b_attn_out_, (size_t)Qdim * sizeof(float), cudaMemcpyDeviceToHost));
        }

        ink_cuda_gate_sigmoid_mul(d_b_attn_out_, d_b_attn_gate_, B * Qdim, stream_);
        ink_cuda_gemm_batch(L.wo.type, L.wo.ptr, d_b_attn_out_, d_b_wo_out_, B, D, Qdim, stream_);
        ink_cuda_shortconv_batch(d_b_wo_out_, kv_.conv_state_ptr(il, 2 * KVdim), (const float*)L.attn_post_conv.ptr, nullptr, d_b_cur_, B, D, 0,
                                 save_spec_states ? kv_.conv_spec_ptr(il, 2 * KVdim) : nullptr, spec_stride, stream_);

        if (ip % hp_.attn_res_block_size == 0) {
            INK_CUDA_CHECK(cudaMemcpyAsync(d_b_blocks_[n_blocks], d_b_inpSA_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            n_blocks++;
        } else {
            ink_cuda_add_inplace(d_b_cur_, d_b_inpSA_, B * D, stream_);
        }

        INK_CUDA_CHECK(cudaMemcpyAsync(d_b_ffn_inp_, d_b_cur_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        {
            BlockPtrList states{};
            for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_b_blocks_[b];
            states.ptrs[states.count++] = d_b_cur_;
            ink_cuda_attn_res_mix_batch(states, (const float*)L.ffn_res_query.ptr, d_b_mixed_, B, D, 1, stream_);
            float* ffn_x = d_b_mixed_;
            if (msg_ptr[3] != nullptr) {
                ink_cuda_loop_control_batch(ffn_x, msg_ptr[3], L.loop_down[3].type,
                                            L.loop_down[3].ptr, L.loop_gain[3].ptr, L.loop_forget[3].ptr,
                                            d_b_loop_z_, d_b_ctrl_out_, B, D, hp_.loop_gate_rank, stream_);
                ffn_x = d_b_ctrl_out_;
            }
            ink_cuda_rms_norm_batch(ffn_x, (const float*)L.ffn_norm.ptr, d_b_ffn_normed_, B, D, hp_.rms_eps, stream_);
        }

        ink_cuda_moe_router_batch((const float*)L.ffn_gate_inp.ptr, (const float*)L.ffn_exp_probs_b.ptr,
                                  d_b_ffn_normed_, d_b_router_logits_, d_b_top_ids_, d_b_top_weights_,
                                  B, hp_.n_expert, D, K_exp, hp_.expert_weights_scale, stream_);

        // Grouped Batched Hybrid MoE (VRAM Tensor Cores + PCIe-Streamed Cold Tensor Cores OR Overlapped CPU AVX2!)
        const auto& lm = experts_.layer_meta(ip);
        const auto& tbl = experts_.layer_device_table(ip);
        const bool all_vram = experts_.is_layer_all_vram(ip);
        const bool may_stream_pcie = (B >= 16 && lm.combined_bytes <= STAGE_SLOT_BYTES && d_prefill_stage_[0] != nullptr);
        const bool use_spec_indirect = (B <= HybridPagedKV::SPEC_MAX_B && !may_stream_pcie && d_b_split_pkt_ != nullptr);

        if (use_spec_indirect) {
            // R13: Zero-Sync (all_vram) & Spin-Poll Zero-Bubble (hybrid) Indirect Batch MoE for Speculative Verify!
            // Eliminates 48x cudaStreamSynchronize stalls, CPU bucketing of VRAM hits, H2D group uploads, and atomicAdd!
            if (all_vram) {
                ink_cuda_moe_pinned_indirect_batch(lm.gate_up_type, lm.down_type, tbl,
                                                   d_b_top_ids_, d_b_top_weights_, B, K_exp,
                                                   d_b_ffn_normed_, d_b_moe_h_, d_b_moe_out_,
                                                   D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
                experts_.record_batch_hit_stats(ip, B * K_exp, 0);
                if (record_batch_routes && h_b_zero_sync_ids_) {
                    INK_CUDA_CHECK(cudaMemcpyAsync(
                        h_b_zero_sync_ids_ + (size_t)il * (size_t)B * (size_t)K_exp,
                        d_b_top_ids_, (size_t)B * (size_t)K_exp * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream_));
                    zero_sync_route_mask[il] = true;
                }
            } else {
                uint32_t seq = (++batch_split_seq_ == 0) ? ++batch_split_seq_ : batch_split_seq_;
                ink_cuda_moe_split_hits_batch(tbl, d_b_top_ids_, d_b_top_weights_, B, K_exp,
                                              d_b_ffn_normed_, D,
                                              d_b_n_gpu_, d_b_hit_ids_, d_b_hit_weights_,
                                              d_b_split_pkt_, d_b_mapped_ffn_in_, seq, stream_);
                ink_cuda_moe_dynamic_indirect_batch(lm.gate_up_type, lm.down_type, tbl,
                                                    d_b_n_gpu_, d_b_hit_ids_, d_b_hit_weights_, B, K_exp,
                                                    d_b_ffn_normed_, d_b_moe_h_, d_b_moe_out_,
                                                    D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);

                wait_batch_split_packet(seq);

                if (record_batch_routes) {
                    std::memcpy(batch_pred_ids_.data() + (size_t)il * (size_t)B * (size_t)K_exp,
                                h_b_split_pkt_->all_ids, (size_t)B * (size_t)K_exp * sizeof(int32_t));
                }
                const int nc_tot = h_b_split_pkt_->n_cpu_tot;
                experts_.record_batch_hit_stats(ip, h_b_split_pkt_->n_gpu_tot, nc_tot);

                if (nc_tot > 0) {
                    std::memset(bucket_cnt, 0, sizeof(bucket_cnt));
                    for (int m = 0; m < nc_tot; ++m) {
                        int e = (int)h_b_split_pkt_->miss_exp[m];
                        int c = bucket_cnt[e]++;
                        bucket_tok[e][c] = (int)h_b_split_pkt_->miss_tok[m];
                        bucket_w[e][c]   = h_b_split_pkt_->miss_w[m];
                    }
                    int n_cpu_grp = 0, cpu_pairs = 0;
                    int cpu_uids[128];
                    for (int e = 0; e < hp_.n_expert; ++e) {
                        int cnt = bucket_cnt[e];
                        if (cnt == 0) continue;
                        cpu_uids[n_cpu_grp] = ip * hp_.n_expert + e;
                        CPUExpertGroupTask& cg = cpu_groups[n_cpu_grp++];
                        cg.gate_up_data  = nullptr;
                        cg.down_data     = nullptr;
                        cg.gate_up_type  = lm.gate_up_type;
                        cg.down_type     = lm.down_type;
                        cg.gate_up_scale = lm.gate_up_scales[e];
                        cg.down_scale    = lm.down_scales[e];
                        cg.tok_offset    = cpu_pairs;
                        cg.tok_count     = cnt;
                        for (int t = 0; t < cnt; ++t) {
                            cpu_tok_ids[cpu_pairs] = bucket_tok[e][t];
                            cpu_tok_w[cpu_pairs]   = bucket_w[e][t];
                            cpu_pairs++;
                        }
                    }

                    int pin_slots[128];
                    std::vector<float> split_acc;
                    bool used_split = false;
                    int base_g = 0;
                    while (base_g < n_cpu_grp) {
                        int cnt_g = 0;
                        for (int k = base_g; k < n_cpu_grp; ++k) {
                            int rslot = -1;
                            const uint8_t* rptr = experts_.pin_ram_expert(cpu_uids[k], &rslot);
                            if (!rptr && cnt_g == 0) {
                                experts_.finish_phased_swaps(stream_);
                                rptr = experts_.pin_ram_expert(cpu_uids[k], &rslot);
                            }
                            if (!rptr) break;
                            cpu_groups[k].gate_up_data = rptr;
                            cpu_groups[k].down_data    = rptr + lm.down_aligned_off;
                            pin_slots[cnt_g++] = rslot;
                        }
                        if (cnt_g <= 0) {
                            record_io_error(experts_.has_io_error()
                                ? experts_.last_error()
                                : ("Unable to pin cold expert uid=" + std::to_string(cpu_uids[base_g]) + " for spec CPU MoE"));
                            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                            return -1;
                        }
                        cpu_pool_->run_moe_experts_grouped(h_b_mapped_ffn_in_, h_b_moe_cpu_out_, B,
                                                           cpu_groups + base_g, cnt_g, cpu_tok_ids, cpu_tok_w, cpu_pairs,
                                                           D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta);
                        experts_.unpin_ram_slots(pin_slots, cnt_g);

                        if (base_g == 0 && cnt_g == n_cpu_grp) break;
                        if (!used_split) {
                            split_acc.assign((size_t)B * D, 0.0f);
                            used_split = true;
                        }
                        for (size_t idx = 0; idx < (size_t)B * D; ++idx) split_acc[idx] += h_b_moe_cpu_out_[idx];
                        base_g += cnt_g;
                    }
                    if (used_split) {
                        std::memcpy(h_b_moe_cpu_out_, split_acc.data(), (size_t)B * D * sizeof(float));
                    }
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_b_moe_cpu_out_, h_b_moe_cpu_out_, (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice, stream_));
                    ink_cuda_add_inplace(d_b_moe_out_, d_b_moe_cpu_out_, B * D, stream_);
                }
            }
        } else {
        INK_CUDA_CHECK(cudaMemcpyAsync(h_b_top_ids_,     d_b_top_ids_,     (size_t)B * K_exp * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(h_b_top_weights_, d_b_top_weights_, (size_t)B * K_exp * sizeof(float),   cudaMemcpyDeviceToHost, stream_));
        if (!all_vram && !may_stream_pcie) {
            INK_CUDA_CHECK(cudaMemcpyAsync(h_b_ffn_in_, d_b_ffn_normed_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        }
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (record_batch_routes) {
            std::memcpy(batch_pred_ids_.data() + (size_t)il * (size_t)B * (size_t)K_exp,
                        h_b_top_ids_, (size_t)B * (size_t)K_exp * sizeof(int32_t));
        }

        std::memset(bucket_cnt, 0, sizeof(bucket_cnt));
        for (int b = 0; b < B; ++b) {
            const int32_t* bids = h_b_top_ids_ + (size_t)b * K_exp;
            const float*   bw   = h_b_top_weights_ + (size_t)b * K_exp;
            for (int k = 0; k < K_exp; ++k) {
                int e = bids[k];
                int c = bucket_cnt[e]++;
                bucket_tok[e][c] = b;
                bucket_w[e][c]   = bw[k];
            }
        }

        int n_gpu_grp = 0, gpu_pairs = 0;
        int n_cpu_grp = 0, cpu_pairs = 0;
        int cpu_uids[128];
        h_grp_tok_off_[0] = 0;
        for (int e = 0; e < hp_.n_expert; ++e) {
            int cnt = bucket_cnt[e];
            if (cnt == 0) continue;
            int uid = ip * hp_.n_expert + e;
            if (experts_.is_in_vram(uid)) {
                h_grp_exp_ids_[n_gpu_grp] = e;
                for (int t = 0; t < cnt; ++t) {
                    h_grp_tok_ids_[gpu_pairs] = bucket_tok[e][t];
                    h_grp_tok_w_[gpu_pairs]   = bucket_w[e][t];
                    gpu_pairs++;
                }
                n_gpu_grp++;
                h_grp_tok_off_[n_gpu_grp] = gpu_pairs;
            } else {
                // Defer RAM pinning until each micro-batch is consumed so tight RAM can NEVER evict an expert before read!
                cpu_uids[n_cpu_grp] = uid;
                CPUExpertGroupTask& cg = cpu_groups[n_cpu_grp++];
                cg.gate_up_data  = nullptr;
                cg.down_data     = nullptr;
                cg.gate_up_type  = lm.gate_up_type;
                cg.down_type     = lm.down_type;
                cg.gate_up_scale = lm.gate_up_scales[e];
                cg.down_scale    = lm.down_scales[e];
                cg.tok_offset    = cpu_pairs;
                cg.tok_count     = cnt;
                for (int t = 0; t < cnt; ++t) {
                    cpu_tok_ids[cpu_pairs] = bucket_tok[e][t];
                    cpu_tok_w[cpu_pairs]   = bucket_w[e][t];
                    cpu_pairs++;
                }
            }
        }
        experts_.record_batch_hit_stats(ip, gpu_pairs, cpu_pairs);

        const bool use_pcie_stream = may_stream_pcie && (n_cpu_grp >= 1);
        if (!use_pcie_stream && n_cpu_grp > 0 && may_stream_pcie) {
            INK_CUDA_CHECK(cudaMemcpyAsync(h_b_ffn_in_, d_b_ffn_normed_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        if (n_gpu_grp > 0) {
            INK_CUDA_CHECK(cudaMemcpyAsync(d_grp_exp_ids_, h_grp_exp_ids_, (size_t)n_gpu_grp * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_grp_tok_off_, h_grp_tok_off_, (size_t)(n_gpu_grp + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_grp_tok_ids_, h_grp_tok_ids_, (size_t)gpu_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_grp_tok_w_,   h_grp_tok_w_,   (size_t)gpu_pairs * sizeof(float),   cudaMemcpyHostToDevice, stream_));
        }
        ink_cuda_moe_grouped_gpu(lm.gate_up_type, lm.down_type, tbl,
                                 d_grp_exp_ids_, d_grp_tok_off_, d_grp_tok_ids_, d_grp_tok_w_,
                                 n_gpu_grp, gpu_pairs, B, d_b_ffn_normed_, d_b_moe_h_, d_b_moe_out_,
                                 D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta,
                                 /*zero_out=*/true, /*preconvert_x=*/true, stream_);

        if (n_cpu_grp > 0 && use_pcie_stream) {
            // Double-Buffered Layer-Wise PCIe Streaming -> 100% RTX 4060 Tensor Core MoE (R15 Direct Pinned DMA!)
            cudaStream_t s_dma = experts_.dma_stream();
            const size_t exp_bytes = lm.combined_bytes;
            bool x_ready = (n_gpu_grp > 0);
            int buf_pin_slots[STAGE_BUFS][STAGE_EXPS];
            int buf_pin_cnt[STAGE_BUFS] = {0, 0};
            auto release_buf_pins = [&](int b) {
                if (buf_pin_cnt[b] > 0) {
                    experts_.unpin_ram_slots(buf_pin_slots[b], buf_pin_cnt[b]);
                    buf_pin_cnt[b] = 0;
                }
            };
            int batch_idx = 0;
            for (int base_g = 0; base_g < n_cpu_grp; ++batch_idx) {
                int buf = batch_idx & 1;
                int want_g = std::min(STAGE_EXPS, n_cpu_grp - base_g);
                if (batch_idx >= STAGE_BUFS) {
                    INK_CUDA_CHECK(cudaEventSynchronize(ev_stage_gpu_[buf]));
                    release_buf_pins(buf);
                }

                // Pin ONLY this micro-batch (<= 4 experts) in RAM until DMA (direct) or bounce memcpy completes!
                int pin_slots[STAGE_EXPS];
                int cnt_g = 0;
                for (int k = 0; k < want_g; ++k) {
                    int rslot = -1;
                    const uint8_t* rptr = experts_.pin_ram_expert(cpu_uids[base_g + k], &rslot);
                    if (!rptr && cnt_g == 0) {
                        // Drain any in-flight direct-DMA buffers first under ultra-tight RAM!
                        for (int b = 0; b < STAGE_BUFS; ++b) {
                            if (buf_pin_cnt[b] > 0) {
                                INK_CUDA_CHECK(cudaEventSynchronize(ev_stage_dma_[b]));
                                release_buf_pins(b);
                            }
                        }
                        experts_.finish_phased_swaps(stream_);
                        rptr = experts_.pin_ram_expert(cpu_uids[base_g + k], &rslot);
                    }
                    if (!rptr) break; // Every fitting RAM slot is already pinned; drain this partial micro-batch!
                    cpu_groups[base_g + k].gate_up_data = rptr;
                    cpu_groups[base_g + k].down_data    = rptr + lm.down_aligned_off;
                    pin_slots[cnt_g++] = rslot;
                }
                if (cnt_g <= 0) {
                    int used_bufs = std::min(batch_idx, STAGE_BUFS);
                    for (int b = 0; b < used_bufs; ++b) {
                        INK_CUDA_CHECK(cudaEventSynchronize(ev_stage_gpu_[b]));
                        release_buf_pins(b);
                    }
                    record_io_error(experts_.has_io_error()
                        ? experts_.last_error()
                        : ("Unable to pin cold expert uid=" + std::to_string(cpu_uids[base_g]) + " in RAM"));
                    return -1;
                }

                // R15: Check if all experts in this micro-batch reside in cudaHostRegister'd Tier-1 RAM!
                bool direct_dma = true;
                for (int k = 0; k < cnt_g; ++k) {
                    if (!experts_.is_ram_ptr_pinned_dma((const uint8_t*)cpu_groups[base_g + k].gate_up_data, exp_bytes)) {
                        direct_dma = false;
                        break;
                    }
                }

                if (direct_dma) {
                    // ZERO-COPY FAST PATH: keep RAM slots pinned until ev_stage_dma_/ev_stage_gpu_ finishes!
                    for (int k = 0; k < cnt_g; ++k) buf_pin_slots[buf][k] = pin_slots[k];
                    buf_pin_cnt[buf] = cnt_g;
                } else {
                    // Fallback bounce memcpy into pinned staging buffer for unregistered RAM slots
                    cpu_pool_->parallel_for_threads([&](int tid, int nth) {
                        size_t total_b = (size_t)cnt_g * exp_bytes;
                        size_t chunk_b = (total_b + (size_t)nth - 1) / (size_t)nth;
                        size_t lo = (size_t)tid * chunk_b;
                        size_t hi = std::min(total_b, lo + chunk_b);
                        while (lo < hi) {
                            int k = (int)(lo / exp_bytes);
                            size_t off_in_e = lo - (size_t)k * exp_bytes;
                            size_t take = std::min(hi - lo, exp_bytes - off_in_e);
                            std::memcpy(h_prefill_stage_[buf] + (size_t)k * exp_bytes + off_in_e,
                                        (const uint8_t*)cpu_groups[base_g + k].gate_up_data + off_in_e, take);
                            lo += take;
                        }
                    });
                    experts_.unpin_ram_slots(pin_slots, cnt_g);
                }

                int st_pairs = 0;
                h_stage_tok_off_[buf][0] = 0;
                for (int k = 0; k < cnt_g; ++k) {
                    const auto& cg = cpu_groups[base_g + k];
                    uint8_t* vbase = d_prefill_stage_[buf] + (size_t)k * exp_bytes;
                    h_stage_gu_ptrs_[buf][k] = (uint64_t)(uintptr_t)vbase;
                    h_stage_dn_ptrs_[buf][k] = (uint64_t)(uintptr_t)(vbase + lm.down_aligned_off);
                    h_stage_gu_sc_[buf][k]   = cg.gate_up_scale;
                    h_stage_dn_sc_[buf][k]   = cg.down_scale;
                    for (int t = 0; t < cg.tok_count; ++t) {
                        h_stage_tok_ids_[buf][st_pairs] = cpu_tok_ids[cg.tok_offset + t];
                        h_stage_tok_w_[buf][st_pairs]   = cpu_tok_w[cg.tok_offset + t];
                        st_pairs++;
                    }
                    h_stage_tok_off_[buf][k + 1] = st_pairs;
                }

                // Async PCIe Gen4 H2D DMA overlapped with previous micro-batch's Tensor Core MoE!
                if (direct_dma) {
                    for (int k = 0; k < cnt_g; ++k) {
                        INK_CUDA_CHECK(cudaMemcpyAsync(
                            d_prefill_stage_[buf] + (size_t)k * exp_bytes,
                            cpu_groups[base_g + k].gate_up_data,
                            exp_bytes, cudaMemcpyHostToDevice, s_dma));
                    }
                } else {
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_prefill_stage_[buf], h_prefill_stage_[buf], (size_t)cnt_g * exp_bytes, cudaMemcpyHostToDevice, s_dma));
                }
                INK_CUDA_CHECK(cudaMemcpyAsync(stage_tbl_[buf].d_gu_ptrs,   h_stage_gu_ptrs_[buf], (size_t)cnt_g * sizeof(uint64_t), cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaMemcpyAsync(stage_tbl_[buf].d_dn_ptrs,   h_stage_dn_ptrs_[buf], (size_t)cnt_g * sizeof(uint64_t), cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaMemcpyAsync(stage_tbl_[buf].d_gu_scales, h_stage_gu_sc_[buf],   (size_t)cnt_g * sizeof(float),    cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaMemcpyAsync(stage_tbl_[buf].d_dn_scales, h_stage_dn_sc_[buf],   (size_t)cnt_g * sizeof(float),    cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaMemcpyAsync(d_stage_tok_off_[buf], h_stage_tok_off_[buf], (size_t)(cnt_g + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaMemcpyAsync(d_stage_tok_ids_[buf], h_stage_tok_ids_[buf], (size_t)st_pairs * sizeof(int32_t),    cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaMemcpyAsync(d_stage_tok_w_[buf],   h_stage_tok_w_[buf],   (size_t)st_pairs * sizeof(float),      cudaMemcpyHostToDevice, s_dma));
                INK_CUDA_CHECK(cudaEventRecord(ev_stage_dma_[buf], s_dma));

                INK_CUDA_CHECK(cudaStreamWaitEvent(stream_, ev_stage_dma_[buf], 0));
                ink_cuda_moe_grouped_gpu(lm.gate_up_type, lm.down_type, stage_tbl_[buf],
                                         d_stage_exp_ids_[buf], d_stage_tok_off_[buf], d_stage_tok_ids_[buf], d_stage_tok_w_[buf],
                                         cnt_g, st_pairs, B, d_b_ffn_normed_, d_b_moe_h_, d_b_moe_out_,
                                         D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta,
                                         /*zero_out=*/false, /*preconvert_x=*/!x_ready, stream_);
                x_ready = true;
                INK_CUDA_CHECK(cudaEventRecord(ev_stage_gpu_[buf], stream_));
                cudaStreamQuery(s_dma);
                cudaStreamQuery(stream_);
                base_g += cnt_g;
            }
            // Ensure staging buffers & direct-DMA RAM pins are done before next layer reuses them
            int used_bufs = std::min(batch_idx, STAGE_BUFS);
            for (int b = 0; b < used_bufs; ++b) {
                INK_CUDA_CHECK(cudaEventSynchronize(ev_stage_gpu_[b]));
                release_buf_pins(b);
            }
        } else if (n_cpu_grp > 0) {
            // Ultra-low-latency CPU AVX2 overlap for tiny batches (e.g. Speculative Verification B <= 7!)
            // Pin experts in RAM slots before reading; if RAM is super tight, split into safe sub-batches!
            cudaStreamQuery(stream_);
            int pin_slots[128];
            std::vector<float> split_acc;
            bool used_split = false;
            int base_g = 0;
            while (base_g < n_cpu_grp) {
                int cnt_g = 0;
                for (int k = base_g; k < n_cpu_grp; ++k) {
                    int rslot = -1;
                    const uint8_t* rptr = experts_.pin_ram_expert(cpu_uids[k], &rslot);
                    if (!rptr && cnt_g == 0) {
                        experts_.finish_phased_swaps(stream_);
                        rptr = experts_.pin_ram_expert(cpu_uids[k], &rslot);
                    }
                    if (!rptr) break;
                    cpu_groups[k].gate_up_data = rptr;
                    cpu_groups[k].down_data    = rptr + lm.down_aligned_off;
                    pin_slots[cnt_g++] = rslot;
                }
                if (cnt_g <= 0) {
                    record_io_error(experts_.has_io_error()
                        ? experts_.last_error()
                        : ("Unable to pin cold expert uid=" + std::to_string(cpu_uids[base_g]) + " for CPU grouped MoE"));
                    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                    return -1;
                }
                cpu_pool_->run_moe_experts_grouped(h_b_ffn_in_, h_b_moe_cpu_out_, B,
                                                   cpu_groups + base_g, cnt_g, cpu_tok_ids, cpu_tok_w, cpu_pairs,
                                                   D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta);
                experts_.unpin_ram_slots(pin_slots, cnt_g);

                if (base_g == 0 && cnt_g == n_cpu_grp) {
                    break; // Fast single-pass path (99.99% of runs!)
                }
                if (!used_split) {
                    split_acc.assign((size_t)B * D, 0.0f);
                    used_split = true;
                }
                for (size_t idx = 0; idx < (size_t)B * D; ++idx) split_acc[idx] += h_b_moe_cpu_out_[idx];
                base_g += cnt_g;
            }
            if (used_split) {
                std::memcpy(h_b_moe_cpu_out_, split_acc.data(), (size_t)B * D * sizeof(float));
            }
            INK_CUDA_CHECK(cudaMemcpyAsync(d_b_moe_cpu_out_, h_b_moe_cpu_out_, (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice, stream_));
            ink_cuda_add_inplace(d_b_moe_out_, d_b_moe_cpu_out_, B * D, stream_);
        }
        } // end else (!use_spec_indirect)

        ink_cuda_shortconv_add_batch(d_b_moe_out_, kv_.conv_state_ptr(il, 2 * KVdim + D),
                                     (const float*)L.ffn_post_conv.ptr, d_b_ffn_inp_, d_b_cur_, B, D,
                                     save_spec_states ? kv_.conv_spec_ptr(il, 2 * KVdim + D) : nullptr, spec_stride, stream_);
    }

    int32_t greedy = -1;
    if (need_all_logits) {
        BlockPtrList states{};
        for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_b_blocks_[b];
        states.ptrs[states.count++] = d_b_cur_;
        ink_cuda_attn_res_mix_batch(states, (const float*)output_res_query_.ptr, d_b_mixed_, B, D, 1, stream_);
        ink_cuda_rms_norm_batch(d_b_mixed_, (const float*)output_norm_.ptr, d_b_cur_, B, D, hp_.rms_eps, stream_);
        ink_cuda_gemm_batch(output_.type, output_.ptr, d_b_cur_, d_b_logits_, B, hp_.vocab_unpadded, D, stream_);
        ink_cuda_logit_softcap_argmax_batch(d_b_logits_, B, hp_.vocab_unpadded, hp_.final_logit_softcap, d_b_argmax_, stream_);
        INK_CUDA_CHECK(cudaMemcpyAsync(h_b_argmax_, d_b_argmax_, (size_t)B * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (out_argmax_batch) {
            for (int b = 0; b < B; ++b) out_argmax_batch[b] = h_b_argmax_[b];
        }
        greedy = h_b_argmax_[B - 1];
        *h_argmax_ = greedy;
    } else if (need_last_logits) {
        BlockPtrList states{};
        for (int b = 0; b < n_blocks; ++b) states.ptrs[states.count++] = d_b_blocks_[b] + (size_t)(B - 1) * D;
        states.ptrs[states.count++] = d_b_cur_ + (size_t)(B - 1) * D;
        ink_cuda_attn_res_mix(states, (const float*)output_res_query_.ptr, d_mixed_, D, 1, stream_);
        ink_cuda_rms_norm(d_mixed_, (const float*)output_norm_.ptr, d_cur_, D, hp_.rms_eps, stream_);
        ink_cuda_gemv(output_.type, output_.ptr, d_cur_, d_logits_, hp_.vocab_unpadded, D, stream_);
        ink_cuda_logit_softcap_argmax(d_logits_, hp_.vocab_unpadded, hp_.final_logit_softcap, d_argmax_, stream_);
        INK_CUDA_CHECK(cudaMemcpyAsync(h_argmax_, d_argmax_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        greedy = *h_argmax_;
        if (out_argmax_batch) out_argmax_batch[B - 1] = greedy;
    } else {
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // R10 (SR3-05): Replay batched routes in exact scalar token order (decay -> record per token!)
    if (record_batch_routes) {
        if (h_b_zero_sync_ids_) {
            for (int il = 0; il < hp_.n_layer_logical; ++il) {
                if (zero_sync_route_mask[il]) {
                    std::memcpy(batch_pred_ids_.data() + (size_t)il * (size_t)B * (size_t)K_exp,
                                h_b_zero_sync_ids_ + (size_t)il * (size_t)B * (size_t)K_exp,
                                (size_t)B * (size_t)K_exp * sizeof(int32_t));
                }
            }
        }
        batch_pred_B_ = B;
        if (track_pred) {
            for (int b = 0; b < B; ++b) {
                if (is_answer) pred_tracker_.step_answer_token();
                else           pred_tracker_.step_prompt_token();
                for (int il = 0; il < hp_.n_layer_logical; ++il) {
                    int ip = hp_.loop_layer(il);
                    pred_tracker_.record_selections(
                        ip, batch_pred_ids_.data() + ((size_t)il * (size_t)B + (size_t)b) * (size_t)K_exp,
                        K_exp, is_answer);
                }
            }
        }
    }

    // Drain any queued predictor swaps cleanly at chunk boundary
    experts_.begin_token_end_swaps(stream_, 4);
    experts_.finish_token_end_swaps(stream_, true);
    return greedy;
}

int32_t InklingEngine::prefill(const std::vector<int32_t>& tokens, int start_pos, double* out_ms) {
    double t0 = now_ms();
    int32_t last_greedy = -1;
    int P = (int)tokens.size();
    if (cfg_.use_batched_prefill && P > 1) {
        int i = 0;
        while (i < P) {
            int chunk = std::min(BATCH_MAX, P - i);
            bool is_last = (i + chunk == P);
            last_greedy = forward_batch(tokens.data() + i, chunk, start_pos + i, is_last, false, false, false, false);
            i += chunk;
        }
    } else {
        for (int i = 0; i < P; ++i) {
            bool is_last = (i + 1 == P);
            last_greedy = forward_token(tokens[i], start_pos + i, is_last, false);
        }
    }
    if (out_ms) *out_ms = now_ms() - t0;
    return last_greedy;
}

void InklingEngine::copy_logits_to_host(std::vector<float>& dst) const {
    dst.resize(hp_.vocab_unpadded);
    INK_CUDA_CHECK(cudaMemcpy(dst.data(), d_logits_, (size_t)hp_.vocab_unpadded * sizeof(float), cudaMemcpyDeviceToHost));
}

void InklingEngine::copy_batch_logits_to_host(int b, std::vector<float>& dst) const {
    if (b < 0 || b >= HybridPagedKV::SPEC_MAX_B || !d_b_logits_) { dst.clear(); return; }
    dst.resize(hp_.vocab_unpadded);
    INK_CUDA_CHECK(cudaMemcpy(dst.data(), d_b_logits_ + (size_t)b * hp_.vocab_unpadded,
                              (size_t)hp_.vocab_unpadded * sizeof(float), cudaMemcpyDeviceToHost));
}

void InklingEngine::clear_io_errors() {
    io_error_ = false;
    last_error_.clear();
    kv_.clear_io_error();
    experts_.clear_io_error();
}

void InklingEngine::record_io_error(const std::string& msg) {
    io_error_ = true;
    if (last_error_.empty()) last_error_ = msg.empty() ? "Engine storage I/O error" : msg;
}

namespace {
int32_t finish_sampling_candidates(
    std::vector<std::pair<float, int32_t>>& cand,
    float max_l, int32_t argmax_val, const SamplingParams& sp,
    uint64_t& rng_state, int32_t draft_tok, bool* out_accepted)
{
    if (cand.empty()) {
        if (out_accepted) *out_accepted = (draft_tok >= 0 && argmax_val == draft_tok);
        return argmax_val;
    }

    if (sp.top_k > 0 || sp.top_p < 1.0f) {
        int keep = (sp.top_k > 0 && (int)cand.size() > sp.top_k) ? sp.top_k : (int)cand.size();
        if (keep < (int)cand.size()) {
            std::partial_sort(cand.begin(), cand.begin() + keep, cand.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            cand.resize(keep);
        } else {
            std::sort(cand.begin(), cand.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
        }
    }

    double sum = 0.0;
    for (auto& c : cand) {
        float w = std::exp(c.first - max_l);
        c.first = w;
        sum += (double)w;
    }
    float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (auto& c : cand) c.first *= inv_sum;

    if (sp.top_p < 1.0f && cand.size() > 1) {
        double cum = 0.0;
        size_t cut = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            cum += (double)cand[i].first;
            cut = i + 1;
            if (cum >= (double)sp.top_p) break;
        }
        cand.resize(cut);
        double renorm = 0.0;
        for (const auto& c : cand) renorm += (double)c.first;
        float inv_r = (renorm > 0.0) ? (float)(1.0 / renorm) : 0.0f;
        for (auto& c : cand) c.first *= inv_r;
    }

    auto next_u = [&]() -> double {
        rng_state ^= rng_state >> 12;
        rng_state ^= rng_state << 25;
        rng_state ^= rng_state >> 27;
        return (double)((rng_state * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
    };

    // Exact lossless Leviathan et al. speculative verification for deterministic draft token `draft_tok`!
    if (draft_tok >= 0 && out_accepted != nullptr) {
        float p_draft = 0.0f;
        int d_idx = -1;
        for (size_t i = 0; i < cand.size(); ++i) {
            if (cand[i].second == draft_tok) { p_draft = cand[i].first; d_idx = (int)i; break; }
        }
        if (next_u() < (double)p_draft) {
            *out_accepted = true;
            return draft_tok;
        }
        *out_accepted = false;
        // Sample exact residual distribution p'(x) = p(x) / (1 - p(draft)) for x != draft.
        // When d_idx < 0 (p_draft == 0), p'(x) == p(x), so fall through to normal random sampling!
        if (d_idx >= 0) {
            if (cand.size() > 1 && p_draft < 1.0f - 1e-7f) {
                double rem = std::max(0.0, 1.0 - (double)p_draft);
                double r = next_u() * rem;
                double acc = 0.0;
                for (size_t i = 0; i < cand.size(); ++i) {
                    if ((int)i == d_idx) continue;
                    acc += (double)cand[i].first;
                    if (r <= acc) return cand[i].second;
                }
                for (int i = (int)cand.size() - 1; i >= 0; --i) {
                    if (i != d_idx) return cand[i].second;
                }
            }
            return cand[0].second;
        }
    }

    double r = next_u();
    double acc = 0.0;
    for (const auto& c : cand) {
        acc += (double)c.first;
        if (r <= acc) return c.second;
    }
    return cand.back().second;
}
} // namespace

int32_t InklingEngine::sample_from_logits_cpu_ref(
    const float* d_logits_ptr, int32_t argmax_val, const SamplingParams& sp,
    uint64_t& rng_state, int32_t draft_tok, bool* out_accepted)
{
    if (sp.temperature <= 0.0f) {
        if (out_accepted) *out_accepted = (draft_tok >= 0 && argmax_val == draft_tok);
        return argmax_val;
    }
    int V = hp_.vocab_unpadded;
    std::vector<float> logits(V);
    INK_CUDA_CHECK(cudaMemcpy(logits.data(), d_logits_ptr, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));

    float inv_t = 1.0f / sp.temperature;
    float max_l = -1e30f;
    for (int i = 0; i < V; ++i) {
        logits[i] *= inv_t;
        if (logits[i] > max_l) max_l = logits[i];
    }

    std::vector<std::pair<float, int32_t>> cand;
    cand.reserve(256);
    // R24 (SR1-08): min_p=0 uses a 1e-6 tail floor (drops negligible <1e-6 * p_max tail mass)
    float thresh = max_l + std::log(std::max(sp.min_p, 1e-6f));
    for (int i = 0; i < V; ++i) {
        if (logits[i] >= thresh) cand.push_back({logits[i], i});
    }
    return finish_sampling_candidates(cand, max_l, argmax_val, sp, rng_state, draft_tok, out_accepted);
}

int32_t InklingEngine::sample_from_logits(
    const float* d_logits_ptr, int32_t argmax_val, const SamplingParams& sp,
    uint64_t& rng_state, int32_t draft_tok, bool* out_accepted)
{
    if (sp.temperature <= 0.0f) {
        if (out_accepted) *out_accepted = (draft_tok >= 0 && argmax_val == draft_tok);
        return argmax_val;
    }

    int V = hp_.vocab_unpadded;
    float inv_t = 1.0f / sp.temperature;
    float log_min_p = std::log(std::max(sp.min_p, 1e-6f));

    // R14 (SR1-03): Bit-exact GPU superset candidate pre-filter (replaces 800 KB sync D2H copy with 4 KB pinned copy!)
    if (d_sample_pkt_ && h_sample_pkt_ && argmax_val >= 0 && argmax_val < V) {
        ink_cuda_sample_prefilter(d_logits_ptr, V, argmax_val, inv_t, log_min_p,
                                  d_sample_pkt_, d_sample_ext_ids_, d_sample_ext_vals_, stream_);
        INK_CUDA_CHECK(cudaMemcpyAsync(h_sample_pkt_, d_sample_pkt_, sizeof(SamplePrefilterPacket),
                                       cudaMemcpyDeviceToHost, stream_));
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (h_sample_pkt_->overflow == 0 && h_sample_pkt_->count > 0 && h_sample_pkt_->count <= INK_SAMPLE_OVERFLOW_CAP) {
            float max_l = h_sample_pkt_->raw_max * inv_t;
            if (std::isfinite(max_l) && max_l > -1e29f) {
                float thresh = max_l + log_min_p;
                int cnt = h_sample_pkt_->count;
                int inline_n = std::min(cnt, INK_SAMPLE_INLINE_CAP);
                h_samp_cand_.clear();
                for (int i = 0; i < inline_n; ++i) {
                    float s = h_sample_pkt_->vals[i] * inv_t;
                    if (s >= thresh) h_samp_cand_.push_back({s, h_sample_pkt_->ids[i]});
                }
                if (cnt > INK_SAMPLE_INLINE_CAP) {
                    int extra = cnt - INK_SAMPLE_INLINE_CAP;
                    INK_CUDA_CHECK(cudaMemcpyAsync(h_sample_ext_ids_,  d_sample_ext_ids_,  (size_t)extra * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
                    INK_CUDA_CHECK(cudaMemcpyAsync(h_sample_ext_vals_, d_sample_ext_vals_, (size_t)extra * sizeof(float),   cudaMemcpyDeviceToHost, stream_));
                    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                    for (int i = 0; i < extra; ++i) {
                        float s = h_sample_ext_vals_[i] * inv_t;
                        if (s >= thresh) h_samp_cand_.push_back({s, h_sample_ext_ids_[i]});
                    }
                }
                // Restore ascending token-ID order so subsequent sort / sum / CDF / RNG are 100% BIT-EXACT!
                std::sort(h_samp_cand_.begin(), h_samp_cand_.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
                return finish_sampling_candidates(h_samp_cand_, max_l, argmax_val, sp, rng_state, draft_tok, out_accepted);
            }
        }
    }

    // Exact full-vocab fallback (only reached on ultra-flat distributions > 8,192 candidates or non-finite values)
    return sample_from_logits_cpu_ref(d_logits_ptr, argmax_val, sp, rng_state, draft_tok, out_accepted);
}

int32_t InklingEngine::sample_next(const SamplingParams& sp, uint64_t& rng_state) {
    return sample_from_logits(d_logits_, *h_argmax_, sp, rng_state);
}

bool InklingEngine::encode_image_input(const std::string& url_or_path, EncodedImage& out, bool allow_local_files) {
    if (!vision_.loaded()) return false;
    std::vector<uint8_t> bytes;
    if (!VisionEncoder::resolve_image_input(url_or_path, bytes, allow_local_files)) return false;
    void* d_stg = batch_ready_ ? (void*)d_prefill_stage_[0] : nullptr;
    void* h_stg = batch_ready_ ? (void*)h_prefill_stage_[0] : nullptr;
    size_t stg_bytes = batch_ready_ ? ((size_t)STAGE_EXPS * STAGE_SLOT_BYTES) : 0;
    return vision_.encode_bytes(bytes.data(), bytes.size(), out, d_stg, h_stg, stg_bytes, stream_);
}

void InklingEngine::expand_multimodal_tokens(
    const std::vector<int32_t>& raw_tokens,
    int32_t image_placeholder_id,
    const std::vector<EncodedImage>& images,
    std::vector<int32_t>& out_tokens,
    MultimodalEmbeddings& out_mm)
{
    out_tokens.clear();
    out_mm.id_to_embd.clear();
    if (images.empty()) {
        out_tokens = raw_tokens;
        return;
    }
    size_t extra = 0;
    for (const auto& im : images) extra += (size_t)im.n_tokens;
    out_tokens.reserve(raw_tokens.size() + extra);

    size_t img_idx = 0;
    for (int32_t tid : raw_tokens) {
        if (tid == image_placeholder_id && img_idx < images.size()) {
            const auto& im = images[img_idx++];
            uint32_t hbase = (uint32_t)((im.hash ^ (im.hash >> 32)) * 2654435761ULL);
            for (int k = 0; k < im.n_tokens; ++k) {
                uint32_t h = hbase ^ ((uint32_t)k * 2246822519U);
                int32_t syn = -(int32_t)((h & 0x3FFFFFFFu) + 1000000u);
                while (out_mm.id_to_embd.find(syn) != out_mm.id_to_embd.end()) --syn;
                out_mm.id_to_embd[syn] = im.embd.data() + (size_t)k * im.n_embd;
                out_tokens.push_back(syn);
            }
        } else {
            out_tokens.push_back(tid);
        }
    }
}

InklingEngine::GenResult InklingEngine::generate(
    const std::vector<int32_t>& prompt_tokens,
    const SamplingParams& sp,
    const std::function<bool(int32_t, const std::string&)>& on_token,
    bool enable_profile,
    bool reuse_kv_prefix,
    int static_prefix_tokens,
    const std::function<bool(int, int)>& on_prefill,
    int tools_prefix_tokens,
    int full_sys_prefix_tokens,
    const MultimodalEmbeddings* mm_embd)
{
    GenResult res;
    res.prompt_tokens = (int)prompt_tokens.size();
    if (prompt_tokens.empty()) return res;
    clear_io_errors();

    int P = (int)prompt_tokens.size();
    if (P >= cfg_.max_ctx) {
        std::fprintf(stderr, "[Engine] Prompt length %d >= max_ctx %d (context_length_exceeded).\n", P, cfg_.max_ctx);
        res.hit_max_ctx = true;
        return res;
    }

    int want_full_ckpt  = (reuse_kv_prefix && full_sys_prefix_tokens >= 128) ? ((full_sys_prefix_tokens / 64) * 64) : 0;
    if (want_full_ckpt >= P) want_full_ckpt = ((P - 1) / 64) * 64;
    int want_ckpt       = (reuse_kv_prefix && static_prefix_tokens >= 128) ? ((static_prefix_tokens / 64) * 64) : 0;
    if (want_ckpt >= P) want_ckpt = ((P - 1) / 64) * 64;
    if (want_full_ckpt > 0 && want_ckpt + 256 > want_full_ckpt) want_ckpt = 0;
    int want_tools_ckpt = (reuse_kv_prefix && tools_prefix_tokens >= 256) ? ((tools_prefix_tokens / 64) * 64) : 0;
    if (want_tools_ckpt >= P) want_tools_ckpt = ((P - 1) / 64) * 64;
    int next_above_tools = (want_ckpt > 0) ? want_ckpt : want_full_ckpt;
    if (next_above_tools > 0 && want_tools_ckpt + 256 > next_above_tools) want_tools_ckpt = 0;

    auto disk_ckpt_path = [&](const char* label) -> std::string {
        if (!cfg_.use_disk_kv_cache || kv_cache_dir_.empty()) return "";
        return kv_cache_dir_ + "\\kv_" + label + ".ckpt";
    };

    int start_pos = 0;
    if (reuse_kv_prefix) {
        int L = (int)kv_tokens_.size();
        auto try_restore = [&](const KVCheckpoint& ckpt, const char* label) -> bool {
            if (ckpt.pos <= 0 || P <= ckpt.pos) return false;
            if (!std::equal(ckpt.tokens.begin(), ckpt.tokens.end(), prompt_tokens.begin())) return false;
            double t_r0 = now_ms();
            if (!kv_.restore_checkpoint(ckpt, stream_)) return false;
            kv_tokens_ = ckpt.tokens;
            start_pos = ckpt.pos;
            res.cached_tokens = ckpt.pos;
            std::printf("[KV] %s checkpoint restored: %d tokens in %.1f ms\n", label, ckpt.pos, now_ms() - t_r0);
            std::fflush(stdout);
            return true;
        };

        auto try_restore_disk = [&](KVCheckpoint& ckpt, int want_pos, const char* label) -> bool {
            if (want_pos < 1024 || P <= want_pos) return false;
            std::string path = disk_ckpt_path(label);
            if (path.empty()) return false;
            double t_d0 = now_ms();
            if (!kv_.load_checkpoint_disk(ckpt, path, model_sig_, prompt_tokens.data(), want_pos)) return false;
            if (!kv_.restore_checkpoint(ckpt, stream_)) return false;
            kv_tokens_ = ckpt.tokens;
            start_pos = ckpt.pos;
            res.cached_tokens = ckpt.pos;
            std::printf("[KV] NVMe persistent %s checkpoint restored: %d tokens in %.1f ms\n",
                        label, ckpt.pos, now_ms() - t_d0);
            std::fflush(stdout);
            return true;
        };

        auto try_rewind_turn = [&]() -> bool {
            if (turn_snap_.pos <= 0 || P <= turn_snap_.pos) return false;
            if (turn_snap_.epoch != kv_.reset_epoch()) return false;
            if (turn_snap_.pos < full_sys_ckpt_.pos) return false;
            if ((int)turn_snap_.tokens.size() != turn_snap_.pos) return false;
            if (!std::equal(turn_snap_.tokens.begin(), turn_snap_.tokens.end(), prompt_tokens.begin())) return false;
            double t_rw0 = now_ms();
            if (!kv_.rewind_to_turn_snapshot(turn_snap_, stream_)) return false;
            kv_tokens_ = turn_snap_.tokens;
            start_pos = turn_snap_.pos;
            res.cached_tokens = turn_snap_.pos;
            std::printf("[KV] Turn-rewind snapshot restored: %d tokens in %.1f ms (zero Global KV copy!)\n",
                        turn_snap_.pos, now_ms() - t_rw0);
            std::fflush(stdout);
            return true;
        };

        const bool incr_hit = (L > 0 && P > L && std::equal(kv_tokens_.begin(), kv_tokens_.end(), prompt_tokens.begin()));
        if (!incr_hit && L > 0) {
            // v4.1 (R2-10): explain every incremental-cache miss so re-prefills can be traced to their cause.
            int lim = std::min(L, P);
            int lcp = 0;
            while (lcp < lim && kv_tokens_[lcp] == prompt_tokens[lcp]) ++lcp;
            auto render = [&](const int32_t* ids, int n) -> std::string {
                std::string out;
                for (int k = 0; k < n && out.size() < 240; ++k) {
                    std::string pc = tok_.decode_token(ids[k]);
                    if (pc.empty()) { out += "<#" + std::to_string(ids[k]) + ">"; continue; }
                    for (unsigned char ch : pc) {
                        if (ch == '\n') out += "\\n";
                        else if (ch == '\r') out += "\\r";
                        else if (ch == '\t') out += "\\t";
                        else if (ch < 0x20) out += "?";
                        else out.push_back((char)ch);
                    }
                    out.push_back('|');
                }
                return out;
            };
            const int before = std::min(lcp, 24);
            std::printf("[KV] Incremental miss: resident=%d prompt=%d common_prefix=%d (%d resident tokens not reusable)\n",
                        L, P, lcp, L - lcp);
            std::printf("[KV]   ...before : %s\n", render(prompt_tokens.data() + lcp - before, before).c_str());
            if (lcp < L) std::printf("[KV]   resident  : %s\n", render(kv_tokens_.data() + lcp, std::min(16, L - lcp)).c_str());
            if (lcp < P) std::printf("[KV]   new prompt: %s\n", render(prompt_tokens.data() + lcp, std::min(16, P - lcp)).c_str());
            std::fflush(stdout);
        }
        if (incr_hit) {
            // Exact incremental continuation (e.g. tool-call follow-up turn -> 0 ms restore!)
            start_pos = L;
            res.cached_tokens = L;
        } else if (try_rewind_turn()) {
            // Tier-0 In-Place Rewind (~4 ms!): recovers up to 512K tokens on retries/cancels without re-prefilling!
        } else if (try_restore(full_sys_ckpt_, "Full-system")) {
            // Tier-1 RAM: Full tools + ALL system messages checkpoint (~99% prompt hit in ~30 ms!)
        } else if (try_restore(prefix_ckpt_, "Static-prefix")) {
            // Tier-2 RAM: Pre-<env> static system+tools checkpoint (~80% hit when <env>/files change!)
        } else if (try_restore(tools_ckpt_, "Tools-prefix")) {
            // Tier-3 RAM: Tool-declarations checkpoint (survives agent build<->plan switches!)
        } else if (try_restore_disk(full_sys_ckpt_, want_full_ckpt, "full-system")) {
            // Tier-1 NVMe Persistent: survives full restart/reboot in <100 ms!
        } else if (try_restore_disk(prefix_ckpt_, want_ckpt, "static-prefix")) {
            // Tier-2 NVMe Persistent: survives restart + <env> change in <100 ms!
        } else if (try_restore_disk(tools_ckpt_, want_tools_ckpt, "tools-prefix")) {
            // Tier-3 NVMe Persistent: survives restart + agent switch in <100 ms!
        } else {
            if (prefix_ckpt_.pos > 0) {
                int lim = std::min(P, prefix_ckpt_.pos);
                int diff = 0;
                while (diff < lim && prefix_ckpt_.tokens[diff] == prompt_tokens[diff]) ++diff;
                std::printf("[KV] Prefix ckpt miss (full=%d, static=%d, tools=%d, first_diff=%d)\n",
                            full_sys_ckpt_.pos, prefix_ckpt_.pos, tools_ckpt_.pos, diff);
                std::fflush(stdout);
            }
            reset_context();
            start_pos = 0;
        }
    } else {
        reset_context();
        start_pos = 0;
    }

    auto maybe_save_ckpt = [&](KVCheckpoint& ckpt, int target_pos, int cur_len, const char* label) {
        if (target_pos < 128 || cur_len != target_pos) return;
        // Never let tiny calls (e.g. OpenCode title/summary) overwrite a large system+tools checkpoint
        if (ckpt.pos > 0 && target_pos < ckpt.pos && target_pos < 1024) return;
        bool same = (ckpt.pos == target_pos &&
                     (int)ckpt.tokens.size() == target_pos &&
                     std::equal(ckpt.tokens.begin(), ckpt.tokens.end(), kv_tokens_.begin()));
        if (!same) {
            kv_.save_checkpoint(ckpt, target_pos, stream_);
            if (ckpt.pos == target_pos) {
                ckpt.tokens = kv_tokens_;
                std::printf("[KV] Saved %s checkpoint: %d tokens\n", label, target_pos);
                if (target_pos >= 1024) {
                    std::string path = disk_ckpt_path(label);
                    if (!path.empty()) {
                        double t_w0 = now_ms();
                        if (kv_.save_checkpoint_disk(ckpt, path, model_sig_)) {
                            std::printf("[KV] Persisted %s checkpoint to NVMe in %.1f ms\n", label, now_ms() - t_w0);
                        }
                    }
                }
                std::fflush(stdout);
            }
        }
    };

    const int want_turn_snap = (reuse_kv_prefix && P >= 128) ? (((P - 1) / 64) * 64) : 0;
    auto maybe_save_turn_snap = [&](int cur_len) {
        if (!reuse_kv_prefix || cur_len < 128 || (cur_len % 64) != 0) return;
        bool at_end_boundary = (cur_len == want_turn_snap);
        bool at_mid_milestone = (cur_len >= 4096 && (cur_len % 4096) == 0 && cur_len < want_turn_snap);
        if (!at_end_boundary && !at_mid_milestone) return;
        bool same = (turn_snap_.epoch == kv_.reset_epoch() &&
                     turn_snap_.pos == cur_len &&
                     (int)turn_snap_.tokens.size() == cur_len &&
                     std::equal(turn_snap_.tokens.begin(), turn_snap_.tokens.end(), kv_tokens_.begin()));
        if (!same) {
            kv_.save_turn_snapshot(turn_snap_, cur_len, stream_);
            if (turn_snap_.pos == cur_len && turn_snap_.epoch == kv_.reset_epoch()) {
                turn_snap_.tokens = kv_tokens_;
            }
        }
    };
    if (start_pos > 0 && start_pos == want_turn_snap) {
        maybe_save_turn_snap(start_pos);
    }

    double t_pf0 = now_ms();
    int total_new = P - start_pos;
    if (predictor_active() && total_new > 0) {
        pred_tracker_.begin_prompt_turn();
        run_start_predictor(prompt_tokens.data() + start_pos, total_new);
    }

    // Stage 4: Batched GEMM + Grouped Hybrid MoE Prefill (B <= 128, ~3x faster TTFT!)
    // Also handles raw multimodal vision embeddings cleanly (splitting image vs text chunks so tok_norm is bypassed on image tokens!)
    bool has_mm = (mm_embd != nullptr && !mm_embd->id_to_embd.empty());
    if ((cfg_.use_batched_prefill && total_new > 1) || has_mm) {
        std::vector<float> img_chunk_buf;
        int i = start_pos;
        while (i < P) {
            int chunk = std::min(BATCH_MAX, P - i);
            bool chunk_is_img = (prompt_tokens[i] < 0);
            if (has_mm) {
                // Keep each chunk homogeneous (all raw image embeddings OR all text tokens)
                for (int k = 1; k < chunk; ++k) {
                    if ((prompt_tokens[i + k] < 0) != chunk_is_img) {
                        chunk = k;
                        break;
                    }
                }
            }
            // Align chunk boundaries cleanly on all 3 prefix KV checkpoints + turn rewind boundary!
            if (want_tools_ckpt >= 256 && i < want_tools_ckpt && i + chunk > want_tools_ckpt) {
                chunk = want_tools_ckpt - i;
            }
            if (want_ckpt >= 128 && i < want_ckpt && i + chunk > want_ckpt) {
                chunk = want_ckpt - i;
            }
            if (want_full_ckpt >= 128 && i < want_full_ckpt && i + chunk > want_full_ckpt) {
                chunk = want_full_ckpt - i;
            }
            if (want_turn_snap >= 128 && i < want_turn_snap && i + chunk > want_turn_snap) {
                chunk = want_turn_snap - i;
            }
            bool is_last = (i + chunk == P);
            const float* raw_ptr = nullptr;
            if (chunk_is_img && has_mm) {
                int D = hp_.n_embd;
                img_chunk_buf.resize((size_t)chunk * D);
                for (int k = 0; k < chunk; ++k) {
                    auto it = mm_embd->id_to_embd.find(prompt_tokens[i + k]);
                    if (it != mm_embd->id_to_embd.end() && it->second) {
                        std::memcpy(img_chunk_buf.data() + (size_t)k * D, it->second, (size_t)D * sizeof(float));
                    } else {
                        std::memset(img_chunk_buf.data() + (size_t)k * D, 0, (size_t)D * sizeof(float));
                    }
                }
                raw_ptr = img_chunk_buf.data();
            }
            forward_batch(prompt_tokens.data() + i, chunk, i,
                          /*need_last_logits=*/is_last, /*need_all_logits=*/false,
                          /*is_answer=*/false, /*capture_last_pred_hidden=*/is_last && pred_main_.loaded(),
                          /*save_spec_states=*/false, nullptr, raw_ptr);
            if (io_error_) {
                res.io_error = true;
                res.error_msg = last_error_;
                res.prefill_ms = now_ms() - t_pf0;
                return res;
            }
            for (int k = 0; k < chunk; ++k) kv_tokens_.push_back(prompt_tokens[i + k]);
            i += chunk;

            maybe_save_ckpt(tools_ckpt_,    want_tools_ckpt, i, "tools-prefix");
            maybe_save_ckpt(prefix_ckpt_,   want_ckpt,       i, "static-prefix");
            maybe_save_ckpt(full_sys_ckpt_, want_full_ckpt,  i, "full-system");
            maybe_save_turn_snap(i);

            int done = i - start_pos;
            if (on_prefill) {
                if (!on_prefill(done, total_new)) {
                    res.prefill_ms = now_ms() - t_pf0;
                    return res; // Client disconnected mid-prefill
                }
            }
        }
    } else {
        for (int i = start_pos; i < P; ++i) {
            bool is_last = (i + 1 == P);
            forward_token(prompt_tokens[i], i, is_last, false, /*is_answer=*/false, /*capture_pred_hidden=*/is_last && pred_main_.loaded());
            if (io_error_) {
                res.io_error = true;
                res.error_msg = last_error_;
                res.prefill_ms = now_ms() - t_pf0;
                return res;
            }
            kv_tokens_.push_back(prompt_tokens[i]);

            maybe_save_ckpt(tools_ckpt_,    want_tools_ckpt, i + 1, "tools-prefix");
            maybe_save_ckpt(prefix_ckpt_,   want_ckpt,       i + 1, "static-prefix");
            maybe_save_ckpt(full_sys_ckpt_, want_full_ckpt,  i + 1, "full-system");
            maybe_save_turn_snap(i + 1);

            int done = i - start_pos + 1;
            if (on_prefill && ((done & 15) == 0 || is_last)) {
                if (!on_prefill(done, total_new)) {
                    res.prefill_ms = now_ms() - t_pf0;
                    return res;
                }
            }
        }
    }

    // Answer-start proactive VRAM rebalance using layer 8 & 16 hidden states from the prompt's final token!
    if (predictor_active()) {
        pred_tracker_.begin_answer();
        if (pred_main_.loaded()) {
            run_main_predictor(/*answer_pos=*/0, /*is_answer_start=*/true);
        }
    }
    res.prefill_ms = now_ms() - t_pf0;

    uint64_t rng = sp.seed ? sp.seed : 42;
    int pos = P;
    const int pred_every = pred_main_.loaded() ? pred_main_.every() : 16;
    int next_pred_at = pred_every;
    const int V = hp_.vocab_unpadded;
    double t_dec0 = now_ms();

    // InkEngine v3.0 Zero-VRAM O(1) Rolling 3-Gram Suffix Hash Index (full 128K+ context coverage in <1 us, K <= 6!)
    constexpr int NG_HASH_BITS = 16;
    constexpr int NG_HASH_SIZE = 1 << NG_HASH_BITS;
    constexpr uint32_t NG_HASH_MASK = (uint32_t)(NG_HASH_SIZE - 1);
    auto hash3 = [&](int32_t a, int32_t b, int32_t c) -> uint32_t {
        return (((uint32_t)a * 2654435761u) ^ ((uint32_t)b * 2246822519u) ^ ((uint32_t)c * 3266489917u)) & NG_HASH_MASK;
    };
    std::vector<int32_t> ng_head(NG_HASH_SIZE, -1);
    std::vector<int32_t> ng_prev;
    ng_prev.reserve(kv_tokens_.size() + (size_t)sp.max_tokens + 16);
    int ng_indexed = 0;
    auto sync_ngram_index = [&]() {
        int H = (int)kv_tokens_.size();
        if (H <= ng_indexed) return;
        ng_prev.resize(H, -1);
        for (int idx = std::max(2, ng_indexed); idx < H; ++idx) {
            uint32_t h = hash3(kv_tokens_[idx - 2], kv_tokens_[idx - 1], kv_tokens_[idx]);
            ng_prev[idx] = ng_head[h];
            ng_head[h] = idx;
        }
        ng_indexed = H;
    };
    if (cfg_.use_speculative) sync_ngram_index();

    // R16: Per-match-length (3/4/5-gram) adaptive draft length tracker (100% lossless!)
    float ngram_ema[3] = {1.5f, 3.0f, 5.0f};
    int   ngram_3_cooldown = 0;
    int   ngram_3_consec_rejects = 0;

    auto draft_ngram = [&](int32_t cur_tok, int max_draft, int32_t* out_draft, int& out_match_len) -> int {
        out_match_len = 0;
        sync_ngram_index();
        const int H = (int)kv_tokens_.size();
        if (H < 6 || max_draft <= 0) return 0;

        int32_t p1 = kv_tokens_[H - 1];
        int32_t p2 = kv_tokens_[H - 2];
        int32_t p3 = kv_tokens_[H - 3];
        int32_t p4 = kv_tokens_[H - 4];

        uint32_t h = hash3(p2, p1, cur_tok);
        int best_idx = -1;
        int best_len = 0;
        int hops = 0;
        for (int idx = ng_head[h]; idx >= 2 && hops < 48; idx = ng_prev[idx], ++hops) {
            if (idx + 1 >= H) continue;
            if (kv_tokens_[idx] != cur_tok || kv_tokens_[idx - 1] != p1 || kv_tokens_[idx - 2] != p2) continue;
            int mlen = 3;
            if (idx >= 3 && kv_tokens_[idx - 3] == p3) {
                mlen = 4;
                if (idx >= 4 && kv_tokens_[idx - 4] == p4) mlen = 5;
            }
            if (mlen > best_len) {
                best_len = mlen;
                best_idx = idx;
                if (best_len == 5) break; // Most recent exact 5-gram match!
            }
        }
        if (best_idx < 0) return 0;
        out_match_len = best_len;

        // R16: Scale draft length K adaptively by match length + acceptance EMA!
        int k_cap = max_draft;
        if (best_len == 3) {
            if (ngram_3_cooldown > 0 || ngram_ema[0] < 0.35f) {
                if (ngram_3_cooldown > 0) ngram_3_cooldown--;
                ngram_ema[0] = std::min(1.0f, ngram_ema[0] + 0.04f); // gradual recovery when switching back to code
                return 0;
            }
            k_cap = std::clamp((int)std::lround(ngram_ema[0] + 1.0f), 1, std::min(4, max_draft));
        } else if (best_len == 4) {
            k_cap = std::clamp((int)std::lround(ngram_ema[1] + 1.5f), 2, max_draft);
        }

        int k = 0;
        for (; k < k_cap && (best_idx + 1 + k) < H; ++k) {
            int32_t cand = kv_tokens_[best_idx + 1 + k];
            if (tok_.is_eog(cand)) break;
            out_draft[k] = cand;
        }
        return k;
    };

    int32_t next_tok = sample_next(sp, rng);
    while (res.gen_tokens < sp.max_tokens) {
        if (tok_.is_eog(next_tok)) { res.stop_token = next_tok; break; }

        std::string piece = tok_.decode_token(next_tok);
        res.tokens.push_back(next_tok);
        res.text += piece;
        res.gen_tokens++;

        if (on_token && !on_token(next_tok, piece)) break;
        if (pos >= cfg_.max_ctx - 1) { res.hit_max_ctx = true; break; }
        if (res.gen_tokens >= sp.max_tokens) break;

        bool need_pred_now = pred_main_.loaded() && (res.gen_tokens >= next_pred_at);

        // Stage 5 / v4.0: Zero-VRAM Lossless Adaptive Speculative Decoding (R13 + R16)!
        //  1) O(1) Rolling 3-Gram Suffix Hash Drafter (adaptive K <= 6 by match_len + EMA, <1 us!)
        //  2) 1-Loop (24L) GPU-Only Self-Speculative Fallback (zero extra VRAM)
        int32_t draft_toks[HybridPagedKV::SPEC_MAX_B];
        int max_d = std::min(HybridPagedKV::SPEC_MAX_B - 2, std::min(sp.max_tokens - res.gen_tokens, cfg_.max_ctx - 2 - pos));
        int ngram_match_len = 0;
        int K_draft = (cfg_.use_speculative && !enable_profile && !need_pred_now && max_d >= 2)
            ? draft_ngram(next_tok, max_d, draft_toks, ngram_match_len)
            : 0;

        if (K_draft == 0 && cfg_.use_speculative && cfg_.self_draft_k > 0 && !enable_profile && !need_pred_now && max_d >= 2) {
            int k_self = std::min(cfg_.self_draft_k, max_d);
            kv_.begin_draft_snapshot(pos, k_self, stream_);
            int32_t d_in = next_tok;
            int drafted = 0;
            for (int s = 0; s < k_self; ++s) {
                int32_t d_out = forward_token_1loop_draft(d_in, pos + s);
                if (io_error_) {
                    kv_.rollback_draft_snapshot(pos, drafted, stream_);
                    res.io_error = true;
                    res.error_msg = last_error_;
                    res.decode_ms = now_ms() - t_dec0;
                    return res;
                }
                drafted++;
                if (tok_.is_eog(d_out)) break;
                draft_toks[K_draft++] = d_out;
                d_in = d_out;
            }
            kv_.rollback_draft_snapshot(pos, drafted, stream_);
        }

        if (K_draft >= 1) {
            int B = 1 + K_draft;
            int32_t batch_toks[HybridPagedKV::SPEC_MAX_B];
            int32_t argm[HybridPagedKV::SPEC_MAX_B];
            batch_toks[0] = next_tok;
            for (int k = 0; k < K_draft; ++k) batch_toks[1 + k] = draft_toks[k];

            const double t_spec0 = now_ms();
            kv_.begin_spec_verify(pos, B, stream_);
            forward_batch(batch_toks, B, pos,
                          /*need_last_logits=*/false, /*need_all_logits=*/true,
                          /*is_answer=*/true, /*capture_last_pred_hidden=*/false,
                          /*save_spec_states=*/true, argm);
            if (io_error_) {
                kv_.commit_spec_verify(1, stream_);
                res.io_error = true;
                res.error_msg = last_error_;
                res.decode_ms = now_ms() - t_dec0;
                return res;
            }

            int accepted = 1; // batch_toks[0] (`next_tok`) is already emitted & committed
            int32_t bonus_tok = -1;
            bool stopped = false;
            bool rejected = false;

            for (int m = 0; m < K_draft; ++m) {
                bool ok = false;
                int32_t sampled = sample_from_logits(d_b_logits_ + (size_t)m * V, argm[m], sp, rng, batch_toks[m + 1], &ok);
                if (ok) {
                    accepted++;
                    res.spec_accepted++;
                    int32_t dtok = batch_toks[m + 1];
                    std::string dpiece = tok_.decode_token(dtok);
                    res.tokens.push_back(dtok);
                    res.text += dpiece;
                    res.gen_tokens++;
                    if (on_token && !on_token(dtok, dpiece)) { stopped = true; break; }
                    if (res.gen_tokens >= sp.max_tokens) { stopped = true; break; }
                } else {
                    bonus_tok = sampled;
                    rejected = true;
                    break;
                }
            }
            if (!stopped && !rejected) {
                // All K drafts accepted! Sample free bonus token from position K_draft!
                bonus_tok = sample_from_logits(d_b_logits_ + (size_t)K_draft * V, argm[K_draft], sp, rng);
            }

            kv_.commit_spec_verify(accepted, stream_);
            res.spec_ms += now_ms() - t_spec0;
            const int K_exp = hp_.n_expert_used;
            const bool replay_spec_routes = predictor_active() && (batch_pred_B_ == B) &&
                (batch_pred_ids_.size() >= (size_t)hp_.n_layer_logical * (size_t)B * (size_t)K_exp);
            for (int a = 0; a < accepted; ++a) {
                kv_tokens_.push_back(batch_toks[a]);
                if (predictor_active()) {
                    // R10 (SR3-05): Replay every accepted speculative token's routes in exact token order!
                    pred_tracker_.step_answer_token();
                    if (replay_spec_routes) {
                        for (int il = 0; il < hp_.n_layer_logical; ++il) {
                            int ip = hp_.loop_layer(il);
                            pred_tracker_.record_selections(
                                ip, batch_pred_ids_.data() + ((size_t)il * (size_t)B + (size_t)a) * (size_t)K_exp,
                                K_exp, /*is_answer=*/true);
                        }
                    }
                }
            }
            pos += accepted;
            res.spec_drafted += K_draft;
            res.spec_steps++;

            // R16: Update adaptive n-gram acceptance EMA + cold-prose 3-gram cooldown
            if (ngram_match_len >= 3) {
                int n_acc = accepted - 1;
                int bi = std::clamp(ngram_match_len - 3, 0, 2);
                ngram_ema[bi] = 0.70f * ngram_ema[bi] + 0.30f * (float)n_acc;
                if (bi == 0 && n_acc == 0) {
                    if (++ngram_3_consec_rejects >= 2) ngram_3_cooldown = 3;
                } else if (n_acc > 0) {
                    ngram_3_consec_rejects = 0;
                    ngram_3_cooldown = 0;
                }
            }

            if (pos >= cfg_.max_ctx - 1) { res.hit_max_ctx = true; break; }
            if (stopped || res.gen_tokens >= sp.max_tokens) break;
            next_tok = bonus_tok;
        } else {
            const double t_sc0 = now_ms();
            forward_token(next_tok, pos, true, enable_profile, /*is_answer=*/true, /*capture_pred_hidden=*/need_pred_now);
            if (io_error_) {
                res.io_error = true;
                res.error_msg = last_error_;
                res.decode_ms = now_ms() - t_dec0;
                return res;
            }
            if (need_pred_now) {
                run_main_predictor(pred_tracker_.answer_pos(), /*is_answer_start=*/false);
                next_pred_at = res.gen_tokens + pred_every;
            }
            kv_tokens_.push_back(next_tok);
            pos++;
            next_tok = sample_next(sp, rng);
            res.scalar_ms += now_ms() - t_sc0;
            res.scalar_steps++;
        }
    }
    res.decode_ms = now_ms() - t_dec0;
    return res;
}

void InklingEngine::probe_early_exit(const std::string& prompt_text, int n_tokens) {
    std::printf("\n[EARLY-EXIT PROBE] Measuring 1-Loop (24L) vs Full 3-Loop (48L) top-1 / top-5 agreement over %d tokens...\n", n_tokens);
    reset_context();
    std::string fmt = tok_.format_chat({{"user", prompt_text}}, "none", true);
    auto ptoks = tok_.encode(fmt, true);
    int32_t cur_tok = prefill(ptoks, 0);
    int pos = (int)ptoks.size();
    const int D = hp_.n_embd;
    const int KVdim = hp_.n_head_kv * hp_.head_dim;

    auto run_1loop_at = [&](int32_t token_id, int p, std::vector<float>& out_logits) -> int32_t {
        const uint8_t* emb_row = h_tok_embd_.data() + (size_t)token_id * tok_embd_row_bytes_;
        CPUThreadPool::dequantize_row(tok_embd_type_, emb_row, h_embd_row_, D);
        INK_CUDA_CHECK(cudaMemcpyAsync(d_cur_, h_embd_row_, (size_t)D * sizeof(float), cudaMemcpyHostToDevice, stream_));
        ink_cuda_rms_norm(d_cur_, (const float*)tok_norm_.ptr, d_cur_, D, hp_.rms_eps, stream_);

        float eff = (float)(p + 1) / (float)hp_.log_scale_floor;
        float tau_g = 1.0f + hp_.log_scale_alpha * std::log(std::max(eff, 1.0f));
        int n_blocks = 0, prelude_n_blocks = 0, n_bank = 0;

        // Run Prelude [0..5], Pass-0 Core [6..17], then skip Passes 1 & 2 straight to Coda [42..47]!
        for (int step = 0; step < 24; ++step) {
            int il = (step < 18) ? step : (step + 24);
            int ip = hp_.loop_layer(il);
            const auto& L = layers_[ip];
            emit_layer_pre(il, n_blocks, prelude_n_blocks, n_bank, tau_g);
            int rel_ext = L.is_swa ? hp_.swa_window : hp_.rel_extent;
            kv_.append_and_attend(il, p, d_Q_, d_K_conv_, d_V_conv_, d_rel_table_, d_sinks_, d_attn_out_, rel_ext, stream_);
            emit_layer_post(il, n_blocks, n_bank);

            const auto& lm = experts_.layer_meta(ip);
            const auto& tbl = experts_.layer_device_table(ip);
            if (experts_.is_layer_all_vram(ip)) {
                ink_cuda_moe_pinned_indirect(lm.gate_up_type, lm.down_type, tbl,
                                             d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                             d_ffn_normed_, d_moe_h_, d_moe_out_,
                                             D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
            } else {
                uint32_t seq = (++split_pkt_seq_ == 0) ? ++split_pkt_seq_ : split_pkt_seq_;
                ink_cuda_moe_split_hits(tbl, d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                        d_ffn_normed_, D, d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_,
                                        d_split_pkt_, d_mapped_ffn_in_, seq, stream_);
                ink_cuda_moe_dynamic_indirect(lm.gate_up_type, lm.down_type, tbl,
                                              d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_, hp_.n_expert_used,
                                              d_ffn_normed_, d_moe_h_, d_moe_out_,
                                              D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
                wait_split_packet(seq);
                CPUExpertTask cpu_tasks[8];
                int n_cpu = 0;
                experts_.resolve_gpu_split(ip, *h_split_pkt_, cpu_tasks, n_cpu);
                if (n_cpu > 0) {
                    cpu_pool_->run_moe_experts(h_ffn_in_, h_moe_cpu_out_, cpu_tasks, n_cpu,
                                               D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta);
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_moe_cpu_out_, h_moe_cpu_out_, (size_t)D * sizeof(float), cudaMemcpyHostToDevice, stream_));
                    ink_cuda_add_inplace(d_moe_out_, d_moe_cpu_out_, D, stream_);
                }
            }
            ink_cuda_shortconv_add(d_moe_out_, kv_.conv_state_ptr(il, 2 * KVdim + D),
                                   (const float*)L.ffn_post_conv.ptr, d_ffn_inp_, d_cur_, D, stream_);
        }
        emit_lm_head(n_blocks);
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        copy_logits_to_host(out_logits);
        return *h_argmax_;
    };

    int top1_match = 0, top5_match = 0, total = 0;
    std::vector<float> l1;
    KVCheckpoint snap;
    for (int t = 0; t < n_tokens; ++t) {
        if (tok_.is_eog(cur_tok)) break;
        kv_.save_checkpoint(snap, pos, stream_);
        int32_t d1 = run_1loop_at(cur_tok, pos, l1);
        kv_.restore_checkpoint(snap, stream_);

        int32_t full_next = forward_token(cur_tok, pos, true, false, true, false);
        if (d1 == full_next) top1_match++;

        // Check if full_next is in top-5 of 1-loop logits
        float target_l = l1[full_next];
        int higher = 0;
        for (float v : l1) { if (v > target_l) { higher++; if (higher >= 5) break; } }
        if (higher < 5) top5_match++;

        total++;
        cur_tok = full_next;
        pos++;
    }
    std::printf("[EARLY-EXIT RESULT] Tested %d tokens | 1-Loop Top-1 agreement = %d/%d (%.1f%%) | Top-5 = %d/%d (%.1f%%)\n",
                total, top1_match, total, total ? 100.0 * top1_match / total : 0.0,
                top5_match, total, total ? 100.0 * top5_match / total : 0.0);
    std::fflush(stdout);
}

// =========================================================================
// R0 (SR1-01): Long-Context Parity Gate Suite (SWA Wrap, Speculative, Checkpoints)
// =========================================================================
bool InklingEngine::run_parity_suite() {
    std::printf("\n================================================================\n");
    std::printf("  [PARITY SUITE R0] Long-Context (>1024), Speculative & Restore Gate\n");
    std::printf("================================================================\n");
    std::fflush(stdout);

    if (cfg_.max_ctx < 2048) {
        std::fprintf(stderr, "[PARITY] Requires --ctx >= 2048 (got %d)\n", cfg_.max_ctx);
        return false;
    }

    auto cmp_logits = [](const std::vector<float>& a, const std::vector<float>& b,
                         float& out_max_abs, double& out_mean_abs, double& out_cos, int& out_top1_a, int& out_top1_b) {
        out_max_abs = 0.0f;
        out_mean_abs = 0.0;
        out_cos = 0.0;
        out_top1_a = 0; out_top1_b = 0;
        float best_a = -1e30f, best_b = -1e30f;
        double dot = 0.0, na = 0.0, nb = 0.0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) {
            float d = std::fabs(a[i] - b[i]);
            if (d > out_max_abs) out_max_abs = d;
            out_mean_abs += (double)d;
            dot += (double)a[i] * (double)b[i];
            na  += (double)a[i] * (double)a[i];
            nb  += (double)b[i] * (double)b[i];
            if (a[i] > best_a) { best_a = a[i]; out_top1_a = (int)i; }
            if (b[i] > best_b) { best_b = b[i]; out_top1_b = (int)i; }
        }
        if (n > 0) out_mean_abs /= (double)n;
        if (na > 0.0 && nb > 0.0) out_cos = dot / (std::sqrt(na) * std::sqrt(nb));
    };

    // Build deterministic realistic code prompt > 1,280 tokens (crosses 1,024 SWA wrap cleanly!)
    std::string code_blk =
        "class MetricsCalculator:\n"
        "    def __init__(self, window_size: int = 1024):\n"
        "        self.window_size = window_size\n"
        "        self.values = []\n"
        "    def record(self, value: float) -> None:\n"
        "        self.values.append(value)\n"
        "        if len(self.values) > self.window_size:\n"
        "            self.values.pop(0)\n"
        "    def mean(self) -> float:\n"
        "        return sum(self.values) / len(self.values) if self.values else 0.0\n\n";
    std::string big_prompt = "Here is a Python module repeated for context testing:\n\n";
    for (int r = 0; r < 14; ++r) {
        big_prompt += "# Section " + std::to_string(r) + "\n" + code_blk;
    }
    big_prompt += "Repeat the exact MetricsCalculator class verbatim below without commentary:\n";
    std::string fmt = tok_.format_chat({{"user", big_prompt}}, "none", true);
    auto toks = tok_.encode(fmt, true);
    if ((int)toks.size() < 1200) {
        std::fprintf(stderr, "[PARITY] Prompt too short (%zu tokens)\n", toks.size());
        return false;
    }

    bool all_ok = true;
    const bool orig_batch = cfg_.use_batched_prefill;
    const bool orig_spec  = cfg_.use_speculative;
    const bool is_q4_kv   = (cfg_.kv_quant == KV_QUANT_Q4_0);
    // Freeze dynamic VRAM/RAM expert swaps across paired parity comparisons so CPU AVX2 vs GPU dp4a
    // 1-ULP FP32 rounding never shifts placement mid-comparison!
    struct SwapFreezeGuard {
        ExpertCache& ec;
        bool prev;
        explicit SwapFreezeGuard(ExpertCache& e) : ec(e), prev(e.swaps_frozen()) { ec.set_swaps_frozen(true); }
        ~SwapFreezeGuard() { ec.set_swaps_frozen(prev); }
    };

    auto max_abs_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
        float md = 0.0f;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
            md = std::max(md, std::fabs(a[i] - b[i]));
        }
        return md;
    };

    // Warm base prefix [0..1023] cleanly (1024 = 16*64 pages, right at the SWA wrap boundary!)
    reset_context();
    std::vector<int32_t> prefix_1024(toks.begin(), toks.begin() + 1024);
    prefill(prefix_1024, 0);
    KVCheckpoint ckpt_1024;
    kv_.save_checkpoint(ckpt_1024, 1024, stream_);
    ckpt_1024.tokens = prefix_1024;

    // Also prepare unaligned prefix [0..1099] to test B=128 wrapping across physical ring boundary 1152!
    std::vector<int32_t> p1024_1100(toks.begin() + 1024, toks.begin() + 1100);
    prefill(p1024_1100, 1024);
    KVCheckpoint ckpt_1100;
    kv_.save_checkpoint(ckpt_1100, 1100, stream_);
    ckpt_1100.tokens.assign(toks.begin(), toks.begin() + 1100);

    // -------------------------------------------------------------------------
    // TEST 1A: Direct SWA Ring-Wrap Parity at pos=1024 (B=7) & pos=1100 (B=128 crossing 1152!)
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 1A] SWA Ring-Wrap Query-0 Parity (pos=1024 B=7 & pos=1100 B=128 vs B=1)...\n");
    float max_d = 0.0f; double mean_d = 0.0, cos_sim = 0.0; int t1a = 0, t1b = 0;
    {
        SwapFreezeGuard guard(experts_);
        std::vector<float> ref_1024, b7_1024, l0_ref, l0_b7;
        kv_.restore_checkpoint(ckpt_1024, stream_);
        g_l0_attn_capture = &l0_ref;
        forward_token(toks[1024], 1024, true, false, false, false);
        g_l0_attn_capture = nullptr;
        copy_logits_to_host(ref_1024);

        kv_.restore_checkpoint(ckpt_1024, stream_);
        int32_t argm7[HybridPagedKV::SPEC_MAX_B];
        g_l0_attn_capture = &l0_b7;
        forward_batch(toks.data() + 1024, 7, 1024, false, true, false, false, false, argm7);
        g_l0_attn_capture = nullptr;
        copy_batch_logits_to_host(0, b7_1024);

        float l0_err = max_abs_diff(l0_ref, l0_b7);
        cmp_logits(ref_1024, b7_1024, max_d, mean_d, cos_sim, t1a, t1b);
        double min_cos = is_q4_kv ? 0.9400 : 0.9850;
        double max_mean = is_q4_kv ? 1.35 : 0.60;
        bool pass_1a_1024 = (l0_err < 1e-4f) && (t1a == t1b) && (cos_sim > min_cos) && (mean_d < max_mean);
        std::printf("  pos=1024 (B=  7[0] vs B=1): L0_SWA_err=%.6f | top1=%d vs %d (%s) | cos=%.6f | mean|d|=%.5f -> %s\n",
                    l0_err, t1b, t1a, (t1a == t1b) ? "MATCH" : "MISMATCH", cos_sim, mean_d, pass_1a_1024 ? "PASS" : "FAIL");
        if (!pass_1a_1024) all_ok = false;

        // Unaligned B=128 at pos=1100 writes slots [1100..1151] + [0..75] (crosses 1152 ring wrap!)
        // In v3.2 (1024 ring), this catastrophically overwrote 127 live SWA keys in L0!
        std::vector<float> l0_1100_ref, l0_1100_b128;
        kv_.restore_checkpoint(ckpt_1100, stream_);
        g_l0_attn_capture = &l0_1100_ref;
        forward_token(toks[1100], 1100, false, false, false, false);
        g_l0_attn_capture = nullptr;

        kv_.restore_checkpoint(ckpt_1100, stream_);
        g_l0_attn_capture = &l0_1100_b128;
        forward_batch(toks.data() + 1100, 128, 1100, false, false, false, false, false);
        g_l0_attn_capture = nullptr;

        float l0_wrap_err = max_abs_diff(l0_1100_ref, l0_1100_b128);
        bool pass_1a_1100 = (l0_wrap_err < 2e-2f); // WMMA FP16 QKV vs dp4a Q8_1 QKV (v3.2 bug gave >0.5-5.0!)
        std::printf("  pos=1100 (B=128[0] crossing 1152 wrap vs B=1): L0_SWA_err=%.6f -> %s\n",
                    l0_wrap_err, pass_1a_1100 ? "PASS" : "FAIL");
        if (!pass_1a_1100) all_ok = false;
    }

    // -------------------------------------------------------------------------
    // TEST 1B: Post-Wrap Chunked Prefill Parity [1024..1151] (B=128, B=32, B=7 vs B=1)
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 1B] Post-Wrap Prefill Parity across pos [1024..1151] (B=128, B=32, B=7 vs B=1)...\n");
    {
        SwapFreezeGuard guard(experts_);
        std::vector<float> ref_1151;
        kv_.restore_checkpoint(ckpt_1024, stream_);
        for (int i = 0; i < 128; ++i) {
            forward_token(toks[1024 + i], 1024 + i, (i == 127), false, false, false);
        }
        copy_logits_to_host(ref_1151);

        int chunk_sizes[3] = {128, 32, 7};
        double min_cos = is_q4_kv ? 0.9850 : 0.9950;
        double max_mean = is_q4_kv ? 0.65 : 0.40;
        for (int cs : chunk_sizes) {
            kv_.restore_checkpoint(ckpt_1024, stream_);
            int done = 0;
            while (done < 128) {
                int cur_b = std::min(cs, 128 - done);
                bool is_last = (done + cur_b == 128);
                forward_batch(toks.data() + 1024 + done, cur_b, 1024 + done, is_last, false, false, false, false);
                done += cur_b;
            }
            std::vector<float> got_1151;
            copy_logits_to_host(got_1151);
            cmp_logits(ref_1151, got_1151, max_d, mean_d, cos_sim, t1a, t1b);
            bool ok = (t1a == t1b) && (cos_sim > min_cos) && (mean_d < max_mean);
            std::printf("  chunk B=%3d vs B=1 @ pos=1151: top1=%d vs %d (%s) | cos=%.6f | max|d|=%.5f | mean|d|=%.5f -> %s\n",
                        cs, t1b, t1a, (t1a == t1b) ? "MATCH" : "MISMATCH", cos_sim, max_d, mean_d, ok ? "PASS" : "FAIL");
            if (!ok) all_ok = false;
        }
    }

    // -------------------------------------------------------------------------
    // TEST 2: Speculative ON vs OFF Past Position 1,024 (96 Greedy Tokens!)
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 2] Speculative ON vs OFF Past pos=1024 (greedy 96 tokens from identical KV)...\n");
    {
        SamplingParams gsp;
        gsp.temperature = 0.0f;
        gsp.max_tokens = 96;

        // Prefill prompt prefix [0..P-2] ONCE so spec_off and spec_on begin from bit-exact KV state!
        reset_context();
        cfg_.use_batched_prefill = true;
        int pref_n = (int)toks.size() - 1;
        std::vector<int32_t> g2_pref(toks.begin(), toks.begin() + pref_n);
        prefill(g2_pref, 0);
        KVCheckpoint ckpt_g2;
        kv_.save_checkpoint(ckpt_g2, pref_n, stream_);
        ckpt_g2.tokens = g2_pref;

        SwapFreezeGuard guard(experts_);

        cfg_.use_speculative = false;
        kv_.restore_checkpoint(ckpt_g2, stream_);
        kv_tokens_ = g2_pref;
        auto res_off = generate(toks, gsp, nullptr, false, true);

        cfg_.use_speculative = true;
        kv_.restore_checkpoint(ckpt_g2, stream_);
        kv_tokens_ = g2_pref;
        auto res_on = generate(toks, gsp, nullptr, false, true);

        bool exact_spec = (res_off.tokens == res_on.tokens);
        int first_diff = -1;
        size_t min_len = std::min(res_off.tokens.size(), res_on.tokens.size());
        for (size_t i = 0; i < min_len; ++i) {
            if (res_off.tokens[i] != res_on.tokens[i]) { first_diff = (int)i; break; }
        }
        bool pass_2 = exact_spec && (res_on.spec_accepted > 0);
        std::printf("  Prompt=%zu tok | spec_off=%zu tok | spec_on=%zu tok (accepted=%d/%d) | exact=%s",
                    toks.size(), res_off.tokens.size(), res_on.tokens.size(),
                    res_on.spec_accepted, res_on.spec_drafted, exact_spec ? "YES" : "NO");
        if (first_diff >= 0) std::printf(" (first_diff=%d)", first_diff);
        std::printf(" -> %s\n", pass_2 ? "PASS" : "FAIL");
        if (!pass_2) all_ok = false;
    }

    // -------------------------------------------------------------------------
    // TEST 3: Turn-Rewind Snapshot & Disk v3 Checkpoint Restore Bit-Exactness
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 3] Turn-Rewind & Disk v3 Checkpoint Restore Parity (pos=1088 -> 1089, scalar deterministic)...\n");
    {
        reset_context();
        std::vector<int32_t> p1088(toks.begin(), toks.begin() + 1088);
        prefill(p1088, 0);
        TurnRewindSnapshot tr_snap;
        kv_.save_turn_snapshot(tr_snap, 1088, stream_);
        tr_snap.tokens = p1088;

        KVCheckpoint ckpt_1088;
        kv_.save_checkpoint(ckpt_1088, 1088, stream_);
        ckpt_1088.tokens = p1088;
        std::string tmp_ckpt = kv_cache_dir_.empty() ? "parity_test_v3.ckpt" : (kv_cache_dir_ + "\\parity_test_v3.ckpt");
        bool disk_saved = kv_.save_checkpoint_disk(ckpt_1088, tmp_ckpt, model_sig_);

        SwapFreezeGuard guard(experts_);

        kv_.rewind_to_turn_snapshot(tr_snap, stream_);
        forward_token(toks[1088], 1088, true, false, false, false);
        std::vector<float> ref_1088;
        copy_logits_to_host(ref_1088);

        // 3A: Turn-rewind in-place (tests live Global KV + restored ShortConv & 1152-slot SWA ring!)
        bool rew_ok = kv_.rewind_to_turn_snapshot(tr_snap, stream_);
        forward_token(toks[1088], 1088, true, false, false, false);
        std::vector<float> got_rew;
        copy_logits_to_host(got_rew);
        cmp_logits(ref_1088, got_rew, max_d, mean_d, cos_sim, t1a, t1b);
        bool pass_3a = rew_ok && (t1a == t1b) && (max_d == 0.0f);
        std::printf("  Turn-Rewind (1088->1089): restored=%s | max|dlogit|=%.6f | top1=%d vs %d -> %s\n",
                    rew_ok ? "YES" : "NO", max_d, t1b, t1a, pass_3a ? "PASS (BIT-EXACT)" : "FAIL");
        if (!pass_3a) all_ok = false;

        // 3B: Disk v3 checkpoint round-trip (tests full serialized ShortConv + 1152-slot SWA + Global KV + Quest Q8!)
        KVCheckpoint loaded_ckpt;
        bool disk_loaded = disk_saved && kv_.load_checkpoint_disk(loaded_ckpt, tmp_ckpt, model_sig_, p1088.data(), 1088);
        DeleteFileA(tmp_ckpt.c_str());
        bool rest_ok = disk_loaded && kv_.restore_checkpoint(loaded_ckpt, stream_);
        forward_token(toks[1088], 1088, true, false, false, false);
        std::vector<float> got_disk;
        copy_logits_to_host(got_disk);
        cmp_logits(ref_1088, got_disk, max_d, mean_d, cos_sim, t1a, t1b);
        bool pass_3b = rest_ok && (t1a == t1b) && (max_d == 0.0f);
        std::printf("  Disk v3 Ckpt (1088->1089): saved=%s loaded=%s | max|dlogit|=%.6f | top1=%d vs %d -> %s\n",
                    disk_saved ? "YES" : "NO", disk_loaded ? "YES" : "NO", max_d, t1b, t1a,
                    pass_3b ? "PASS (BIT-EXACT)" : "FAIL");
        if (!pass_3b) all_ok = false;
    }

    // -------------------------------------------------------------------------
    // TEST 4 (R4 / SR3-02): Exact Global Attention under Severe VRAM Pressure + Multi-Chunk Streaming
    // TEST 5 (R6 / SR3-03): Checked I/O & Atomic Publish Fault-Injection (KV + ExpertCache)
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 4 & 5] R4 Multi-Chunk Exact Global Attention + R6 Fault-Injected Checked I/O...\n");
    std::fflush(stdout);
    {
        const int n_gl = 4;
        std::vector<uint8_t> all_glob(n_gl, 0);
        const int n_h = hp_.n_head, n_hkv = hp_.n_head_kv, hd = hp_.head_dim, D = hp_.n_embd;
        const int qdim = n_h * hd, kvdim = n_hkv * hd, rel_ext = hp_.rel_extent;
        const int max_c = 2048; // 32 pages/layer * 4 layers = 128 total pages
        const int fill_toks = 1216; // 19 pages/layer (76 pages total >> 6 VRAM frames -> 4 chunks/layer!)
        const int B_test = 7;

        float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_rel = nullptr, *d_snk = nullptr, *d_o1 = nullptr, *d_o2 = nullptr;
        INK_CUDA_CHECK(cudaMalloc(&d_Q,   (size_t)64 * qdim * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_K,   (size_t)64 * kvdim * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_V,   (size_t)64 * kvdim * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_rel, (size_t)64 * n_h * rel_ext * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_snk, (size_t)64 * n_h * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_o1,  (size_t)64 * qdim * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_o2,  (size_t)64 * qdim * sizeof(float)));

        std::vector<float> h_Q((size_t)64 * qdim), h_K((size_t)64 * kvdim), h_V((size_t)64 * kvdim);
        std::vector<float> h_rel((size_t)64 * n_h * rel_ext), h_snk((size_t)64 * n_h);
        std::vector<float> h_o1((size_t)B_test * qdim), h_o2((size_t)B_test * qdim);

        auto fill_synth = [&](int seed_base, int B_n) {
            for (int t = 0; t < B_n; ++t) {
                uint32_t s = (uint32_t)(seed_base + t * 977 + 13);
                auto next_f = [&]() -> float {
                    s = s * 1664525u + 1013904223u;
                    return ((float)(int32_t)(s >> 16) * (1.0f / 32768.0f)) - 1.0f;
                };
                for (int i = 0; i < qdim; ++i)  h_Q[(size_t)t * qdim + i] = next_f() * 0.5f;
                for (int i = 0; i < kvdim; ++i) h_K[(size_t)t * kvdim + i] = next_f() * 0.6f;
                for (int i = 0; i < kvdim; ++i) h_V[(size_t)t * kvdim + i] = next_f() * 0.6f;
                for (int i = 0; i < n_h; ++i)   h_snk[(size_t)t * n_h + i] = -1.5f + 0.2f * next_f();
                for (int i = 0; i < n_h * rel_ext; ++i) h_rel[(size_t)t * n_h * rel_ext + i] = -0.001f * (float)(i % rel_ext);
            }
            INK_CUDA_CHECK(cudaMemcpyAsync(d_Q,   h_Q.data(),   (size_t)B_n * qdim * sizeof(float), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_K,   h_K.data(),   (size_t)B_n * kvdim * sizeof(float), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_V,   h_V.data(),   (size_t)B_n * kvdim * sizeof(float), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_snk, h_snk.data(), (size_t)B_n * n_h * sizeof(float), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_rel, h_rel.data(), (size_t)B_n * n_h * rel_ext * sizeof(float), cudaMemcpyHostToDevice, stream_));
        };

        char tmpdir[MAX_PATH] = {};
        GetTempPathA(MAX_PATH, tmpdir);
        std::string r4_swap = std::string(tmpdir) + "inkengine_r4_test_swap.bin";
        size_t pg_bytes = 64ull * 2 * (size_t)n_hkv * (cfg_.kv_quant == KV_QUANT_Q4_0 ? 72ull : (cfg_.kv_quant == KV_QUANT_Q8_0 ? 136ull : 256ull));

        HybridPagedKV kv_ref, kv_press;
        bool ok_init = kv_ref.init(n_gl, all_glob, n_h, n_hkv, hd, D, 1024, max_c, cfg_.kv_quant, KV_ATTN_EXACT_STREAM,
                                   200ull * pg_bytes, 0, r4_swap + ".ref", 16, /*allow_tiny_vram_for_test=*/true)
                    && kv_press.init(n_gl, all_glob, n_h, n_hkv, hd, D, 1024, max_c, cfg_.kv_quant, KV_ATTN_EXACT_STREAM,
                                     6ull * pg_bytes, 8ull * pg_bytes, r4_swap, 16, /*allow_tiny_vram_for_test=*/true);

        bool r4_ok = ok_init;
        float max_e1 = 0.0f, max_e7 = 0.0f;
        if (ok_init) {
            for (int pos0 = 0; pos0 < fill_toks && r4_ok; pos0 += 64) {
                for (int il = 0; il < n_gl && r4_ok; ++il) {
                    fill_synth(pos0 * 17 + il * 100003, 64);
                    r4_ok = kv_ref.append_and_attend_batch(il, pos0, 64, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_)
                         && kv_press.append_and_attend_batch(il, pos0, 64, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                }
            }
            // Compare B=1 exact chunked attention across all 4 global layers (19 pages > 6 VRAM frames = 4 chunks!)
            for (int il = 0; il < n_gl && r4_ok; ++il) {
                fill_synth(99911 + il * 7919, 1);
                r4_ok = kv_ref.append_and_attend(il, fill_toks, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_)
                     && kv_press.append_and_attend(il, fill_toks, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                INK_CUDA_CHECK(cudaMemcpy(h_o1.data(), d_o1, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
                INK_CUDA_CHECK(cudaMemcpy(h_o2.data(), d_o2, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < qdim; ++i) max_e1 = std::max(max_e1, std::fabs(h_o1[i] - h_o2[i]));
            }
            // Compare B=7 exact chunked batched attention across all 4 global layers!
            for (int il = 0; il < n_gl && r4_ok; ++il) {
                fill_synth(177773 + il * 3571, B_test);
                r4_ok = kv_ref.append_and_attend_batch(il, fill_toks + 1, B_test, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_)
                     && kv_press.append_and_attend_batch(il, fill_toks + 1, B_test, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                INK_CUDA_CHECK(cudaMemcpy(h_o1.data(), d_o1, (size_t)B_test * qdim * sizeof(float), cudaMemcpyDeviceToHost));
                INK_CUDA_CHECK(cudaMemcpy(h_o2.data(), d_o2, (size_t)B_test * qdim * sizeof(float), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < (size_t)B_test * qdim; ++i) max_e7 = std::max(max_e7, std::fabs(h_o1[i] - h_o2[i]));
            }
        }
        auto pst = kv_press.stats();
        bool pass_4 = r4_ok && (max_e1 < 1e-4f) && (max_e7 < 1e-4f) && (pst.ssd_page_writes > 0) && (pst.ssd_page_reads > 0);
        std::printf("  [GATE 4] R4 Exact Multi-Chunk (6 VRAM / 8 RAM / SSD, 20p/L): B=1 err=%.6f | B=7 err=%.6f | ssd_rd=%llu ssd_wr=%llu -> %s\n",
                    max_e1, max_e7, (unsigned long long)pst.ssd_page_reads, (unsigned long long)pst.ssd_page_writes,
                    pass_4 ? "PASS" : "FAIL");
        if (!pass_4) all_ok = false;

        // GATE 5: R6 Fault-Injected Checked I/O & Atomic Publish (Spill Fail, Read Fail, Short Read, ExpertCache SSD Fail)
        bool pass_5a = false, pass_5b = false, pass_5c = false, pass_5d = false;
        if (ok_init) {
            int p_next = fill_toks + 1 + B_test;
            // 5A: Inject SSD spill write failure -> must fail cleanly AND keep victim page intact!
            fill_synth(555001, 1);
            kv_press.set_test_io_fault(/*fail_write_after=*/0, -1, false);
            bool wr_ret = kv_press.append_and_attend(0, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
            bool wr_err = kv_press.has_io_error();
            kv_press.set_test_io_fault(-1, -1, false);
            kv_press.clear_io_error();
            bool rec_a = kv_ref.append_and_attend(0, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_)
                      && kv_press.append_and_attend(0, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            INK_CUDA_CHECK(cudaMemcpy(h_o1.data(), d_o1, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
            INK_CUDA_CHECK(cudaMemcpy(h_o2.data(), d_o2, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
            float ea = 0.0f; for (int i = 0; i < qdim; ++i) ea = std::max(ea, std::fabs(h_o1[i] - h_o2[i]));
            pass_5a = (!wr_ret) && wr_err && rec_a && (ea < 1e-4f);

            // 5B: Inject SSD read failure -> must fail cleanly WITHOUT publishing bad page, then recover!
            fill_synth(555002, 1);
            kv_press.set_test_io_fault(-1, /*fail_read_after=*/0, false);
            bool rd_ret = kv_press.append_and_attend(1, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
            bool rd_err = kv_press.has_io_error();
            kv_press.set_test_io_fault(-1, -1, false);
            kv_press.clear_io_error();
            bool rec_b = kv_ref.append_and_attend(1, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_)
                      && kv_press.append_and_attend(1, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            INK_CUDA_CHECK(cudaMemcpy(h_o1.data(), d_o1, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
            INK_CUDA_CHECK(cudaMemcpy(h_o2.data(), d_o2, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
            float eb = 0.0f; for (int i = 0; i < qdim; ++i) eb = std::max(eb, std::fabs(h_o1[i] - h_o2[i]));
            pass_5b = (!rd_ret) && rd_err && rec_b && (eb < 1e-4f);

            // 5C: Inject partial/short read in QD-16 batch fetch -> must complete remaining bytes & match exact!
            fill_synth(555003, 1);
            kv_press.set_test_io_fault(-1, -1, /*short_read_once=*/true);
            bool sr_ret = kv_ref.append_and_attend(2, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_)
                       && kv_press.append_and_attend(2, p_next, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            INK_CUDA_CHECK(cudaMemcpy(h_o1.data(), d_o1, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
            INK_CUDA_CHECK(cudaMemcpy(h_o2.data(), d_o2, (size_t)qdim * sizeof(float), cudaMemcpyDeviceToHost));
            float ec = 0.0f; for (int i = 0; i < qdim; ++i) ec = std::max(ec, std::fabs(h_o1[i] - h_o2[i]));
            pass_5c = sr_ret && (!kv_press.has_io_error()) && (ec < 1e-4f);

            // 5D: ExpertCache SSD read failure -> fails cleanly without publishing corrupt slot or crashing!
            int test_uid = -1;
            for (int uid = 0; uid < hp_.n_layer_phys * hp_.n_expert; ++uid) {
                if (!experts_.is_in_vram(uid) && experts_.evict_ram_expert_for_test(uid)) {
                    test_uid = uid;
                    break;
                }
            }
            if (test_uid >= 0) {
                experts_.clear_io_error();
                experts_.set_test_ssd_fail(true);
                int rslot_bad = -1;
                const uint8_t* p_bad = experts_.pin_ram_expert(test_uid, &rslot_bad);
                MoESplitPacket fake_pkt{};
                fake_pkt.n_gpu = 7;
                fake_pkt.n_cpu = 1;
                for (int k = 0; k < 8; ++k) fake_pkt.all_ids[k] = k;
                fake_pkt.miss_ids[0] = test_uid % hp_.n_expert;
                fake_pkt.miss_weights[0] = 1.0f;
                CPUExpertTask tmp_tasks[8];
                int n_cpu_out = 0;
                bool split_ok = experts_.resolve_gpu_split(test_uid / hp_.n_expert, fake_pkt, tmp_tasks, n_cpu_out);
                bool exp_err = experts_.has_io_error();
                experts_.set_test_ssd_fail(false);
                experts_.clear_io_error();
                int rslot_ok = -1;
                const uint8_t* p_ok = experts_.pin_ram_expert(test_uid, &rslot_ok);
                if (p_ok && rslot_ok >= 0) experts_.unpin_ram_slots(&rslot_ok, 1);
                pass_5d = (p_bad == nullptr) && (!split_ok) && exp_err && (p_ok != nullptr);
            } else {
                pass_5d = true; // All experts fit 100% in VRAM (e.g. >=16GB GPU)
            }
        }
        bool pass_5 = pass_5a && pass_5b && pass_5c && pass_5d;
        std::printf("  [GATE 5] R6 Checked I/O (spill_fail=%s | read_fail=%s | short_read=%s | exp_ssd_fail=%s) -> %s\n",
                    pass_5a ? "PASS" : "FAIL", pass_5b ? "PASS" : "FAIL",
                    pass_5c ? "PASS" : "FAIL", pass_5d ? "PASS" : "FAIL",
                    pass_5 ? "PASS" : "FAIL");
        if (!pass_5) all_ok = false;

        // -------------------------------------------------------------------------
        // TEST 6 (R8 / SR3-07): Quest Extrema Rollback Parity Across 64-Token Page Boundary!
        // -------------------------------------------------------------------------
        std::printf("\n[GATE 6] R8 Quest Extrema Speculative & Draft Rollback Parity (across 64-tok boundary)...\n");
        {
            const int q_gl = 2;
            std::vector<uint8_t> q_mask(q_gl, 0);
            HybridPagedKV q_clean, q_spec;
            bool q_init = q_clean.init(q_gl, q_mask, n_h, n_hkv, hd, D, 1024, 512, cfg_.kv_quant, KV_ATTN_QUEST,
                                       16ull * pg_bytes, 0, r4_swap + ".q1", 8, true)
                       && q_spec.init(q_gl, q_mask, n_h, n_hkv, hd, D, 1024, 512, cfg_.kv_quant, KV_ATTN_QUEST,
                                      16ull * pg_bytes, 0, r4_swap + ".q2", 8, true);
            bool pass_6a = false, pass_6b = false;
            if (q_init) {
                // Feed identical 61 tokens [0..60] into page 0
                for (int il = 0; il < q_gl; ++il) {
                    fill_synth(70001 + il * 997, 61);
                    q_clean.append_and_attend_batch(il, 0, 61, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_);
                    q_spec.append_and_attend_batch(il, 0, 61, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                }
                // 6A: Spec verify of B=6 tokens [61..66] crossing page boundary 63->64; tokens [63..66] are +/-50 outliers!
                q_spec.begin_spec_verify(61, 6, stream_);
                for (int il = 0; il < q_gl; ++il) {
                    fill_synth(71001 + il * 997, 6);
                    // Clean gets only accepted tokens [61, 62]
                    q_clean.append_and_attend_batch(il, 61, 2, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_);
                    // Inject huge +/-50.0 outlier keys into rejected draft tokens [2..5] (pos 63..66)!
                    for (int t = 2; t < 6; ++t) {
                        for (int i = 0; i < kvdim; ++i) h_K[(size_t)t * kvdim + i] = ((i & 1) ? 50.0f : -50.0f);
                    }
                    INK_CUDA_CHECK(cudaMemcpyAsync(d_K, h_K.data(), (size_t)6 * kvdim * sizeof(float), cudaMemcpyHostToDevice, stream_));
                    q_spec.append_and_attend_batch(il, 61, 6, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                }
                q_spec.commit_spec_verify(/*accepted=*/2, stream_); // Rejects [63..66], restores page 0 extrema!
                // Append real token 63 to both so active_k_min/max is ALSO proven bit-exact!
                for (int il = 0; il < q_gl; ++il) {
                    fill_synth(72001 + il * 997, 1);
                    q_clean.append_and_attend(il, 63, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_);
                    q_spec.append_and_attend(il, 63, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                }
                pass_6a = true;
                for (int g = 0; g < q_gl; ++g) {
                    std::vector<int8_t> mn1, mx1, mn2, mx2;
                    std::vector<float> sc1, sc2, amn1, amx1, amn2, amx2;
                    bool ok_c1 = q_clean.copy_quest_page_summary(g, 0, mn1, mx1, sc1, amn1, amx1, stream_);
                    bool ok_c2 = q_spec.copy_quest_page_summary(g, 0, mn2, mx2, sc2, amn2, amx2, stream_);
                    // Also assert scale < 0.1f (proves +/-50.0 outlier whose scale would be ~0.394 was truly erased!)
                    if (!ok_c1 || !ok_c2 || mn1.empty() || sc1.empty() || sc1[0] >= 0.1f ||
                        mn1 != mn2 || mx1 != mx2 || sc1 != sc2 || amn1 != amn2 || amx1 != amx2) {
                        pass_6a = false;
                    }
                }

                // 6B: Draft snapshot full rollback on page 1 with +/-50 outliers
                for (int il = 0; il < q_gl; ++il) {
                    fill_synth(73001 + il * 997, 2);
                    q_clean.append_and_attend_batch(il, 64, 2, d_Q, d_K, d_V, d_rel, d_snk, d_o1, rel_ext, stream_);
                    q_spec.append_and_attend_batch(il, 64, 2, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                }
                q_spec.begin_draft_snapshot(66, 3, stream_);
                for (int t = 0; t < 3; ++t) {
                    for (int il = 0; il < q_gl; ++il) {
                        fill_synth(74001 + t * 31 + il * 997, 1);
                        for (int i = 0; i < kvdim; ++i) h_K[i] = ((i & 1) ? 50.0f : -50.0f);
                        INK_CUDA_CHECK(cudaMemcpyAsync(d_K, h_K.data(), (size_t)kvdim * sizeof(float), cudaMemcpyHostToDevice, stream_));
                        q_spec.append_and_attend(il, 66 + t, d_Q, d_K, d_V, d_rel, d_snk, d_o2, rel_ext, stream_);
                    }
                }
                q_spec.rollback_draft_snapshot(66, 3, stream_);
                pass_6b = true;
                for (int g = 0; g < q_gl; ++g) {
                    std::vector<int8_t> mn1, mx1, mn2, mx2;
                    std::vector<float> sc1, sc2, amn1, amx1, amn2, amx2;
                    bool ok_c1 = q_clean.copy_quest_page_summary(g, 1, mn1, mx1, sc1, amn1, amx1, stream_);
                    bool ok_c2 = q_spec.copy_quest_page_summary(g, 1, mn2, mx2, sc2, amn2, amx2, stream_);
                    if (!ok_c1 || !ok_c2 || mn1.empty() || sc1.empty() || sc1[0] >= 0.1f ||
                        mn1 != mn2 || mx1 != mx2 || sc1 != sc2 || amn1 != amn2 || amx1 != amx2) {
                        pass_6b = false;
                    }
                }
            }
            bool pass_6 = pass_6a && pass_6b;
            std::printf("  Spec partial rollback (p0->p1): %s | Draft full rollback (p1): %s -> %s\n",
                        pass_6a ? "BIT-EXACT" : "MISMATCH", pass_6b ? "BIT-EXACT" : "MISMATCH",
                        pass_6 ? "PASS" : "FAIL");
            if (!pass_6) all_ok = false;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_rel); cudaFree(d_snk); cudaFree(d_o1); cudaFree(d_o2);
    }

    // -------------------------------------------------------------------------
    // TEST 7 (R10 / SR3-05): Predictor Route Bookkeeping Exactness (Batched Prefill + Spec Commit)
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 7] R10 Predictor Route Bookkeeping Exactness (batched decay & speculative commit)...\n");
    if (predictor_active()) {
        SwapFreezeGuard guard(experts_);
        auto sum_map = [&](MoePredMapKind k) -> double {
            double s = 0.0;
            for (float v : pred_tracker_.map(k)) s += (double)v;
            return s;
        };
        const double per_tok = (double)hp_.n_layer_logical * (double)hp_.n_expert_used; // 48 * 8 = 384

        // 7A: Batched prefill (B=128) must match exact per-token geometric decay sum_{t=0}^{127} d^t!
        pred_tracker_.reset_conversation();
        pred_tracker_.begin_prompt_turn();
        kv_.restore_checkpoint(ckpt_1024, stream_);
        forward_batch(toks.data() + 1024, 128, 1024, false, false, /*is_answer=*/false, false, false);
        double d64 = std::pow(2.0, -1.0 / 64.0);
        double expected_live64 = per_tok * (1.0 - std::pow(d64, 128)) / (1.0 - d64);
        double expected_turn   = per_tok * 128.0;
        double got_live64 = sum_map(MP_LIVE64);
        double got_turn   = sum_map(MP_TURN);
        bool pass_7a = (std::fabs(got_turn - expected_turn) < 1e-2) &&
                       (std::fabs(got_live64 - expected_live64) < 1.0);

        // 7B: Speculative decode (accepted > 0) must record EVERY accepted token's routes in token order!
        SamplingParams gsp;
        gsp.temperature = 0.0f;
        gsp.max_tokens = 64;
        reset_context();
        cfg_.use_batched_prefill = true;
        cfg_.use_speculative = true;
        auto res_p = generate(toks, gsp, nullptr, false, false);
        int64_t ans_n = pred_tracker_.answer_pos();
        double d8 = std::pow(2.0, -1.0 / 8.0);
        double expected_ans  = per_tok * (double)ans_n;
        double expected_ans8 = (ans_n > 0) ? (per_tok * (1.0 - std::pow(d8, (double)ans_n)) / (1.0 - d8)) : 0.0;
        double got_ans  = sum_map(MP_ANSWER);
        double got_ans8 = sum_map(MP_ANSWER8);
        bool pass_7b = (res_p.spec_accepted > 0) && (ans_n > 0) &&
                       (std::fabs(got_ans - expected_ans) < 1e-2) &&
                       (std::fabs(got_ans8 - expected_ans8) < 1.0);

        bool pass_7 = pass_7a && pass_7b;
        std::printf("  Batched B=128 LIVE64=%.2f (exp=%.2f) TURN=%.0f (exp=%.0f) -> %s\n",
                    got_live64, expected_live64, got_turn, expected_turn, pass_7a ? "PASS" : "FAIL");
        std::printf("  Spec Commit (acc=%d/%d, ans_n=%lld) ANS=%.0f (exp=%.0f) ANS8=%.2f (exp=%.2f) -> %s\n",
                    res_p.spec_accepted, res_p.spec_drafted, (long long)ans_n,
                    got_ans, expected_ans, got_ans8, expected_ans8, pass_7b ? "PASS" : "FAIL");
        if (!pass_7) all_ok = false;
    } else {
        std::printf("  Predictor not loaded -> SKIP\n");
    }

    // -------------------------------------------------------------------------
    // TEST 8 (R14 / SR1-03): Bit-Exact GPU Sampling Pre-Filter Parity & Speed
    // -------------------------------------------------------------------------
    std::printf("\n[GATE 8] R14 Bit-Exact GPU Sampling Pre-Filter Parity (200 trials across 4 regimes) & Speed...\n");
    {
        SwapFreezeGuard guard(experts_);
        kv_.restore_checkpoint(ckpt_1024, stream_);
        int32_t argm = forward_token(toks[1024], 1024, true, false, false, false);

        SamplingParams regimes[4];
        regimes[0] = {1.0f, 0,  1.0f,  0.0f,  1, 1}; // Official Inkling-2 default
        regimes[1] = {0.6f, 40, 0.95f, 0.05f, 1, 1}; // Tight coding
        regimes[2] = {1.8f, 0,  0.98f, 0.0f,  1, 1}; // High-temperature wide tail
        regimes[3] = {1.0f, 0,  1.0f,  0.0f,  1, 1}; // Speculative verify with draft_tok

        int mismatches = 0;
        int total_trials = 0;
        for (int r = 0; r < 4; ++r) {
            for (int s = 1; s <= 50; ++s) {
                uint64_t rng1 = 0x9E3779B97F4A7C15ULL * (uint64_t)(r * 100 + s) + 17;
                uint64_t rng2 = rng1;
                int32_t dtok = (r == 3) ? ((s & 1) ? argm : ((argm + 7) % hp_.vocab_unpadded)) : -1;
                bool acc1 = false, acc2 = false;
                int32_t t_gpu = sample_from_logits(d_logits_, argm, regimes[r], rng1, dtok, r == 3 ? &acc1 : nullptr);
                int32_t t_cpu = sample_from_logits_cpu_ref(d_logits_, argm, regimes[r], rng2, dtok, r == 3 ? &acc2 : nullptr);
                total_trials++;
                if (t_gpu != t_cpu || rng1 != rng2 || (r == 3 && acc1 != acc2)) mismatches++;
            }
        }
        // Micro-benchmark GPU pre-filter vs full 800 KB host copy at official T=1.0, min_p=0.0
        uint64_t rb1 = 1234567ULL, rb2 = 1234567ULL;
        double tg0 = now_ms();
        for (int i = 0; i < 100; ++i) sample_from_logits(d_logits_, argm, regimes[0], rb1);
        double us_gpu = (now_ms() - tg0) * 10.0;
        double tc0 = now_ms();
        for (int i = 0; i < 100; ++i) sample_from_logits_cpu_ref(d_logits_, argm, regimes[0], rb2);
        double us_cpu = (now_ms() - tc0) * 10.0;

        bool pass_8 = (mismatches == 0);
        std::printf("  Bit-Exact Trials: %d/%d MATCH | Speed: %.1f us (GPU pre-filter) vs %.1f us (800 KB CPU) = %.2fx faster -> %s\n",
                    total_trials - mismatches, total_trials, us_gpu, us_cpu,
                    us_gpu > 0.0 ? (us_cpu / us_gpu) : 1.0, pass_8 ? "PASS (BIT-EXACT)" : "FAIL");
        if (!pass_8) all_ok = false;
    }

    cfg_.use_batched_prefill = orig_batch;
    cfg_.use_speculative = orig_spec;
    std::printf("================================================================\n");
    std::printf("  [PARITY SUITE R0/v4.0] OVERALL: %s\n", all_ok ? "ALL GATES PASSED!" : "FAILED!");
    std::printf("================================================================\n");
    std::fflush(stdout);
    return all_ok;
}
