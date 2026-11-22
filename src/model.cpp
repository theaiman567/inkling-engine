#include "model.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

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
    last_tau_global_ = -1.0f;

    free_h(h_embd_row_); free_h(h_top_ids_); free_h(h_top_weights_);
    free_h(h_ffn_in_); free_h(h_split_pkt_); free_h(h_zero_sync_ids_);
    free_h(h_moe_cpu_out_); free_h(h_argmax_); free_h(h_pred_hidden_);
    pred_hidden_slots_ = 0;
    d_mapped_ffn_in_ = nullptr;
    d_split_pkt_ = nullptr;

    if (ev_split_) { cudaEventDestroy(ev_split_); ev_split_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    kv_tokens_.clear();
    full_sys_ckpt_ = KVCheckpoint{};
    prefix_ckpt_ = KVCheckpoint{};
    tools_ckpt_ = KVCheckpoint{};
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
    std::printf("  InkEngine v3.0 - Custom Looped-MoE 3-Tier Serving Engine\n");
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

    INK_CUDA_CHECK(cudaHostAlloc(&h_embd_row_,      D * sizeof(float), cudaHostAllocWriteCombined));
    INK_CUDA_CHECK(cudaHostAlloc(&h_top_ids_,       hp_.n_expert_used * sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_top_weights_,   hp_.n_expert_used * sizeof(float), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_ffn_in_,        D * sizeof(float), cudaHostAllocMapped));
    INK_CUDA_CHECK(cudaHostGetDevicePointer(&d_mapped_ffn_in_, h_ffn_in_, 0));
    INK_CUDA_CHECK(cudaHostAlloc(&h_split_pkt_,     sizeof(MoESplitPacket), cudaHostAllocMapped));
    INK_CUDA_CHECK(cudaHostGetDevicePointer(&d_split_pkt_, h_split_pkt_, 0));
    INK_CUDA_CHECK(cudaHostAlloc(&h_zero_sync_ids_, (size_t)hp_.n_layer_logical * hp_.n_expert_used * sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_moe_cpu_out_,   D * sizeof(float), cudaHostAllocWriteCombined));
    INK_CUDA_CHECK(cudaHostAlloc(&h_argmax_,        sizeof(int32_t), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_split_, cudaEventDisableTiming));

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
        INK_CUDA_CHECK(cudaMemcpyAsync(d_tau_global_, &tau_global, sizeof(float), cudaMemcpyHostToDevice, stream_));
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
        kv_.append_and_attend(il, pos, d_Q_, d_K_conv_, d_V_conv_, d_rel_table_, d_sinks_, d_attn_out_, rel_ext, stream_);

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
            // GPU-FIRST ZERO-BUBBLE HYBRID PATH!
            ink_cuda_moe_split_hits(tbl, d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                    d_ffn_normed_, D,
                                    d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_,
                                    d_split_pkt_, d_mapped_ffn_in_, stream_);
            INK_CUDA_CHECK(cudaEventRecord(ev_split_, stream_));

            ink_cuda_moe_dynamic_indirect(lm.gate_up_type, lm.down_type, tbl,
                                          d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_, hp_.n_expert_used,
                                          d_ffn_normed_, d_moe_h_, d_moe_out_,
                                          D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);

            cudaStreamQuery(stream_);
            INK_CUDA_CHECK(cudaEventSynchronize(ev_split_));

            CPUExpertTask cpu_tasks[8];
            int n_cpu = 0;
            experts_.resolve_gpu_split(ip, *h_split_pkt_, cpu_tasks, n_cpu);
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
    } else if (n_zero_sync > 0 || capture_pred_hidden) {
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
        INK_CUDA_CHECK(cudaMemcpyAsync(d_tau_global_, &tau_global, sizeof(float), cudaMemcpyHostToDevice, stream_));
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
        kv_.append_and_attend(il, pos, d_Q_, d_K_conv_, d_V_conv_, d_rel_table_, d_sinks_, d_attn_out_, rel_ext, stream_);

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
                                    nullptr, nullptr, stream_);
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

    // Double-buffered PCIe staging ring for Layer-Wise Tensor-Core Prefill (28 MiB VRAM + 28 MiB pinned RAM)
    int32_t seq_ids[STAGE_EXPS];
    for (int i = 0; i < STAGE_EXPS; ++i) seq_ids[i] = i;
    for (int s = 0; s < STAGE_BUFS; ++s) {
        INK_CUDA_CHECK(cudaMalloc(&d_prefill_stage_[s], (size_t)STAGE_EXPS * STAGE_SLOT_BYTES));
        INK_CUDA_CHECK(cudaHostAlloc(&h_prefill_stage_[s], (size_t)STAGE_EXPS * STAGE_SLOT_BYTES, cudaHostAllocWriteCombined));
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
    batch_ready_ = false;
}

int32_t InklingEngine::forward_batch(
    const int32_t* tokens, int B, int pos0,
    bool need_last_logits, bool need_all_logits,
    bool is_answer, bool capture_last_pred_hidden,
    bool save_spec_states,
    int32_t* out_argmax_batch)
{
    if (B <= 0) return -1;
    if (B == 1 && !save_spec_states && !need_all_logits) {
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
    const auto& pred_hid_layers = pred_main_.hidden_layers();
    const size_t spec_stride = kv_.conv_spec_stride();

    if (track_pred) {
        for (int b = 0; b < B; ++b) {
            if (is_answer) pred_tracker_.step_answer_token();
            else           pred_tracker_.step_prompt_token();
        }
    }

    // 1. Parallel CPU token embedding dequantize -> single H2D upload + Batched RMSNorm
    cpu_pool_->parallel_for_threads([&](int tid, int nth) {
        for (int b = tid; b < B; b += nth) {
            const uint8_t* src = h_tok_embd_.data() + (size_t)tokens[b] * tok_embd_row_bytes_;
            CPUThreadPool::dequantize_row(tok_embd_type_, src, h_b_embd_ + (size_t)b * D, D);
        }
    });
    INK_CUDA_CHECK(cudaMemcpyAsync(d_b_cur_, h_b_embd_, (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice, stream_));
    ink_cuda_rms_norm_batch(d_b_cur_, (const float*)tok_norm_.ptr, d_b_cur_, B, D, hp_.rms_eps, stream_);

    const int n_loop_core = (hp_.num_loops > 1) ? (hp_.core_end - hp_.core_start) : 0;
    int n_blocks = 0, prelude_n_blocks = 0, n_bank = 0;

    // Reusable per-expert bucket vectors for grouped hybrid MoE
    int bucket_cnt[128];
    int bucket_tok[128][BATCH_MAX];
    float bucket_w[128][BATCH_MAX];
    CPUExpertGroupTask cpu_groups[128];
    int32_t cpu_tok_ids[BATCH_MAX * 8];
    float   cpu_tok_w[BATCH_MAX * 8];

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

        kv_.append_and_attend_batch(il, pos0, B, d_b_Q_, d_b_K_conv_, d_b_V_conv_,
                                    d_b_rel_table_, d_b_sinks_, d_b_attn_out_, rel_ext, stream_);

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

        INK_CUDA_CHECK(cudaMemcpyAsync(h_b_top_ids_,     d_b_top_ids_,     (size_t)B * K_exp * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(h_b_top_weights_, d_b_top_weights_, (size_t)B * K_exp * sizeof(float),   cudaMemcpyDeviceToHost, stream_));
        if (!all_vram && !may_stream_pcie) {
            INK_CUDA_CHECK(cudaMemcpyAsync(h_b_ffn_in_, d_b_ffn_normed_, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        }
        INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

        std::memset(bucket_cnt, 0, sizeof(bucket_cnt));
        for (int b = 0; b < B; ++b) {
            const int32_t* bids = h_b_top_ids_ + (size_t)b * K_exp;
            const float*   bw   = h_b_top_weights_ + (size_t)b * K_exp;
            if (track_pred) pred_tracker_.record_selections(ip, bids, K_exp, is_answer);
            for (int k = 0; k < K_exp; ++k) {
                int e = bids[k];
                int c = bucket_cnt[e]++;
                bucket_tok[e][c] = b;
                bucket_w[e][c]   = bw[k];
            }
        }

        int n_gpu_grp = 0, gpu_pairs = 0;
        int n_cpu_grp = 0, cpu_pairs = 0;
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
                const uint8_t* rptr = experts_.get_ram_expert_ptr(uid);
                CPUExpertGroupTask& cg = cpu_groups[n_cpu_grp++];
                cg.gate_up_data  = rptr;
                cg.down_data     = rptr + lm.down_aligned_off;
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
            // Double-Buffered Layer-Wise PCIe Streaming -> 100% RTX 4060 Tensor Core MoE (ZERO CPU bottleneck!)
            cudaStream_t s_dma = experts_.dma_stream();
            const size_t exp_bytes = lm.combined_bytes;
            bool x_ready = (n_gpu_grp > 0);
            int batch_idx = 0;
            for (int base_g = 0; base_g < n_cpu_grp; base_g += STAGE_EXPS, ++batch_idx) {
                int buf = batch_idx & 1;
                int cnt_g = std::min(STAGE_EXPS, n_cpu_grp - base_g);
                if (batch_idx >= STAGE_BUFS) {
                    INK_CUDA_CHECK(cudaEventSynchronize(ev_stage_gpu_[buf]));
                }
                // Parallel DDR4 memcpy of 1..4 cold experts into pinned WC bounce buffer (~0.35 ms!)
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
                INK_CUDA_CHECK(cudaMemcpyAsync(d_prefill_stage_[buf], h_prefill_stage_[buf], (size_t)cnt_g * exp_bytes, cudaMemcpyHostToDevice, s_dma));
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
            }
            // Ensure staging buffers are done on GPU before next layer reuses them
            int used_bufs = std::min(batch_idx, STAGE_BUFS);
            for (int b = 0; b < used_bufs; ++b) {
                INK_CUDA_CHECK(cudaEventSynchronize(ev_stage_gpu_[b]));
            }
        } else if (n_cpu_grp > 0) {
            // Ultra-low-latency CPU AVX2 overlap for tiny batches (e.g. Speculative Verification B <= 7!)
            cudaStreamQuery(stream_);
            cpu_pool_->run_moe_experts_grouped(h_b_ffn_in_, h_b_moe_cpu_out_, B,
                                               cpu_groups, n_cpu_grp, cpu_tok_ids, cpu_tok_w, cpu_pairs,
                                               D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta);
            INK_CUDA_CHECK(cudaMemcpyAsync(d_b_moe_cpu_out_, h_b_moe_cpu_out_, (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice, stream_));
            ink_cuda_add_inplace(d_b_moe_out_, d_b_moe_cpu_out_, B * D, stream_);
        }

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

int32_t InklingEngine::sample_from_logits(
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
    float thresh = max_l + std::log(std::max(sp.min_p, 1e-6f));
    for (int i = 0; i < V; ++i) {
        if (logits[i] >= thresh) cand.push_back({logits[i], i});
    }
    if (cand.empty()) {
        if (out_accepted) *out_accepted = (draft_tok >= 0 && argmax_val == draft_tok);
        return argmax_val;
    }

    int keep = (sp.top_k > 0 && (int)cand.size() > sp.top_k) ? sp.top_k : (int)cand.size();
    std::partial_sort(cand.begin(), cand.begin() + keep, cand.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    cand.resize(keep);

    float sum = 0.0f;
    for (auto& c : cand) { c.first = std::exp(c.first - max_l); sum += c.first; }
    for (auto& c : cand) c.first /= sum;

    if (sp.top_p < 1.0f && cand.size() > 1) {
        float cum = 0.0f;
        size_t cut = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            cum += cand[i].first;
            cut = i + 1;
            if (cum >= sp.top_p) break;
        }
        cand.resize(cut);
        sum = 0.0f;
        for (auto& c : cand) sum += c.first;
        for (auto& c : cand) c.first /= sum;
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
        // Sample residual distribution p'(x) = p(x) / (1 - p(draft)) for x != draft
        if (d_idx >= 0 && cand.size() > 1) {
            float rem = 1.0f - p_draft;
            double r = next_u() * (double)rem;
            float acc = 0.0f;
            for (size_t i = 0; i < cand.size(); ++i) {
                if ((int)i == d_idx) continue;
                acc += cand[i].first;
                if (r <= acc) return cand[i].second;
            }
            for (int i = (int)cand.size() - 1; i >= 0; --i) {
                if (i != d_idx) return cand[i].second;
            }
        }
        return cand[0].second;
    }

    double r = next_u();
    float acc = 0.0f;
    for (const auto& c : cand) {
        acc += c.first;
        if (r <= acc) return c.second;
    }
    return cand.back().second;
}

int32_t InklingEngine::sample_next(const SamplingParams& sp, uint64_t& rng_state) {
    return sample_from_logits(d_logits_, *h_argmax_, sp, rng_state);
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
    int full_sys_prefix_tokens)
{
    GenResult res;
    res.prompt_tokens = (int)prompt_tokens.size();
    if (prompt_tokens.empty()) return res;

    int P = (int)prompt_tokens.size();
    if (P >= cfg_.max_ctx) {
        std::fprintf(stderr, "[Engine] Prompt length %d >= max_ctx %d; truncating to fit.\n", P, cfg_.max_ctx);
        P = cfg_.max_ctx - 1;
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

        if (L > 0 && P > L && std::equal(kv_tokens_.begin(), kv_tokens_.end(), prompt_tokens.begin())) {
            // Exact incremental continuation (e.g. tool-call follow-up turn -> 0 ms restore!)
            start_pos = L;
            res.cached_tokens = L;
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

    double t_pf0 = now_ms();
    int total_new = P - start_pos;
    if (predictor_active() && total_new > 0) {
        pred_tracker_.begin_prompt_turn();
        run_start_predictor(prompt_tokens.data() + start_pos, total_new);
    }

    // Stage 4: Batched GEMM + Grouped Hybrid MoE Prefill (B <= 64, ~3x faster TTFT!)
    if (cfg_.use_batched_prefill && total_new > 1) {
        int i = start_pos;
        while (i < P) {
            int chunk = std::min(BATCH_MAX, P - i);
            // Align chunk boundaries cleanly on all 3 prefix KV checkpoints!
            if (want_tools_ckpt >= 256 && i < want_tools_ckpt && i + chunk > want_tools_ckpt) {
                chunk = want_tools_ckpt - i;
            }
            if (want_ckpt >= 128 && i < want_ckpt && i + chunk > want_ckpt) {
                chunk = want_ckpt - i;
            }
            if (want_full_ckpt >= 128 && i < want_full_ckpt && i + chunk > want_full_ckpt) {
                chunk = want_full_ckpt - i;
            }
            bool is_last = (i + chunk == P);
            forward_batch(prompt_tokens.data() + i, chunk, i,
                          /*need_last_logits=*/is_last, /*need_all_logits=*/false,
                          /*is_answer=*/false, /*capture_last_pred_hidden=*/is_last && pred_main_.loaded(),
                          /*save_spec_states=*/false);
            for (int k = 0; k < chunk; ++k) kv_tokens_.push_back(prompt_tokens[i + k]);
            i += chunk;

            maybe_save_ckpt(tools_ckpt_,    want_tools_ckpt, i, "tools-prefix");
            maybe_save_ckpt(prefix_ckpt_,   want_ckpt,       i, "static-prefix");
            maybe_save_ckpt(full_sys_ckpt_, want_full_ckpt,  i, "full-system");

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
            kv_tokens_.push_back(prompt_tokens[i]);

            maybe_save_ckpt(tools_ckpt_,    want_tools_ckpt, i + 1, "tools-prefix");
            maybe_save_ckpt(prefix_ckpt_,   want_ckpt,       i + 1, "static-prefix");
            maybe_save_ckpt(full_sys_ckpt_, want_full_ckpt,  i + 1, "full-system");

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

    auto draft_ngram = [&](int32_t cur_tok, int max_draft, int32_t* out_draft) -> int {
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

        int k = 0;
        for (; k < max_draft && (best_idx + 1 + k) < H; ++k) {
            int32_t cand = kv_tokens_[best_idx + 1 + k];
            if (tok_.is_eog(cand)) break;
            out_draft[k] = cand;
        }
        return k;
    };

    int32_t next_tok = sample_next(sp, rng);
    while (res.gen_tokens < sp.max_tokens) {
        if (tok_.is_eog(next_tok)) break;

        std::string piece = tok_.decode_token(next_tok);
        res.tokens.push_back(next_tok);
        res.text += piece;
        res.gen_tokens++;

        if (on_token && !on_token(next_tok, piece)) break;
        if (res.gen_tokens >= sp.max_tokens || pos >= cfg_.max_ctx - 1) break;

        bool need_pred_now = pred_main_.loaded() && (res.gen_tokens >= next_pred_at);

        // Stage 5 / v3.0: Zero-VRAM Lossless Speculative Decoding!
        //  1) O(1) Rolling 3-Gram Suffix Hash Drafter (K <= 6, <1 us, instant on code/JSON/edits!)
        //  2) 1-Loop (24L) GPU-Only Self-Speculative Fallback (zero extra VRAM, 79% Top-1 agreement on general prose!)
        int32_t draft_toks[HybridPagedKV::SPEC_MAX_B];
        int max_d = std::min(HybridPagedKV::SPEC_MAX_B - 2, std::min(sp.max_tokens - res.gen_tokens, cfg_.max_ctx - 2 - pos));
        int K_draft = (cfg_.use_speculative && !enable_profile && !need_pred_now && max_d >= 2)
            ? draft_ngram(next_tok, max_d, draft_toks)
            : 0;

        if (K_draft == 0 && cfg_.use_speculative && cfg_.self_draft_k > 0 && !enable_profile && !need_pred_now && max_d >= 2) {
            int k_self = std::min(cfg_.self_draft_k, max_d);
            kv_.begin_draft_snapshot(pos, k_self, stream_);
            int32_t d_in = next_tok;
            int drafted = 0;
            for (int s = 0; s < k_self; ++s) {
                int32_t d_out = forward_token_1loop_draft(d_in, pos + s);
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

            kv_.begin_spec_verify(pos, B, stream_);
            forward_batch(batch_toks, B, pos,
                          /*need_last_logits=*/false, /*need_all_logits=*/true,
                          /*is_answer=*/true, /*capture_last_pred_hidden=*/false,
                          /*save_spec_states=*/true, argm);

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
            for (int a = 0; a < accepted; ++a) {
                kv_tokens_.push_back(batch_toks[a]);
                if (predictor_active()) pred_tracker_.step_answer_token();
            }
            pos += accepted;
            res.spec_drafted += K_draft;
            res.spec_steps++;

            if (stopped || res.gen_tokens >= sp.max_tokens || pos >= cfg_.max_ctx - 1) break;
            next_tok = bonus_tok;
        } else {
            forward_token(next_tok, pos, true, enable_profile, /*is_answer=*/true, /*capture_pred_hidden=*/need_pred_now);
            if (need_pred_now) {
                run_main_predictor(pred_tracker_.answer_pos(), /*is_answer_start=*/false);
                next_pred_at = res.gen_tokens + pred_every;
            }
            kv_tokens_.push_back(next_tok);
            pos++;
            next_tok = sample_next(sp, rng);
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
                ink_cuda_moe_split_hits(tbl, d_top_ids_, d_top_weights_, hp_.n_expert_used,
                                        d_ffn_normed_, D, d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_,
                                        d_split_pkt_, d_mapped_ffn_in_, stream_);
                INK_CUDA_CHECK(cudaEventRecord(ev_split_, stream_));
                ink_cuda_moe_dynamic_indirect(lm.gate_up_type, lm.down_type, tbl,
                                              d_n_gpu_, d_gpu_hit_ids_, d_gpu_hit_weights_, hp_.n_expert_used,
                                              d_ffn_normed_, d_moe_h_, d_moe_out_,
                                              D, hp_.n_ff_exp, hp_.situ_beta, hp_.situ_linear_beta, stream_);
                INK_CUDA_CHECK(cudaEventSynchronize(ev_split_));
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
