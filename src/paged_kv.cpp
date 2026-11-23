#include "paged_kv.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

HybridPagedKV::~HybridPagedKV() { shutdown(); }

void HybridPagedKV::shutdown() {
    if (d_conv_state_)   { cudaFree(d_conv_state_);   d_conv_state_ = nullptr; }
    if (d_swa_k_)        { cudaFree(d_swa_k_);        d_swa_k_ = nullptr; }
    if (d_swa_v_)        { cudaFree(d_swa_v_);        d_swa_v_ = nullptr; }
    if (d_swa_pos_)      { cudaFree(d_swa_pos_);      d_swa_pos_ = nullptr; }
    if (d_vram_pages_)   { cudaFree(d_vram_pages_);   d_vram_pages_ = nullptr; }
    if (d_k_mins_q8_)    { cudaFree(d_k_mins_q8_);    d_k_mins_q8_ = nullptr; }
    if (d_k_maxs_q8_)    { cudaFree(d_k_maxs_q8_);    d_k_maxs_q8_ = nullptr; }
    if (d_k_scales_)     { cudaFree(d_k_scales_);     d_k_scales_ = nullptr; }
    if (d_active_k_min_) { cudaFree(d_active_k_min_); d_active_k_min_ = nullptr; }
    if (d_active_k_max_) { cudaFree(d_active_k_max_); d_active_k_max_ = nullptr; }
    if (d_split_out_)    { cudaFree(d_split_out_);    d_split_out_ = nullptr; }
    if (d_split_ml_)     { cudaFree(d_split_ml_);     d_split_ml_ = nullptr; }
    if (d_page_k_ptrs_)  { cudaFree(d_page_k_ptrs_);  d_page_k_ptrs_ = nullptr; }
    if (d_page_v_ptrs_)  { cudaFree(d_page_v_ptrs_);  d_page_v_ptrs_ = nullptr; }
    if (d_page_pos0_)    { cudaFree(d_page_pos0_);    d_page_pos0_ = nullptr; }
    if (d_page_lens_)    { cudaFree(d_page_lens_);    d_page_lens_ = nullptr; }
    if (d_quest_scores_)   { cudaFree(d_quest_scores_);   d_quest_scores_ = nullptr; }
    if (d_quest_topk_ids_) { cudaFree(d_quest_topk_ids_); d_quest_topk_ids_ = nullptr; }
    if (d_conv_spec_)      { cudaFree(d_conv_spec_);      d_conv_spec_ = nullptr; }
    if (d_swa_bak_k_)      { cudaFree(d_swa_bak_k_);      d_swa_bak_k_ = nullptr; }
    if (d_swa_bak_v_)      { cudaFree(d_swa_bak_v_);      d_swa_bak_v_ = nullptr; }
    if (d_swa_bak_pos_)    { cudaFree(d_swa_bak_pos_);    d_swa_bak_pos_ = nullptr; }

    if (h_page_k_ptrs_)    { cudaFreeHost(h_page_k_ptrs_);    h_page_k_ptrs_ = nullptr; }
    if (h_page_v_ptrs_)    { cudaFreeHost(h_page_v_ptrs_);    h_page_v_ptrs_ = nullptr; }
    if (h_page_pos0_)      { cudaFreeHost(h_page_pos0_);      h_page_pos0_ = nullptr; }
    if (h_page_lens_)      { cudaFreeHost(h_page_lens_);      h_page_lens_ = nullptr; }
    if (h_quest_scores_)   { cudaFreeHost(h_quest_scores_);   h_quest_scores_ = nullptr; }
    if (h_quest_topk_ids_) { cudaFreeHost(h_quest_topk_ids_); h_quest_topk_ids_ = nullptr; }
    if (h_page_bounce_)    { cudaFreeHost(h_page_bounce_);    h_page_bounce_ = nullptr; }
    if (h_ssd_async_buf_)  { cudaFreeHost(h_ssd_async_buf_);  h_ssd_async_buf_ = nullptr; }
    for (int i = 0; i < SSD_ASYNC_QD; ++i) {
        if (ssd_ov_events_[i]) { CloseHandle((HANDLE)ssd_ov_events_[i]); ssd_ov_events_[i] = nullptr; }
    }
    if (ssd_sync_event_) { CloseHandle((HANDLE)ssd_sync_event_); ssd_sync_event_ = nullptr; }

    if (h_ram_pages_) {
        VirtualFree(h_ram_pages_, 0, MEM_RELEASE);
        h_ram_pages_ = nullptr;
    }
    if (h_ssd_file_ && h_ssd_file_ != INVALID_HANDLE_VALUE) {
        CloseHandle((HANDLE)h_ssd_file_);
        h_ssd_file_ = nullptr;
    }
}

bool HybridPagedKV::init(
    int n_logical_layers,
    const std::vector<uint8_t>& is_swa,
    int n_head,
    int n_head_kv,
    int head_dim,
    int n_embd,
    int swa_window,
    int max_ctx,
    KVQuantMode kv_mode,
    KVAttentionMode attn_mode,
    size_t vram_global_kv_bytes,
    size_t ram_global_kv_bytes,
    const std::string& ssd_swap_path,
    int quest_top_pages)
{
    shutdown();
    n_layers_ = n_logical_layers;
    is_swa_ = is_swa;
    n_head_ = n_head;
    n_head_kv_ = n_head_kv;
    head_dim_ = head_dim;
    n_embd_ = n_embd;
    swa_window_ = swa_window;
    max_ctx_ = max_ctx;
    kv_mode_ = kv_mode;
    attn_mode_ = attn_mode;
    ssd_path_ = ssd_swap_path;
    quest_top_pages_ = quest_top_pages;

    head_bytes_ = (kv_mode_ == KV_QUANT_Q4_0) ? 72 : ((kv_mode_ == KV_QUANT_Q8_0) ? 136 : 256);
    tok_k_bytes_ = (size_t)n_head_kv_ * head_bytes_;
    page_half_bytes_ = (size_t)INK_PAGE_TOKENS * tok_k_bytes_;
    page_full_bytes_ = 2 * page_half_bytes_;

    layer_to_swa_idx_.assign(n_layers_, -1);
    layer_to_glob_idx_.assign(n_layers_, -1);
    n_swa_layers_ = 0;
    n_glob_layers_ = 0;
    for (int il = 0; il < n_layers_; ++il) {
        if (is_swa_[il]) layer_to_swa_idx_[il] = n_swa_layers_++;
        else             layer_to_glob_idx_[il] = n_glob_layers_++;
    }

    // 1. ShortConv recurrent state for all 48 logical layers (3.83 MB total!)
    int n_sconv_ch = 2 * (n_head_kv_ * head_dim_) + 2 * n_embd_;
    conv_layer_floats_ = (size_t)n_sconv_ch * 3;
    INK_CUDA_CHECK(cudaMalloc(&d_conv_state_, (size_t)n_layers_ * conv_layer_floats_ * sizeof(float)));

    // 2. Fixed-size SWA ring buffers in VRAM (never grows with context!)
    size_t swa_layer_bytes = (size_t)swa_window_ * tok_k_bytes_;
    INK_CUDA_CHECK(cudaMalloc(&d_swa_k_, (size_t)n_swa_layers_ * swa_layer_bytes));
    INK_CUDA_CHECK(cudaMalloc(&d_swa_v_, (size_t)n_swa_layers_ * swa_layer_bytes));
    INK_CUDA_CHECK(cudaMalloc(&d_swa_pos_, (size_t)n_swa_layers_ * swa_window_ * sizeof(int32_t)));
    swa_counts_.assign(n_swa_layers_, 0);

    // 3. Global Paged KV pools
    max_pages_per_layer_ = (max_ctx_ + INK_PAGE_TOKENS - 1) / INK_PAGE_TOKENS;
    int total_max_pages = n_glob_layers_ * max_pages_per_layer_;

    // Cap VRAM pages at total_max_pages so we never waste VRAM when max_ctx is small!
    int want_vram_pages = (int)(vram_global_kv_bytes / page_full_bytes_);
    int min_vram_pages = n_glob_layers_ * (quest_top_pages_ + 24);
    if (want_vram_pages < std::min(total_max_pages, min_vram_pages)) {
        want_vram_pages = std::min(total_max_pages, min_vram_pages);
    }
    vram_page_cap_ = std::min(total_max_pages, std::max(want_vram_pages, n_glob_layers_ * 32));

    INK_CUDA_CHECK(cudaMalloc(&d_vram_pages_, (size_t)vram_page_cap_ * page_full_bytes_));
    vram_owner_.assign(vram_page_cap_, {-1, -1});

    // Only allocate RAM KV pool if max_ctx exceeds VRAM page capacity!
    int overflow_pages = std::max(0, total_max_pages - vram_page_cap_);
    if (overflow_pages > 0 && ram_global_kv_bytes >= page_full_bytes_) {
        ram_page_cap_ = std::min(overflow_pages, (int)(ram_global_kv_bytes / page_full_bytes_));
        if (ram_page_cap_ < 64) ram_page_cap_ = std::min(overflow_pages, 64);
        h_ram_pages_ = (uint8_t*)VirtualAlloc(nullptr, (size_t)ram_page_cap_ * page_full_bytes_,
                                              MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!h_ram_pages_) ram_page_cap_ = 0;
        ram_owner_.assign(ram_page_cap_, {-1, -1});
        ram_lru_.assign(ram_page_cap_, 0);
    } else {
        ram_page_cap_ = 0;
    }

    // INT8-quantized Quest static K_min/K_max summaries in VRAM (4x smaller & 3.5x faster scoring!)
    bool need_quest = (overflow_pages > 0) || (attn_mode_ == KV_ATTN_QUEST);
    if (need_quest) {
        size_t q_elems  = (size_t)total_max_pages * n_head_kv_ * head_dim_;
        size_t q_scales = (size_t)total_max_pages * n_head_kv_;
        size_t act_elems = (size_t)n_glob_layers_ * n_head_kv_ * head_dim_;
        INK_CUDA_CHECK(cudaMalloc(&d_k_mins_q8_,    q_elems * sizeof(int8_t)));
        INK_CUDA_CHECK(cudaMalloc(&d_k_maxs_q8_,    q_elems * sizeof(int8_t)));
        INK_CUDA_CHECK(cudaMalloc(&d_k_scales_,     q_scales * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_active_k_min_, act_elems * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_active_k_max_, act_elems * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_quest_scores_,   (size_t)max_pages_per_layer_ * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&d_quest_topk_ids_, (size_t)max_pages_per_layer_ * sizeof(int32_t)));
        INK_CUDA_CHECK(cudaHostAlloc(&h_quest_scores_,   (size_t)max_pages_per_layer_ * sizeof(float), cudaHostAllocDefault));
        INK_CUDA_CHECK(cudaHostAlloc(&h_quest_topk_ids_, (size_t)max_pages_per_layer_ * sizeof(int32_t), cudaHostAllocDefault));
    }

    pages_.assign(n_glob_layers_, std::vector<PageMeta>(max_pages_per_layer_));
    glob_num_pages_.assign(n_glob_layers_, 0);

    // Scratch buffers for Split-K FlashDecoding (single + batched up to 256 B*splits = 3.0 MiB!)
    INK_CUDA_CHECK(cudaMalloc(&d_split_out_, (size_t)256 * n_head_ * head_dim_ * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_split_ml_,  (size_t)256 * n_head_ * 2 * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_page_k_ptrs_, (size_t)max_pages_per_layer_ * sizeof(uint64_t)));
    INK_CUDA_CHECK(cudaMalloc(&d_page_v_ptrs_, (size_t)max_pages_per_layer_ * sizeof(uint64_t)));
    INK_CUDA_CHECK(cudaMalloc(&d_page_pos0_,   (size_t)max_pages_per_layer_ * sizeof(int32_t)));
    INK_CUDA_CHECK(cudaMalloc(&d_page_lens_,   (size_t)max_pages_per_layer_ * sizeof(int32_t)));

    // Speculative verification rollback buffers (23 MiB ShortConv history + 0.27 MiB SWA slot backup)
    INK_CUDA_CHECK(cudaMalloc(&d_conv_spec_,   (size_t)SPEC_MAX_B * (size_t)n_layers_ * conv_layer_floats_ * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_swa_bak_k_,   (size_t)n_swa_layers_ * (size_t)SPEC_MAX_B * tok_k_bytes_));
    INK_CUDA_CHECK(cudaMalloc(&d_swa_bak_v_,   (size_t)n_swa_layers_ * (size_t)SPEC_MAX_B * tok_k_bytes_));
    INK_CUDA_CHECK(cudaMalloc(&d_swa_bak_pos_, (size_t)n_swa_layers_ * (size_t)SPEC_MAX_B * sizeof(int32_t)));

    INK_CUDA_CHECK(cudaHostAlloc(&h_page_k_ptrs_, (size_t)max_pages_per_layer_ * sizeof(uint64_t), cudaHostAllocWriteCombined));
    INK_CUDA_CHECK(cudaHostAlloc(&h_page_v_ptrs_, (size_t)max_pages_per_layer_ * sizeof(uint64_t), cudaHostAllocWriteCombined));
    INK_CUDA_CHECK(cudaHostAlloc(&h_page_pos0_,   (size_t)max_pages_per_layer_ * sizeof(int32_t),  cudaHostAllocWriteCombined));
    INK_CUDA_CHECK(cudaHostAlloc(&h_page_lens_,   (size_t)max_pages_per_layer_ * sizeof(int32_t),  cudaHostAllocWriteCombined));
    INK_CUDA_CHECK(cudaHostAlloc(&h_page_bounce_, page_full_bytes_, cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_ssd_async_buf_, (size_t)SSD_ASYNC_QD * page_full_bytes_, cudaHostAllocDefault));
    for (int i = 0; i < SSD_ASYNC_QD; ++i) {
        ssd_ov_events_[i] = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    }
    ssd_sync_event_ = CreateEventA(nullptr, TRUE, FALSE, nullptr);

    reset();

    auto st = stats();
    std::printf("[PagedKV] MaxCtx=%d | KV=%s | SWA(%dL)=%.1f MiB VRAM | Global(%dL)=%.1f MiB VRAM (%d pages = %dK tok in VRAM) | RAM=%.1f MiB | QuestSummary=%.1f MiB | SSD=%s\n",
                max_ctx_,
                kv_mode_ == KV_QUANT_Q4_0 ? "Q4_0" : (kv_mode_ == KV_QUANT_Q8_0 ? "Q8_0" : "F16"),
                n_swa_layers_, st.swa_vram_mb,
                n_glob_layers_, st.global_vram_mb, vram_page_cap_,
                (vram_page_cap_ / n_glob_layers_) * INK_PAGE_TOKENS / 1024,
                st.ram_kv_mb, st.quest_summary_vram_mb,
                overflow_pages > 0 ? ssd_path_.c_str() : "standby (fits 100% in VRAM)");
    return true;
}

void HybridPagedKV::reset() {
    if (d_conv_state_) {
        INK_CUDA_CHECK(cudaMemset(d_conv_state_, 0, (size_t)n_layers_ * conv_layer_floats_ * sizeof(float)));
    }
    if (d_swa_pos_) {
        INK_CUDA_CHECK(cudaMemset(d_swa_pos_, 0xFF, (size_t)n_swa_layers_ * swa_window_ * sizeof(int32_t))); // -1
    }
    std::fill(swa_counts_.begin(), swa_counts_.end(), 0);
    std::fill(vram_owner_.begin(), vram_owner_.end(), std::make_pair(-1, -1));
    std::fill(ram_owner_.begin(), ram_owner_.end(), std::make_pair(-1, -1));
    std::fill(ram_lru_.begin(), ram_lru_.end(), 0);
    for (int g = 0; g < n_glob_layers_; ++g) {
        std::fill(pages_[g].begin(), pages_[g].end(), PageMeta{});
        glob_num_pages_[g] = 0;
    }
    next_ssd_slot_ = 0;
    cur_pos_ = 0;
    clock_ = 0;
    reset_epoch_++;
}

void HybridPagedKV::reset_counters() {
    stat_vram_hits_ = stat_ram_hits_ = stat_ssd_reads_ = stat_ssd_writes_ = 0;
}

void HybridPagedKV::save_checkpoint(KVCheckpoint& ckpt, int pos, cudaStream_t stream) {
    int n_pages = (pos + INK_PAGE_TOKENS - 1) / INK_PAGE_TOKENS;
    size_t glob_total_bytes = (size_t)n_glob_layers_ * (size_t)n_pages * page_full_bytes_;
    if (pos <= 0 || n_pages <= 0 || glob_total_bytes > 256ull * 1024 * 1024) {
        return; // Safety cap: keep RAM prefix checkpoint <= 256 MiB (~28K tokens)
    }

    INK_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify all pages [0..n_pages) are resident in VRAM or RAM (never SSD) before saving
    for (int g = 0; g < n_glob_layers_; ++g) {
        if (glob_num_pages_[g] < n_pages) return;
        for (int p = 0; p < n_pages; ++p) {
            if (pages_[g][p].tier != TIER_VRAM && pages_[g][p].tier != TIER_RAM) return;
        }
    }

    ckpt.pos = pos;
    ckpt.glob_n_pages = n_pages;

    size_t conv_n = (size_t)n_layers_ * conv_layer_floats_;
    ckpt.conv_state.resize(conv_n);
    INK_CUDA_CHECK(cudaMemcpy(ckpt.conv_state.data(), d_conv_state_, conv_n * sizeof(float), cudaMemcpyDeviceToHost));

    size_t swa_bytes = (size_t)n_swa_layers_ * (size_t)swa_window_ * tok_k_bytes_;
    ckpt.swa_k.resize(swa_bytes);
    ckpt.swa_v.resize(swa_bytes);
    INK_CUDA_CHECK(cudaMemcpy(ckpt.swa_k.data(), d_swa_k_, swa_bytes, cudaMemcpyDeviceToHost));
    INK_CUDA_CHECK(cudaMemcpy(ckpt.swa_v.data(), d_swa_v_, swa_bytes, cudaMemcpyDeviceToHost));

    size_t spos_n = (size_t)n_swa_layers_ * (size_t)swa_window_;
    ckpt.swa_pos.resize(spos_n);
    INK_CUDA_CHECK(cudaMemcpy(ckpt.swa_pos.data(), d_swa_pos_, spos_n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ckpt.swa_counts = swa_counts_;

    // Snapshot Global KV pages so interleaved requests (e.g. OpenCode title/summary calls) NEVER invalidate the cache!
    ckpt.glob_pages.resize(glob_total_bytes);
    for (int g = 0; g < n_glob_layers_; ++g) {
        for (int p = 0; p < n_pages; ++p) {
            const PageMeta& pm = pages_[g][p];
            uint8_t* dst = ckpt.glob_pages.data() + ((size_t)g * n_pages + (size_t)p) * page_full_bytes_;
            if (pm.tier == TIER_VRAM) {
                INK_CUDA_CHECK(cudaMemcpy(dst, d_vram_pages_ + (size_t)pm.slot * page_full_bytes_, page_full_bytes_, cudaMemcpyDeviceToHost));
            } else if (pm.tier == TIER_RAM) {
                std::memcpy(dst, h_ram_pages_ + (size_t)pm.slot * page_full_bytes_, page_full_bytes_);
            }
        }
    }

    if (d_k_mins_q8_ && d_k_maxs_q8_ && d_k_scales_) {
        size_t kd = (size_t)n_head_kv_ * head_dim_;
        size_t q_elems  = (size_t)n_glob_layers_ * (size_t)n_pages * kd;
        size_t q_scales = (size_t)n_glob_layers_ * (size_t)n_pages * n_head_kv_;
        ckpt.k_mins_q8.resize(q_elems);
        ckpt.k_maxs_q8.resize(q_elems);
        ckpt.k_scales.resize(q_scales);
        for (int g = 0; g < n_glob_layers_; ++g) {
            size_t src_off = (size_t)g * max_pages_per_layer_ * kd;
            size_t dst_off = (size_t)g * n_pages * kd;
            INK_CUDA_CHECK(cudaMemcpy(ckpt.k_mins_q8.data() + dst_off, d_k_mins_q8_ + src_off, (size_t)n_pages * kd * sizeof(int8_t), cudaMemcpyDeviceToHost));
            INK_CUDA_CHECK(cudaMemcpy(ckpt.k_maxs_q8.data() + dst_off, d_k_maxs_q8_ + src_off, (size_t)n_pages * kd * sizeof(int8_t), cudaMemcpyDeviceToHost));
            size_t s_src = (size_t)g * max_pages_per_layer_ * n_head_kv_;
            size_t s_dst = (size_t)g * n_pages * n_head_kv_;
            INK_CUDA_CHECK(cudaMemcpy(ckpt.k_scales.data() + s_dst, d_k_scales_ + s_src, (size_t)n_pages * n_head_kv_ * sizeof(float), cudaMemcpyDeviceToHost));
        }
    } else {
        ckpt.k_mins_q8.clear();
        ckpt.k_maxs_q8.clear();
        ckpt.k_scales.clear();
    }
}

bool HybridPagedKV::restore_checkpoint(const KVCheckpoint& ckpt, cudaStream_t stream) {
    if (ckpt.pos <= 0 || ckpt.glob_n_pages <= 0 || ckpt.conv_state.empty() || ckpt.glob_pages.empty()) return false;
    int n_pages = ckpt.glob_n_pages;
    if (n_pages > max_pages_per_layer_) return false;

    INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    reset();

    INK_CUDA_CHECK(cudaMemcpyAsync(d_conv_state_, ckpt.conv_state.data(), ckpt.conv_state.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_k_,      ckpt.swa_k.data(),      ckpt.swa_k.size(), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_v_,      ckpt.swa_v.data(),      ckpt.swa_v.size(), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_pos_,    ckpt.swa_pos.data(),    ckpt.swa_pos.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    swa_counts_ = ckpt.swa_counts;

    int next_vram = 0;
    int next_ram  = 0;
    for (int g = 0; g < n_glob_layers_; ++g) {
        glob_num_pages_[g] = n_pages;
        for (int p = 0; p < n_pages; ++p) {
            PageMeta& pm = pages_[g][p];
            pm.pos0 = p * INK_PAGE_TOKENS;
            pm.len  = std::min(INK_PAGE_TOKENS, ckpt.pos - pm.pos0);
            pm.dirty = true;
            pm.last_use = ++clock_;
            const uint8_t* src = ckpt.glob_pages.data() + ((size_t)g * n_pages + (size_t)p) * page_full_bytes_;
            if (next_vram < vram_page_cap_) {
                int vs = next_vram++;
                INK_CUDA_CHECK(cudaMemcpyAsync(d_vram_pages_ + (size_t)vs * page_full_bytes_, src, page_full_bytes_, cudaMemcpyHostToDevice, stream));
                pm.tier = TIER_VRAM;
                pm.slot = vs;
                vram_owner_[vs] = {g, p};
            } else if (next_ram < ram_page_cap_) {
                int rs = next_ram++;
                std::memcpy(h_ram_pages_ + (size_t)rs * page_full_bytes_, src, page_full_bytes_);
                pm.tier = TIER_RAM;
                pm.slot = rs;
                ram_owner_[rs] = {g, p};
                ram_lru_[rs] = ++clock_;
            } else {
                return false;
            }
        }
        if (d_k_mins_q8_ && d_k_maxs_q8_ && d_k_scales_ && !ckpt.k_mins_q8.empty()) {
            size_t kd = (size_t)n_head_kv_ * head_dim_;
            size_t dst_off = (size_t)g * max_pages_per_layer_ * kd;
            size_t src_off = (size_t)g * n_pages * kd;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_k_mins_q8_ + dst_off, ckpt.k_mins_q8.data() + src_off, (size_t)n_pages * kd * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_k_maxs_q8_ + dst_off, ckpt.k_maxs_q8.data() + src_off, (size_t)n_pages * kd * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            size_t s_dst = (size_t)g * max_pages_per_layer_ * n_head_kv_;
            size_t s_src = (size_t)g * n_pages * n_head_kv_;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_k_scales_ + s_dst, ckpt.k_scales.data() + s_src, (size_t)n_pages * n_head_kv_ * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
    }
    cur_pos_ = ckpt.pos;
    INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

void HybridPagedKV::save_turn_snapshot(TurnRewindSnapshot& snap, int pos, cudaStream_t stream) {
    if (pos <= 0 || (pos % INK_PAGE_TOKENS) != 0) return;
    int n_pages = pos / INK_PAGE_TOKENS;
    for (int g = 0; g < n_glob_layers_; ++g) {
        if (glob_num_pages_[g] < n_pages) return;
    }

    INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    snap.pos = pos;
    snap.epoch = reset_epoch_;

    size_t conv_n = (size_t)n_layers_ * conv_layer_floats_;
    snap.conv_state.resize(conv_n);
    INK_CUDA_CHECK(cudaMemcpy(snap.conv_state.data(), d_conv_state_, conv_n * sizeof(float), cudaMemcpyDeviceToHost));

    size_t swa_bytes = (size_t)n_swa_layers_ * (size_t)swa_window_ * tok_k_bytes_;
    snap.swa_k.resize(swa_bytes);
    snap.swa_v.resize(swa_bytes);
    INK_CUDA_CHECK(cudaMemcpy(snap.swa_k.data(), d_swa_k_, swa_bytes, cudaMemcpyDeviceToHost));
    INK_CUDA_CHECK(cudaMemcpy(snap.swa_v.data(), d_swa_v_, swa_bytes, cudaMemcpyDeviceToHost));

    size_t spos_n = (size_t)n_swa_layers_ * (size_t)swa_window_;
    snap.swa_pos.resize(spos_n);
    INK_CUDA_CHECK(cudaMemcpy(snap.swa_pos.data(), d_swa_pos_, spos_n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    snap.swa_counts = swa_counts_;
}

bool HybridPagedKV::rewind_to_turn_snapshot(const TurnRewindSnapshot& snap, cudaStream_t stream) {
    if (snap.pos <= 0 || (snap.pos % INK_PAGE_TOKENS) != 0 || snap.epoch != reset_epoch_ ||
        cur_pos_ < snap.pos || snap.conv_state.empty()) {
        return false;
    }
    int snap_pages = snap.pos / INK_PAGE_TOKENS;
    for (int g = 0; g < n_glob_layers_; ++g) {
        if (glob_num_pages_[g] < snap_pages) return false;
    }

    INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_conv_state_, snap.conv_state.data(), snap.conv_state.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_k_,      snap.swa_k.data(),      snap.swa_k.size(), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_v_,      snap.swa_v.data(),      snap.swa_v.size(), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_pos_,    snap.swa_pos.data(),    snap.swa_pos.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    swa_counts_ = snap.swa_counts;

    // Free only pages strictly beyond `snap_pages` — pages [0..snap_pages) remain 100% intact across VRAM/RAM/SSD!
    for (int g = 0; g < n_glob_layers_; ++g) {
        int cur_np = glob_num_pages_[g];
        for (int p = snap_pages; p < cur_np; ++p) {
            PageMeta& pm = pages_[g][p];
            if (pm.tier == TIER_VRAM && pm.slot >= 0 && pm.slot < vram_page_cap_) {
                if (vram_owner_[pm.slot].first == g && vram_owner_[pm.slot].second == p) {
                    vram_owner_[pm.slot] = {-1, -1};
                }
            } else if (pm.tier == TIER_RAM && pm.slot >= 0 && pm.slot < ram_page_cap_) {
                if (ram_owner_[pm.slot].first == g && ram_owner_[pm.slot].second == p) {
                    ram_owner_[pm.slot] = {-1, -1};
                }
            }
            pm = PageMeta{};
        }
        glob_num_pages_[g] = snap_pages;
    }
    cur_pos_ = snap.pos;
    INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

namespace {
struct DiskCkptHeader {
    uint32_t magic;       // 'KCK1' = 0x314B434B
    uint32_t version;     // 1
    uint64_t model_sig;
    int32_t  pos;
    int32_t  glob_n_pages;
    int32_t  kv_mode;
    int32_t  n_swa_layers;
    int32_t  n_glob_layers;
    int32_t  swa_window;
    uint64_t n_tokens;
    uint64_t conv_floats;
    uint64_t swa_bytes;
    uint64_t spos_count;
    uint64_t glob_bytes;
    uint64_t q8_elems;
    uint64_t q8_scales;
    uint64_t token_hash;
};

uint64_t fnv1a_tokens(const int32_t* toks, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(toks);
    for (size_t i = 0; i < n * sizeof(int32_t); ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}
} // namespace

bool HybridPagedKV::save_checkpoint_disk(const KVCheckpoint& ckpt, const std::string& path, uint64_t model_sig) const {
    if (ckpt.pos <= 0 || ckpt.glob_n_pages <= 0 || ckpt.tokens.empty()) return false;

    std::string tmp_path = path + ".tmp";
    FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) return false;

    DiskCkptHeader hdr{};
    hdr.magic         = 0x314B434Bu;
    hdr.version       = 2u; // v2: exact unbiased Q4_0 KV rounding (+8.0f with __float2int_rn)
    hdr.model_sig     = model_sig;
    hdr.pos           = ckpt.pos;
    hdr.glob_n_pages  = ckpt.glob_n_pages;
    hdr.kv_mode       = (int32_t)kv_mode_;
    hdr.n_swa_layers  = n_swa_layers_;
    hdr.n_glob_layers = n_glob_layers_;
    hdr.swa_window    = swa_window_;
    hdr.n_tokens      = ckpt.tokens.size();
    hdr.conv_floats   = ckpt.conv_state.size();
    hdr.swa_bytes     = ckpt.swa_k.size();
    hdr.spos_count    = ckpt.swa_pos.size();
    hdr.glob_bytes    = ckpt.glob_pages.size();
    hdr.q8_elems      = ckpt.k_mins_q8.size();
    hdr.q8_scales     = ckpt.k_scales.size();
    hdr.token_hash    = fnv1a_tokens(ckpt.tokens.data(), ckpt.tokens.size());

    bool ok = (std::fwrite(&hdr, sizeof(hdr), 1, f) == 1);
    auto wr = [&](const void* ptr, size_t nbytes) {
        if (ok && nbytes > 0) ok = (std::fwrite(ptr, 1, nbytes, f) == nbytes);
    };
    wr(ckpt.tokens.data(),     ckpt.tokens.size()     * sizeof(int32_t));
    wr(ckpt.conv_state.data(), ckpt.conv_state.size() * sizeof(float));
    wr(ckpt.swa_k.data(),      ckpt.swa_k.size());
    wr(ckpt.swa_v.data(),      ckpt.swa_v.size());
    wr(ckpt.swa_pos.data(),    ckpt.swa_pos.size()    * sizeof(int32_t));
    wr(ckpt.swa_counts.data(), ckpt.swa_counts.size() * sizeof(int));
    wr(ckpt.glob_pages.data(), ckpt.glob_pages.size());
    wr(ckpt.k_mins_q8.data(),  ckpt.k_mins_q8.size()  * sizeof(int8_t));
    wr(ckpt.k_maxs_q8.data(),  ckpt.k_maxs_q8.size()  * sizeof(int8_t));
    wr(ckpt.k_scales.data(),   ckpt.k_scales.size()   * sizeof(float));
    std::fclose(f);

    if (!ok) {
        DeleteFileA(tmp_path.c_str());
        return false;
    }
    return MoveFileExA(tmp_path.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
}

bool HybridPagedKV::load_checkpoint_disk(KVCheckpoint& ckpt, const std::string& path, uint64_t model_sig,
                                         const int32_t* expect_tokens, int expect_pos) const {
    if (!expect_tokens || expect_pos <= 0) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    DiskCkptHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, f) != 1) { std::fclose(f); return false; }

    uint64_t want_hash = fnv1a_tokens(expect_tokens, (size_t)expect_pos);
    size_t expected_conv = (size_t)n_layers_ * conv_layer_floats_;
    size_t expected_swa  = (size_t)n_swa_layers_ * (size_t)swa_window_ * tok_k_bytes_;
    size_t expected_glob = (size_t)n_glob_layers_ * (size_t)hdr.glob_n_pages * page_full_bytes_;

    if (hdr.magic != 0x314B434Bu || hdr.version != 2u ||
        hdr.model_sig != model_sig || hdr.pos != expect_pos ||
        hdr.kv_mode != (int32_t)kv_mode_ || hdr.n_swa_layers != n_swa_layers_ ||
        hdr.n_glob_layers != n_glob_layers_ || hdr.swa_window != swa_window_ ||
        hdr.n_tokens != (uint64_t)expect_pos || hdr.token_hash != want_hash ||
        hdr.glob_n_pages <= 0 || hdr.glob_n_pages > max_pages_per_layer_ ||
        hdr.conv_floats != expected_conv || hdr.swa_bytes != expected_swa ||
        hdr.glob_bytes != expected_glob) {
        std::fclose(f);
        return false;
    }

    KVCheckpoint tmp;
    tmp.pos = hdr.pos;
    tmp.glob_n_pages = hdr.glob_n_pages;
    tmp.tokens.resize((size_t)hdr.n_tokens);
    tmp.conv_state.resize((size_t)hdr.conv_floats);
    tmp.swa_k.resize((size_t)hdr.swa_bytes);
    tmp.swa_v.resize((size_t)hdr.swa_bytes);
    tmp.swa_pos.resize((size_t)hdr.spos_count);
    tmp.swa_counts.resize((size_t)n_swa_layers_);
    tmp.glob_pages.resize((size_t)hdr.glob_bytes);
    tmp.k_mins_q8.resize((size_t)hdr.q8_elems);
    tmp.k_maxs_q8.resize((size_t)hdr.q8_elems);
    tmp.k_scales.resize((size_t)hdr.q8_scales);

    bool ok = true;
    auto rd = [&](void* ptr, size_t nbytes) {
        if (ok && nbytes > 0) ok = (std::fread(ptr, 1, nbytes, f) == nbytes);
    };
    rd(tmp.tokens.data(),     tmp.tokens.size()     * sizeof(int32_t));
    if (!ok || std::memcmp(tmp.tokens.data(), expect_tokens, (size_t)expect_pos * sizeof(int32_t)) != 0) {
        std::fclose(f);
        return false;
    }
    rd(tmp.conv_state.data(), tmp.conv_state.size() * sizeof(float));
    rd(tmp.swa_k.data(),      tmp.swa_k.size());
    rd(tmp.swa_v.data(),      tmp.swa_v.size());
    rd(tmp.swa_pos.data(),    tmp.swa_pos.size()    * sizeof(int32_t));
    rd(tmp.swa_counts.data(), tmp.swa_counts.size() * sizeof(int));
    rd(tmp.glob_pages.data(), tmp.glob_pages.size());
    rd(tmp.k_mins_q8.data(),  tmp.k_mins_q8.size()  * sizeof(int8_t));
    rd(tmp.k_maxs_q8.data(),  tmp.k_maxs_q8.size()  * sizeof(int8_t));
    rd(tmp.k_scales.data(),   tmp.k_scales.size()   * sizeof(float));
    std::fclose(f);
    if (!ok) return false;

    ckpt = std::move(tmp);
    return true;
}

bool HybridPagedKV::benchmark_populate(int target_pos, cudaStream_t stream) {
    if (target_pos <= 0) return true;
    if (target_pos > max_ctx_) {
        std::fprintf(stderr, "[PagedKV] benchmark_populate target %d > max_ctx %d\n", target_pos, max_ctx_);
        return false;
    }
    reset();

    // 1. Build realistic finite quantized 1-page K+V template (well-conditioned non-zero patterns)
    std::vector<uint8_t> synth_page(page_full_bytes_, 0);
    if (kv_mode_ == KV_QUANT_Q4_0) {
        uint16_t d_h = fp32_to_fp16_host(0.04f);
        size_t n_blocks = page_full_bytes_ / 18;
        for (size_t b = 0; b < n_blocks; ++b) {
            uint8_t* bp = synth_page.data() + b * 18;
            bp[0] = (uint8_t)(d_h & 0xFF);
            bp[1] = (uint8_t)(d_h >> 8);
            for (int j = 0; j < 16; ++j) {
                uint8_t lo = (uint8_t)(6 + ((b + j) % 5));      // in [6..10] -> [-2..+2]
                uint8_t hi = (uint8_t)(6 + ((b * 3 + j) % 5));
                bp[2 + j] = (uint8_t)(lo | (hi << 4));
            }
        }
    } else if (kv_mode_ == KV_QUANT_Q8_0) {
        uint16_t d_h = fp32_to_fp16_host(0.01f);
        size_t n_blocks = page_full_bytes_ / 34;
        for (size_t b = 0; b < n_blocks; ++b) {
            uint8_t* bp = synth_page.data() + b * 34;
            bp[0] = (uint8_t)(d_h & 0xFF);
            bp[1] = (uint8_t)(d_h >> 8);
            for (int j = 0; j < 32; ++j) bp[2 + j] = (uint8_t)(int8_t)((int)((b + j) % 17) - 8);
        }
    } else {
        uint16_t* hp = (uint16_t*)synth_page.data();
        size_t n_h = page_full_bytes_ / 2;
        for (size_t i = 0; i < n_h; ++i) hp[i] = fp32_to_fp16_host(0.02f * (float)((int)(i % 9) - 4));
    }

    // 2. Populate all 40 SWA ring buffers in VRAM up to min(target_pos, swa_window_)
    int swa_len = std::min(target_pos, swa_window_);
    std::vector<int32_t> swa_pos(swa_window_, -1);
    for (int i = 0; i < swa_len; ++i) {
        int p = target_pos - swa_len + i;
        swa_pos[p % swa_window_] = p;
    }
    std::vector<uint8_t> swa_buf((size_t)swa_window_ * tok_k_bytes_);
    for (int s = 0; s < swa_window_; ++s) {
        std::memcpy(swa_buf.data() + (size_t)s * tok_k_bytes_,
                    synth_page.data() + (size_t)(s % INK_PAGE_TOKENS) * tok_k_bytes_,
                    tok_k_bytes_);
    }
    size_t swa_layer_bytes = (size_t)swa_window_ * tok_k_bytes_;
    for (int si = 0; si < n_swa_layers_; ++si) {
        INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_k_ + (size_t)si * swa_layer_bytes, swa_buf.data(), swa_layer_bytes, cudaMemcpyHostToDevice, stream));
        INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_v_ + (size_t)si * swa_layer_bytes, swa_buf.data(), swa_layer_bytes, cudaMemcpyHostToDevice, stream));
        INK_CUDA_CHECK(cudaMemcpyAsync(d_swa_pos_ + (size_t)si * swa_window_, swa_pos.data(), (size_t)swa_window_ * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        swa_counts_[si] = swa_len;
    }

    // 3. Populate all 8 Global layers across Tier 0 VRAM -> Tier 1 RAM -> Tier 2 NVMe SSD!
    int n_pages = (target_pos + INK_PAGE_TOKENS - 1) / INK_PAGE_TOKENS;
    int vram_per_g = vram_page_cap_ / n_glob_layers_;
    int ram_per_g  = ram_page_cap_  / n_glob_layers_;

    int next_vram = 0;
    int next_ram  = 0;
    int cnt_vram = 0, cnt_ram = 0, cnt_ssd = 0;

    // Prepare batched SSD buffer (64 pages = 4.5 MiB sequential NVMe writes -> >1.5 GB/s!)
    constexpr int SSD_BATCH = 64;
    std::vector<uint8_t> ssd_batch_buf;

    for (int g = 0; g < n_glob_layers_; ++g) {
        glob_num_pages_[g] = n_pages;
        // Decide which pages of layer g reside in VRAM vs RAM vs SSD:
        // Sink [0..1] + most recent (vram_per_g - 2) pages in VRAM; next ram_per_g in RAM; rest on NVMe SSD!
        int vram_recent_start = std::max(2, n_pages - std::max(0, vram_per_g - 2));
        int ram_recent_start  = std::max(2, vram_recent_start - ram_per_g);

        int ssd_start_slot = -1;
        int ssd_batch_cnt  = 0;

        auto flush_ssd_batch = [&]() {
            if (ssd_batch_cnt <= 0) return;
            if (!ensure_ssd_file()) { ssd_batch_cnt = 0; return; }
            if (ssd_batch_buf.size() < (size_t)SSD_BATCH * page_full_bytes_) {
                ssd_batch_buf.resize((size_t)SSD_BATCH * page_full_bytes_);
                for (int b = 0; b < SSD_BATCH; ++b) {
                    std::memcpy(ssd_batch_buf.data() + (size_t)b * page_full_bytes_, synth_page.data(), page_full_bytes_);
                }
            }
            uint64_t off = (uint64_t)ssd_start_slot * page_full_bytes_;
            ssd_pwrite_sync(off, ssd_batch_buf.data(), (size_t)ssd_batch_cnt * page_full_bytes_);
            stat_ssd_writes_ += (uint64_t)ssd_batch_cnt;
            ssd_batch_cnt = 0;
            ssd_start_slot = -1;
        };

        for (int p = 0; p < n_pages; ++p) {
            PageMeta& pm = pages_[g][p];
            pm.pos0 = p * INK_PAGE_TOKENS;
            pm.len  = std::min(INK_PAGE_TOKENS, target_pos - pm.pos0);
            pm.dirty = false;
            pm.last_use = ++clock_;

            if ((p < 2 || p >= vram_recent_start) && next_vram < vram_page_cap_) {
                int vs = next_vram++;
                INK_CUDA_CHECK(cudaMemcpyAsync(d_vram_pages_ + (size_t)vs * page_full_bytes_,
                                               synth_page.data(), page_full_bytes_, cudaMemcpyHostToDevice, stream));
                pm.tier = TIER_VRAM;
                pm.slot = vs;
                vram_owner_[vs] = {g, p};
                cnt_vram++;
            } else if (p >= ram_recent_start && next_ram < ram_page_cap_) {
                int rs = next_ram++;
                std::memcpy(h_ram_pages_ + (size_t)rs * page_full_bytes_, synth_page.data(), page_full_bytes_);
                pm.tier = TIER_RAM;
                pm.slot = rs;
                ram_owner_[rs] = {g, p};
                ram_lru_[rs] = ++clock_;
                cnt_ram++;
            } else {
                int ss = next_ssd_slot_++;
                pm.tier = TIER_SSD;
                pm.slot = ss;
                pm.ssd_slot = ss;
                cnt_ssd++;
                if (ssd_batch_cnt == 0) ssd_start_slot = ss;
                ssd_batch_cnt++;
                if (ssd_batch_cnt == SSD_BATCH) flush_ssd_batch();
            }
        }
        flush_ssd_batch();

        // Populate INT8 Quest static K_min / K_max summaries with realistic per-page variance
        if (d_k_mins_q8_ && d_k_maxs_q8_ && d_k_scales_) {
            size_t kd = (size_t)n_head_kv_ * head_dim_;
            std::vector<int8_t> kmin((size_t)n_pages * kd);
            std::vector<int8_t> kmax((size_t)n_pages * kd);
            std::vector<float>  ksc((size_t)n_pages * n_head_kv_);
            for (int p = 0; p < n_pages; ++p) {
                float amp = 0.08f + 0.04f * (float)((p * 17 + g * 7) % 11);
                float bias = 0.02f * (float)(((p * 31 + g * 13) % 7) - 3);
                float sc = std::max(std::fabs(bias) + amp, 1e-4f) / 127.0f;
                for (int h = 0; h < n_head_kv_; ++h) ksc[(size_t)p * n_head_kv_ + h] = sc;
                float inv_sc = 1.0f / sc;
                int8_t* mn = kmin.data() + (size_t)p * kd;
                int8_t* mx = kmax.data() + (size_t)p * kd;
                for (size_t d = 0; d < kd; ++d) {
                    float sgn = ((p + d) & 1) ? 1.0f : -1.0f;
                    mn[d] = (int8_t)std::max(-127, std::min(127, (int)std::floor((bias * sgn - amp) * inv_sc)));
                    mx[d] = (int8_t)std::max(-127, std::min(127, (int)std::ceil((bias * sgn + amp) * inv_sc)));
                }
            }
            size_t base_off = (size_t)g * max_pages_per_layer_ * kd;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_k_mins_q8_ + base_off, kmin.data(), kmin.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_k_maxs_q8_ + base_off, kmax.data(), kmax.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            size_t sc_off = (size_t)g * max_pages_per_layer_ * n_head_kv_;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_k_scales_ + sc_off, ksc.data(), ksc.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
    }

    cur_pos_ = target_pos;
    INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::printf("[PagedKV] Populated %d tokens (%d pages/layer x %d global layers): VRAM=%d pages (%.1f MiB), RAM=%d pages (%.1f MiB), NVMe SSD=%d pages (%.1f MiB)\n",
                target_pos, n_pages, n_glob_layers_,
                cnt_vram, (cnt_vram * page_full_bytes_) / (1024.0 * 1024.0),
                cnt_ram,  (cnt_ram  * page_full_bytes_) / (1024.0 * 1024.0),
                cnt_ssd,  (cnt_ssd  * page_full_bytes_) / (1024.0 * 1024.0));
    std::fflush(stdout);
    return true;
}

float* HybridPagedKV::conv_state_ptr(int il, int channel_offset) const {
    return d_conv_state_ + (size_t)il * conv_layer_floats_ + (size_t)channel_offset * 3;
}

float* HybridPagedKV::conv_spec_ptr(int il, int channel_offset) const {
    return d_conv_spec_ + (size_t)il * conv_layer_floats_ + (size_t)channel_offset * 3;
}

void HybridPagedKV::begin_spec_verify(int pos0, int B, cudaStream_t stream) {
    spec_pos0_ = pos0;
    spec_B_    = B;
    ink_cuda_swa_spec_backup_restore(
        d_swa_k_, d_swa_v_, d_swa_pos_,
        d_swa_bak_k_, d_swa_bak_v_, d_swa_bak_pos_,
        n_swa_layers_, swa_window_, pos0, 0, B, SPEC_MAX_B, tok_k_bytes_, /*is_restore=*/false, stream);
}

void HybridPagedKV::commit_spec_verify(int accepted, cudaStream_t stream) {
    if (accepted <= 0 || accepted > spec_B_) return;
    int new_pos = spec_pos0_ + accepted;

    if (accepted < spec_B_) {
        // 1. Exact ShortConv rollback to post-token-(accepted-1) state (3.83 MiB D2D = 0.02 ms!)
        size_t total_conv = (size_t)n_layers_ * conv_layer_floats_;
        INK_CUDA_CHECK(cudaMemcpyAsync(
            d_conv_state_,
            d_conv_spec_ + (size_t)(accepted - 1) * total_conv,
            total_conv * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        // 2. Restore rejected SWA ring slots [accepted .. spec_B_ - 1] cleanly (0.002 ms!)
        ink_cuda_swa_spec_backup_restore(
            d_swa_k_, d_swa_v_, d_swa_pos_,
            d_swa_bak_k_, d_swa_bak_v_, d_swa_bak_pos_,
            n_swa_layers_, swa_window_, spec_pos0_, accepted, spec_B_, SPEC_MAX_B,
            tok_k_bytes_, /*is_restore=*/true, stream);
        for (int si = 0; si < n_swa_layers_; ++si) {
            swa_counts_[si] = std::min(swa_window_, new_pos);
        }

        // 3. Truncate Global KV pages back to exact `new_pos`
        int last_p = (new_pos - 1) / INK_PAGE_TOKENS;
        int last_len = ((new_pos - 1) % INK_PAGE_TOKENS) + 1;
        for (int g = 0; g < n_glob_layers_; ++g) {
            int cur_np = glob_num_pages_[g];
            for (int p = last_p + 1; p < cur_np; ++p) {
                PageMeta& pm = pages_[g][p];
                if (pm.tier == TIER_VRAM && pm.slot >= 0 && pm.slot < vram_page_cap_) {
                    if (vram_owner_[pm.slot].first == g && vram_owner_[pm.slot].second == p) {
                        vram_owner_[pm.slot] = {-1, -1};
                    }
                } else if (pm.tier == TIER_RAM && pm.slot >= 0 && pm.slot < ram_page_cap_) {
                    if (ram_owner_[pm.slot].first == g && ram_owner_[pm.slot].second == p) {
                        ram_owner_[pm.slot] = {-1, -1};
                    }
                }
                pm = PageMeta{};
            }
            if (last_p < cur_np) {
                pages_[g][last_p].len = last_len;
            }
            glob_num_pages_[g] = std::min(cur_np, last_p + 1);
        }
    }
    cur_pos_ = new_pos;
}

void HybridPagedKV::begin_draft_snapshot(int pos0, int max_k, cudaStream_t stream) {
    // Save pre-draft ShortConv state into reserved slot SPEC_MAX_B - 1 (0.015 ms D2D!)
    size_t total_conv = (size_t)n_layers_ * conv_layer_floats_;
    INK_CUDA_CHECK(cudaMemcpyAsync(
        d_conv_spec_ + (size_t)(SPEC_MAX_B - 1) * total_conv,
        d_conv_state_,
        total_conv * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    // Backup SWA slots [pos0 .. pos0 + max_k - 1]
    ink_cuda_swa_spec_backup_restore(
        d_swa_k_, d_swa_v_, d_swa_pos_,
        d_swa_bak_k_, d_swa_bak_v_, d_swa_bak_pos_,
        n_swa_layers_, swa_window_, pos0, 0, max_k, SPEC_MAX_B, tok_k_bytes_, /*is_restore=*/false, stream);
}

void HybridPagedKV::rollback_draft_snapshot(int pos0, int drafted, cudaStream_t stream) {
    if (drafted <= 0) return;
    size_t total_conv = (size_t)n_layers_ * conv_layer_floats_;
    INK_CUDA_CHECK(cudaMemcpyAsync(
        d_conv_state_,
        d_conv_spec_ + (size_t)(SPEC_MAX_B - 1) * total_conv,
        total_conv * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    ink_cuda_swa_spec_backup_restore(
        d_swa_k_, d_swa_v_, d_swa_pos_,
        d_swa_bak_k_, d_swa_bak_v_, d_swa_bak_pos_,
        n_swa_layers_, swa_window_, pos0, 0, drafted, SPEC_MAX_B, tok_k_bytes_, /*is_restore=*/true, stream);
    for (int si = 0; si < n_swa_layers_; ++si) {
        swa_counts_[si] = std::min(swa_window_, pos0);
    }
    if (pos0 > 0) {
        int last_p = (pos0 - 1) / INK_PAGE_TOKENS;
        int last_len = ((pos0 - 1) % INK_PAGE_TOKENS) + 1;
        for (int g = 0; g < n_glob_layers_; ++g) {
            int cur_np = glob_num_pages_[g];
            for (int p = last_p + 1; p < cur_np; ++p) {
                PageMeta& pm = pages_[g][p];
                if (pm.tier == TIER_VRAM && pm.slot >= 0 && pm.slot < vram_page_cap_) {
                    if (vram_owner_[pm.slot].first == g && vram_owner_[pm.slot].second == p) {
                        vram_owner_[pm.slot] = {-1, -1};
                    }
                } else if (pm.tier == TIER_RAM && pm.slot >= 0 && pm.slot < ram_page_cap_) {
                    if (ram_owner_[pm.slot].first == g && ram_owner_[pm.slot].second == p) {
                        ram_owner_[pm.slot] = {-1, -1};
                    }
                }
                pm = PageMeta{};
            }
            if (last_p < cur_np) {
                pages_[g][last_p].len = last_len;
            }
            glob_num_pages_[g] = std::min(cur_np, last_p + 1);
        }
    }
    cur_pos_ = pos0;
}

bool HybridPagedKV::ensure_ssd_file() {
    if (h_ssd_file_ && h_ssd_file_ != INVALID_HANDLE_VALUE) return true;
    HANDLE h = CreateFileA(
        ssd_path_.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED | FILE_FLAG_DELETE_ON_CLOSE,
        nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[PagedKV] Failed creating SSD KV swap '%s' (err=%lu)\n", ssd_path_.c_str(), GetLastError());
        return false;
    }
    h_ssd_file_ = h;
    return true;
}

bool HybridPagedKV::ssd_pwrite_sync(uint64_t offset, const void* src, size_t nbytes) {
    if (!ensure_ssd_file()) return false;
    HANDLE h = (HANDLE)h_ssd_file_;
    HANDLE ev = (HANDLE)ssd_sync_event_;
    const uint8_t* ptr = (const uint8_t*)src;
    while (nbytes > 0) {
        DWORD chunk = (DWORD)std::min<size_t>(nbytes, 64ull * 1024 * 1024);
        OVERLAPPED ov{};
        ov.Offset     = (DWORD)(offset & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent     = ev;
        ResetEvent(ev);
        DWORD written = 0;
        if (!WriteFile(h, ptr, chunk, &written, &ov)) {
            if (GetLastError() != ERROR_IO_PENDING) return false;
            if (!GetOverlappedResult(h, &ov, &written, TRUE)) return false;
        }
        ptr += written;
        offset += written;
        nbytes -= written;
    }
    return true;
}

bool HybridPagedKV::ssd_pread_sync(uint64_t offset, void* dst, size_t nbytes) {
    if (!h_ssd_file_ || h_ssd_file_ == INVALID_HANDLE_VALUE) return false;
    HANDLE h = (HANDLE)h_ssd_file_;
    HANDLE ev = (HANDLE)ssd_sync_event_;
    uint8_t* ptr = (uint8_t*)dst;
    while (nbytes > 0) {
        DWORD chunk = (DWORD)std::min<size_t>(nbytes, 64ull * 1024 * 1024);
        OVERLAPPED ov{};
        ov.Offset     = (DWORD)(offset & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent     = ev;
        ResetEvent(ev);
        DWORD got = 0;
        if (!ReadFile(h, ptr, chunk, &got, &ov)) {
            if (GetLastError() != ERROR_IO_PENDING) return false;
            if (!GetOverlappedResult(h, &ov, &got, TRUE)) return false;
        }
        if (got == 0) return false;
        ptr += got;
        offset += got;
        nbytes -= got;
    }
    return true;
}

void HybridPagedKV::spill_ram_to_ssd(int g, int p, int rslot) {
    if (!ensure_ssd_file()) return;
    PageMeta& pm = pages_[g][p];
    if (pm.ssd_slot < 0) pm.ssd_slot = next_ssd_slot_++;
    if (!pm.dirty) return;

    uint64_t off = (uint64_t)pm.ssd_slot * page_full_bytes_;
    ssd_pwrite_sync(off, h_ram_pages_ + (size_t)rslot * page_full_bytes_, page_full_bytes_);
    pm.dirty = false;
    stat_ssd_writes_++;
}

void HybridPagedKV::read_ssd_to_ram(int g, int p, int rslot) {
    PageMeta& pm = pages_[g][p];
    uint64_t off = (uint64_t)pm.ssd_slot * page_full_bytes_;
    ssd_pread_sync(off, h_ram_pages_ + (size_t)rslot * page_full_bytes_, page_full_bytes_);
    stat_ssd_reads_++;
}

int HybridPagedKV::alloc_ram_page_slot(int& evicted_g, int& evicted_p) {
    evicted_g = -1; evicted_p = -1;
    int best = 0;
    uint64_t oldest = UINT64_MAX;
    for (int s = 0; s < ram_page_cap_; ++s) {
        if (ram_owner_[s].first == -1) return s;
        if (ram_lru_[s] < oldest) { oldest = ram_lru_[s]; best = s; }
    }
    evicted_g = ram_owner_[best].first;
    evicted_p = ram_owner_[best].second;
    spill_ram_to_ssd(evicted_g, evicted_p, best);
    pages_[evicted_g][evicted_p].tier = TIER_SSD;
    pages_[evicted_g][evicted_p].slot = pages_[evicted_g][evicted_p].ssd_slot;
    ram_owner_[best] = {-1, -1};
    return best;
}

int HybridPagedKV::alloc_vram_page_slot(int protect_g, int protect_recent_start, cudaStream_t stream) {
    int best = -1;
    uint64_t oldest = UINT64_MAX;
    for (int s = 0; s < vram_page_cap_; ++s) {
        auto [og, op] = vram_owner_[s];
        if (og == -1) return s;
        // Never evict sink pages [0..1] or recent relative-bias window of current layer
        if (op < 2) continue;
        if (og == protect_g && op >= protect_recent_start) continue;
        // Also avoid evicting the active append page of any layer
        if (op >= glob_num_pages_[og] - 2) continue;
        uint64_t lu = pages_[og][op].last_use;
        if (lu < oldest) { oldest = lu; best = s; }
    }
    if (best < 0) best = 0;

    auto [og, op] = vram_owner_[best];
    PageMeta& old_pm = pages_[og][op];

    // Copy evicted VRAM page down to Tier-1 RAM (or directly to Tier-2 SSD if RAM cap == 0)
    if (ram_page_cap_ > 0) {
        int eg, ep;
        int rslot = alloc_ram_page_slot(eg, ep);
        INK_CUDA_CHECK(cudaMemcpyAsync(h_page_bounce_, d_vram_pages_ + (size_t)best * page_full_bytes_,
                                       page_full_bytes_, cudaMemcpyDeviceToHost, stream));
        INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::memcpy(h_ram_pages_ + (size_t)rslot * page_full_bytes_, h_page_bounce_, page_full_bytes_);
        ram_owner_[rslot] = {og, op};
        ram_lru_[rslot] = ++clock_;
        old_pm.tier = TIER_RAM;
        old_pm.slot = rslot;
    } else if (ensure_ssd_file()) {
        INK_CUDA_CHECK(cudaMemcpyAsync(h_page_bounce_, d_vram_pages_ + (size_t)best * page_full_bytes_,
                                       page_full_bytes_, cudaMemcpyDeviceToHost, stream));
        INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (old_pm.ssd_slot < 0) old_pm.ssd_slot = next_ssd_slot_++;
        if (old_pm.dirty) {
            uint64_t off = (uint64_t)old_pm.ssd_slot * page_full_bytes_;
            ssd_pwrite_sync(off, h_page_bounce_, page_full_bytes_);
            old_pm.dirty = false;
            stat_ssd_writes_++;
        }
        old_pm.tier = TIER_SSD;
        old_pm.slot = old_pm.ssd_slot;
    }

    vram_owner_[best] = {-1, -1};
    return best;
}

int HybridPagedKV::ensure_page_in_vram(int g, int p, int protect_recent_start, cudaStream_t stream) {
    PageMeta& pm = pages_[g][p];
    if (pm.tier == TIER_VRAM) {
        pm.last_use = ++clock_;
        stat_vram_hits_++;
        return pm.slot;
    }

    int vslot = alloc_vram_page_slot(g, protect_recent_start, stream);
    uint8_t* vdst = d_vram_pages_ + (size_t)vslot * page_full_bytes_;

    if (pm.tier == TIER_RAM) {
        int rslot = pm.slot;
        std::memcpy(h_page_bounce_, h_ram_pages_ + (size_t)rslot * page_full_bytes_, page_full_bytes_);
        INK_CUDA_CHECK(cudaMemcpyAsync(vdst, h_page_bounce_, page_full_bytes_, cudaMemcpyHostToDevice, stream));
        ram_owner_[rslot] = {-1, -1};
        stat_ram_hits_++;
    } else if (pm.tier == TIER_SSD) {
        uint64_t off = (uint64_t)pm.ssd_slot * page_full_bytes_;
        ssd_pread_sync(off, h_page_bounce_, page_full_bytes_);
        INK_CUDA_CHECK(cudaMemcpyAsync(vdst, h_page_bounce_, page_full_bytes_, cudaMemcpyHostToDevice, stream));
        stat_ssd_reads_++;
    }
    // Note: TIER_NONE means freshly allocated empty page (no copy needed!)

    pm.tier = TIER_VRAM;
    pm.slot = vslot;
    pm.last_use = ++clock_;
    vram_owner_[vslot] = {g, p};
    return vslot;
}

// Async Queue-Depth-16 Overlapped NVMe Batch Fetch (hits full NVMe parallel IOPS at 256K-512K!)
void HybridPagedKV::fetch_ssd_pages_batch(int g, const std::vector<int>& ssd_pids, int protect_recent_start, cudaStream_t stream) {
    if (ssd_pids.empty() || !h_ssd_file_ || h_ssd_file_ == INVALID_HANDLE_VALUE) return;
    HANDLE h = (HANDLE)h_ssd_file_;
    int total = (int)ssd_pids.size();

    for (int base = 0; base < total; base += SSD_ASYNC_QD) {
        int cnt = std::min(SSD_ASYNC_QD, total - base);
        int vslots[SSD_ASYNC_QD];
        OVERLAPPED ovs[SSD_ASYNC_QD];
        bool pending[SSD_ASYNC_QD];

        // 1. Allocate VRAM slots & kick off up to 16 concurrent NVMe ReadFile I/Os in parallel!
        for (int i = 0; i < cnt; ++i) {
            int pid = ssd_pids[base + i];
            vslots[i] = alloc_vram_page_slot(g, protect_recent_start, stream);
            // Temporarily reserve slot so subsequent allocs in this batch cannot evict it!
            vram_owner_[vslots[i]] = {g, pid};
            pages_[g][pid].last_use = ++clock_;

            uint64_t off = (uint64_t)pages_[g][pid].ssd_slot * page_full_bytes_;
            std::memset(&ovs[i], 0, sizeof(OVERLAPPED));
            ovs[i].Offset     = (DWORD)(off & 0xFFFFFFFFull);
            ovs[i].OffsetHigh = (DWORD)(off >> 32);
            ovs[i].hEvent     = (HANDLE)ssd_ov_events_[i];
            ResetEvent(ovs[i].hEvent);

            uint8_t* dst = h_ssd_async_buf_ + (size_t)i * page_full_bytes_;
            DWORD got = 0;
            BOOL ok = ReadFile(h, dst, (DWORD)page_full_bytes_, &got, &ovs[i]);
            pending[i] = (!ok && GetLastError() == ERROR_IO_PENDING);
        }

        // 2. Complete overlapped NVMe reads and immediately DMA each completed page into VRAM!
        for (int i = 0; i < cnt; ++i) {
            if (pending[i]) {
                DWORD got = 0;
                GetOverlappedResult(h, &ovs[i], &got, TRUE);
            }
            int pid = ssd_pids[base + i];
            uint8_t* src  = h_ssd_async_buf_ + (size_t)i * page_full_bytes_;
            uint8_t* vdst = d_vram_pages_ + (size_t)vslots[i] * page_full_bytes_;
            INK_CUDA_CHECK(cudaMemcpyAsync(vdst, src, page_full_bytes_, cudaMemcpyHostToDevice, stream));
            pages_[g][pid].tier = TIER_VRAM;
            pages_[g][pid].slot = vslots[i];
            pages_[g][pid].last_use = ++clock_;
            stat_ssd_reads_++;
        }
        // Sync DMA before reusing h_ssd_async_buf_ for the next QD-16 batch (only if more batches remain)
        if (base + cnt < total) {
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
}

void HybridPagedKV::append_and_attend(
    int il, int pos,
    const float* d_Q, const float* d_K, const float* d_V,
    const float* d_rel_table, const float* d_sinks, float* d_out,
    int rel_extent, cudaStream_t stream)
{
    cur_pos_ = pos + 1;

    if (is_swa_[il]) {
        int si = layer_to_swa_idx_[il];
        size_t swa_layer_bytes = (size_t)swa_window_ * tok_k_bytes_;
        uint8_t* k_ring = d_swa_k_ + (size_t)si * swa_layer_bytes;
        uint8_t* v_ring = d_swa_v_ + (size_t)si * swa_layer_bytes;
        int32_t* pos_ring = d_swa_pos_ + (size_t)si * swa_window_;

        int slot = pos % swa_window_;
        ink_cuda_kv_store(d_K, d_V,
                          k_ring + (size_t)slot * tok_k_bytes_,
                          v_ring + (size_t)slot * tok_k_bytes_,
                          nullptr, nullptr, nullptr, nullptr, nullptr, false,
                          pos_ring + slot, pos,
                          n_head_kv_, head_dim_, kv_mode_, stream);

        if (swa_counts_[si] < swa_window_) swa_counts_[si] = std::min(swa_window_, pos + 1);
        ink_cuda_attn_swa(d_Q, k_ring, v_ring, pos_ring, d_rel_table, d_sinks, d_split_out_, d_split_ml_, d_out,
                          n_head_, n_head_kv_, head_dim_, swa_counts_[si], pos, rel_extent, kv_mode_, stream);
        return;
    }

    // Global attention layer!
    int g = layer_to_glob_idx_[il];
    int p = pos / INK_PAGE_TOKENS;
    int t_in_p = pos % INK_PAGE_TOKENS;
    if (p >= max_pages_per_layer_) {
        std::fprintf(stderr, "[PagedKV] Position %d exceeds max_ctx %d!\n", pos, max_ctx_);
        std::abort();
    }

    int recent_start = std::max(0, p - 17); // protects last 17 pages (>= 1024 rel_extent tokens!)
    if (p >= glob_num_pages_[g]) glob_num_pages_[g] = p + 1;

    int vslot = ensure_page_in_vram(g, p, recent_start, stream);
    PageMeta& cur_pm = pages_[g][p];
    cur_pm.pos0 = p * INK_PAGE_TOKENS;
    cur_pm.len  = std::max(cur_pm.len, t_in_p + 1);
    cur_pm.dirty = true;

    uint8_t* page_base = d_vram_pages_ + (size_t)vslot * page_full_bytes_;
    uint8_t* k_dst = page_base + (size_t)t_in_p * tok_k_bytes_;
    uint8_t* v_dst = page_base + page_half_bytes_ + (size_t)t_in_p * tok_k_bytes_;

    float*  act_min_ptr = nullptr;
    float*  act_max_ptr = nullptr;
    int8_t* k_min_q8 = nullptr;
    int8_t* k_max_q8 = nullptr;
    float*  k_sc_ptr = nullptr;
    if (d_k_mins_q8_) {
        size_t kd = (size_t)n_head_kv_ * head_dim_;
        size_t sum_off = ((size_t)g * max_pages_per_layer_ + (size_t)p) * kd;
        size_t sc_off  = ((size_t)g * max_pages_per_layer_ + (size_t)p) * n_head_kv_;
        act_min_ptr = d_active_k_min_ + (size_t)g * kd;
        act_max_ptr = d_active_k_max_ + (size_t)g * kd;
        k_min_q8 = d_k_mins_q8_ + sum_off;
        k_max_q8 = d_k_maxs_q8_ + sum_off;
        k_sc_ptr = d_k_scales_  + sc_off;
    }

    ink_cuda_kv_store(d_K, d_V, k_dst, v_dst,
                      act_min_ptr, act_max_ptr, k_min_q8, k_max_q8, k_sc_ptr,
                      t_in_p == 0,
                      nullptr, 0,
                      n_head_kv_, head_dim_, kv_mode_, stream);

    int n_pages = glob_num_pages_[g];
    bool all_in_vram = true;
    for (int i = 0; i < n_pages; ++i) {
        if (pages_[g][i].tier != TIER_VRAM) { all_in_vram = false; break; }
    }

    bool use_quest = (!all_in_vram && attn_mode_ != KV_ATTN_EXACT_STREAM) ||
                     (attn_mode_ == KV_ATTN_QUEST && n_pages > quest_top_pages_ + 20);

    int n_sel = 0;
    if (!use_quest) {
        // 100% EXACT FULL ATTENTION over all n_pages!
        for (int i = 0; i < n_pages; ++i) {
            int vs = ensure_page_in_vram(g, i, recent_start, stream);
            uint8_t* pb = d_vram_pages_ + (size_t)vs * page_full_bytes_;
            h_page_k_ptrs_[n_sel] = (uint64_t)(uintptr_t)pb;
            h_page_v_ptrs_[n_sel] = (uint64_t)(uintptr_t)(pb + page_half_bytes_);
            h_page_pos0_[n_sel]   = pages_[g][i].pos0;
            h_page_lens_[n_sel]   = pages_[g][i].len;
            n_sel++;
        }
    } else {
        // Quest Exact Upper-Bound Page Selection (for 128K - 1M SSD-backed context!)
        // Always pin sink pages [0..1] + recent window [recent_start .. n_pages-1]
        int hist_start = 2;
        int hist_end = std::max(hist_start, recent_start);
        std::vector<int> chosen;
        chosen.reserve(quest_top_pages_ + 24);
        for (int i = 0; i < std::min(2, n_pages); ++i) chosen.push_back(i);

        if (hist_end > hist_start && d_k_mins_q8_) {
            size_t base_off = (size_t)g * max_pages_per_layer_ * n_head_kv_ * head_dim_;
            size_t sc_off   = (size_t)g * max_pages_per_layer_ * n_head_kv_;
            int hist_n = hist_end - hist_start;
            int take = std::min<int>(quest_top_pages_, hist_n);
            ink_cuda_quest_score_pages(d_Q, d_k_mins_q8_ + base_off, d_k_maxs_q8_ + base_off,
                                       d_k_scales_ + sc_off,
                                       d_quest_scores_, hist_end, n_head_, n_head_kv_, head_dim_, stream);
            if (take < hist_n && d_quest_topk_ids_) {
                // Exact O(N) GPU Radix-Select Top-K (~3 us, copies only K int32s = 512 B D2H!)
                ink_cuda_quest_topk(d_quest_scores_, hist_start, hist_end, take, d_quest_topk_ids_, stream);
                INK_CUDA_CHECK(cudaMemcpyAsync(h_quest_topk_ids_, d_quest_topk_ids_,
                                               (size_t)take * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                INK_CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < take; ++i) chosen.push_back(h_quest_topk_ids_[i]);
            } else {
                for (int i = hist_start; i < hist_end; ++i) chosen.push_back(i);
            }
        }
        for (int i = std::max(2, recent_start); i < n_pages; ++i) chosen.push_back(i);
        std::sort(chosen.begin(), chosen.end());

        // Pin timestamp on already-VRAM chosen pages & batch-fetch ALL SSD misses at Queue Depth 16!
        std::vector<int> ssd_misses;
        for (int pid : chosen) {
            if (pages_[g][pid].tier == TIER_VRAM) {
                pages_[g][pid].last_use = ++clock_;
            } else if (pages_[g][pid].tier == TIER_SSD) {
                ssd_misses.push_back(pid);
            }
        }
        if (ssd_misses.size() > 1) {
            fetch_ssd_pages_batch(g, ssd_misses, recent_start, stream);
        }

        for (int pid : chosen) {
            int vs = ensure_page_in_vram(g, pid, recent_start, stream);
            uint8_t* pb = d_vram_pages_ + (size_t)vs * page_full_bytes_;
            h_page_k_ptrs_[n_sel] = (uint64_t)(uintptr_t)pb;
            h_page_v_ptrs_[n_sel] = (uint64_t)(uintptr_t)(pb + page_half_bytes_);
            h_page_pos0_[n_sel]   = pages_[g][pid].pos0;
            h_page_lens_[n_sel]   = pages_[g][pid].len;
            n_sel++;
        }
    }

    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_k_ptrs_, h_page_k_ptrs_, (size_t)n_sel * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_v_ptrs_, h_page_v_ptrs_, (size_t)n_sel * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_pos0_,   h_page_pos0_,   (size_t)n_sel * sizeof(int32_t),  cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_lens_,   h_page_lens_,   (size_t)n_sel * sizeof(int32_t),  cudaMemcpyHostToDevice, stream));

    ink_cuda_attn_global_paged(d_Q, d_page_k_ptrs_, d_page_v_ptrs_, d_page_pos0_, d_page_lens_, n_sel,
                               d_rel_table, d_sinks, d_split_out_, d_split_ml_, d_out,
                               n_head_, n_head_kv_, head_dim_, pos, rel_extent, kv_mode_, stream);
}

void HybridPagedKV::append_and_attend_batch(
    int il, int pos0, int B,
    const float* d_Q, const float* d_K, const float* d_V,
    const float* d_rel_table, const float* d_sinks, float* d_out,
    int rel_extent, cudaStream_t stream)
{
    if (B == 1) {
        append_and_attend(il, pos0, d_Q, d_K, d_V, d_rel_table, d_sinks, d_out, rel_extent, stream);
        return;
    }
    cur_pos_ = pos0 + B;

    if (is_swa_[il]) {
        int si = layer_to_swa_idx_[il];
        size_t swa_layer_bytes = (size_t)swa_window_ * tok_k_bytes_;
        uint8_t* k_ring = d_swa_k_ + (size_t)si * swa_layer_bytes;
        uint8_t* v_ring = d_swa_v_ + (size_t)si * swa_layer_bytes;
        int32_t* pos_ring = d_swa_pos_ + (size_t)si * swa_window_;

        ink_cuda_kv_store_swa_batch(d_K, d_V, k_ring, v_ring, pos_ring, swa_window_, pos0, B,
                                    n_head_kv_, head_dim_, kv_mode_, stream);
        swa_counts_[si] = std::min(swa_window_, pos0 + B);
        ink_cuda_attn_swa_batch(d_Q, k_ring, v_ring, pos_ring, d_rel_table, d_sinks,
                                d_split_out_, d_split_ml_, d_out, B,
                                n_head_, n_head_kv_, head_dim_, swa_counts_[si], pos0, rel_extent, kv_mode_, stream);
        return;
    }

    // Global attention layer (batch append + causal batched attention)
    int g = layer_to_glob_idx_[il];
    int last_pos = pos0 + B - 1;
    int last_p = last_pos / INK_PAGE_TOKENS;
    if (last_p >= max_pages_per_layer_) {
        std::fprintf(stderr, "[PagedKV] Batch position %d exceeds max_ctx %d!\n", last_pos, max_ctx_);
        std::abort();
    }
    int recent_start = std::max(0, last_p - 18);

    const size_t kv_stride = (size_t)n_head_kv_ * head_dim_;
    for (int t = 0; t < B; ++t) {
        int pos = pos0 + t;
        int p = pos / INK_PAGE_TOKENS;
        int t_in_p = pos % INK_PAGE_TOKENS;
        if (p >= glob_num_pages_[g]) glob_num_pages_[g] = p + 1;

        int vslot = ensure_page_in_vram(g, p, recent_start, stream);
        PageMeta& cur_pm = pages_[g][p];
        cur_pm.pos0 = p * INK_PAGE_TOKENS;
        cur_pm.len  = std::max(cur_pm.len, t_in_p + 1);
        cur_pm.dirty = true;

        uint8_t* page_base = d_vram_pages_ + (size_t)vslot * page_full_bytes_;
        uint8_t* k_dst = page_base + (size_t)t_in_p * tok_k_bytes_;
        uint8_t* v_dst = page_base + page_half_bytes_ + (size_t)t_in_p * tok_k_bytes_;

        float*  act_min_ptr = nullptr;
        float*  act_max_ptr = nullptr;
        int8_t* k_min_q8 = nullptr;
        int8_t* k_max_q8 = nullptr;
        float*  k_sc_ptr = nullptr;
        if (d_k_mins_q8_) {
            size_t sum_off = ((size_t)g * max_pages_per_layer_ + (size_t)p) * kv_stride;
            size_t sc_off  = ((size_t)g * max_pages_per_layer_ + (size_t)p) * n_head_kv_;
            act_min_ptr = d_active_k_min_ + (size_t)g * kv_stride;
            act_max_ptr = d_active_k_max_ + (size_t)g * kv_stride;
            k_min_q8 = d_k_mins_q8_ + sum_off;
            k_max_q8 = d_k_maxs_q8_ + sum_off;
            k_sc_ptr = d_k_scales_  + sc_off;
        }
        ink_cuda_kv_store(d_K + (size_t)t * kv_stride, d_V + (size_t)t * kv_stride,
                          k_dst, v_dst, act_min_ptr, act_max_ptr, k_min_q8, k_max_q8, k_sc_ptr,
                          t_in_p == 0, nullptr, 0, n_head_kv_, head_dim_, kv_mode_, stream);
    }

    int n_pages = glob_num_pages_[g];
    bool all_in_vram = true;
    for (int i = 0; i < n_pages; ++i) {
        if (pages_[g][i].tier != TIER_VRAM) { all_in_vram = false; break; }
    }
    bool use_quest = (!all_in_vram && attn_mode_ != KV_ATTN_EXACT_STREAM) ||
                     (attn_mode_ == KV_ATTN_QUEST && n_pages > quest_top_pages_ + 20);

    int n_sel = 0;
    if (!use_quest) {
        for (int i = 0; i < n_pages; ++i) {
            int vs = ensure_page_in_vram(g, i, recent_start, stream);
            uint8_t* pb = d_vram_pages_ + (size_t)vs * page_full_bytes_;
            h_page_k_ptrs_[n_sel] = (uint64_t)(uintptr_t)pb;
            h_page_v_ptrs_[n_sel] = (uint64_t)(uintptr_t)(pb + page_half_bytes_);
            h_page_pos0_[n_sel]   = pages_[g][i].pos0;
            h_page_lens_[n_sel]   = pages_[g][i].len;
            n_sel++;
        }
    } else {
        int hist_start = 2;
        int hist_end = std::max(hist_start, recent_start);
        std::vector<int> chosen;
        chosen.reserve(quest_top_pages_ + 24);
        for (int i = 0; i < std::min(2, n_pages); ++i) chosen.push_back(i);

        if (hist_end > hist_start && d_k_mins_q8_) {
            size_t base_off = (size_t)g * max_pages_per_layer_ * kv_stride;
            size_t sc_off   = (size_t)g * max_pages_per_layer_ * n_head_kv_;
            // Score with newest query in batch (and blend with first query if B > 1)
            const float* d_Q_last = d_Q + (size_t)(B - 1) * n_head_ * head_dim_;
            ink_cuda_quest_score_pages(d_Q_last, d_k_mins_q8_ + base_off, d_k_maxs_q8_ + base_off,
                                       d_k_scales_ + sc_off, d_quest_scores_, hist_end,
                                       n_head_, n_head_kv_, head_dim_, stream);
            INK_CUDA_CHECK(cudaMemcpyAsync(h_quest_scores_, d_quest_scores_,
                                           (size_t)hist_end * sizeof(float), cudaMemcpyDeviceToHost, stream));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));

            std::vector<int> cand;
            cand.reserve(hist_end - hist_start);
            for (int i = hist_start; i < hist_end; ++i) cand.push_back(i);
            int take = std::min<int>(quest_top_pages_, (int)cand.size());
            std::nth_element(cand.begin(), cand.begin() + take, cand.end(),
                             [&](int a, int b) { return h_quest_scores_[a] > h_quest_scores_[b]; });
            for (int i = 0; i < take; ++i) chosen.push_back(cand[i]);
        }
        for (int i = std::max(2, recent_start); i < n_pages; ++i) chosen.push_back(i);
        std::sort(chosen.begin(), chosen.end());

        for (int pid : chosen) {
            int vs = ensure_page_in_vram(g, pid, recent_start, stream);
            uint8_t* pb = d_vram_pages_ + (size_t)vs * page_full_bytes_;
            h_page_k_ptrs_[n_sel] = (uint64_t)(uintptr_t)pb;
            h_page_v_ptrs_[n_sel] = (uint64_t)(uintptr_t)(pb + page_half_bytes_);
            h_page_pos0_[n_sel]   = pages_[g][pid].pos0;
            h_page_lens_[n_sel]   = pages_[g][pid].len;
            n_sel++;
        }
    }

    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_k_ptrs_, h_page_k_ptrs_, (size_t)n_sel * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_v_ptrs_, h_page_v_ptrs_, (size_t)n_sel * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_pos0_,   h_page_pos0_,   (size_t)n_sel * sizeof(int32_t),  cudaMemcpyHostToDevice, stream));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_page_lens_,   h_page_lens_,   (size_t)n_sel * sizeof(int32_t),  cudaMemcpyHostToDevice, stream));

    ink_cuda_attn_global_paged_batch(d_Q, d_page_k_ptrs_, d_page_v_ptrs_, d_page_pos0_, d_page_lens_, n_sel,
                                     d_rel_table, d_sinks, d_split_out_, d_split_ml_, d_out, B,
                                     n_head_, n_head_kv_, head_dim_, pos0, rel_extent, kv_mode_, stream);
}

PagedKVStats HybridPagedKV::stats() const {
    PagedKVStats s{};
    s.max_ctx = max_ctx_;
    s.cur_pos = cur_pos_;
    s.swa_vram_mb = (double)(2 * (size_t)n_swa_layers_ * swa_window_ * tok_k_bytes_) / (1024.0 * 1024.0);
    s.global_vram_mb = (double)((size_t)vram_page_cap_ * page_full_bytes_) / (1024.0 * 1024.0);
    if (d_k_mins_q8_) {
        size_t tot_p = (size_t)n_glob_layers_ * max_pages_per_layer_;
        size_t q_bytes = tot_p * n_head_kv_ * (2ull * head_dim_ * sizeof(int8_t) + sizeof(float))
                       + 2ull * n_glob_layers_ * n_head_kv_ * head_dim_ * sizeof(float);
        s.quest_summary_vram_mb = (double)q_bytes / (1024.0 * 1024.0);
    }
    s.ram_kv_mb = (double)((size_t)ram_page_cap_ * page_full_bytes_) / (1024.0 * 1024.0);
    s.ssd_kv_mb = (double)((size_t)next_ssd_slot_ * page_full_bytes_) / (1024.0 * 1024.0);
    s.vram_page_cap = vram_page_cap_;
    s.ram_page_cap = ram_page_cap_;
    int tot = 0;
    for (int g = 0; g < n_glob_layers_; ++g) tot += glob_num_pages_[g];
    s.total_pages_allocated = tot;
    s.vram_page_hits = stat_vram_hits_;
    s.ram_page_hits = stat_ram_hits_;
    s.ssd_page_reads = stat_ssd_reads_;
    s.ssd_page_writes = stat_ssd_writes_;
    return s;
}
