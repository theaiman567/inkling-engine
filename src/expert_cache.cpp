#include "expert_cache.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
inline size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }
}

ExpertCache::~ExpertCache() { shutdown(); }

void ExpertCache::wait_for_ram_preload() {
    if (preload_thread_.joinable()) {
        preload_thread_.join();
    }
}

void ExpertCache::shutdown() {
    stop_preload_.store(true, std::memory_order_release);
    if (preload_thread_.joinable()) {
        preload_thread_.join();
    }
    ram_preload_done_.store(true, std::memory_order_release);

    if (stream_dma_) {
        cudaStreamSynchronize(stream_dma_);
    }
    for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
        if (dma_[i].event)         { cudaEventDestroy(dma_[i].event);     dma_[i].event = nullptr; }
        if (dma_[i].h_in_staging)  { cudaFreeHost(dma_[i].h_in_staging);  dma_[i].h_in_staging = nullptr; }
        if (dma_[i].h_out_staging) { cudaFreeHost(dma_[i].h_out_staging); dma_[i].h_out_staging = nullptr; }
        dma_[i].active = false;
    }
    if (ev_compute_done_) { cudaEventDestroy(ev_compute_done_); ev_compute_done_ = nullptr; }
    if (h_ptr_update_)    { cudaFreeHost(h_ptr_update_);        h_ptr_update_ = nullptr; }
    if (stream_dma_)      { cudaStreamDestroy(stream_dma_);     stream_dma_ = nullptr; }
    for (auto& tbl : layer_tables_) {
        if (tbl.d_gu_ptrs)   { cudaFree(tbl.d_gu_ptrs);   tbl.d_gu_ptrs = nullptr; }
        if (tbl.d_dn_ptrs)   { cudaFree(tbl.d_dn_ptrs);   tbl.d_dn_ptrs = nullptr; }
        if (tbl.d_gu_scales) { cudaFree(tbl.d_gu_scales); tbl.d_gu_scales = nullptr; }
        if (tbl.d_dn_scales) { cudaFree(tbl.d_dn_scales); tbl.d_dn_scales = nullptr; }
        tbl.valid = false;
    }
    layer_tables_.clear();
    layer_vram_count_.clear();
    pending_promos_.clear();
    vram_slot_ptr_.clear();
    vram_slot_cap_.clear();
    ram_slot_ptr_.clear();
    ram_slot_cap_.clear();
    if (d_core_pinned_) { cudaFree(d_core_pinned_); d_core_pinned_ = nullptr; }
    if (d_vram_pool_) { cudaFree(d_vram_pool_); d_vram_pool_ = nullptr; }
    if (h_ram_pool_) { VirtualFree(h_ram_pool_, 0, MEM_RELEASE); h_ram_pool_ = nullptr; }
}

bool ExpertCache::init(
    const GGUFFile& gguf,
    int n_phys_layers,
    int n_experts,
    int core_start,
    int core_end,
    size_t vram_budget_bytes,
    size_t ram_budget_bytes,
    ExpertCachePolicy policy,
    bool preload_ram,
    bool async_ram_preload)
{
    shutdown();
    gguf_ = &gguf;
    n_layers_ = n_phys_layers;
    n_experts_ = n_experts;
    total_experts_ = n_phys_layers * n_experts;
    core_start_ = core_start;
    core_end_ = core_end;
    policy_ = policy;

    layers_.resize(n_phys_layers);
    max_slot_bytes_ = 0;
    size_t total_core_bytes = 0;
    size_t total_outer_bytes = 0;

    for (int ip = 0; ip < n_phys_layers; ++ip) {
        auto& lm = layers_[ip];
        lm.is_core = (ip >= core_start && ip < core_end);
        char name[128];

        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate_up_exps.weight", ip);
        const auto& gu = gguf.require_tensor(name);
        lm.gate_up_type = gu.type;
        lm.gate_up_file_off = gu.abs_offset;
        lm.gate_up_exp_bytes = (size_t)(gu.nbytes / (uint64_t)n_experts);

        std::snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", ip);
        const auto& dn = gguf.require_tensor(name);
        lm.down_type = dn.type;
        lm.down_file_off = dn.abs_offset;
        lm.down_exp_bytes = (size_t)(dn.nbytes / (uint64_t)n_experts);

        lm.down_aligned_off = align_up(lm.gate_up_exp_bytes, 256);
        lm.combined_bytes = align_up(lm.down_aligned_off + lm.down_exp_bytes, 256);
        max_slot_bytes_ = std::max(max_slot_bytes_, lm.combined_bytes);

        if (lm.is_core) total_core_bytes += lm.combined_bytes * n_experts;
        else            total_outer_bytes += lm.combined_bytes * n_experts;

        // Per-expert scales (optional; absent in blk.20..23 -> 1.0f)
        lm.gate_up_scales.assign(n_experts, 1.0f);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate_up_exps.scale", ip);
        if (const auto* gs = gguf.find_tensor(name)) {
            gguf.pread(gs->abs_offset, lm.gate_up_scales.data(), n_experts * sizeof(float));
        }

        lm.down_scales.assign(n_experts, 1.0f);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.scale", ip);
        if (const auto* ds = gguf.find_tensor(name)) {
            gguf.pread(ds->abs_offset, lm.down_scales.data(), n_experts * sizeof(float));
        }
    }

    // Determine compact Bucket-0 size (all layers strictly smaller than max_slot_bytes_, e.g. IQ2_* = 3.04 MiB vs IQ3 = 3.36 MiB!)
    small_slot_bytes_ = 0;
    int small_core_experts = 0;
    int small_outer_experts = 0;
    for (int ip = 0; ip < n_phys_layers; ++ip) {
        if (layers_[ip].combined_bytes < max_slot_bytes_) {
            small_slot_bytes_ = std::max(small_slot_bytes_, layers_[ip].combined_bytes);
        }
    }
    if (small_slot_bytes_ == 0 || small_slot_bytes_ + 65536 >= max_slot_bytes_) {
        small_slot_bytes_ = max_slot_bytes_;
    } else {
        for (int ip = 0; ip < n_phys_layers; ++ip) {
            if (layers_[ip].combined_bytes <= small_slot_bytes_) {
                if (layers_[ip].is_core) small_core_experts += n_experts;
                else                     small_outer_experts += n_experts;
            }
        }
    }
    const bool use_two_buckets = (small_slot_bytes_ < max_slot_bytes_);

    uid_to_vram_slot_.assign(total_experts_, -1);
    uid_to_ram_slot_.assign(total_experts_, -1);
    pinned_gu_ptr_.assign(total_experts_, nullptr);
    pinned_dn_ptr_.assign(total_experts_, nullptr);
    layer_tables_.assign(n_phys_layers, PinnedLayerDeviceTable{});
    layer_vram_count_.assign(n_phys_layers, 0);
    scores_.assign(total_experts_, 0.0f);
    last_used_.assign(total_experts_, 0);
    last_token_.assign(total_experts_, 0);
    pending_promos_.reserve(64);
    token_seq_ = 1;

    INK_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_dma_, cudaStreamNonBlocking));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_compute_done_, cudaEventDisableTiming));
    INK_CUDA_CHECK(cudaHostAlloc(&h_ptr_update_, MAX_INFLIGHT_DMA * 4 * sizeof(uint64_t), cudaHostAllocDefault));
    for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
        INK_CUDA_CHECK(cudaHostAlloc(&dma_[i].h_in_staging,  max_slot_bytes_, cudaHostAllocDefault));
        INK_CUDA_CHECK(cudaHostAlloc(&dma_[i].h_out_staging, max_slot_bytes_, cudaHostAllocDefault));
        INK_CUDA_CHECK(cudaEventCreateWithFlags(&dma_[i].event, cudaEventDisableTiming));
    }

    auto t0 = std::chrono::steady_clock::now();

    // Allocate VRAM tiers!
    size_t vram_left = vram_budget_bytes;
    int pinned_count = 0;

    // If CORE_PINNED and budget fits the entire looped core, pin 100% tightly in VRAM!
    if (policy_ == CACHE_CORE_PINNED && vram_left >= total_core_bytes) {
        INK_CUDA_CHECK(cudaMalloc(&d_core_pinned_, total_core_bytes));
        core_pinned_bytes_ = total_core_bytes;
        vram_left -= total_core_bytes;

        std::printf("[ExpertCache] Pinning 100%% of looped core (blk.%d..%d, %d experts = %.1f MiB) into VRAM...\n",
                    core_start_, core_end_ - 1, (core_end_ - core_start_) * n_experts_,
                    total_core_bytes / (1024.0 * 1024.0));
        std::fflush(stdout);

        size_t offset = 0;
        for (int ip = core_start_; ip < core_end_; ++ip) {
            const auto& lm = layers_[ip];
            std::vector<uint8_t> tmp_gu(lm.gate_up_exp_bytes * n_experts_);
            std::vector<uint8_t> tmp_dn(lm.down_exp_bytes * n_experts_);
            if (!gguf.pread(lm.gate_up_file_off, tmp_gu.data(), tmp_gu.size()) ||
                !gguf.pread(lm.down_file_off, tmp_dn.data(), tmp_dn.size())) {
                std::fprintf(stderr, "[ExpertCache] Failed reading core layer %d from SSD\n", ip);
                return false;
            }
            std::vector<uint8_t> packed(lm.combined_bytes * n_experts_, 0);
            for (int e = 0; e < n_experts_; ++e) {
                uint8_t* dst = packed.data() + (size_t)e * lm.combined_bytes;
                std::memcpy(dst, tmp_gu.data() + (size_t)e * lm.gate_up_exp_bytes, lm.gate_up_exp_bytes);
                std::memcpy(dst + lm.down_aligned_off, tmp_dn.data() + (size_t)e * lm.down_exp_bytes, lm.down_exp_bytes);
                int uid = ip * n_experts_ + e;
                pinned_gu_ptr_[uid] = d_core_pinned_ + offset + (size_t)e * lm.combined_bytes;
                pinned_dn_ptr_[uid] = pinned_gu_ptr_[uid] + lm.down_aligned_off;
                uid_to_vram_slot_[uid] = -2; // tightly pinned!
                pinned_count++;
            }
            INK_CUDA_CHECK(cudaMemcpy(d_core_pinned_ + offset, packed.data(), packed.size(), cudaMemcpyHostToDevice));
            offset += packed.size();
        }
    } else {
        core_pinned_bytes_ = 0;
    }

    // 2-Bucket Variable-Size Dynamic VRAM Pool:
    //  - Bucket 0: compact `small_slot_bytes_` slots (for IQ2_* layers blk.0..8, 12, 18 -> saves ~200+ MiB!)
    //  - Bucket 1: universal `max_slot_bytes_` slots (accepts BOTH IQ3_* layers and overflow IQ2_* layers!)
    int max_dynamic_needed = total_experts_ - pinned_count;
    n_vram_small_ = 0;
    n_vram_large_ = 0;
    if (use_two_buckets && vram_left >= small_slot_bytes_) {
        int avail_small_exps = (pinned_count > 0) ? small_outer_experts : (small_core_experts + small_outer_experts);
        int target_small = (pinned_count > 0)
            ? std::min(avail_small_exps, (int)((double)vram_left * 0.55 / (double)small_slot_bytes_))
            : std::min(small_core_experts + small_outer_experts / 6,
                       (int)((double)vram_left * 0.36 / (double)small_slot_bytes_));
        n_vram_small_ = std::max(0, std::min(target_small, max_dynamic_needed));
        size_t small_bytes_used = (size_t)n_vram_small_ * small_slot_bytes_;
        size_t rem_vram = (vram_left > small_bytes_used) ? (vram_left - small_bytes_used) : 0;
        n_vram_large_ = (int)std::min<size_t>(rem_vram / max_slot_bytes_, (size_t)(max_dynamic_needed - n_vram_small_));
    } else {
        n_vram_large_ = (int)std::min<size_t>(vram_left / max_slot_bytes_, (size_t)max_dynamic_needed);
    }
    n_vram_slots_ = n_vram_small_ + n_vram_large_;
    vram_pool_bytes_ = (size_t)n_vram_small_ * small_slot_bytes_ + (size_t)n_vram_large_ * max_slot_bytes_;

    if (n_vram_slots_ > 0) {
        INK_CUDA_CHECK(cudaMalloc(&d_vram_pool_, vram_pool_bytes_));
        vram_slot_ptr_.resize(n_vram_slots_);
        vram_slot_cap_.resize(n_vram_slots_);
        vram_slot_owner_.assign(n_vram_slots_, -1);
        size_t off = 0;
        for (int s = 0; s < n_vram_small_; ++s) {
            vram_slot_ptr_[s] = d_vram_pool_ + off;
            vram_slot_cap_[s] = small_slot_bytes_;
            off += small_slot_bytes_;
        }
        for (int s = n_vram_small_; s < n_vram_slots_; ++s) {
            vram_slot_ptr_[s] = d_vram_pool_ + off;
            vram_slot_cap_[s] = max_slot_bytes_;
            off += max_slot_bytes_;
        }
    }

    // Pre-warm dynamic VRAM slots (Core-first, placing small layers into Bucket 0 and large layers into Bucket 1!)
    if (n_vram_slots_ > 0) {
        std::printf("[ExpertCache] Pre-warming %d dynamic VRAM slots (%.1f MiB: %d compact@%.2fM + %d universal@%.2fM)...\n",
                    n_vram_slots_, vram_pool_bytes_ / (1024.0 * 1024.0),
                    n_vram_small_, small_slot_bytes_ / (1024.0 * 1024.0),
                    n_vram_large_, max_slot_bytes_ / (1024.0 * 1024.0));
        std::fflush(stdout);

        std::vector<int> layer_order;
        if (pinned_count == 0) {
            for (int ip = core_start_; ip < core_end_; ++ip) layer_order.push_back(ip);
        }
        for (int ip = 0; ip < core_start_; ++ip) layer_order.push_back(ip);
        for (int ip = core_end_; ip < n_layers_; ++ip) layer_order.push_back(ip);

        int next_small = 0;
        int next_large = n_vram_small_;
        for (int ip : layer_order) {
            const auto& lm = layers_[ip];
            bool fits_small = (lm.combined_bytes <= small_slot_bytes_);
            int avail_s = fits_small ? (n_vram_small_ - next_small) : 0;
            int avail_l = n_vram_slots_ - next_large;
            int take = std::min(n_experts_, avail_s + avail_l);
            if (take <= 0) continue;

            std::vector<uint8_t> tmp_gu(lm.gate_up_exp_bytes * take);
            std::vector<uint8_t> tmp_dn(lm.down_exp_bytes * take);
            gguf.pread(lm.gate_up_file_off, tmp_gu.data(), tmp_gu.size());
            gguf.pread(lm.down_file_off, tmp_dn.data(), tmp_dn.size());

            // Upload contiguous chunks directly into Bucket 0 and/or Bucket 1
            int take_s = std::min(take, avail_s);
            int take_l = take - take_s;
            if (take_s > 0) {
                std::vector<uint8_t> packed((size_t)take_s * small_slot_bytes_, 0);
                for (int e = 0; e < take_s; ++e) {
                    uint8_t* dst = packed.data() + (size_t)e * small_slot_bytes_;
                    std::memcpy(dst, tmp_gu.data() + (size_t)e * lm.gate_up_exp_bytes, lm.gate_up_exp_bytes);
                    std::memcpy(dst + lm.down_aligned_off, tmp_dn.data() + (size_t)e * lm.down_exp_bytes, lm.down_exp_bytes);
                    int uid = ip * n_experts_ + e;
                    int s = next_small + e;
                    vram_slot_owner_[s] = uid;
                    uid_to_vram_slot_[uid] = s;
                    scores_[uid] = lm.is_core ? 0.5f : 0.25f;
                }
                INK_CUDA_CHECK(cudaMemcpy(vram_slot_ptr_[next_small], packed.data(), packed.size(), cudaMemcpyHostToDevice));
                next_small += take_s;
            }
            if (take_l > 0) {
                std::vector<uint8_t> packed((size_t)take_l * max_slot_bytes_, 0);
                for (int j = 0; j < take_l; ++j) {
                    int e = take_s + j;
                    uint8_t* dst = packed.data() + (size_t)j * max_slot_bytes_;
                    std::memcpy(dst, tmp_gu.data() + (size_t)e * lm.gate_up_exp_bytes, lm.gate_up_exp_bytes);
                    std::memcpy(dst + lm.down_aligned_off, tmp_dn.data() + (size_t)e * lm.down_exp_bytes, lm.down_exp_bytes);
                    int uid = ip * n_experts_ + e;
                    int s = next_large + j;
                    vram_slot_owner_[s] = uid;
                    uid_to_vram_slot_[uid] = s;
                    scores_[uid] = lm.is_core ? 0.5f : 0.25f;
                }
                INK_CUDA_CHECK(cudaMemcpy(vram_slot_ptr_[next_large], packed.data(), packed.size(), cudaMemcpyHostToDevice));
                next_large += take_l;
            }
        }
    }

    // Build GPU-side 24x128 expert pointer & scale tables (72 KB VRAM total!)
    for (int ip = 0; ip < n_layers_; ++ip) {
        const auto& lm = layers_[ip];
        std::vector<uint64_t> gu_ptrs(n_experts_, 0), dn_ptrs(n_experts_, 0);
        int cnt = 0;
        for (int e = 0; e < n_experts_; ++e) {
            int uid = ip * n_experts_ + e;
            int vs = uid_to_vram_slot_[uid];
            if (vs == -2) {
                gu_ptrs[e] = (uint64_t)(uintptr_t)pinned_gu_ptr_[uid];
                dn_ptrs[e] = (uint64_t)(uintptr_t)pinned_dn_ptr_[uid];
                cnt++;
            } else if (vs >= 0) {
                uint8_t* base = vram_slot_ptr_[vs];
                gu_ptrs[e] = (uint64_t)(uintptr_t)base;
                dn_ptrs[e] = (uint64_t)(uintptr_t)(base + lm.down_aligned_off);
                cnt++;
            }
        }
        layer_vram_count_[ip] = cnt;
        auto& tbl = layer_tables_[ip];
        INK_CUDA_CHECK(cudaMalloc(&tbl.d_gu_ptrs,   n_experts_ * sizeof(uint64_t)));
        INK_CUDA_CHECK(cudaMalloc(&tbl.d_dn_ptrs,   n_experts_ * sizeof(uint64_t)));
        INK_CUDA_CHECK(cudaMalloc(&tbl.d_gu_scales, n_experts_ * sizeof(float)));
        INK_CUDA_CHECK(cudaMalloc(&tbl.d_dn_scales, n_experts_ * sizeof(float)));
        INK_CUDA_CHECK(cudaMemcpy(tbl.d_gu_ptrs,   gu_ptrs.data(),           n_experts_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
        INK_CUDA_CHECK(cudaMemcpy(tbl.d_dn_ptrs,   dn_ptrs.data(),           n_experts_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
        INK_CUDA_CHECK(cudaMemcpy(tbl.d_gu_scales, lm.gate_up_scales.data(), n_experts_ * sizeof(float),    cudaMemcpyHostToDevice));
        INK_CUDA_CHECK(cudaMemcpy(tbl.d_dn_scales, lm.down_scales.data(),    n_experts_ * sizeof(float),    cudaMemcpyHostToDevice));
        tbl.valid = true;
    }

    // Allocate 2-Bucket Variable-Size Tier-1 Host RAM pool (saves ~230 MiB RAM vs uniform padding!)
    int rem_small = 0, rem_large = 0;
    for (int uid = 0; uid < total_experts_; ++uid) {
        if (uid_to_vram_slot_[uid] == -1) {
            if (layers_[uid / n_experts_].combined_bytes <= small_slot_bytes_) rem_small++;
            else rem_large++;
        }
    }
    int total_cold = rem_small + rem_large;
    // Leave a generous +96 universal slot cushion so cross-bucket VRAM<->RAM swaps NEVER evict live RAM experts (0 SSD reads!)
    int ram_cap_slots = std::min(total_experts_, total_cold + 96);
    n_ram_small_ = 0;
    n_ram_large_ = 0;
    if (use_two_buckets && ram_budget_bytes >= small_slot_bytes_) {
        int want_s = std::min(rem_small, (int)(ram_budget_bytes / small_slot_bytes_));
        size_t min_large_bytes = (size_t)std::min(rem_large + 96, ram_cap_slots) * max_slot_bytes_;
        if (ram_budget_bytes > min_large_bytes) {
            n_ram_small_ = std::min(want_s, (int)((ram_budget_bytes - min_large_bytes) / small_slot_bytes_));
        } else {
            n_ram_small_ = std::min(want_s, (int)((double)ram_budget_bytes * 0.42 / (double)small_slot_bytes_));
        }
        size_t s_bytes = (size_t)n_ram_small_ * small_slot_bytes_;
        size_t rem_ram = (ram_budget_bytes > s_bytes) ? (ram_budget_bytes - s_bytes) : 0;
        n_ram_large_ = (int)std::min<size_t>(rem_ram / max_slot_bytes_, (size_t)std::max(0, ram_cap_slots - n_ram_small_));
    } else {
        n_ram_large_ = (int)std::min<size_t>(ram_budget_bytes / max_slot_bytes_, (size_t)ram_cap_slots);
    }
    n_ram_slots_ = std::max(n_ram_small_ + n_ram_large_, 0);
    ram_pool_bytes_ = (size_t)n_ram_small_ * small_slot_bytes_ + (size_t)n_ram_large_ * max_slot_bytes_;

    if (n_ram_slots_ > 0) {
        h_ram_pool_ = (uint8_t*)VirtualAlloc(nullptr, ram_pool_bytes_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!h_ram_pool_) {
            std::fprintf(stderr, "[ExpertCache] Failed allocating %.1f MiB Tier-1 RAM pool\n",
                         ram_pool_bytes_ / (1024.0 * 1024.0));
            return false;
        }
        ram_slot_ptr_.resize(n_ram_slots_);
        ram_slot_cap_.resize(n_ram_slots_);
        ram_slot_owner_.assign(n_ram_slots_, -1);
        ram_slot_pin_.assign(n_ram_slots_, 0);
        ram_slot_lru_.assign(n_ram_slots_, 0);
        size_t off = 0;
        for (int s = 0; s < n_ram_small_; ++s) {
            ram_slot_ptr_[s] = h_ram_pool_ + off;
            ram_slot_cap_[s] = small_slot_bytes_;
            off += small_slot_bytes_;
        }
        for (int s = n_ram_small_; s < n_ram_slots_; ++s) {
            ram_slot_ptr_[s] = h_ram_pool_ + off;
            ram_slot_cap_[s] = max_slot_bytes_;
            off += max_slot_bytes_;
        }
    }

    // Preload cold experts into Tier-1 RAM (either async background thread for ~14s instant startup, or inline!)
    stop_preload_.store(false, std::memory_order_release);
    if (preload_ram && n_ram_slots_ > 0 && total_cold > 0) {
        if (async_ram_preload) {
            ram_preload_done_.store(false, std::memory_order_release);
            std::printf("[ExpertCache] Spawning async background RAM preload for %d cold experts (%.1f MiB)...\n",
                        std::min(total_cold, n_ram_slots_), ram_pool_bytes_ / (1024.0 * 1024.0));
            std::fflush(stdout);
            preload_thread_ = std::thread(&ExpertCache::bg_preload_ram_worker, this);
        } else {
            ram_preload_done_.store(false, std::memory_order_release);
            std::printf("[ExpertCache] Preloading %d/%d cold experts into Tier-1 RAM (%.1f MiB)...\n",
                        std::min(total_cold, n_ram_slots_), total_cold, ram_pool_bytes_ / (1024.0 * 1024.0));
            std::fflush(stdout);
            bg_preload_ram_worker();
        }
    } else {
        ram_preload_done_.store(true, std::memory_order_release);
    }

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    stats_.vram_mb = (core_pinned_bytes_ + vram_pool_bytes_) / (1024.0 * 1024.0);
    stats_.ram_mb  = ram_pool_bytes_ / (1024.0 * 1024.0);
    stats_.vram_slots = n_vram_slots_;
    stats_.ram_slots  = n_ram_slots_;
    stats_.pinned_core_experts = pinned_count;

    std::printf("[ExpertCache] Ready in %.2fs | Policy=%s | VRAM=%.1f MiB (pinned=%d, dynamic=%d) | RAM=%.1f MiB (%d slots%s) | SSD backing=ON\n",
                sec,
                policy_ == CACHE_CORE_PINNED ? "core-pinned" : "loop-lfu",
                stats_.vram_mb, pinned_count, n_vram_slots_,
                stats_.ram_mb, n_ram_slots_,
                (preload_ram && async_ram_preload) ? ", async warming" : "");
    std::fflush(stdout);
    return true;
}

void ExpertCache::bg_preload_ram_worker() {
    auto t0 = std::chrono::steady_clock::now();
    // Prioritize Outer layers first (most likely to miss VRAM during initial tokens), then any cold Core layers!
    std::vector<int> order;
    for (int ip = 0; ip < core_start_; ++ip) order.push_back(ip);
    for (int ip = core_end_; ip < n_layers_; ++ip) order.push_back(ip);
    for (int ip = core_start_; ip < core_end_; ++ip) order.push_back(ip);

    int loaded = 0;
    for (int ip : order) {
        if (stop_preload_.load(std::memory_order_acquire)) break;
        const auto& lm = layers_[ip];

        // Quick check under lock whether this layer has uncached experts & free fitting RAM slots
        int need = 0;
        {
            std::lock_guard<std::mutex> lk(ram_mu_);
            for (int e = 0; e < n_experts_; ++e) {
                int uid = ip * n_experts_ + e;
                if (uid_to_vram_slot_[uid] == -1 && uid_to_ram_slot_[uid] == -1) need++;
            }
        }
        if (need == 0) continue;

        // Bulk sequential read from SSD OUTSIDE mutex (zero inference blocking!)
        std::vector<uint8_t> tmp_gu(lm.gate_up_exp_bytes * n_experts_);
        std::vector<uint8_t> tmp_dn(lm.down_exp_bytes * n_experts_);
        if (!gguf_->pread(lm.gate_up_file_off, tmp_gu.data(), tmp_gu.size()) ||
            !gguf_->pread(lm.down_file_off, tmp_dn.data(), tmp_dn.size())) {
            continue;
        }
        if (stop_preload_.load(std::memory_order_acquire)) break;

        // Insert uncached experts into free fitting RAM slots under brief lock
        std::lock_guard<std::mutex> lk(ram_mu_);
        int scan_s = 0;
        int scan_l = n_ram_small_;
        bool fits_small = (lm.combined_bytes <= small_slot_bytes_);
        for (int e = 0; e < n_experts_; ++e) {
            int uid = ip * n_experts_ + e;
            if (uid_to_vram_slot_[uid] != -1 || uid_to_ram_slot_[uid] != -1) continue;

            int rslot = -1;
            if (fits_small) {
                while (scan_s < n_ram_small_ && (ram_slot_owner_[scan_s] != -1 || ram_slot_pin_[scan_s] != 0)) scan_s++;
                if (scan_s < n_ram_small_) rslot = scan_s++;
            }
            if (rslot < 0) {
                while (scan_l < n_ram_slots_ && (ram_slot_owner_[scan_l] != -1 || ram_slot_pin_[scan_l] != 0)) scan_l++;
                if (scan_l < n_ram_slots_) rslot = scan_l++;
            }
            if (rslot < 0) break; // RAM pool full

            uint8_t* dst = ram_slot_ptr_[rslot];
            std::memcpy(dst, tmp_gu.data() + (size_t)e * lm.gate_up_exp_bytes, lm.gate_up_exp_bytes);
            std::memcpy(dst + lm.down_aligned_off, tmp_dn.data() + (size_t)e * lm.down_exp_bytes, lm.down_exp_bytes);
            ram_slot_owner_[rslot] = uid;
            uid_to_ram_slot_[uid] = rslot;
            ram_slot_lru_[rslot] = ++clock_;
            loaded++;
        }
    }

    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ram_preload_done_.store(true, std::memory_order_release);
    if (!stop_preload_.load(std::memory_order_acquire)) {
        std::printf("[ExpertCache] Tier-1 RAM preload complete in %.2fs (%d cold experts warm in RAM)\n", sec, loaded);
        std::fflush(stdout);
    }
}

void ExpertCache::reset_stats() {
    stats_.total_lookups = 0;
    stats_.vram_hits = 0;
    stats_.ram_hits = 0;
    stats_.ssd_reads = 0;
    stats_.core_lookups = 0;
    stats_.core_vram_hits = 0;
    stats_.outer_lookups = 0;
    stats_.outer_vram_hits = 0;
    stats_.dma_promotions = 0;
}

bool ExpertCache::read_expert_from_ssd(int uid, uint8_t* dst) {
    if (test_ssd_fail_) {
        io_error_ = true;
        last_error_ = "Injected SSD expert read failure";
        return false;
    }
    int ip = uid / n_experts_;
    int e  = uid % n_experts_;
    const auto& lm = layers_[ip];
    bool ok1 = gguf_->pread(lm.gate_up_file_off + (uint64_t)e * lm.gate_up_exp_bytes, dst, lm.gate_up_exp_bytes);
    bool ok2 = ok1 && gguf_->pread(lm.down_file_off + (uint64_t)e * lm.down_exp_bytes, dst + lm.down_aligned_off, lm.down_exp_bytes);
    stats_.ssd_reads++;
    if (!ok2) {
        io_error_ = true;
        last_error_ = "Short or failed GGUF SSD read for expert uid=" + std::to_string(uid);
    }
    return ok2;
}

bool ExpertCache::evict_ram_expert_for_test(int uid) {
    auto work = [&]() -> bool {
        if (uid < 0 || uid >= total_experts_) return false;
        int rs = uid_to_ram_slot_[uid];
        if (rs < 0 || ram_slot_pin_[rs] > 0) return false;
        ram_slot_owner_[rs] = -1;
        uid_to_ram_slot_[uid] = -1;
        return true;
    };
    if (ram_preload_done_.load(std::memory_order_acquire)) return work();
    std::lock_guard<std::mutex> lk(ram_mu_);
    return work();
}

int ExpertCache::find_ram_victim_unlocked(size_t need_bytes) const {
    auto is_busy_ram = [&](int s) {
        if (ram_slot_pin_[s] > 0) return true;
        for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
            if (dma_[i].active && dma_[i].ram_slot == s) return true;
        }
        return false;
    };
    int start_s = (need_bytes <= small_slot_bytes_) ? 0 : n_ram_small_;
    int best = -1;
    uint64_t oldest = UINT64_MAX;
    for (int s = start_s; s < n_ram_slots_; ++s) {
        if (ram_slot_cap_[s] < need_bytes) continue;
        if (is_busy_ram(s)) continue;
        if (ram_slot_owner_[s] == -1) return s;
        if (ram_slot_lru_[s] < oldest) {
            oldest = ram_slot_lru_[s];
            best = s;
        }
    }
    return best;
}

int ExpertCache::ensure_in_ram_unlocked(int uid) {
    int rslot = uid_to_ram_slot_[uid];
    if (rslot >= 0) {
        ram_slot_lru_[rslot] = ++clock_;
        stats_.ram_hits++;
        return rslot;
    }
    // Tier-2 SSD miss! Demand-page from GGUF file directly into an unpinned fitting LRU RAM slot!
    size_t need = layers_[uid / n_experts_].combined_bytes;
    rslot = find_ram_victim_unlocked(need);
    if (rslot < 0) return -1;
    int old_uid = ram_slot_owner_[rslot];
    if (old_uid >= 0) {
        uid_to_ram_slot_[old_uid] = -1;
        ram_slot_owner_[rslot] = -1;
    }
    uint8_t* dst = ram_slot_ptr_[rslot];
    if (!read_expert_from_ssd(uid, dst)) {
        // R6 (SR3-03): Never publish incomplete/failed expert bytes into RAM slot!
        ram_slot_owner_[rslot] = -1;
        uid_to_ram_slot_[uid] = -1;
        std::fprintf(stderr, "[ExpertCache] ERROR: SSD read failed for expert uid=%d\n", uid);
        return -1;
    }
    ram_slot_owner_[rslot] = uid;
    uid_to_ram_slot_[uid] = rslot;
    ram_slot_lru_[rslot] = ++clock_;
    return rslot;
}

int ExpertCache::ensure_in_ram(int uid) {
    if (ram_preload_done_.load(std::memory_order_acquire)) {
        return ensure_in_ram_unlocked(uid);
    }
    std::lock_guard<std::mutex> lk(ram_mu_);
    return ensure_in_ram_unlocked(uid);
}

void ExpertCache::complete_dma_unlocked(int i, cudaStream_t compute_stream) {
    auto& d = dma_[i];
    if (!d.active) return;

    // 1. Deactivate evicted VRAM expert (if any)
    if (d.uid_out >= 0) {
        int ip_out = d.uid_out / n_experts_;
        int e_out  = d.uid_out % n_experts_;
        uid_to_vram_slot_[d.uid_out] = -1;
        layer_vram_count_[ip_out]--;
        h_ptr_update_[i * 4 + 0] = 0;
        if (compute_stream) {
            INK_CUDA_CHECK(cudaMemcpyAsync(layer_tables_[ip_out].d_gu_ptrs + e_out, &h_ptr_update_[i * 4 + 0], sizeof(uint64_t), cudaMemcpyHostToDevice, compute_stream));
            INK_CUDA_CHECK(cudaMemcpyAsync(layer_tables_[ip_out].d_dn_ptrs + e_out, &h_ptr_update_[i * 4 + 0], sizeof(uint64_t), cudaMemcpyHostToDevice, compute_stream));
        } else {
            INK_CUDA_CHECK(cudaMemcpy(layer_tables_[ip_out].d_gu_ptrs + e_out, &h_ptr_update_[i * 4 + 0], sizeof(uint64_t), cudaMemcpyHostToDevice));
            INK_CUDA_CHECK(cudaMemcpy(layer_tables_[ip_out].d_dn_ptrs + e_out, &h_ptr_update_[i * 4 + 0], sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
    }

    // 2. Activate promoted expert in VRAM + update GPU device pointer table
    {
        int ip_in = d.uid_in / n_experts_;
        int e_in  = d.uid_in % n_experts_;
        vram_slot_owner_[d.vram_slot] = d.uid_in;
        uid_to_vram_slot_[d.uid_in]   = d.vram_slot;
        layer_vram_count_[ip_in]++;

        uint8_t* vbase = vram_slot_ptr_[d.vram_slot];
        h_ptr_update_[i * 4 + 2] = (uint64_t)(uintptr_t)vbase;
        h_ptr_update_[i * 4 + 3] = (uint64_t)(uintptr_t)(vbase + layers_[ip_in].down_aligned_off);
        if (compute_stream) {
            INK_CUDA_CHECK(cudaMemcpyAsync(layer_tables_[ip_in].d_gu_ptrs + e_in, &h_ptr_update_[i * 4 + 2], sizeof(uint64_t), cudaMemcpyHostToDevice, compute_stream));
            INK_CUDA_CHECK(cudaMemcpyAsync(layer_tables_[ip_in].d_dn_ptrs + e_in, &h_ptr_update_[i * 4 + 3], sizeof(uint64_t), cudaMemcpyHostToDevice, compute_stream));
        } else {
            INK_CUDA_CHECK(cudaMemcpy(layer_tables_[ip_in].d_gu_ptrs + e_in, &h_ptr_update_[i * 4 + 2], sizeof(uint64_t), cudaMemcpyHostToDevice));
            INK_CUDA_CHECK(cudaMemcpy(layer_tables_[ip_in].d_dn_ptrs + e_in, &h_ptr_update_[i * 4 + 3], sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
    }

    // 3. Place evicted expert into a fitting RAM slot (overflow-proof across 2-bucket sizes!)
    if (d.ram_slot >= 0) {
        if (uid_to_ram_slot_[d.uid_in] == d.ram_slot) {
            uid_to_ram_slot_[d.uid_in] = -1;
        }
        if (d.uid_out >= 0) {
            int dst_rslot = d.ram_slot;
            if (ram_slot_cap_[dst_rslot] < d.out_bytes) {
                // Free the smaller RAM slot and acquire a fitting universal RAM slot
                ram_slot_owner_[dst_rslot] = -1;
                dst_rslot = find_ram_victim_unlocked(d.out_bytes);
                if (dst_rslot >= 0) {
                    int prev = ram_slot_owner_[dst_rslot];
                    if (prev >= 0) uid_to_ram_slot_[prev] = -1;
                }
            }
            if (dst_rslot >= 0) {
                uint8_t* rdst = ram_slot_ptr_[dst_rslot];
                std::memcpy(rdst, d.h_out_staging, d.out_bytes);
                ram_slot_owner_[dst_rslot] = d.uid_out;
                uid_to_ram_slot_[d.uid_out] = dst_rslot;
                ram_slot_lru_[dst_rslot]   = ++clock_;
            }
        } else {
            ram_slot_owner_[d.ram_slot] = -1;
        }
    }
    d.active = false;
    stats_.dma_promotions++;
}

void ExpertCache::poll_promotions() {
    auto work = [&]() {
        for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
            if (!dma_[i].active) continue;
            if (cudaEventQuery(dma_[i].event) == cudaSuccess) {
                complete_dma_unlocked(i, nullptr);
            }
        }
    };
    if (ram_preload_done_.load(std::memory_order_acquire)) work();
    else { std::lock_guard<std::mutex> lk(ram_mu_); work(); }
}

int ExpertCache::find_vram_victim(float incoming_score, size_t need_bytes, int phase_filter) const {
    if (n_vram_slots_ <= 0) return -1;
    if (incoming_score < 2.0f) return -1;

    auto is_busy = [&](int s) {
        for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
            if (dma_[i].active && dma_[i].vram_slot == s) return true;
        }
        return false;
    };
    int start_s = (need_bytes <= small_slot_bytes_) ? 0 : n_vram_small_;
    int best_slot = -1;
    float best_score = incoming_score * 0.80f; // 20% hysteresis prevents PCIe thrashing
    for (int s = start_s; s < n_vram_slots_; ++s) {
        if (vram_slot_cap_[s] < need_bytes) continue;
        if (is_busy(s)) continue;
        int owner = vram_slot_owner_[s];
        if (owner == -1) return s;
        bool owner_core = layers_[owner / n_experts_].is_core;
        if (phase_filter == 1 && owner_core)  continue; // Core pass: NEVER touch core slots!
        if (phase_filter == 2 && !owner_core) continue; // Outer coda: NEVER touch outer slots!
        if (phase_filter == 0 && last_token_[owner] == token_seq_) continue;
        float sc = scores_[owner];
        if (sc < best_score) {
            best_score = sc;
            best_slot = s;
        }
    }
    return best_slot;
}

void ExpertCache::maybe_promote_to_vram(int uid, int ram_slot) {
    (void)ram_slot;
    if (n_vram_slots_ <= 0 || scores_[uid] < 2.0f) return;
    for (int p : pending_promos_) {
        if (p == uid) return;
    }
    pending_promos_.push_back(uid);
}

bool ExpertCache::resolve_gpu_split(
    int ip,
    const MoESplitPacket& pkt,
    CPUExpertTask* cpu_tasks,
    int& n_cpu)
{
    const auto& lm = layers_[ip];
    const float weight_mult = lm.is_core ? 3.0f : 1.0f;

    for (int i = 0; i < 8; ++i) {
        int uid = ip * n_experts_ + pkt.all_ids[i];
        scores_[uid] += weight_mult;
        last_used_[uid] = ++clock_;
        last_token_[uid] = token_seq_;
    }

    stats_.total_lookups += 8;
    stats_.vram_hits += pkt.n_gpu;
    if (lm.is_core) { stats_.core_lookups += 8; stats_.core_vram_hits += pkt.n_gpu; }
    else            { stats_.outer_lookups += 8; stats_.outer_vram_hits += pkt.n_gpu; }

    n_cpu = 0;
    for (int i = 0; i < pkt.n_cpu; ++i) {
        int e = pkt.miss_ids[i];
        int uid = ip * n_experts_ + e;
        int rslot = ensure_in_ram(uid);
        if (rslot < 0) {
            io_error_ = true;
            if (last_error_.empty()) last_error_ = "Failed loading missed expert uid=" + std::to_string(uid) + " into RAM";
            return false;
        }
        const uint8_t* base = ram_slot_ptr_[rslot];
        CPUExpertTask& tk = cpu_tasks[n_cpu++];
        tk.gate_up_data  = base;
        tk.down_data     = base + lm.down_aligned_off;
        tk.gate_up_type  = lm.gate_up_type;
        tk.down_type     = lm.down_type;
        tk.gate_up_scale = lm.gate_up_scales[e];
        tk.down_scale    = lm.down_scales[e];
        tk.router_weight = pkt.miss_weights[i];

        if (n_vram_slots_ > 0 && scores_[uid] >= 2.0f) {
            bool dup = false;
            for (int p : pending_promos_) { if (p == uid) { dup = true; break; } }
            if (!dup) pending_promos_.push_back(uid);
        }
    }
    return true;
}

int ExpertCache::find_vram_victim_pred(float incoming_score, float min_gain, size_t need_bytes, int phase_filter) const {
    if (n_vram_slots_ <= 0) return -1;
    auto is_busy = [&](int s) {
        for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
            if (dma_[i].active && dma_[i].vram_slot == s) return true;
        }
        return false;
    };
    int start_s = (need_bytes <= small_slot_bytes_) ? 0 : n_vram_small_;
    int best_slot = -1;
    float best_score = incoming_score - min_gain;
    for (int s = start_s; s < n_vram_slots_; ++s) {
        if (vram_slot_cap_[s] < need_bytes) continue;
        if (is_busy(s)) continue;
        int owner = vram_slot_owner_[s];
        if (owner == -1) return s;
        bool owner_core = layers_[owner / n_experts_].is_core;
        if (phase_filter == 1 && owner_core)  continue;
        if (phase_filter == 2 && !owner_core) continue;
        float sc = pred_scores_[owner];
        if (sc < best_score) {
            best_score = sc;
            best_slot = s;
        }
    }
    return best_slot;
}

bool ExpertCache::launch_single_swap_unlocked(int uid_in, int victim_slot, int rslot, cudaStream_t compute_stream, bool record_compute_dep) {
    int ch = -1;
    for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
        if (dma_[i].active) {
            if (dma_[i].uid_in == uid_in || dma_[i].vram_slot == victim_slot || dma_[i].ram_slot == rslot) return false;
        } else if (ch < 0) {
            ch = i;
        }
    }
    if (ch < 0) return false;

    if (record_compute_dep && compute_stream) {
        INK_CUDA_CHECK(cudaEventRecord(ev_compute_done_, compute_stream));
        INK_CUDA_CHECK(cudaStreamWaitEvent(stream_dma_, ev_compute_done_, 0));
    }

    int old_uid = vram_slot_owner_[victim_slot];
    int ip_in = uid_in / n_experts_;
    size_t copy_in = layers_[ip_in].combined_bytes;
    const uint8_t* src_ram = ram_slot_ptr_[rslot];
    std::memcpy(dma_[ch].h_in_staging, src_ram, copy_in);

    uint8_t* vram_ptr = vram_slot_ptr_[victim_slot];
    size_t copy_out = 0;
    if (old_uid >= 0) {
        int ip_out = old_uid / n_experts_;
        copy_out = layers_[ip_out].combined_bytes;
        INK_CUDA_CHECK(cudaMemcpyAsync(dma_[ch].h_out_staging, vram_ptr, copy_out, cudaMemcpyDeviceToHost, stream_dma_));
    }
    INK_CUDA_CHECK(cudaMemcpyAsync(vram_ptr, dma_[ch].h_in_staging, copy_in, cudaMemcpyHostToDevice, stream_dma_));
    INK_CUDA_CHECK(cudaEventRecord(dma_[ch].event, stream_dma_));

    dma_[ch].active    = true;
    dma_[ch].uid_in    = uid_in;
    dma_[ch].uid_out   = old_uid;
    dma_[ch].vram_slot = victim_slot;
    dma_[ch].ram_slot  = rslot;
    dma_[ch].out_bytes = copy_out;
    return true;
}

void ExpertCache::clear_predictor_scores() {
    use_predictor_ = false;
    pred_queue_all_.clear();
    pred_queue_outer_.clear();
    pred_queue_core_.clear();
    pred_pos_all_ = pred_pos_outer_ = pred_pos_core_ = 0;
}

int ExpertCache::apply_predictor_scores(
    const std::vector<float>& pred_scores,
    float min_bulk_gain,
    int max_bulk_swaps,
    float min_bg_gain,
    cudaStream_t compute_stream)
{
    if (swaps_frozen_ || (int)pred_scores.size() != total_experts_ || n_vram_slots_ <= 0) return 0;

    finish_token_end_swaps(compute_stream, false);

    pred_scores_ = pred_scores;
    pred_min_bg_gain_ = min_bg_gain;
    use_predictor_ = true;

    std::vector<int> cand_in;
    cand_in.reserve(total_experts_);
    {
        auto gather = [&]() {
            for (int uid = 0; uid < total_experts_; ++uid) {
                if (uid_to_vram_slot_[uid] == -1 && uid_to_ram_slot_[uid] >= 0) {
                    cand_in.push_back(uid);
                }
            }
        };
        if (ram_preload_done_.load(std::memory_order_acquire)) gather();
        else { std::lock_guard<std::mutex> lk(ram_mu_); gather(); }
    }
    std::sort(cand_in.begin(), cand_in.end(),
              [&](int a, int b) { return pred_scores_[a] > pred_scores_[b]; });

    int total_swapped = 0;
    size_t ci = 0;
    if (max_bulk_swaps > 0 && !cand_in.empty()) {
        if (compute_stream) INK_CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        while (total_swapped < max_bulk_swaps && ci < cand_in.size()) {
            int batch = 0;
            auto launch_batch = [&]() {
                while (batch < MAX_INFLIGHT_DMA && total_swapped < max_bulk_swaps && ci < cand_in.size()) {
                    int uid_in = cand_in[ci++];
                    if (uid_to_vram_slot_[uid_in] != -1) continue;
                    int rslot = uid_to_ram_slot_[uid_in];
                    if (rslot < 0) continue;
                    size_t need = layers_[uid_in / n_experts_].combined_bytes;
                    int victim = find_vram_victim_pred(pred_scores_[uid_in], min_bulk_gain, need, 0);
                    if (victim < 0) {
                        // A large expert might not find a large victim while a smaller expert still could
                        if (need <= small_slot_bytes_) { ci = cand_in.size(); break; }
                        continue;
                    }
                    if (launch_single_swap_unlocked(uid_in, victim, rslot, nullptr, false)) {
                        batch++;
                        total_swapped++;
                    }
                }
            };
            if (ram_preload_done_.load(std::memory_order_acquire)) launch_batch();
            else { std::lock_guard<std::mutex> lk(ram_mu_); launch_batch(); }
            if (batch == 0) break;
            finish_token_end_swaps(compute_stream, false);
        }
        if (compute_stream) INK_CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }

    // Build O(1) phase-separated predictor queues for zero-stall Loop-Phased Continuous DMA!
    pred_queue_all_.clear();
    pred_queue_outer_.clear();
    pred_queue_core_.clear();
    pred_pos_all_ = pred_pos_outer_ = pred_pos_core_ = 0;

    auto fill_queues = [&]() {
        for (int uid : cand_in) {
            if (uid_to_vram_slot_[uid] == -1 && uid_to_ram_slot_[uid] >= 0) {
                pred_queue_all_.push_back(uid);
                if (layers_[uid / n_experts_].is_core) pred_queue_core_.push_back(uid);
                else                                   pred_queue_outer_.push_back(uid);
            }
        }
    };
    if (ram_preload_done_.load(std::memory_order_acquire)) fill_queues();
    else { std::lock_guard<std::mutex> lk(ram_mu_); fill_queues(); }

    return total_swapped;
}

const uint8_t* ExpertCache::get_ram_expert_ptr(int uid) {
    int rslot = ensure_in_ram(uid);
    return (rslot >= 0) ? ram_slot_ptr_[rslot] : nullptr;
}

const uint8_t* ExpertCache::pin_ram_expert(int uid, int* out_slot) {
    auto work = [&]() -> int {
        int rslot = ensure_in_ram_unlocked(uid);
        if (rslot >= 0 && ram_slot_pin_[rslot] < 255) {
            ram_slot_pin_[rslot]++;
        }
        return rslot;
    };
    int rslot = -1;
    if (ram_preload_done_.load(std::memory_order_acquire)) rslot = work();
    else { std::lock_guard<std::mutex> lk(ram_mu_); rslot = work(); }
    if (out_slot) *out_slot = rslot;
    return (rslot >= 0) ? ram_slot_ptr_[rslot] : nullptr;
}

void ExpertCache::unpin_ram_slots(const int* slots, int n) {
    if (!slots || n <= 0) return;
    auto work = [&]() {
        for (int i = 0; i < n; ++i) {
            int s = slots[i];
            if (s >= 0 && s < n_ram_slots_ && ram_slot_pin_[s] > 0) {
                ram_slot_pin_[s]--;
            }
        }
    };
    if (ram_preload_done_.load(std::memory_order_acquire)) work();
    else { std::lock_guard<std::mutex> lk(ram_mu_); work(); }
}

void ExpertCache::record_batch_hit_stats(int ip, int n_vram_hits, int n_cpu_misses) {
    int tot = n_vram_hits + n_cpu_misses;
    stats_.total_lookups += tot;
    stats_.vram_hits += n_vram_hits;
    if (layers_[ip].is_core) { stats_.core_lookups += tot; stats_.core_vram_hits += n_vram_hits; }
    else                     { stats_.outer_lookups += tot; stats_.outer_vram_hits += n_vram_hits; }
}

void ExpertCache::begin_phased_swaps(cudaStream_t compute_stream, int phase_filter, int max_swaps) {
    if (swaps_frozen_ || n_vram_slots_ <= 0 || max_swaps <= 0) return;

    int free_ch = 0;
    for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
        if (!dma_[i].active) free_ch++;
    }
    int limit = std::min(max_swaps, free_ch);
    if (limit <= 0) return;

    auto run = [&]() {
        int launched = 0;
        if (use_predictor_) {
            auto& q   = (phase_filter == 1) ? pred_queue_outer_ : ((phase_filter == 2) ? pred_queue_core_ : pred_queue_all_);
            auto& pos = (phase_filter == 1) ? pred_pos_outer_   : ((phase_filter == 2) ? pred_pos_core_   : pred_pos_all_);

            while (launched < limit && pos < q.size()) {
                int uid = q[pos++];
                if (uid_to_vram_slot_[uid] != -1) continue;
                int rslot = uid_to_ram_slot_[uid];
                if (rslot < 0) continue;
                size_t need = layers_[uid / n_experts_].combined_bytes;
                float gain_req = (phase_filter == 1) ? std::max(pred_min_bg_gain_, 6.0f) : pred_min_bg_gain_;
                int victim = find_vram_victim_pred(pred_scores_[uid], gain_req, need, phase_filter);
                if (victim < 0) {
                    if (need <= small_slot_bytes_) { pos = q.size(); break; }
                    continue;
                }
                if (launch_single_swap_unlocked(uid, victim, rslot, compute_stream, launched == 0)) {
                    launched++;
                }
            }
            return;
        }

        if (pending_promos_.empty()) return;
        std::sort(pending_promos_.begin(), pending_promos_.end(),
                  [&](int a, int b) { return scores_[a] > scores_[b]; });

        for (int uid : pending_promos_) {
            if (launched >= limit) break;
            if (uid_to_vram_slot_[uid] != -1) continue;
            bool is_c = layers_[uid / n_experts_].is_core;
            if (phase_filter == 1 && is_c)  continue;
            if (phase_filter == 2 && !is_c) continue;
            int rslot = uid_to_ram_slot_[uid];
            if (rslot < 0) continue;
            size_t need = layers_[uid / n_experts_].combined_bytes;
            int victim = find_vram_victim(scores_[uid], need, phase_filter);
            if (victim < 0) continue;
            if (launch_single_swap_unlocked(uid, victim, rslot, compute_stream, launched == 0)) {
                launched++;
            }
        }
    };

    if (ram_preload_done_.load(std::memory_order_acquire)) run();
    else { std::lock_guard<std::mutex> lk(ram_mu_); run(); }
}

void ExpertCache::finish_phased_swaps(cudaStream_t compute_stream) {
    auto run = [&]() {
        for (int i = 0; i < MAX_INFLIGHT_DMA; ++i) {
            if (!dma_[i].active) continue;
            INK_CUDA_CHECK(cudaEventSynchronize(dma_[i].event));
            complete_dma_unlocked(i, compute_stream);
        }
    };
    if (ram_preload_done_.load(std::memory_order_acquire)) run();
    else { std::lock_guard<std::mutex> lk(ram_mu_); run(); }
}

void ExpertCache::begin_token_end_swaps(cudaStream_t compute_stream, int max_swaps) {
    begin_phased_swaps(compute_stream, /*phase_filter=*/0, max_swaps);
}

void ExpertCache::finish_token_end_swaps(cudaStream_t compute_stream, bool advance_token) {
    finish_phased_swaps(compute_stream);
    if (advance_token) {
        pending_promos_.clear();
        token_seq_++;
        for (int i = 0; i < total_experts_; ++i) {
            scores_[i] *= 0.996f;
        }
    }
}

void ExpertCache::on_token_end() {
    finish_token_end_swaps(nullptr);
}

bool ExpertCache::resolve_layer_experts(
    int ip,
    const int32_t* expert_ids,
    const float* router_weights,
    int k,
    GPUExpertBatch& gpu_batch,
    CPUExpertTask* cpu_tasks,
    int& n_cpu)
{
    const auto& lm = layers_[ip];
    const float weight_mult = lm.is_core ? 3.0f : 1.0f;

    gpu_batch.count = 0;
    n_cpu = 0;

    for (int i = 0; i < k; ++i) {
        int e = expert_ids[i];
        float rw = router_weights[i];
        int uid = ip * n_experts_ + e;

        scores_[uid] += weight_mult;
        last_used_[uid] = ++clock_;
        last_token_[uid] = token_seq_;
        stats_.total_lookups++;
        if (lm.is_core) stats_.core_lookups++;
        else            stats_.outer_lookups++;

        int vslot = uid_to_vram_slot_[uid];
        if (vslot == -2) {
            stats_.vram_hits++;
            if (lm.is_core) stats_.core_vram_hits++; else stats_.outer_vram_hits++;
            int b = gpu_batch.count++;
            gpu_batch.gate_up_ptrs[b] = pinned_gu_ptr_[uid];
            gpu_batch.down_ptrs[b]    = pinned_dn_ptr_[uid];
            gpu_batch.gate_up_scales[b] = lm.gate_up_scales[e];
            gpu_batch.down_w_scales[b]  = lm.down_scales[e] * rw;
        } else if (vslot >= 0) {
            stats_.vram_hits++;
            if (lm.is_core) stats_.core_vram_hits++; else stats_.outer_vram_hits++;
            uint8_t* base = vram_slot_ptr_[vslot];
            int b = gpu_batch.count++;
            gpu_batch.gate_up_ptrs[b] = base;
            gpu_batch.down_ptrs[b]    = base + lm.down_aligned_off;
            gpu_batch.gate_up_scales[b] = lm.gate_up_scales[e];
            gpu_batch.down_w_scales[b]  = lm.down_scales[e] * rw;
        } else {
            int rslot = ensure_in_ram(uid);
            if (rslot < 0) {
                io_error_ = true;
                if (last_error_.empty()) last_error_ = "Failed loading expert uid=" + std::to_string(uid) + " into RAM";
                return false;
            }
            const uint8_t* base = ram_slot_ptr_[rslot];
            CPUExpertTask& tk = cpu_tasks[n_cpu++];
            tk.gate_up_data  = base;
            tk.down_data     = base + lm.down_aligned_off;
            tk.gate_up_type  = lm.gate_up_type;
            tk.down_type     = lm.down_type;
            tk.gate_up_scale = lm.gate_up_scales[e];
            tk.down_scale    = lm.down_scales[e];
            tk.router_weight = rw;
            maybe_promote_to_vram(uid, rslot);
        }
    }
    return true;
}
