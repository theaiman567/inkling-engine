#pragma once
#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <string>
#include <cuda_runtime.h>
#include "gguf.h"
#include "cuda_kernels.cuh"
#include "cpu_kernels.h"

enum ExpertCachePolicy {
    CACHE_LOOP_LFU    = 0, // Dynamic 3x-core-weighted LFU-aging cache across all 24 layers + async DMA promotion
    CACHE_CORE_PINNED = 1, // 100% of looped core (blk.6..17) pinned tightly in VRAM + dynamic outer cache
};

struct LayerExpertMeta {
    uint32_t gate_up_type = 0;
    uint32_t down_type = 0;
    uint64_t gate_up_file_off = 0;
    uint64_t down_file_off = 0;
    size_t gate_up_exp_bytes = 0; // bytes per single expert
    size_t down_exp_bytes = 0;    // bytes per single expert
    size_t down_aligned_off = 0;  // offset of down inside combined expert slot (256-byte aligned)
    size_t combined_bytes = 0;
    bool is_core = false;
    std::vector<float> gate_up_scales; // [128] (1.0f if absent)
    std::vector<float> down_scales;    // [128] (1.0f if absent)
};

struct ExpertCacheStats {
    uint64_t total_lookups = 0;
    uint64_t vram_hits = 0;
    uint64_t ram_hits = 0;
    uint64_t ssd_reads = 0;
    uint64_t core_lookups = 0;
    uint64_t core_vram_hits = 0;
    uint64_t outer_lookups = 0;
    uint64_t outer_vram_hits = 0;
    uint64_t dma_promotions = 0;
    double   vram_mb = 0.0;
    double   ram_mb = 0.0;
    int      vram_slots = 0;
    int      ram_slots = 0;
    int      pinned_core_experts = 0;
};

class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    bool init(
        const GGUFFile& gguf,
        int n_phys_layers,
        int n_experts,
        int core_start,
        int core_end,
        size_t vram_budget_bytes,
        size_t ram_budget_bytes,
        ExpertCachePolicy policy,
        bool preload_ram,
        bool async_ram_preload = false);

    void shutdown();
    void wait_for_ram_preload();
    bool is_ram_preload_done() const { return ram_preload_done_.load(std::memory_order_acquire); }

    // Poll async DMA promotions (legacy/no-op for token-end swaps)
    void poll_promotions();

    // GPU-first split resolution: updates LFU scores & stats, resolves ONLY CPU-missed experts
    // from RAM/SSD, and queues hot misses for zero-stall phased / end-of-token exclusive swap!
    void resolve_gpu_split(
        int ip,
        const MoESplitPacket& pkt,
        CPUExpertTask* cpu_tasks,
        int& n_cpu);

    // Loop-Phased Continuous Background DMA (InkEngine v3.0):
    //   phase_filter = 1 (OUTER_ONLY): safe during Core passes [il=6..41], swaps outer experts into outer victims!
    //   phase_filter = 2 (CORE_ONLY) : safe during Outer Coda [il=42..47] + LM Head, swaps core experts into core victims!
    //   phase_filter = 0 (ANY)       : safe behind LM Head, allows cross-tier Core <-> Outer rebalancing!
    void begin_phased_swaps(cudaStream_t compute_stream, int phase_filter, int max_swaps);
    void finish_phased_swaps(cudaStream_t compute_stream);

    // Begin up to `max_swaps` exclusive VRAM <-> RAM swaps on stream_dma_ (overlapped with LM Head!)
    void begin_token_end_swaps(cudaStream_t compute_stream, int max_swaps);

    // Complete in-flight token-end swaps, update GPU pointer tables, and optionally decay LFU scores.
    void finish_token_end_swaps(cudaStream_t compute_stream, bool advance_token = true);

    // Apply learned MoE predictor scores [total_experts]:
    //  - Executes up to `max_bulk_swaps` immediate pipelined swaps with gain >= `min_bulk_gain`
    //  - Queues remaining beneficial swaps (gain >= `min_bg_gain`) for zero-stall phased & LM-Head streaming!
    int apply_predictor_scores(
        const std::vector<float>& pred_scores,
        float min_bulk_gain,
        int max_bulk_swaps,
        float min_bg_gain,
        cudaStream_t compute_stream);

    void clear_predictor_scores();

    // Direct lookup helpers for grouped batched prefill (Stage 4 & v3.0 PCIe streaming)
    bool is_in_vram(int uid) const { return uid >= 0 && uid < total_experts_ && uid_to_vram_slot_[uid] != -1; }
    const uint8_t* get_ram_expert_ptr(int uid); // ensures expert is in RAM (demand-pages SSD if needed)
    // Pin a cold expert in RAM so it CANNOT be evicted until unpin_ram_slots() is called!
    // Returns nullptr if every fitting RAM slot is already pinned (caller should flush current batch first).
    const uint8_t* pin_ram_expert(int uid, int* out_slot = nullptr);
    void unpin_ram_slots(const int* slots, int n);
    void record_batch_hit_stats(int ip, int n_vram_hits, int n_cpu_misses);
    cudaStream_t dma_stream() const { return stream_dma_; }
    size_t max_slot_bytes() const { return max_slot_bytes_; }

    // Legacy decay hook (used if begin/finish aren't invoked)
    void on_token_end();

    // Legacy host-resolved lookup (kept for compatibility)
    void resolve_layer_experts(
        int ip,
        const int32_t* expert_ids,
        const float* router_weights,
        int k,
        GPUExpertBatch& gpu_batch,
        CPUExpertTask* cpu_tasks,
        int& n_cpu);

    const LayerExpertMeta& layer_meta(int ip) const { return layers_[ip]; }
    const PinnedLayerDeviceTable& layer_device_table(int ip) const { return layer_tables_[ip]; }
    bool is_layer_all_vram(int ip) const { return layer_vram_count_[ip] == n_experts_; }
    bool is_layer_pinned(int ip) const { return policy_ == CACHE_CORE_PINNED && layers_[ip].is_core; }
    void record_zero_sync_layer_ids(int ip, const int32_t* ids, int k) {
        const float weight_mult = layers_[ip].is_core ? 3.0f : 1.0f;
        for (int i = 0; i < k; ++i) {
            int uid = ip * n_experts_ + ids[i];
            scores_[uid] += weight_mult;
            last_used_[uid] = ++clock_;
            last_token_[uid] = token_seq_;
        }
    }
    void record_pinned_layer_hits(int ip, int k) {
        stats_.total_lookups += k;
        stats_.vram_hits += k;
        if (layers_[ip].is_core) { stats_.core_lookups += k; stats_.core_vram_hits += k; }
        else                     { stats_.outer_lookups += k; stats_.outer_vram_hits += k; }
    }
    ExpertCacheStats stats() const { return stats_; }
    void reset_stats();
    void set_swaps_frozen(bool frozen) { swaps_frozen_ = frozen; }
    bool swaps_frozen() const { return swaps_frozen_; }

private:
    int ensure_in_ram_unlocked(int uid);
    int ensure_in_ram(int uid); // returns RAM slot index (reads from SSD on RAM miss!)
    void maybe_promote_to_vram(int uid, int ram_slot);
    int find_vram_victim(float incoming_score, size_t need_bytes, int phase_filter) const;
    int find_vram_victim_pred(float incoming_score, float min_gain, size_t need_bytes, int phase_filter) const;
    int find_ram_victim_unlocked(size_t need_bytes) const;
    bool read_expert_from_ssd(int uid, uint8_t* dst);
    bool launch_single_swap_unlocked(int uid_in, int victim_slot, int rslot, cudaStream_t compute_stream, bool record_compute_dep);
    void complete_dma_unlocked(int idx, cudaStream_t compute_stream);
    void bg_preload_ram_worker();

    const GGUFFile* gguf_ = nullptr;
    int n_layers_ = 0;
    int n_experts_ = 0;
    int total_experts_ = 0;
    int core_start_ = 6;
    int core_end_ = 18;
    ExpertCachePolicy policy_ = CACHE_LOOP_LFU;

    std::vector<LayerExpertMeta> layers_;
    size_t small_slot_bytes_ = 0;
    size_t max_slot_bytes_ = 0;

    // Pinned core tight VRAM buffer (used when policy_ == CACHE_CORE_PINNED)
    uint8_t* d_core_pinned_ = nullptr;
    size_t   core_pinned_bytes_ = 0;
    std::vector<uint8_t*> pinned_gu_ptr_; // [total_experts], non-null if tightly pinned in VRAM
    std::vector<uint8_t*> pinned_dn_ptr_;
    std::vector<PinnedLayerDeviceTable> layer_tables_; // [n_layers] GPU pointer tables for ALL 24 layers!
    std::vector<int> layer_vram_count_;                // [n_layers] how many experts in this layer are currently in VRAM

    // 2-Bucket Variable-Size Dynamic VRAM slab pool (Bucket 0: compact IQ2, Bucket 1: universal IQ3)
    uint8_t* d_vram_pool_ = nullptr;
    size_t   vram_pool_bytes_ = 0;
    int      n_vram_slots_ = 0;
    int      n_vram_small_ = 0;
    int      n_vram_large_ = 0;
    std::vector<uint8_t*> vram_slot_ptr_;   // [n_vram_slots]
    std::vector<size_t>   vram_slot_cap_;   // [n_vram_slots]
    std::vector<int>      vram_slot_owner_; // [n_vram_slots] -> uid or -1
    std::vector<int>      uid_to_vram_slot_; // [total_experts] -> slot or -1 (-2 = pinned)

    // 2-Bucket Variable-Size Tier-1 Host RAM slab pool
    uint8_t* h_ram_pool_ = nullptr;
    size_t   ram_pool_bytes_ = 0;
    int      n_ram_slots_ = 0;
    int      n_ram_small_ = 0;
    int      n_ram_large_ = 0;
    std::vector<uint8_t*> ram_slot_ptr_;    // [n_ram_slots]
    std::vector<size_t>   ram_slot_cap_;    // [n_ram_slots]
    std::vector<int>      ram_slot_owner_;  // [n_ram_slots] -> uid or -1
    std::vector<uint8_t>  ram_slot_pin_;    // [n_ram_slots] active consumer pin count (0 = evictable)
    std::vector<int>      uid_to_ram_slot_; // [total_experts] -> slot or -1
    std::vector<uint64_t> ram_slot_lru_;
    uint64_t clock_ = 0;
    uint64_t token_seq_ = 1;

    // Async background RAM preload thread (cuts boot time 34s -> ~14s!)
    std::thread       preload_thread_;
    std::mutex        ram_mu_;
    std::atomic<bool> ram_preload_done_{true};
    std::atomic<bool> stop_preload_{false};

    // Loop-weighted frequency-recency scores
    std::vector<float> scores_;         // [total_experts]
    std::vector<uint64_t> last_used_;   // [total_experts]
    std::vector<uint64_t> last_token_;  // [total_experts] token_seq_ of last access

    // Deferred hot miss candidates collected during the 48 layers (swapped cleanly in phased / token-end DMA!)
    std::vector<int> pending_promos_;

    // Learned predictor state (overrides LFU ranking when active!)
    bool use_predictor_ = false;
    float pred_min_bg_gain_ = 1.5f;
    std::vector<float> pred_scores_;      // [total_experts]
    std::vector<int>   pred_queue_all_;   // descending all candidates (for LM-Head cross-tier swaps)
    std::vector<int>   pred_queue_outer_; // descending outer-only candidates (for Core-phase swaps!)
    std::vector<int>   pred_queue_core_;  // descending core-only candidates (for Coda-phase swaps!)
    size_t             pred_pos_all_ = 0;
    size_t             pred_pos_outer_ = 0;
    size_t             pred_pos_core_ = 0;

    // Async phased & end-of-token exclusive VRAM <-> RAM swaps (ZERO mid-token stalls!)
    static constexpr int MAX_INFLIGHT_DMA = 8;
    struct PendingDMA {
        bool active = false;
        int uid_in = -1;
        int uid_out = -1;
        int vram_slot = -1;
        int ram_slot = -1;
        size_t out_bytes = 0;
        uint8_t* h_in_staging = nullptr;
        uint8_t* h_out_staging = nullptr;
        cudaEvent_t event = nullptr;
    };
    PendingDMA dma_[MAX_INFLIGHT_DMA];
    cudaStream_t stream_dma_ = nullptr;
    cudaEvent_t  ev_compute_done_ = nullptr;
    uint64_t*    h_ptr_update_ = nullptr; // pinned [MAX_INFLIGHT_DMA * 4] for zero-stall 8-byte table updates
    bool         swaps_frozen_ = false;

    ExpertCacheStats stats_{};
};
