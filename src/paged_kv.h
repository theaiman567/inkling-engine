#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.cuh"

enum KVAttentionMode {
    KV_ATTN_AUTO         = 0, // Exact when all pages fit in VRAM, Quest sparse-page when spilling to RAM/SSD
    KV_ATTN_QUEST        = 1, // Force Quest sparse-page retrieval past quest_budget
    KV_ATTN_EXACT_STREAM = 2, // 100% exact full attention even when streaming pages from RAM/SSD
};

struct PagedKVStats {
    int max_ctx = 0;
    int cur_pos = 0;
    double swa_vram_mb = 0.0;
    double global_vram_mb = 0.0;
    double quest_summary_vram_mb = 0.0;
    double ram_kv_mb = 0.0;
    double ssd_kv_mb = 0.0;
    int vram_page_cap = 0;
    int ram_page_cap = 0;
    int total_pages_allocated = 0;
    uint64_t vram_page_hits = 0;
    uint64_t ram_page_hits = 0;
    uint64_t ssd_page_reads = 0;
    uint64_t ssd_page_writes = 0;
};

struct KVCheckpoint {
    int pos = 0;
    int glob_n_pages = 0;
    std::vector<float>   conv_state;
    std::vector<uint8_t> swa_k;
    std::vector<uint8_t> swa_v;
    std::vector<int32_t> swa_pos;
    std::vector<int>     swa_counts;
    std::vector<uint8_t> glob_pages; // [n_glob_layers, glob_n_pages, page_full_bytes]
    std::vector<int8_t>  k_mins_q8;  // [n_glob_layers, glob_n_pages, n_head_kv * head_dim]
    std::vector<int8_t>  k_maxs_q8;  // [n_glob_layers, glob_n_pages, n_head_kv * head_dim]
    std::vector<float>   k_scales;   // [n_glob_layers, glob_n_pages, n_head_kv]
    std::vector<float>   active_k_min; // [n_glob_layers, n_head_kv * head_dim] FP32 active-page extrema
    std::vector<float>   active_k_max;
    std::vector<int32_t> tokens;
};

// Lightweight ~49 MiB in-place rewind snapshot (ShortConv + SWA only, ZERO Global KV copy!)
// Enables ~4 ms rewind to any 64-token page boundary up to 512K+ context on client retries/cancels!
struct TurnRewindSnapshot {
    int pos = 0;
    uint64_t epoch = 0;
    std::vector<float>   conv_state;
    std::vector<uint8_t> swa_k;
    std::vector<uint8_t> swa_v;
    std::vector<int32_t> swa_pos;
    std::vector<int>     swa_counts;
    std::vector<int32_t> tokens;
};

class HybridPagedKV {
public:
    HybridPagedKV() = default;
    ~HybridPagedKV();
    HybridPagedKV(const HybridPagedKV&) = delete;
    HybridPagedKV& operator=(const HybridPagedKV&) = delete;

    bool init(
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
        int quest_top_pages = 128,
        bool allow_tiny_vram_for_test = false);

    void shutdown();
    void reset(); // Clear all tokens & recurrent conv states for a fresh conversation

    // R6 (SR3-03): Explicit I/O error inspection & deterministic test fault-injection hooks
    bool has_io_error() const { return io_error_; }
    const std::string& last_error() const { return last_error_; }
    void clear_io_error() { io_error_ = false; last_error_.clear(); }
    void set_test_io_fault(int fail_write_after, int fail_read_after, bool short_read_once = false) {
        test_fail_write_after_ = fail_write_after;
        test_fail_read_after_  = fail_read_after;
        test_short_read_once_  = short_read_once;
    }

    // R8 (SR3-07): Test inspection helper to copy Quest summaries for parity verification
    bool copy_quest_page_summary(int g, int p, std::vector<int8_t>& kmin, std::vector<int8_t>& kmax,
                                 std::vector<float>& ksc, std::vector<float>& act_min, std::vector<float>& act_max,
                                 cudaStream_t stream) const;

    // Fast realistic synthetic population up to target_pos tokens across VRAM -> RAM -> NVMe SSD
    // (enables instant 32K/64K/128K/256K/512K real paged-attention & NVMe-swap benchmarks!)
    bool benchmark_populate(int target_pos, cudaStream_t stream);

    // Save/restore lightweight RAM checkpoint of ShortConv + SWA + Global KV state at a 64-token page boundary
    // so multi-turn agent sessions (OpenCode) NEVER re-prefill the giant system+tools prompt!
    void save_checkpoint(KVCheckpoint& ckpt, int pos, cudaStream_t stream);
    bool restore_checkpoint(const KVCheckpoint& ckpt, cudaStream_t stream);

    // Lightweight turn-level in-place rewind (preserves live Global KV pages across VRAM/RAM/SSD!)
    uint64_t reset_epoch() const { return reset_epoch_; }
    void save_turn_snapshot(TurnRewindSnapshot& snap, int pos, cudaStream_t stream);
    bool rewind_to_turn_snapshot(const TurnRewindSnapshot& snap, cudaStream_t stream);

    // Persistent NVMe disk checkpoint save/load (InkEngine v3.0: survives full restarts & reboots in <100 ms!)
    bool save_checkpoint_disk(const KVCheckpoint& ckpt, const std::string& path, uint64_t model_sig) const;
    bool load_checkpoint_disk(KVCheckpoint& ckpt, const std::string& path, uint64_t model_sig,
                              const int32_t* expect_tokens, int expect_pos) const;
    KVQuantMode kv_mode() const { return kv_mode_; }

    // Pointer to ShortConv recurrent state for logical layer `il` and stream offset:
    // stream 0: K (1024 ch), stream 1: V (1024 ch), stream 2: attn_post (2304 ch), stream 3: ffn_post (2304 ch)
    float* conv_state_ptr(int il, int channel_offset) const;
    float* conv_spec_ptr(int il, int channel_offset) const;
    size_t conv_spec_stride() const { return (size_t)n_layers_ * conv_layer_floats_; }

    // Append current token's K & V at position `pos` for logical layer `il`, and run attention!
    // Returns false if a tier transfer / I/O error occurred (R6 / SR3-03).
    bool append_and_attend(
        int il,
        int pos,
        const float* d_Q,         // [n_head, head_dim]
        const float* d_K,         // [n_head_kv, head_dim]
        const float* d_V,         // [n_head_kv, head_dim]
        const float* d_rel_table, // [n_head, rel_extent]
        const float* d_sinks,     // [n_head]
        float* d_out,             // [n_head, head_dim]
        int rel_extent,
        cudaStream_t stream);

    // Batched append & causal attention for B tokens [pos0 .. pos0 + B - 1] (Stage 4 Prefill & Stage 5 Verify!)
    // Returns false if a tier transfer / I/O error occurred (R6 / SR3-03).
    bool append_and_attend_batch(
        int il,
        int pos0,
        int B,
        const float* d_Q,         // [B, n_head, head_dim]
        const float* d_K,         // [B, n_head_kv, head_dim]
        const float* d_V,         // [B, n_head_kv, head_dim]
        const float* d_rel_table, // [B, n_head, rel_extent]
        const float* d_sinks,     // [B, n_head]
        float* d_out,             // [B, n_head, head_dim]
        int rel_extent,
        cudaStream_t stream);

    // Exact 0.02 ms GPU-side rollback helpers for Stage 5 / v3.0 lossless speculative verification (B <= 8 -> K <= 6!)
    static constexpr int SPEC_MAX_B = 8;
    static constexpr int SWA_BATCH_PAD = 128; // >= InklingEngine::BATCH_MAX so batched SWA never overwrites live keys!
    static constexpr int SSD_ASYNC_QD = 16;
    void begin_spec_verify(int pos0, int B, cudaStream_t stream);
    void commit_spec_verify(int accepted, cudaStream_t stream);
    void begin_draft_snapshot(int pos0, int max_k, cudaStream_t stream);
    void rollback_draft_snapshot(int pos0, int drafted, cudaStream_t stream);

    PagedKVStats stats() const;
    void reset_counters();

private:
    enum PageTier : uint8_t { TIER_NONE = 0, TIER_VRAM = 1, TIER_RAM = 2, TIER_SSD = 3 };

    struct PageMeta {
        PageTier tier = TIER_NONE;
        int32_t  slot = -1;       // index in VRAM pool, RAM pool, or SSD slot
        int32_t  ssd_slot = -1;   // persistent slot in SSD swap file once spilled
        int32_t  pos0 = 0;
        int32_t  len = 0;
        bool     dirty = false;   // modified in VRAM since last spill
        uint64_t last_use = 0;
    };

    // Ensure logical page (g, p) is resident in VRAM (evicting LRU unpinned VRAM page to RAM/SSD if needed).
    // Returns -1 on unrecoverable eviction/I/O failure (sets io_error_).
    int ensure_page_in_vram(int g, int p, int protect_recent_start, cudaStream_t stream);
    bool fetch_ssd_pages_batch(int g, const std::vector<int>& ssd_pids, int protect_recent_start,
                               std::vector<int>& out_pinned_slots, cudaStream_t stream);
    bool attend_global_pages(int g, int B, int pos0, const std::vector<int>& page_ids, int recent_start,
                             const float* d_Q, const float* d_rel_table, const float* d_sinks,
                             float* d_out, int rel_extent, cudaStream_t stream);
    bool ensure_chunk_accum(int B);
    void unpin_vram_slots(const std::vector<int>& slots);
    void rebuild_free_lists();
    void release_vram_slot(int s);
    void release_ram_slot(int s);
    int alloc_vram_page_slot(int protect_g, int protect_recent_start, cudaStream_t stream);
    int alloc_ram_page_slot(int& evicted_g, int& evicted_p);
    bool spill_ram_to_ssd(int g, int p, int rslot);
    bool read_ssd_to_ram(int g, int p, int rslot);
    bool ssd_pwrite_sync(uint64_t offset, const void* src, size_t nbytes);
    bool ssd_pread_sync(uint64_t offset, void* dst, size_t nbytes);
    bool ensure_ssd_file();
    void set_io_error(const char* msg);

    int n_layers_ = 0;
    int n_head_ = 24;
    int n_head_kv_ = 8;
    int head_dim_ = 128;
    int n_embd_ = 2304;
    int swa_window_ = 1024;    // Logical sliding window length (1024)
    int swa_ring_cap_ = 1152;  // Physical ring capacity = swa_window_ + SWA_BATCH_PAD (prevents batched overwrite!)
    int max_ctx_ = 32768;
    int quest_top_pages_ = 128;
    KVQuantMode kv_mode_ = KV_QUANT_Q4_0;
    KVAttentionMode attn_mode_ = KV_ATTN_AUTO;

    size_t head_bytes_ = 72;
    size_t tok_k_bytes_ = 576;
    size_t page_half_bytes_ = 36864; // K or V for 64 tokens
    size_t page_full_bytes_ = 73728; // K + V for 64 tokens

    std::vector<uint8_t> is_swa_;
    std::vector<int> layer_to_swa_idx_;
    std::vector<int> layer_to_glob_idx_;
    int n_swa_layers_ = 0;
    int n_glob_layers_ = 0;

    // ShortConv state in VRAM: [n_layers, n_sconv_channels * 3]
    float* d_conv_state_ = nullptr;
    size_t conv_layer_floats_ = 0;

    // SWA Ring Buffers in VRAM (physical capacity = swa_ring_cap_ = swa_window_ + 128)
    uint8_t* d_swa_k_ = nullptr;
    uint8_t* d_swa_v_ = nullptr;
    int32_t* d_swa_pos_ = nullptr; // [n_swa_layers, swa_ring_cap_]
    std::vector<int> swa_counts_;  // valid physical ring entries (up to swa_ring_cap_) per SWA layer

    // Global Paged KV: Tier 0 (VRAM), Tier 1 (RAM), Tier 2 (SSD)
    uint8_t* d_vram_pages_ = nullptr;
    int vram_page_cap_ = 0;
    std::vector<std::pair<int, int>> vram_owner_; // [slot] -> (g, page_id)
    std::vector<uint8_t> vram_pin_;               // [slot] > 0 means pinned for in-flight attention/append (R4!)
    std::vector<int> vram_free_list_;             // R18: O(1) free VRAM frame stack
    int vram_evict_cursor_ = 0;                   // R18: bounded clock-LRU eviction hand

    // R4 lazy online-softmax chunk accumulators (0 bytes VRAM when single-chunk suffices!)
    float* d_chunk_out_ = nullptr;
    float* d_chunk_ml_  = nullptr;
    int    chunk_acc_cap_b_ = 0;

    uint8_t* h_ram_pages_ = nullptr;
    int ram_page_cap_ = 0;
    std::vector<std::pair<int, int>> ram_owner_;  // [slot] -> (g, page_id)
    std::vector<uint64_t> ram_lru_;
    std::vector<int> ram_free_list_;              // R18: O(1) free RAM frame stack
    int ram_evict_cursor_ = 0;                    // R18: bounded clock-LRU eviction hand

    std::string ssd_path_;
    void* h_ssd_file_ = nullptr;
    int next_ssd_slot_ = 0;

    // Per global layer metadata & INT8-quantized Quest static K_min/K_max summaries in VRAM (4x smaller!)
    int max_pages_per_layer_ = 0;
    std::vector<std::vector<PageMeta>> pages_; // [n_glob_layers][max_pages_per_layer]
    std::vector<int> glob_num_pages_;          // [n_glob_layers]
    int8_t* d_k_mins_q8_ = nullptr;   // [n_glob_layers, max_pages_per_layer, n_head_kv * head_dim]
    int8_t* d_k_maxs_q8_ = nullptr;
    float*  d_k_scales_  = nullptr;   // [n_glob_layers, max_pages_per_layer, n_head_kv]
    float*  d_active_k_min_ = nullptr; // [n_glob_layers, n_head_kv * head_dim] (64 KB FP32 accumulator)
    float*  d_active_k_max_ = nullptr;

    // Scratch buffers for Split-K attention (single + batched up to 128 B*splits) & Quest page selection
    float*    d_split_out_ = nullptr; // [128 * n_head * head_dim] = 1.5 MiB
    float*    d_split_ml_  = nullptr; // [128 * n_head * 2]
    uint64_t* d_page_k_ptrs_ = nullptr;
    uint64_t* d_page_v_ptrs_ = nullptr;
    int32_t*  d_page_pos0_ = nullptr;
    int32_t*  d_page_lens_ = nullptr;
    float*    d_quest_scores_ = nullptr;
    int32_t*  d_quest_topk_ids_ = nullptr; // [max_pages_per_layer] GPU Radix-Select Top-K output!

    // Speculative verification rollback buffers (exact ShortConv + SWA + R8 Quest extrema in VRAM)
    float*    d_conv_spec_ = nullptr;     // [SPEC_MAX_B, n_layers, conv_layer_floats]
    uint8_t*  d_swa_bak_k_ = nullptr;     // [n_swa_layers, SPEC_MAX_B, tok_k_bytes]
    uint8_t*  d_swa_bak_v_ = nullptr;
    int32_t*  d_swa_bak_pos_ = nullptr;   // [n_swa_layers, SPEC_MAX_B]
    float*    d_quest_spec_min_ = nullptr; // [SPEC_MAX_B, n_glob_layers, kd] FP32 post-token extrema (R8!)
    float*    d_quest_spec_max_ = nullptr; // [SPEC_MAX_B, n_glob_layers, kd]
    float*    d_quest_draft_min_ = nullptr;// [n_glob_layers, kd] pre-draft FP32 extrema (R8!)
    float*    d_quest_draft_max_ = nullptr;
    int8_t*   d_quest_draft_q8_min_ = nullptr; // [n_glob_layers, kd] pre-draft page Q8 summary (R8!)
    int8_t*   d_quest_draft_q8_max_ = nullptr;
    float*    d_quest_draft_sc_ = nullptr;     // [n_glob_layers, n_head_kv]
    int       draft_saved_page_ = -1;
    bool      spec_recording_ = false;
    int       spec_pos0_ = 0;
    int       spec_B_ = 0;

    // Pinned host staging arrays for zero-stall page pointer upload, Quest Top-K, & QD-16 Async NVMe reads
    uint64_t* h_page_k_ptrs_ = nullptr;
    uint64_t* h_page_v_ptrs_ = nullptr;
    int32_t*  h_page_pos0_ = nullptr;
    int32_t*  h_page_lens_ = nullptr;
    float*    h_quest_scores_ = nullptr;
    int32_t*  h_quest_topk_ids_ = nullptr;
    uint8_t*  h_page_bounce_ = nullptr; // pinned 1-page bounce buffer for fast VRAM<->RAM DMA
    uint8_t*  h_ssd_async_buf_ = nullptr; // pinned SSD_ASYNC_QD-page bounce buffer for QD-16 NVMe reads!
    void*     ssd_ov_events_[SSD_ASYNC_QD] = {};
    void*     ssd_sync_event_ = nullptr;

    int cur_pos_ = 0;
    uint64_t reset_epoch_ = 1;
    uint64_t clock_ = 0;
    uint64_t stat_vram_hits_ = 0;
    uint64_t stat_ram_hits_ = 0;
    uint64_t stat_ssd_reads_ = 0;
    uint64_t stat_ssd_writes_ = 0;

    bool        io_error_ = false;
    std::string last_error_;
    int         test_fail_write_after_ = -1;
    int         test_fail_read_after_  = -1;
    bool        test_short_read_once_  = false;
};
