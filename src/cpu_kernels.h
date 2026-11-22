#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "quant_tables.h"

struct CPUExpertTask {
    const void* gate_up_data;
    const void* down_data;
    uint32_t gate_up_type;
    uint32_t down_type;
    float gate_up_scale;
    float down_scale;
    float router_weight;
};

struct CPUExpertGroupTask {
    const void* gate_up_data;
    const void* down_data;
    uint32_t gate_up_type;
    uint32_t down_type;
    float gate_up_scale;
    float down_scale;
    int   tok_offset; // index into flat tok_ids / tok_weights arrays
    int   tok_count;  // number of tokens [1..B] routed to this expert
};

class CPUThreadPool {
public:
    explicit CPUThreadPool(int n_threads = 6);
    ~CPUThreadPool();
    CPUThreadPool(const CPUThreadPool&) = delete;
    CPUThreadPool& operator=(const CPUThreadPool&) = delete;

    int num_threads() const { return n_threads_; }

    // Dispatch fn(tid, n_threads) across all workers (including main thread as tid=0).
    template <typename Fn>
    void parallel_for_threads(Fn&& fn) {
        if (n_threads_ <= 1) {
            fn(0, 1);
            return;
        }
        struct Wrapper {
            Fn* f;
            static void call(void* ctx, int tid, int nth) { (*static_cast<Fn*>(ctx))(tid, nth); }
        };
        dispatch(&Wrapper::call, &fn);
    }

    // Run a batch of missed MoE experts for one token:
    // x: [2304] F32 input
    // out: [2304] F32 output (overwritten with weighted sum of all tasks)
    void run_moe_experts(
        const float* x,
        float* out,
        const CPUExpertTask* tasks,
        int n_tasks,
        int n_embd,
        int n_ff,
        float situ_beta,
        float situ_linear_beta);

    // Grouped Batched MoE on CPU: reads each unique missed expert's weights from RAM ONCE
    // and dots it against all tokens routed to that expert (6x-10x less DDR4 bandwidth during prefill & verify!)
    void run_moe_experts_grouped(
        const float* x_batch,       // [B, n_embd]
        float* out_batch,           // [B, n_embd] (overwritten with weighted sum of CPU misses)
        int B,
        const CPUExpertGroupTask* groups,
        int n_groups,
        const int32_t* tok_ids,     // [total_pairs]
        const float* tok_weights,   // [total_pairs]
        int total_pairs,
        int n_embd,
        int n_ff,
        float situ_beta,
        float situ_linear_beta);

    // Dequantize one embedding row (supports Q4_K, Q5_K, Q6_K, Q8_0, BF16, F32)
    static void dequantize_row(uint32_t type, const void* src, float* dst, int n);

    // Quantized row dot product (public for unit tests & verification)
    static float vec_dot_row(uint32_t type, const void* row_data, const float* x, int n);

private:
    using TaskFn = void(*)(void* ctx, int tid, int nth);
    void dispatch(TaskFn fn, void* ctx);
    void worker_loop(int tid);

    int n_threads_ = 1;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> done_count_{0};
    std::atomic<int> sleeping_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
    TaskFn cur_fn_ = nullptr;
    void* cur_ctx_ = nullptr;

    // Reusable scratch buffer for intermediate expert hidden states [max_tasks * 1152]
    std::vector<float> h_buf_;
};
