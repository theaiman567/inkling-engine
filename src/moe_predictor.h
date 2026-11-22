#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "gguf.h"

// Standalone native C++17 / AVX2 implementation of the GGUF v1 `moe-predictor`
// contract (compatible with Lanilen/Inkling-2-Mini-Expert-Predictor `.moepred` files,
// zero ggml / llama.cpp dependencies).

enum MoePredMapKind : int {
    MP_LIVE64      = 0,
    MP_CTX512      = 1,
    MP_CONTEXT     = 2,
    MP_TURN        = 3,
    MP_PREV_ANSWER = 4,
    MP_ANSWER      = 5,
    MP_ANSWER32    = 6,
    MP_ANSWER8     = 7,
    MP_PRIOR       = 8,
    MP_MAP_COUNT   = 9,
};

class MoePredictorModel {
public:
    MoePredictorModel() = default;

    bool load(const std::string& path, int expected_layers = 24, int expected_experts = 128, int expected_embd = 2304);
    bool loaded() const { return loaded_; }
    bool is_start() const { return is_start_; }
    bool reads_turn_embd() const { return use_turn_; }
    int  every() const { return every_; }
    int  n_layer() const { return n_layer_; }
    int  n_expert() const { return n_expert_; }
    int  n_embd() const { return n_embd_; }

    const std::vector<int32_t>& hidden_layers() const { return hidden_layers_; }
    const std::vector<int32_t>& horizons() const { return horizons_; }
    const std::vector<MoePredMapKind>& maps() const { return maps_; }

    // Predict per-layer expert probability distributions for each horizon.
    // `raw_maps`: size == maps_.size(); each entry points to [n_layer * n_expert] floats (ignored for MP_PRIOR).
    // `hidden`:   size == hidden_layers_.size(); each entry points to [n_embd] floats.
    // `turn_embd`: [n_embd] mean token embedding (required when reads_turn_embd() == true).
    // `out_probs`: resized to [horizons_.size()][n_layer * n_expert] (softmax per layer).
    bool predict(
        const std::vector<const float*>& raw_maps,
        const std::vector<const float*>& hidden,
        int64_t answer_pos,
        const float* turn_embd,
        std::vector<std::vector<float>>& out_probs) const;

    // Verify numerical parity against PyTorch exported `.vec` golden test vectors!
    bool verify_vectors(const std::string& vec_path, float* out_max_abs_diff = nullptr) const;

private:
    struct Head {
        int32_t horizon = 0;
        std::vector<float> w;    // [NL, NF, NB] (ggml [NB, NF, NL])
        std::vector<float> bias; // [NL, NE]
        std::vector<float> v;    // [NL * NE, R] (ggml [R, NL * NE])
    };

    bool loaded_ = false;
    bool is_start_ = false;
    bool use_last_ = false;
    bool use_turn_ = false;
    bool turn_first_ = false;

    int n_layer_ = 0;
    int n_expert_ = 0;
    int n_embd_ = 0;
    int every_ = 16;
    int D_ = 0;
    int NB_ = 0;
    int NF_ = 0;
    int F_ = 0;
    int R_ = 0;
    float norm_eps_ = 1e-5f;
    float map_eps_  = 1e-4f;
    float turn_scale_ = 1.0f;

    std::vector<int32_t> layers_;
    std::vector<int32_t> hidden_layers_;
    std::vector<int32_t> buckets_;
    std::vector<int32_t> horizons_;
    std::vector<MoePredMapKind> maps_;

    std::vector<float> prior_;     // [NL, NE]
    std::vector<float> norm_w_;    // [D]
    std::vector<float> norm_b_;    // [D]
    std::vector<float> mlp0_w_;    // [F, D + NB]
    std::vector<float> mlp0_b_;    // [F]
    std::vector<float> mlp2_w_;    // [R, F]
    std::vector<float> mlp2_b_;    // [R]
    std::vector<Head>  heads_;
};

// Runtime state tracker for the 9 expert-selection count maps across turns, prompts & answers.
class MoePredictorTracker {
public:
    void init(int n_layer, int n_expert, int core_start, int core_end, int num_loops);
    void reset_conversation();
    void begin_prompt_turn();
    void begin_answer();

    // Advance per-token decays once per token (prompt or answer)
    void step_prompt_token();
    void step_answer_token();

    // Record `k` routed expert ids for physical layer `ip` (called on every pass of looped layers!)
    void record_selections(int ip, const int32_t* ids, int k, bool is_answer);

    // Combine predictor probabilities into loop-weighted expected-selection scores [n_layer * n_expert]
    bool compute_start_scores(
        const MoePredictorModel& start_model,
        const float* mean_turn_embd,
        std::vector<float>& out_scores) const;

    bool compute_main_scores(
        const MoePredictorModel& main_model,
        const std::vector<const float*>& hidden_ptrs,
        int64_t answer_pos,
        std::vector<float>& out_scores) const;

    int64_t answer_pos() const { return answer_tokens_; }

private:
    void build_raw_map_ptrs(const MoePredictorModel& model, std::vector<const float*>& ptrs) const;
    void blend_horizons_to_scores(const MoePredictorModel& model,
                                  const std::vector<std::vector<float>>& probs,
                                  std::vector<float>& out_scores) const;

    int n_layer_ = 24;
    int n_expert_ = 128;
    int total_ = 3072;
    int core_start_ = 6;
    int core_end_ = 18;
    int num_loops_ = 3;
    int64_t answer_tokens_ = 0;

    std::vector<float> maps_[MP_MAP_COUNT]; // [total_] each (MP_PRIOR unused here)
};
