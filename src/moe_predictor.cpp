#include "moe_predictor.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

static const char* const kMapNames[MP_MAP_COUNT] = {
    "live64", "ctx512", "context", "turn", "prev_answer", "answer", "answer32", "answer8", "prior"
};

// Fast AVX2/FMA3 dot product of two contiguous float arrays
inline float dot_f32_avx2(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 7 < n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s4);
    __m128 s2 = _mm_add_ps(s4, shuf);
    shuf = _mm_movehl_ps(shuf, s2);
    float sum = _mm_cvtss_f32(_mm_add_ss(s2, shuf));
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void scale_vec_avx2(float* x, int n, float s) {
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    }
    for (; i < n; ++i) x[i] *= s;
}

bool read_f32_tensor(const GGUFFile& g, const std::string& name, std::vector<float>& dst, size_t expected_elems) {
    const GGUFTensorInfo* ti = g.find_tensor(name);
    if (!ti || ti->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[MoePred] Missing or non-F32 tensor '%s'\n", name.c_str());
        return false;
    }
    size_t count = ti->nbytes / sizeof(float);
    if (expected_elems > 0 && count != expected_elems) {
        std::fprintf(stderr, "[MoePred] Tensor '%s' size mismatch: got %zu floats, expected %zu\n",
                     name.c_str(), count, expected_elems);
        return false;
    }
    dst.resize(count);
    return g.pread(ti->abs_offset, dst.data(), ti->nbytes);
}

} // namespace

bool MoePredictorModel::load(const std::string& path, int expected_layers, int expected_experts, int expected_embd) {
    loaded_ = false;
    GGUFFile g;
    if (!g.open(path)) {
        std::fprintf(stderr, "[MoePred] Cannot open '%s'\n", path.c_str());
        return false;
    }
    if (g.get_str("general.architecture") != "moe-predictor") {
        std::fprintf(stderr, "[MoePred] '%s' is not a moe-predictor GGUF\n", path.c_str());
        return false;
    }
    uint32_t ver = g.get_u32("moe_predictor.version", 0);
    if (ver != 1) {
        std::fprintf(stderr, "[MoePred] Unsupported predictor version %u\n", ver);
        return false;
    }

    n_layer_    = (int)g.get_u32("moe_predictor.n_layer", 0);
    n_expert_   = (int)g.get_u32("moe_predictor.n_expert", 0);
    n_embd_     = (int)g.get_u32("moe_predictor.n_embd", 0);
    every_      = std::max(1, (int)g.get_u32("moe_predictor.every", 16));
    norm_eps_   = g.get_f32("moe_predictor.norm_eps", 1e-5f);
    map_eps_    = g.get_f32("moe_predictor.map_eps",  1e-4f);
    turn_scale_ = g.get_f32("moe_predictor.turn_embd_scale", 1.0f);
    is_start_   = (g.get_str("moe_predictor.role", "main") == "start");

    if ((expected_layers > 0 && n_layer_ != expected_layers) ||
        (expected_experts > 0 && n_expert_ != expected_experts) ||
        (expected_embd > 0 && n_embd_ != expected_embd)) {
        std::fprintf(stderr, "[MoePred] Dimension mismatch in '%s': got (%dL, %dE, %dD), expected (%dL, %dE, %dD)\n",
                     path.c_str(), n_layer_, n_expert_, n_embd_, expected_layers, expected_experts, expected_embd);
        return false;
    }

    if (g.has_kv("moe_predictor.layers")) {
        layers_ = g.get_i32_arr("moe_predictor.layers");
    } else {
        layers_.resize(n_layer_);
        for (int i = 0; i < n_layer_; ++i) layers_[i] = i;
    }

    hidden_layers_ = g.has_kv("moe_predictor.hidden_layers") ? g.get_i32_arr("moe_predictor.hidden_layers") : std::vector<int32_t>{};
    buckets_       = g.get_i32_arr("moe_predictor.buckets");
    horizons_      = g.get_i32_arr("moe_predictor.horizons");
    const auto& slots = g.get_str_arr("moe_predictor.hidden_slots");
    const auto& map_names = g.get_str_arr("moe_predictor.maps");

    use_last_ = false;
    use_turn_ = false;
    turn_first_ = false;
    for (const auto& sl : slots) {
        if (sl == "last") {
            use_last_ = true;
        } else if (sl == "turn_embd_mean") {
            use_turn_ = true;
            turn_first_ = !use_last_;
        } else {
            std::fprintf(stderr, "[MoePred] Unknown hidden_slot '%s'\n", sl.c_str());
            return false;
        }
    }

    maps_.clear();
    for (const auto& mn : map_names) {
        int found = -1;
        for (int k = 0; k < MP_MAP_COUNT; ++k) {
            if (mn == kMapNames[k]) { found = k; break; }
        }
        if (found < 0) {
            std::fprintf(stderr, "[MoePred] Unknown count map '%s'\n", mn.c_str());
            return false;
        }
        maps_.push_back((MoePredMapKind)found);
    }

    if (buckets_.empty() || horizons_.empty() || maps_.empty()) {
        std::fprintf(stderr, "[MoePred] Missing buckets/horizons/maps in '%s'\n", path.c_str());
        return false;
    }

    D_  = (int)hidden_layers_.size() * n_embd_ + (use_turn_ ? n_embd_ : 0);
    NB_ = (int)buckets_.size();
    NF_ = (int)maps_.size();

    const GGUFTensorInfo* t_m0 = g.find_tensor("mlp.0.weight");
    const GGUFTensorInfo* t_m2 = g.find_tensor("mlp.2.weight");
    if (!t_m0 || !t_m2) {
        std::fprintf(stderr, "[MoePred] Missing mlp.0.weight or mlp.2.weight\n");
        return false;
    }
    F_ = (int)t_m0->ne[1];
    R_ = (int)t_m2->ne[1];

    const size_t NLNE = (size_t)n_layer_ * (size_t)n_expert_;
    if (!read_f32_tensor(g, "prior",        prior_,  NLNE) ||
        !read_f32_tensor(g, "norm.weight",  norm_w_, (size_t)D_) ||
        !read_f32_tensor(g, "norm.bias",    norm_b_, (size_t)D_) ||
        !read_f32_tensor(g, "mlp.0.weight", mlp0_w_, (size_t)(D_ + NB_) * (size_t)F_) ||
        !read_f32_tensor(g, "mlp.0.bias",   mlp0_b_, (size_t)F_) ||
        !read_f32_tensor(g, "mlp.2.weight", mlp2_w_, (size_t)F_ * (size_t)R_) ||
        !read_f32_tensor(g, "mlp.2.bias",   mlp2_b_, (size_t)R_)) {
        return false;
    }

    heads_.clear();
    heads_.resize(horizons_.size());
    for (size_t hi = 0; hi < horizons_.size(); ++hi) {
        int32_t H = horizons_[hi];
        std::string pfx = "head." + std::to_string(H) + ".";
        heads_[hi].horizon = H;
        if (!read_f32_tensor(g, pfx + "w",    heads_[hi].w,    (size_t)n_layer_ * (size_t)NF_ * (size_t)NB_) ||
            !read_f32_tensor(g, pfx + "bias", heads_[hi].bias, NLNE) ||
            !read_f32_tensor(g, pfx + "v",    heads_[hi].v,    NLNE * (size_t)R_)) {
            return false;
        }
    }

    loaded_ = true;
    std::printf("[MoePred] Loaded %s (%s): %dL x %dE | D=%d -> F=%d -> R=%d | horizons=[",
                path.c_str(), is_start_ ? "start" : "main", n_layer_, n_expert_, D_, F_, R_);
    for (size_t i = 0; i < horizons_.size(); ++i) std::printf("%s%d", i ? "," : "", horizons_[i]);
    std::printf("] | every=%d | maps=%d\n", every_, NF_);
    std::fflush(stdout);
    return true;
}

bool MoePredictorModel::predict(
    const std::vector<const float*>& raw_maps,
    const std::vector<const float*>& hidden,
    int64_t answer_pos,
    const float* turn_embd,
    std::vector<std::vector<float>>& out_probs) const
{
    if (!loaded_) return false;
    if ((int)raw_maps.size() != NF_ || hidden.size() != hidden_layers_.size()) return false;
    if (use_turn_ && !turn_embd) return false;

    // 1. Assemble hidden input vector [D_] and apply LayerNorm(x, norm_eps) * norm_w + norm_b
    std::vector<float> x(D_ + NB_, 0.0f);
    const size_t off_hid = turn_first_ ? 1 : 0;
    for (size_t i = 0; i < hidden.size(); ++i) {
        std::memcpy(x.data() + (i + off_hid) * n_embd_, hidden[i], (size_t)n_embd_ * sizeof(float));
    }
    if (use_turn_) {
        const size_t at = turn_first_ ? 0 : hidden.size();
        float* dst = x.data() + at * n_embd_;
        for (int i = 0; i < n_embd_; ++i) dst[i] = turn_embd[i] * turn_scale_;
    }

    double mean = 0.0;
    for (int i = 0; i < D_; ++i) mean += x[i];
    mean /= (double)D_;
    double var = 0.0;
    for (int i = 0; i < D_; ++i) {
        double d = (double)x[i] - mean;
        var += d * d;
    }
    var /= (double)D_;
    float inv_std = 1.0f / std::sqrt((float)var + norm_eps_);
    for (int i = 0; i < D_; ++i) {
        x[i] = ((x[i] - (float)mean) * inv_std) * norm_w_[i] + norm_b_[i];
    }

    // 2. Nearest bucket (lower index on tie) -> one-hot appended to x[D_ .. D_ + NB_ - 1]
    int best_b = 0;
    int64_t best_dist = std::llabs((int64_t)buckets_[0] - answer_pos);
    for (int b = 1; b < NB_; ++b) {
        int64_t dist = std::llabs((int64_t)buckets_[b] - answer_pos);
        if (dist < best_dist) {
            best_dist = dist;
            best_b = b;
        }
    }
    x[D_ + best_b] = 1.0f;

    // 3. 2-layer MLP: y = GELU_erf(mlp0_w * x + mlp0_b), z = mlp2_w * y + mlp2_b
    const int in0 = D_ + NB_;
    std::vector<float> y(F_);
    const float inv_sqrt2 = 0.7071067811865475244f;
    for (int f = 0; f < F_; ++f) {
        float v = mlp0_b_[f] + dot_f32_avx2(mlp0_w_.data() + (size_t)f * in0, x.data(), in0);
        y[f] = 0.5f * v * (1.0f + std::erff(v * inv_sqrt2));
    }
    std::vector<float> z(R_);
    for (int r = 0; r < R_; ++r) {
        z[r] = mlp2_b_[r] + dot_f32_avx2(mlp2_w_.data() + (size_t)r * F_, y.data(), F_);
    }

    // 4. Normalize all NF count maps per layer into log(p + map_eps), layout [NL, NE, NF]
    const int NL = n_layer_, NE = n_expert_;
    std::vector<float> logp((size_t)NL * NE * NF_);
    const float inv_ne = 1.0f / (float)NE;
    for (int f = 0; f < NF_; ++f) {
        const float* src = (maps_[f] == MP_PRIOR) ? prior_.data() : raw_maps[f];
        for (int l = 0; l < NL; ++l) {
            const float* row = src + (size_t)l * NE;
            double sum = 0.0;
            for (int e = 0; e < NE; ++e) sum += row[e];
            if (sum > 0.0) {
                double inv = 1.0 / std::max(sum, 1e-9);
                for (int e = 0; e < NE; ++e) {
                    float p = (float)(row[e] * inv);
                    logp[((size_t)l * NE + e) * NF_ + f] = std::log(p + map_eps_);
                }
            } else {
                float lv = std::log(inv_ne + map_eps_);
                for (int e = 0; e < NE; ++e) {
                    logp[((size_t)l * NE + e) * NF_ + f] = lv;
                }
            }
        }
    }

    // 5. Evaluate each horizon head + per-layer softmax
    out_probs.resize(heads_.size());
    std::vector<float> wb(NF_);
    for (size_t hi = 0; hi < heads_.size(); ++hi) {
        const Head& hd = heads_[hi];
        auto& prob = out_probs[hi];
        prob.resize((size_t)NL * NE);

        for (int l = 0; l < NL; ++l) {
            for (int f = 0; f < NF_; ++f) {
                // head.H.w layout: [NL, NF, NB] -> index ((l * NF + f) * NB + best_b)
                wb[f] = hd.w[((size_t)l * NF_ + f) * NB_ + best_b];
            }
            float* out_row = prob.data() + (size_t)l * NE;
            float max_v = -1e30f;
            for (int e = 0; e < NE; ++e) {
                const float* lp = logp.data() + ((size_t)l * NE + e) * NF_;
                float csum = 0.0f;
                for (int f = 0; f < NF_; ++f) csum += wb[f] * lp[f];
                const float* vrow = hd.v.data() + ((size_t)l * NE + e) * R_;
                float hv = dot_f32_avx2(vrow, z.data(), R_);
                float logit = hd.bias[(size_t)l * NE + e] + csum + hv;
                out_row[e] = logit;
                if (logit > max_v) max_v = logit;
            }
            double esum = 0.0;
            for (int e = 0; e < NE; ++e) {
                float ex = std::exp(out_row[e] - max_v);
                out_row[e] = ex;
                esum += ex;
            }
            float inv_e = (float)(1.0 / esum);
            for (int e = 0; e < NE; ++e) out_row[e] *= inv_e;
        }
    }
    return true;
}

bool MoePredictorModel::verify_vectors(const std::string& vec_path, float* out_max_abs_diff) const {
    if (!loaded_) return false;
    FILE* f = std::fopen(vec_path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[MoePred] Cannot open vector file '%s'\n", vec_path.c_str());
        return false;
    }
    int32_t hdr[8] = {};
    if (std::fread(hdr, sizeof(int32_t), 8, f) != 8) {
        std::fclose(f);
        return false;
    }
    int n_vec = hdr[0], nl = hdr[1], ne = hdr[2], nf = hdr[3], nd = hdr[4], nhid = hdr[5], has_turn = hdr[6], nh = hdr[7];
    if (nl != n_layer_ || ne != n_expert_ || nf != NF_ || nd != n_embd_ ||
        nhid != (int)hidden_layers_.size() || (has_turn != 0) != use_turn_ || nh != (int)horizons_.size()) {
        std::fprintf(stderr, "[MoePred] Vector header mismatch in '%s': [%d,%d,%d,%d,%d,%d,%d,%d]\n",
                     vec_path.c_str(), hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]);
        std::fclose(f);
        return false;
    }

    std::vector<float> raw((size_t)nf * nl * ne);
    std::vector<float> hid((size_t)nhid * nd);
    std::vector<float> turn(has_turn ? nd : 0);
    std::vector<float> exp_prob((size_t)nl * ne);
    std::vector<const float*> map_ptrs(nf);
    std::vector<const float*> hid_ptrs(nhid);
    std::vector<std::vector<float>> got;

    float max_diff = 0.0f;
    for (int vi = 0; vi < n_vec; ++vi) {
        int32_t t = 0;
        if (std::fread(&t, sizeof(int32_t), 1, f) != 1 ||
            std::fread(raw.data(), sizeof(float), raw.size(), f) != raw.size() ||
            (nhid > 0 && std::fread(hid.data(), sizeof(float), hid.size(), f) != hid.size()) ||
            (has_turn && std::fread(turn.data(), sizeof(float), turn.size(), f) != turn.size())) {
            std::fclose(f);
            return false;
        }
        for (int k = 0; k < nf; ++k) map_ptrs[k] = raw.data() + (size_t)k * nl * ne;
        for (int k = 0; k < nhid; ++k) hid_ptrs[k] = hid.data() + (size_t)k * nd;

        if (!predict(map_ptrs, hid_ptrs, t, has_turn ? turn.data() : nullptr, got)) {
            std::fclose(f);
            return false;
        }
        for (int hi = 0; hi < nh; ++hi) {
            if (std::fread(exp_prob.data(), sizeof(float), exp_prob.size(), f) != exp_prob.size()) {
                std::fclose(f);
                return false;
            }
            for (size_t i = 0; i < exp_prob.size(); ++i) {
                float d = std::fabs(got[hi][i] - exp_prob[i]);
                if (d > max_diff) max_diff = d;
            }
        }
    }
    std::fclose(f);
    if (out_max_abs_diff) *out_max_abs_diff = max_diff;
    std::printf("[MoePred] Verified '%s' (%d PyTorch test vectors): max_abs_diff = %.3e (%s)\n",
                vec_path.c_str(), n_vec, max_diff, max_diff < 1e-5f ? "EXACT PASS" : "MISMATCH");
    return max_diff < 1e-5f;
}

// =========================================================================
// MoePredictorTracker
// =========================================================================
void MoePredictorTracker::init(int n_layer, int n_expert, int core_start, int core_end, int num_loops) {
    n_layer_ = n_layer;
    n_expert_ = n_expert;
    total_ = n_layer * n_expert;
    core_start_ = core_start;
    core_end_ = core_end;
    num_loops_ = num_loops;
    for (int k = 0; k < MP_MAP_COUNT; ++k) {
        maps_[k].assign(total_, 0.0f);
    }
    answer_tokens_ = 0;
}

void MoePredictorTracker::reset_conversation() {
    for (int k = 0; k < MP_MAP_COUNT; ++k) {
        std::fill(maps_[k].begin(), maps_[k].end(), 0.0f);
    }
    answer_tokens_ = 0;
}

void MoePredictorTracker::begin_prompt_turn() {
    // Shift previous answer counts -> prev_answer, clear turn & answer maps
    maps_[MP_PREV_ANSWER] = maps_[MP_ANSWER];
    std::fill(maps_[MP_TURN].begin(),     maps_[MP_TURN].end(),     0.0f);
    std::fill(maps_[MP_ANSWER].begin(),   maps_[MP_ANSWER].end(),   0.0f);
    std::fill(maps_[MP_ANSWER32].begin(), maps_[MP_ANSWER32].end(), 0.0f);
    std::fill(maps_[MP_ANSWER8].begin(),  maps_[MP_ANSWER8].end(),  0.0f);
    answer_tokens_ = 0;
}

void MoePredictorTracker::begin_answer() {
    std::fill(maps_[MP_ANSWER].begin(),   maps_[MP_ANSWER].end(),   0.0f);
    std::fill(maps_[MP_ANSWER32].begin(), maps_[MP_ANSWER32].end(), 0.0f);
    std::fill(maps_[MP_ANSWER8].begin(),  maps_[MP_ANSWER8].end(),  0.0f);
    answer_tokens_ = 0;
}

void MoePredictorTracker::step_prompt_token() {
    // Half-lives: live64 = 2^(-1/64), ctx512 = 2^(-1/512)
    static const float kDecay64  = std::pow(2.0f, -1.0f / 64.0f);
    static const float kDecay512 = std::pow(2.0f, -1.0f / 512.0f);
    scale_vec_avx2(maps_[MP_LIVE64].data(), total_, kDecay64);
    scale_vec_avx2(maps_[MP_CTX512].data(), total_, kDecay512);
}

void MoePredictorTracker::step_answer_token() {
    static const float kDecay64  = std::pow(2.0f, -1.0f / 64.0f);
    static const float kDecay512 = std::pow(2.0f, -1.0f / 512.0f);
    static const float kDecay32  = std::pow(2.0f, -1.0f / 32.0f);
    static const float kDecay8   = std::pow(2.0f, -1.0f / 8.0f);
    scale_vec_avx2(maps_[MP_LIVE64].data(),   total_, kDecay64);
    scale_vec_avx2(maps_[MP_CTX512].data(),   total_, kDecay512);
    scale_vec_avx2(maps_[MP_ANSWER32].data(), total_, kDecay32);
    scale_vec_avx2(maps_[MP_ANSWER8].data(),  total_, kDecay8);
    answer_tokens_++;
}

void MoePredictorTracker::record_selections(int ip, const int32_t* ids, int k, bool is_answer) {
    int base = ip * n_expert_;
    for (int i = 0; i < k; ++i) {
        int uid = base + ids[i];
        maps_[MP_LIVE64][uid]  += 1.0f;
        maps_[MP_CTX512][uid]  += 1.0f;
        maps_[MP_CONTEXT][uid] += 1.0f;
        if (is_answer) {
            maps_[MP_ANSWER][uid]   += 1.0f;
            maps_[MP_ANSWER32][uid] += 1.0f;
            maps_[MP_ANSWER8][uid]  += 1.0f;
        } else {
            maps_[MP_TURN][uid] += 1.0f;
        }
    }
}

void MoePredictorTracker::build_raw_map_ptrs(const MoePredictorModel& model, std::vector<const float*>& ptrs) const {
    const auto& ms = model.maps();
    ptrs.resize(ms.size());
    for (size_t i = 0; i < ms.size(); ++i) {
        ptrs[i] = maps_[ms[i]].data();
    }
}

void MoePredictorTracker::blend_horizons_to_scores(
    const MoePredictorModel& model,
    const std::vector<std::vector<float>>& probs,
    std::vector<float>& out_scores) const
{
    out_scores.assign(total_, 0.0f);
    const auto& hs = model.horizons();
    // Weight short (128-token) and long (512-token) horizons cleanly, scaled by expected selections:
    // Each token selects 8 experts per layer pass; looped core layers [6..17] run `num_loops_` (3x) passes!
    std::vector<float> hw(hs.size(), 1.0f / (float)hs.size());
    if (hs.size() == 2) {
        hw[0] = 0.65f; // nearer horizon (128 tok)
        hw[1] = 0.35f; // longer horizon (512 tok)
    }
    for (int l = 0; l < n_layer_; ++l) {
        float pass_mult = (l >= core_start_ && l < core_end_) ? (float)num_loops_ : 1.0f;
        // Scale to expected selections over 128 tokens * 8 experts/pass * pass_mult -> directly comparable to answer-gain G!
        float scale = pass_mult * 8.0f * 128.0f;
        int base = l * n_expert_;
        for (int e = 0; e < n_expert_; ++e) {
            float p = 0.0f;
            for (size_t hi = 0; hi < hs.size(); ++hi) {
                p += hw[hi] * probs[hi][base + e];
            }
            out_scores[base + e] = p * scale;
        }
    }
}

bool MoePredictorTracker::compute_start_scores(
    const MoePredictorModel& start_model,
    const float* mean_turn_embd,
    std::vector<float>& out_scores) const
{
    if (!start_model.loaded()) return false;
    std::vector<const float*> mptrs;
    build_raw_map_ptrs(start_model, mptrs);
    std::vector<const float*> hptrs; // empty for start-v3
    std::vector<std::vector<float>> probs;
    if (!start_model.predict(mptrs, hptrs, 0, mean_turn_embd, probs)) return false;
    blend_horizons_to_scores(start_model, probs, out_scores);
    return true;
}

bool MoePredictorTracker::compute_main_scores(
    const MoePredictorModel& main_model,
    const std::vector<const float*>& hidden_ptrs,
    int64_t answer_pos,
    std::vector<float>& out_scores) const
{
    if (!main_model.loaded()) return false;
    std::vector<const float*> mptrs;
    build_raw_map_ptrs(main_model, mptrs);
    std::vector<std::vector<float>> probs;
    if (!main_model.predict(mptrs, hidden_ptrs, answer_pos, nullptr, probs)) return false;
    blend_horizons_to_scores(main_model, probs, out_scores);
    return true;
}
