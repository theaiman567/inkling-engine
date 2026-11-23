#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <cuda_runtime.h>
#include "gguf.h"

struct EncodedImage {
    uint64_t hash = 0;           // 64-bit content hash for zero-cost multi-turn cache & KV prefix hits
    int orig_w = 0, orig_h = 0;
    int proc_w = 0, proc_h = 0;
    int n_tokens = 0;            // LLM vision tokens = (proc_w / 40) * (proc_h / 40)
    int n_embd = 0;              // 2304 (Inkling-2-Mini text hidden dim)
    double encode_ms = 0.0;
    std::vector<float> embd;     // [n_tokens * n_embd] raw FP32 embeddings (bypass tok_norm!)
};

// Standalone Zero-Permanent-VRAM / Zero-Permanent-RAM Streamed Vision Encoder for Inkling-2-Mini (inkling2v).
// Streams mmproj-F16.gguf layer-by-layer (8-9 MiB slices) through the engine's existing prefill PCIe staging
// buffer on GPU Tensor Cores, freeing all temporary activation VRAM immediately after each image!
class VisionEncoder {
public:
    VisionEncoder() = default;
    ~VisionEncoder();
    VisionEncoder(const VisionEncoder&) = delete;
    VisionEncoder& operator=(const VisionEncoder&) = delete;

    // Opens mmproj GGUF and validates tensors (<100 KB RAM, 0 MiB VRAM!).
    bool load(const std::string& mmproj_path);
    void unload();
    bool loaded() const { return loaded_; }
    const std::string& path() const { return mmproj_path_; }
    int projection_dim() const { return proj_dim_; }

    // Decode base64 data URI ("data:image/...;base64,..."), file:// URI, or local image path -> raw encoded file bytes.
    static bool resolve_image_input(const std::string& url_or_path, std::vector<uint8_t>& out_bytes);
    static bool base64_decode(const std::string& b64, std::vector<uint8_t>& out_bytes);
    static uint64_t hash_bytes(const uint8_t* data, size_t len);

    // Encode raw PNG/JPEG/BMP/GIF/WebP file bytes into [n_tokens, 2304] FP32 embeddings.
    // Reuses engine's already-allocated d_stage_buf / h_stage_buf (>= 10 MiB) when provided -> 0 MiB permanent VRAM/RAM!
    bool encode_bytes(
        const uint8_t* file_bytes,
        size_t file_len,
        EncodedImage& out,
        void* d_stage_buf,
        void* h_stage_buf,
        size_t stage_bytes,
        cudaStream_t stream);

    bool encode_file(
        const std::string& file_path,
        EncodedImage& out,
        void* d_stage_buf,
        void* h_stage_buf,
        size_t stage_bytes,
        cudaStream_t stream);

private:
    struct ViTLayerRefs {
        const GGUFTensorInfo* ln1_w = nullptr;
        const GGUFTensorInfo* ln1_b = nullptr;
        const GGUFTensorInfo* ln2_w = nullptr;
        const GGUFTensorInfo* ln2_b = nullptr;
        const GGUFTensorInfo* ls1_w = nullptr;
        const GGUFTensorInfo* ls2_w = nullptr;
        const GGUFTensorInfo* q_w   = nullptr;
        const GGUFTensorInfo* q_b   = nullptr;
        const GGUFTensorInfo* k_w   = nullptr;
        const GGUFTensorInfo* k_b   = nullptr;
        const GGUFTensorInfo* v_w   = nullptr;
        const GGUFTensorInfo* v_b   = nullptr;
        const GGUFTensorInfo* o_w   = nullptr;
        const GGUFTensorInfo* o_b   = nullptr;
        const GGUFTensorInfo* up_w  = nullptr;
        const GGUFTensorInfo* up_b  = nullptr;
        const GGUFTensorInfo* dn_w  = nullptr;
        const GGUFTensorInfo* dn_b  = nullptr;
    };

    bool decode_image_wic(const uint8_t* bytes, size_t len, int& out_w, int& out_h, std::vector<uint8_t>& out_rgb) const;

    GGUFFile gguf_;
    bool loaded_ = false;
    std::string mmproj_path_;

    // Hyperparameters read from mmproj GGUF metadata
    int image_size_ = 640;
    int patch_size_ = 20;
    int n_embd_     = 1024;
    int n_ff_       = 4096;
    int n_layers_   = 23;
    int n_heads_    = 16;
    int d_head_     = 64;
    int merge_size_ = 2;
    int proj_dim_   = 2304;
    float ln_eps_   = 1e-5f;
    float rope_theta_ = 10000.0f;
    float image_mean_[3] = {0.5f, 0.5f, 0.5f};
    float image_std_[3]  = {0.5f, 0.5f, 0.5f};

    const GGUFTensorInfo* t_cls_embd_     = nullptr;
    const GGUFTensorInfo* t_cls_pos_embd_ = nullptr;
    const GGUFTensorInfo* t_patch_embd_   = nullptr;
    const GGUFTensorInfo* t_pos_embd_     = nullptr;
    const GGUFTensorInfo* t_pre_ln_w_     = nullptr;
    const GGUFTensorInfo* t_pre_ln_b_     = nullptr;
    const GGUFTensorInfo* t_mm1_w_        = nullptr;
    const GGUFTensorInfo* t_mm1_b_        = nullptr;
    const GGUFTensorInfo* t_mm2_w_        = nullptr;
    const GGUFTensorInfo* t_mm2_b_        = nullptr;
    std::vector<ViTLayerRefs> layers_;

    // On-demand LRU cache of encoded image embeddings (0 MiB until images are actually sent; capped at 16 images / 64 MiB)
    static constexpr size_t MAX_IMG_CACHE = 16;
    static constexpr size_t MAX_IMG_CACHE_BYTES = 64ull * 1024 * 1024;
    std::deque<EncodedImage> cache_;
};
