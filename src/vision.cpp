#include "vision.h"
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincodec.h>
#include <objbase.h>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

namespace {

double now_ms_v() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline size_t align_up_256(size_t v) { return (v + 255ull) & ~255ull; }

// Bilinear resize of RGB uint8 image using align_corners=false pixel-center mapping
void resize_rgb_bilinear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    float sx = (float)sw / (float)dw;
    float sy = (float)sh / (float)dh;
    for (int y = 0; y < dh; ++y) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        float wy1 = fy - (float)y0;
        y0 = std::max(0, std::min(sh - 1, y0));
        y1 = std::max(0, std::min(sh - 1, y1));
        wy1 = std::max(0.0f, std::min(1.0f, wy1));
        float wy0 = 1.0f - wy1;

        const uint8_t* row0 = src + (size_t)y0 * sw * 3;
        const uint8_t* row1 = src + (size_t)y1 * sw * 3;
        uint8_t* out_row = dst + (size_t)y * dw * 3;

        for (int x = 0; x < dw; ++x) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            int x1 = x0 + 1;
            float wx1 = fx - (float)x0;
            x0 = std::max(0, std::min(sw - 1, x0));
            x1 = std::max(0, std::min(sw - 1, x1));
            wx1 = std::max(0.0f, std::min(1.0f, wx1));
            float wx0 = 1.0f - wx1;

            float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
            for (int c = 0; c < 3; ++c) {
                float v = w00 * row0[x0 * 3 + c] + w01 * row0[x1 * 3 + c]
                        + w10 * row1[x0 * 3 + c] + w11 * row1[x1 * 3 + c];
                int iv = (int)std::lround(v);
                out_row[x * 3 + c] = (uint8_t)std::max(0, std::min(255, iv));
            }
        }
    }
}

// Bilinear resize of learned 2D position embeddings [grid_src, grid_src, D] -> [grid_h, grid_w, D]
// Matches ggml_interpolate(GGML_SCALE_MODE_BILINEAR) with align_corners=false pixel-center mapping.
void resize_pos_embd_bilinear(const float* src, int n_side, int D, int out_w, int out_h, float* dst) {
    if (out_w == n_side && out_h == n_side) {
        std::memcpy(dst, src, (size_t)n_side * n_side * D * sizeof(float));
        return;
    }
    float sx = (float)n_side / (float)out_w;
    float sy = (float)n_side / (float)out_h;
    for (int y = 0; y < out_h; ++y) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        float wy1 = fy - (float)y0;
        y0 = std::max(0, std::min(n_side - 1, y0));
        y1 = std::max(0, std::min(n_side - 1, y1));
        wy1 = std::max(0.0f, std::min(1.0f, wy1));
        float wy0 = 1.0f - wy1;

        for (int x = 0; x < out_w; ++x) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            int x1 = x0 + 1;
            float wx1 = fx - (float)x0;
            x0 = std::max(0, std::min(n_side - 1, x0));
            x1 = std::max(0, std::min(n_side - 1, x1));
            wx1 = std::max(0.0f, std::min(1.0f, wx1));
            float wx0 = 1.0f - wx1;

            float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
            const float* p00 = src + ((size_t)y0 * n_side + x0) * D;
            const float* p01 = src + ((size_t)y0 * n_side + x1) * D;
            const float* p10 = src + ((size_t)y1 * n_side + x0) * D;
            const float* p11 = src + ((size_t)y1 * n_side + x1) * D;
            float* out = dst + ((size_t)y * out_w + x) * D;
            for (int d = 0; d < D; ++d) {
                out[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
            }
        }
    }
}

// Helper to convert F16/BF16/F32 GGUF tensor on host into F32 vector (for tiny 1D/pos tensors)
bool read_tensor_as_f32(const GGUFFile& gguf, const GGUFTensorInfo* ti, std::vector<float>& out) {
    if (!ti) return false;
    size_t n = 1;
    for (uint32_t i = 0; i < ti->n_dims; ++i) n *= (size_t)ti->ne[i];
    out.resize(n);
    if (ti->type == GGML_TYPE_F32) {
        return gguf.pread(ti->abs_offset, out.data(), n * sizeof(float));
    }
    std::vector<uint16_t> raw(n);
    if (!gguf.pread(ti->abs_offset, raw.data(), n * sizeof(uint16_t))) return false;
    if (ti->type == GGML_TYPE_F16) {
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32_host(raw[i]);
        return true;
    }
    if (ti->type == GGML_TYPE_BF16) {
        for (size_t i = 0; i < n; ++i) {
            uint32_t u = (uint32_t)raw[i] << 16;
            std::memcpy(&out[i], &u, 4);
        }
        return true;
    }
    return false;
}

} // namespace

VisionEncoder::~VisionEncoder() { unload(); }

void VisionEncoder::unload() {
    gguf_.close();
    layers_.clear();
    cache_.clear();
    loaded_ = false;
}

bool VisionEncoder::load(const std::string& mmproj_path) {
    unload();
    if (mmproj_path.empty()) return false;
    if (!gguf_.open(mmproj_path)) return false;

    std::string proj_type = gguf_.get_str("clip.vision.projector_type");
    if (proj_type != "inkling2v") {
        std::fprintf(stderr, "[Vision] Unsupported mmproj projector_type '%s' (expected 'inkling2v')\n", proj_type.c_str());
        gguf_.close();
        return false;
    }

    mmproj_path_ = mmproj_path;
    image_size_  = (int)gguf_.get_u32("clip.vision.image_size", 640);
    patch_size_  = (int)gguf_.get_u32("clip.vision.patch_size", 20);
    n_embd_      = (int)gguf_.get_u32("clip.vision.embedding_length", 1024);
    n_ff_        = (int)gguf_.get_u32("clip.vision.feed_forward_length", 4096);
    n_layers_    = (int)gguf_.get_u32("clip.vision.block_count", 23);
    n_heads_     = (int)gguf_.get_u32("clip.vision.attention.head_count", 16);
    d_head_      = (n_heads_ > 0) ? (n_embd_ / n_heads_) : 64;
    merge_size_  = (int)gguf_.get_u32("clip.vision.spatial_merge_size", 2);
    proj_dim_    = (int)gguf_.get_u32("clip.vision.projection_dim", 2304);
    ln_eps_      = gguf_.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
    rope_theta_  = 10000.0f;

    const auto& mean_arr = gguf_.get_f32_arr("clip.vision.image_mean");
    if (mean_arr.size() >= 3) {
        for (int c = 0; c < 3; ++c) image_mean_[c] = mean_arr[c];
    }
    const auto& std_arr = gguf_.get_f32_arr("clip.vision.image_std");
    if (std_arr.size() >= 3) {
        for (int c = 0; c < 3; ++c) image_std_[c] = std_arr[c];
    }

    auto req = [&](const std::string& name) -> const GGUFTensorInfo* {
        const auto* t = gguf_.find_tensor(name);
        if (!t) std::fprintf(stderr, "[Vision] Missing required tensor '%s' in %s\n", name.c_str(), mmproj_path.c_str());
        return t;
    };

    t_cls_embd_     = req("v.class_embd");
    t_cls_pos_embd_ = req("v.class_position_embd.weight");
    t_patch_embd_   = req("v.patch_embd.weight");
    t_pos_embd_     = req("v.position_embd.weight");
    t_pre_ln_w_     = req("v.pre_ln.weight");
    t_pre_ln_b_     = req("v.pre_ln.bias");
    t_mm1_w_        = req("mm.1.weight");
    t_mm1_b_        = gguf_.find_tensor("mm.1.bias");
    t_mm2_w_        = req("mm.2.weight");
    t_mm2_b_        = gguf_.find_tensor("mm.2.bias");

    if (!t_cls_embd_ || !t_cls_pos_embd_ || !t_patch_embd_ || !t_pos_embd_ ||
        !t_pre_ln_w_ || !t_pre_ln_b_ || !t_mm1_w_ || !t_mm2_w_) {
        gguf_.close();
        return false;
    }

    layers_.resize(n_layers_);
    for (int il = 0; il < n_layers_; ++il) {
        std::string p = "v.blk." + std::to_string(il) + ".";
        auto& L = layers_[il];
        L.ln1_w = req(p + "ln1.weight");
        L.ln1_b = req(p + "ln1.bias");
        L.ln2_w = req(p + "ln2.weight");
        L.ln2_b = req(p + "ln2.bias");
        L.ls1_w = gguf_.find_tensor(p + "ls1.weight");
        L.ls2_w = gguf_.find_tensor(p + "ls2.weight");
        L.q_w   = req(p + "attn_q.weight");
        L.q_b   = gguf_.find_tensor(p + "attn_q.bias");
        L.k_w   = req(p + "attn_k.weight");
        L.k_b   = gguf_.find_tensor(p + "attn_k.bias");
        L.v_w   = req(p + "attn_v.weight");
        L.v_b   = gguf_.find_tensor(p + "attn_v.bias");
        L.o_w   = req(p + "attn_out.weight");
        L.o_b   = gguf_.find_tensor(p + "attn_out.bias");
        L.up_w  = req(p + "ffn_up.weight");
        L.up_b  = gguf_.find_tensor(p + "ffn_up.bias");
        L.dn_w  = req(p + "ffn_down.weight");
        L.dn_b  = gguf_.find_tensor(p + "ffn_down.bias");
        if (!L.ln1_w || !L.ln1_b || !L.ln2_w || !L.ln2_b ||
            !L.q_w || !L.k_w || !L.v_w || !L.o_w || !L.up_w || !L.dn_w) {
            gguf_.close();
            return false;
        }
    }

    loaded_ = true;
    std::printf("[Vision] Loaded %s (inkling2v: %dL ViT d=%d, %dx%d patch -> %dx%d merge -> %d | 0 MiB permanent VRAM/RAM, on-demand GPU streamed!)\n",
                mmproj_path.c_str(), n_layers_, n_embd_, patch_size_, patch_size_, merge_size_, merge_size_, proj_dim_);
    std::fflush(stdout);
    return true;
}

uint64_t VisionEncoder::hash_bytes(const uint8_t* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}

bool VisionEncoder::base64_decode(const std::string& b64, std::vector<uint8_t>& out) {
    static int8_t T[256];
    static bool inited = false;
    if (!inited) {
        std::memset(T, -1, sizeof(T));
        for (int i = 0; i < 26; ++i) T['A' + i] = (int8_t)i;
        for (int i = 0; i < 26; ++i) T['a' + i] = (int8_t)(26 + i);
        for (int i = 0; i < 10; ++i) T['0' + i] = (int8_t)(52 + i);
        T['+'] = 62; T['/'] = 63;
        T['-'] = 62; T['_'] = 63; // URL-safe base64 support too
        inited = true;
    }
    out.clear();
    out.reserve((b64.size() * 3) / 4);
    uint32_t val = 0;
    int valb = -8;
    for (unsigned char c : b64) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
        int8_t d = T[c];
        if (d < 0) continue;
        val = (val << 6) | (uint32_t)d;
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return !out.empty();
}

bool VisionEncoder::resolve_image_input(const std::string& url_or_path, std::vector<uint8_t>& out_bytes) {
    out_bytes.clear();
    if (url_or_path.empty()) return false;

    // 1. data:image/...;base64,<payload>
    if (url_or_path.rfind("data:", 0) == 0) {
        size_t comma = url_or_path.find(',');
        if (comma == std::string::npos) return false;
        return base64_decode(url_or_path.substr(comma + 1), out_bytes);
    }

    // 2. file:///C:/... or raw local file path
    std::string path = url_or_path;
    if (path.rfind("file:///", 0) == 0) {
        path = path.substr(8);
        // Basic %20 URL unescape
        std::string unesc;
        unesc.reserve(path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            if (path[i] == '%' && i + 2 < path.size()) {
                char hex[3] = {path[i + 1], path[i + 2], 0};
                unesc.push_back((char)std::strtol(hex, nullptr, 16));
                i += 2;
            } else {
                unesc.push_back(path[i] == '/' ? '\\' : path[i]);
            }
        }
        path = std::move(unesc);
    }

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (ifs.good()) {
        auto sz = ifs.tellg();
        if (sz > 0 && sz < 128 * 1024 * 1024) {
            out_bytes.resize((size_t)sz);
            ifs.seekg(0, std::ios::beg);
            if (ifs.read((char*)out_bytes.data(), sz)) return true;
        }
    }

    // 3. Fallback: raw base64 without data: prefix (some SDKs send bare base64)
    if (url_or_path.size() > 64 && url_or_path.find("://") == std::string::npos) {
        return base64_decode(url_or_path, out_bytes);
    }
    return false;
}

bool VisionEncoder::decode_image_wic(const uint8_t* bytes, size_t len, int& out_w, int& out_h, std::vector<uint8_t>& out_rgb) const {
    if (!bytes || len == 0) return false;
    HRESULT hr_init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool need_uninit = SUCCEEDED(hr_init);

    IWICImagingFactory* factory = nullptr;
    IWICStream* wstream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* conv = nullptr;
    bool ok = false;

    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) hr = factory->CreateStream(&wstream);
    if (SUCCEEDED(hr)) hr = wstream->InitializeFromMemory(const_cast<BYTE*>(bytes), (DWORD)len);
    if (SUCCEEDED(hr)) hr = factory->CreateDecoderFromStream(wstream, nullptr, WICDecodeMetadataCacheOnDemand, &decoder);
    if (SUCCEEDED(hr)) hr = decoder->GetFrame(0, &frame);
    if (SUCCEEDED(hr)) {
        UINT w = 0, h = 0;
        hr = frame->GetSize(&w, &h);
        if (SUCCEEDED(hr) && w > 0 && h > 0 && w <= 16384 && h <= 16384) {
            hr = factory->CreateFormatConverter(&conv);
            if (SUCCEEDED(hr)) {
                hr = conv->Initialize(frame, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
            }
            if (SUCCEEDED(hr)) {
                out_w = (int)w;
                out_h = (int)h;
                out_rgb.resize((size_t)w * h * 3);
                hr = conv->CopyPixels(nullptr, w * 3, (UINT)out_rgb.size(), out_rgb.data());
                ok = SUCCEEDED(hr);
            }
        }
    }

    if (conv) conv->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (wstream) wstream->Release();
    if (factory) factory->Release();
    if (need_uninit) CoUninitialize();
    return ok;
}

bool VisionEncoder::encode_file(const std::string& file_path, EncodedImage& out,
                                void* d_stage_buf, void* h_stage_buf, size_t stage_bytes, cudaStream_t stream) {
    std::vector<uint8_t> bytes;
    if (!resolve_image_input(file_path, bytes)) {
        std::fprintf(stderr, "[Vision] Failed to read image file: %s\n", file_path.c_str());
        return false;
    }
    return encode_bytes(bytes.data(), bytes.size(), out, d_stage_buf, h_stage_buf, stage_bytes, stream);
}

bool VisionEncoder::encode_bytes(const uint8_t* file_bytes, size_t file_len, EncodedImage& out,
                                 void* d_stage_buf, void* h_stage_buf, size_t stage_bytes, cudaStream_t stream) {
    if (!loaded_ || !file_bytes || file_len == 0) return false;
    double t0 = now_ms_v();

    uint64_t h64 = hash_bytes(file_bytes, file_len);
    for (const auto& c : cache_) {
        if (c.hash == h64) {
            out = c;
            out.encode_ms = 0.0;
            return true;
        }
    }

    int orig_w = 0, orig_h = 0;
    std::vector<uint8_t> rgb;
    if (!decode_image_wic(file_bytes, file_len, orig_w, orig_h, rgb)) {
        std::fprintf(stderr, "[Vision] WIC failed to decode image (%zu bytes)\n", file_len);
        return false;
    }

    // 1. Dynamic VRAM-safe max pixel budget (never OOMs on 6-8 GB GPUs; full HD/4K on larger GPUs!)
    size_t vram_free = 0, vram_tot = 0;
    INK_CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_tot));
    bool own_stage = (d_stage_buf == nullptr || h_stage_buf == nullptr || stage_bytes < 11ull * 1024 * 1024);
    size_t stage_need = own_stage ? (12ull * 1024 * 1024) : 0;
    size_t safety_vram = 48ull * 1024 * 1024 + stage_need;
    size_t avail_act = (vram_free > safety_vram) ? (vram_free - safety_vram) : (32ull * 1024 * 1024);
    // ~22 KB transient activation VRAM per ViT patch (5*N*1024*4B = 20 KB + small FFN chunk)
    int max_patches = (int)std::max<size_t>(1024, std::min<size_t>(16384, avail_act / 22528ull));
    int max_img_tokens = std::max(256, std::min(4096, max_patches / (merge_size_ * merge_size_)));
    // Cap at 640 tokens on 8 GB GPUs (crisp 1280x720 HD / 1000x1000) so FFN stays single-pass & prefill stays fast!
    if (vram_tot <= 9000ull * 1024 * 1024) max_img_tokens = std::min(max_img_tokens, 640);

    const int align = patch_size_ * merge_size_; // 40
    const int64_t max_pixels = (int64_t)max_img_tokens * align * align;

    auto align_up_i = [&](int v) { return ((v + align - 1) / align) * align; };
    int cur_w = orig_w, cur_h = orig_h;
    std::vector<uint8_t> resized_rgb;
    const uint8_t* use_rgb = rgb.data();

    if ((int64_t)align_up_i(cur_w) * align_up_i(cur_h) > max_pixels) {
        double scale = std::sqrt((double)max_pixels / ((double)cur_w * cur_h));
        int nw = std::max(align, (int)std::floor(cur_w * scale / align) * align);
        int nh = std::max(align, (int)std::floor(cur_h * scale / align) * align);
        while ((int64_t)nw * nh > max_pixels) {
            if (nw >= nh && nw > align) nw -= align;
            else if (nh > align) nh -= align;
            else break;
        }
        resized_rgb.resize((size_t)nw * nh * 3);
        resize_rgb_bilinear(rgb.data(), orig_w, orig_h, resized_rgb.data(), nw, nh);
        cur_w = nw; cur_h = nh;
        use_rgb = resized_rgb.data();
    }

    // 2. Exact inkling2v normalization (mean*255, std*255) on valid pixels + exact -1.0f bottom-right padding
    const int proc_w = align_up_i(cur_w);
    const int proc_h = align_up_i(cur_h);
    const int grid_w = proc_w / patch_size_;
    const int grid_h = proc_h / patch_size_;
    const int n_patches = grid_w * grid_h;
    const int n_pos = n_patches + 1; // +1 CLS token at index 0
    const int n_out_tokens = (grid_w / merge_size_) * (grid_h / merge_size_);

    const float m0 = image_mean_[0] * 255.0f, inv_s0 = 1.0f / (image_std_[0] * 255.0f);
    const float m1 = image_mean_[1] * 255.0f, inv_s1 = 1.0f / (image_std_[1] * 255.0f);
    const float m2 = image_mean_[2] * 255.0f, inv_s2 = 1.0f / (image_std_[2] * 255.0f);
    std::vector<float> norm_img((size_t)proc_w * proc_h * 3, -1.0f);
    for (int y = 0; y < cur_h; ++y) {
        const uint8_t* sr = use_rgb + (size_t)y * cur_w * 3;
        float* dr = norm_img.data() + (size_t)y * proc_w * 3;
        for (int x = 0; x < cur_w; ++x) {
            dr[x * 3 + 0] = ((float)sr[x * 3 + 0] - m0) * inv_s0;
            dr[x * 3 + 1] = ((float)sr[x * 3 + 1] - m1) * inv_s1;
            dr[x * 3 + 2] = ((float)sr[x * 3 + 2] - m2) * inv_s2;
        }
    }

    // 3. Extract patches [n_patches, patch_dim = 3 * 20 * 20 = 1200] in [c][dy][dx] order matching ggml conv_2d kernel!
    const int patch_dim = 3 * patch_size_ * patch_size_;
    std::vector<float> h_patches((size_t)n_patches * patch_dim);
    for (int py = 0; py < grid_h; ++py) {
        for (int px = 0; px < grid_w; ++px) {
            float* dst_p = h_patches.data() + ((size_t)py * grid_w + px) * patch_dim;
            for (int c = 0; c < 3; ++c) {
                for (int dy = 0; dy < patch_size_; ++dy) {
                    const float* row = norm_img.data() + ((size_t)(py * patch_size_ + dy) * proc_w + px * patch_size_) * 3;
                    float* out_row = dst_p + (c * patch_size_ + dy) * patch_size_;
                    for (int dx = 0; dx < patch_size_; ++dx) {
                        out_row[dx] = row[dx * 3 + c];
                    }
                }
            }
        }
    }

    // 4. Prepare temporary GPU buffers (ALL freed before returning -> 0 MiB permanent VRAM!)
    uint8_t* d_stage = (uint8_t*)d_stage_buf;
    uint8_t* h_stage = (uint8_t*)h_stage_buf;
    size_t stage_cap = stage_bytes;
    if (own_stage) {
        stage_cap = 12ull * 1024 * 1024;
        INK_CUDA_CHECK(cudaMalloc(&d_stage, stage_cap));
        INK_CUDA_CHECK(cudaHostAlloc(&h_stage, stage_cap, cudaHostAllocWriteCombined));
    }

    float* d_X = nullptr;       // [n_pos, 1024]
    float* d_H = nullptr;       // [n_pos, 1024]
    float* d_work = nullptr;    // Reused as [d_Q, d_K, d_V] during Attention AND d_ffn_mid [ffn_chunk, 4096] during FFN!
    int32_t* d_pos_w = nullptr;
    int32_t* d_pos_h = nullptr;

    // Prefer single-pass FFN (reads mmproj only once per layer!) whenever [n_pos, 4096] fits in transient budget
    const int max_single_ffn = (n_pos <= 2600) ? n_pos : std::max(1088, (int)(((size_t)3 * n_pos * n_embd_) / (size_t)n_ff_));
    const int ffn_chunk = std::min(n_pos, max_single_ffn);
    const size_t work_floats = std::max((size_t)3 * n_pos * n_embd_, (size_t)ffn_chunk * n_ff_);

    INK_CUDA_CHECK(cudaMalloc(&d_X, (size_t)n_pos * n_embd_ * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_H, (size_t)n_pos * n_embd_ * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_work, work_floats * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_pos_w, (size_t)n_pos * sizeof(int32_t)));
    INK_CUDA_CHECK(cudaMalloc(&d_pos_h, (size_t)n_pos * sizeof(int32_t)));

    float* d_Q = d_work;
    float* d_K = d_work + (size_t)n_pos * n_embd_;
    float* d_V = d_work + (size_t)2 * n_pos * n_embd_;
    float* d_ffn_mid = d_work;

    // Helper: pack a slice of tensors from disk into h_stage -> H2D into d_stage!
    auto load_slice_to_stage = [&](const std::vector<const GGUFTensorInfo*>& list, std::vector<void*>& out_ptrs) -> bool {
        out_ptrs.assign(list.size(), nullptr);
        size_t off = 0;
        for (size_t i = 0; i < list.size(); ++i) {
            const auto* ti = list[i];
            if (!ti) continue;
            off = align_up_256(off);
            if (off + ti->nbytes > stage_cap) {
                std::fprintf(stderr, "[Vision] Stage overflow (%zu > %zu)\n", off + ti->nbytes, stage_cap);
                return false;
            }
            if (!gguf_.pread(ti->abs_offset, h_stage + off, ti->nbytes)) return false;
            out_ptrs[i] = d_stage + off;
            off += ti->nbytes;
        }
        if (off > 0) {
            INK_CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, off, cudaMemcpyHostToDevice, stream));
        }
        return true;
    };

    // 5. Patch embedding conv2d via Tensor Core GEMM -> d_X[1..n_patches]
    {
        float* d_pin = nullptr;
        INK_CUDA_CHECK(cudaMalloc(&d_pin, h_patches.size() * sizeof(float)));
        INK_CUDA_CHECK(cudaMemcpyAsync(d_pin, h_patches.data(), h_patches.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        std::vector<void*> ptrs;
        load_slice_to_stage({t_patch_embd_}, ptrs);
        ink_cuda_vision_gemm(t_patch_embd_->type, ptrs[0], nullptr, d_pin, d_X + (size_t)n_embd_, n_patches, n_embd_, patch_dim, n_embd_, stream);
        INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_pin);
    }

    // 6. Add resized learned pos_embd + prepend CLS token + run v.pre_ln!
    {
        std::vector<float> cls_e, cls_p, pos_raw, pos_resized((size_t)n_patches * n_embd_);
        read_tensor_as_f32(gguf_, t_cls_embd_, cls_e);
        read_tensor_as_f32(gguf_, t_cls_pos_embd_, cls_p);
        read_tensor_as_f32(gguf_, t_pos_embd_, pos_raw);
        int n_side = (int)std::lround(std::sqrt((double)(pos_raw.size() / n_embd_)));
        resize_pos_embd_bilinear(pos_raw.data(), n_side, n_embd_, grid_w, grid_h, pos_resized.data());

        std::vector<float> h_X((size_t)n_pos * n_embd_);
        INK_CUDA_CHECK(cudaMemcpy(h_X.data() + n_embd_, d_X + n_embd_, (size_t)n_patches * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost));
        for (int d = 0; d < n_embd_; ++d) h_X[d] = cls_e[d] + cls_p[d];
        for (size_t i = 0; i < (size_t)n_patches * n_embd_; ++i) h_X[n_embd_ + i] += pos_resized[i];
        INK_CUDA_CHECK(cudaMemcpyAsync(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Upload 2D RoPE coordinates: CLS=(0,0), patch(y,x)=(x+1, y+1)
        std::vector<int32_t> pw(n_pos), ph(n_pos);
        pw[0] = 0; ph[0] = 0;
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                int idx = 1 + y * grid_w + x;
                pw[idx] = x + 1;
                ph[idx] = y + 1;
            }
        }
        INK_CUDA_CHECK(cudaMemcpyAsync(d_pos_w, pw.data(), pw.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        INK_CUDA_CHECK(cudaMemcpyAsync(d_pos_h, ph.data(), ph.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        // Pre-LayerNorm
        std::vector<void*> ptrs;
        load_slice_to_stage({t_pre_ln_w_, t_pre_ln_b_}, ptrs);
        ink_cuda_vision_layernorm(d_X, (const float*)ptrs[0], (const float*)ptrs[1], d_X, n_pos, n_embd_, ln_eps_, stream);
        INK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // 7. Stream all 23 ViT layers through the 10-14 MiB staging buffer (3 small slices per layer!)
    const float attn_scale = 1.0f / std::sqrt((float)d_head_);
    for (int il = 0; il < n_layers_; ++il) {
        const auto& L = layers_[il];
        std::vector<void*> ptrs;

        // Slice A (~8.05 MiB): ln1 + q/k/v/o weights & biases + ls1
        load_slice_to_stage({L.ln1_w, L.ln1_b, L.q_w, L.q_b, L.k_w, L.k_b, L.v_w, L.v_b, L.o_w, L.o_b, L.ls1_w}, ptrs);
        ink_cuda_vision_layernorm(d_X, (const float*)ptrs[0], (const float*)ptrs[1], d_H, n_pos, n_embd_, ln_eps_, stream);
        ink_cuda_vision_gemm(L.q_w->type, ptrs[2], (const float*)ptrs[3], d_H, d_Q, n_pos, n_embd_, n_embd_, n_embd_, stream);
        ink_cuda_vision_gemm(L.k_w->type, ptrs[4], (const float*)ptrs[5], d_H, d_K, n_pos, n_embd_, n_embd_, n_embd_, stream);
        ink_cuda_vision_gemm(L.v_w->type, ptrs[6], (const float*)ptrs[7], d_H, d_V, n_pos, n_embd_, n_embd_, n_embd_, stream);
        ink_cuda_vision_rope_2d(d_Q, d_pos_w, d_pos_h, n_pos, n_heads_, d_head_, rope_theta_, stream);
        ink_cuda_vision_rope_2d(d_K, d_pos_w, d_pos_h, n_pos, n_heads_, d_head_, rope_theta_, stream);
        ink_cuda_vision_flash_attn(d_Q, d_K, d_V, d_H, n_pos, n_heads_, d_head_, attn_scale, stream);
        ink_cuda_vision_gemm(L.o_w->type, ptrs[8], (const float*)ptrs[9], d_H, d_Q, n_pos, n_embd_, n_embd_, n_embd_, stream);
        if (ptrs[10]) {
            ink_cuda_vision_layerscale_add(d_X, d_Q, (const float*)ptrs[10], n_pos, n_embd_, stream);
        } else {
            ink_cuda_add_inplace(d_X, d_Q, n_pos * n_embd_, stream);
        }
        INK_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Slice B & C: ln2 + FFN reusing d_work (Q/K/V slab) for d_ffn_mid and d_H slice for FFN output!
        // Single pass (only 3 slices/layer total!) for all images up to 640x640 (n_pos <= 1088), 2 passes for 1280x720!
        if (ffn_chunk >= n_pos) {
            load_slice_to_stage({L.ln2_w, L.ln2_b, L.up_w, L.up_b}, ptrs);
            ink_cuda_vision_layernorm(d_X, (const float*)ptrs[0], (const float*)ptrs[1], d_H, n_pos, n_embd_, ln_eps_, stream);
            ink_cuda_vision_gemm(L.up_w->type, ptrs[2], (const float*)ptrs[3], d_H, d_ffn_mid, n_pos, n_ff_, n_embd_, n_ff_, stream);
            ink_cuda_vision_gelu_erf(d_ffn_mid, n_pos * n_ff_, stream);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));

            load_slice_to_stage({L.dn_w, L.dn_b, L.ls2_w}, ptrs);
            ink_cuda_vision_gemm(L.dn_w->type, ptrs[0], (const float*)ptrs[1], d_ffn_mid, d_H, n_pos, n_embd_, n_ff_, n_embd_, stream);
            if (ptrs[2]) {
                ink_cuda_vision_layerscale_add(d_X, d_H, (const float*)ptrs[2], n_pos, n_embd_, stream);
            } else {
                ink_cuda_add_inplace(d_X, d_H, n_pos * n_embd_, stream);
            }
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            load_slice_to_stage({L.ln2_w, L.ln2_b}, ptrs);
            ink_cuda_vision_layernorm(d_X, (const float*)ptrs[0], (const float*)ptrs[1], d_H, n_pos, n_embd_, ln_eps_, stream);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int n0 = 0; n0 < n_pos; n0 += ffn_chunk) {
                int cn = std::min(ffn_chunk, n_pos - n0);
                float* d_H_slice = d_H + (size_t)n0 * n_embd_;
                load_slice_to_stage({L.up_w, L.up_b}, ptrs);
                ink_cuda_vision_gemm(L.up_w->type, ptrs[0], (const float*)ptrs[1], d_H_slice, d_ffn_mid, cn, n_ff_, n_embd_, n_ff_, stream);
                ink_cuda_vision_gelu_erf(d_ffn_mid, cn * n_ff_, stream);
                INK_CUDA_CHECK(cudaStreamSynchronize(stream));

                load_slice_to_stage({L.dn_w, L.dn_b, L.ls2_w}, ptrs);
                ink_cuda_vision_gemm(L.dn_w->type, ptrs[0], (const float*)ptrs[1], d_ffn_mid, d_H_slice, cn, n_embd_, n_ff_, n_embd_, stream);
                if (ptrs[2]) {
                    ink_cuda_vision_layerscale_add(d_X + (size_t)n0 * n_embd_, d_H_slice, (const float*)ptrs[2], cn, n_embd_, stream);
                } else {
                    ink_cuda_add_inplace(d_X + (size_t)n0 * n_embd_, d_H_slice, cn * n_embd_, stream);
                }
                INK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
    }

    // 8. Drop CLS token (index 0) and perform exact 2x2 Spatial Merge -> [n_out_tokens, 4096]
    // Order matches mtmd_graph::build_patch_merge_permute(cur, 2):
    //   merged[by * out_w + bx] = [P(2*by, 2*bx), P(2*by, 2*bx+1), P(2*by+1, 2*bx), P(2*by+1, 2*bx+1)]
    std::vector<float> h_patches_out((size_t)n_patches * n_embd_);
    INK_CUDA_CHECK(cudaMemcpy(h_patches_out.data(), d_X + n_embd_,
                              (size_t)n_patches * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost));

    // Free ViT activations right away BEFORE allocating projector buffers!
    cudaFree(d_X); cudaFree(d_H); cudaFree(d_work);
    cudaFree(d_pos_w); cudaFree(d_pos_h);

    const int merge_dim = n_embd_ * merge_size_ * merge_size_; // 4096
    const int out_w = grid_w / merge_size_;
    const int out_h = grid_h / merge_size_;
    std::vector<float> h_merged((size_t)n_out_tokens * merge_dim);
    for (int by = 0; by < out_h; ++by) {
        for (int bx = 0; bx < out_w; ++bx) {
            float* dst = h_merged.data() + ((size_t)by * out_w + bx) * merge_dim;
            int sub = 0;
            for (int dy = 0; dy < merge_size_; ++dy) {
                for (int dx = 0; dx < merge_size_; ++dx) {
                    const float* src = h_patches_out.data()
                        + ((size_t)(by * merge_size_ + dy) * grid_w + (bx * merge_size_ + dx)) * n_embd_;
                    std::memcpy(dst + (size_t)sub * n_embd_, src, (size_t)n_embd_ * sizeof(float));
                    sub++;
                }
            }
        }
    }

    // 9. Multimodal Projector: mm.1 (4096 -> 4096) + GELU_ERF + mm.2 (4096 -> 2304), streamed in <= 9 MiB row slices!
    const int mm1_out = (int)t_mm1_w_->ne[1]; // 4096
    const int mm1_in  = (int)t_mm1_w_->ne[0]; // 4096
    const int mm2_out = (int)t_mm2_w_->ne[1]; // 2304
    const int mm2_in  = (int)t_mm2_w_->ne[0]; // 4096

    float* d_proj_in  = nullptr;
    float* d_proj_mid = nullptr;
    float* d_proj_out = nullptr;
    INK_CUDA_CHECK(cudaMalloc(&d_proj_in,  (size_t)n_out_tokens * mm1_in  * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_proj_mid, (size_t)n_out_tokens * mm1_out * sizeof(float)));
    INK_CUDA_CHECK(cudaMalloc(&d_proj_out, (size_t)n_out_tokens * mm2_out * sizeof(float)));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_proj_in, h_merged.data(), h_merged.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Stream mm.1 in row blocks fitting inside stage_cap (e.g. 1024 rows x 4096 x 2B = 8 MiB!)
    {
        size_t elem_b = (t_mm1_w_->type == GGML_TYPE_F32) ? 4 : 2;
        size_t row_b  = (size_t)mm1_in * elem_b;
        int max_rows  = std::max(64, (int)((stage_cap - 65536ull) / row_b));
        max_rows = (max_rows / 32) * 32;
        for (int r0 = 0; r0 < mm1_out; r0 += max_rows) {
            int cr = std::min(max_rows, mm1_out - r0);
            size_t wb = (size_t)cr * row_b;
            gguf_.pread(t_mm1_w_->abs_offset + (size_t)r0 * row_b, h_stage, wb);
            float* d_b = nullptr;
            size_t tot_copy = wb;
            if (t_mm1_b_) {
                size_t boff = align_up_256(wb);
                gguf_.pread(t_mm1_b_->abs_offset + (size_t)r0 * sizeof(float), h_stage + boff, (size_t)cr * sizeof(float));
                d_b = (float*)(d_stage + boff);
                tot_copy = boff + (size_t)cr * sizeof(float);
            }
            INK_CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, tot_copy, cudaMemcpyHostToDevice, stream));
            ink_cuda_vision_gemm(t_mm1_w_->type, d_stage, d_b, d_proj_in, d_proj_mid + r0, n_out_tokens, cr, mm1_in, mm1_out, stream);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        ink_cuda_vision_gelu_erf(d_proj_mid, n_out_tokens * mm1_out, stream);
    }

    // Stream mm.2 in row blocks fitting inside stage_cap (e.g. 1152 rows x 4096 x 2B = 9 MiB!)
    {
        size_t elem_b = (t_mm2_w_->type == GGML_TYPE_F32) ? 4 : 2;
        size_t row_b  = (size_t)mm2_in * elem_b;
        int max_rows  = std::max(64, (int)((stage_cap - 65536ull) / row_b));
        max_rows = (max_rows / 32) * 32;
        for (int r0 = 0; r0 < mm2_out; r0 += max_rows) {
            int cr = std::min(max_rows, mm2_out - r0);
            size_t wb = (size_t)cr * row_b;
            gguf_.pread(t_mm2_w_->abs_offset + (size_t)r0 * row_b, h_stage, wb);
            float* d_b = nullptr;
            size_t tot_copy = wb;
            if (t_mm2_b_) {
                size_t boff = align_up_256(wb);
                gguf_.pread(t_mm2_b_->abs_offset + (size_t)r0 * sizeof(float), h_stage + boff, (size_t)cr * sizeof(float));
                d_b = (float*)(d_stage + boff);
                tot_copy = boff + (size_t)cr * sizeof(float);
            }
            INK_CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, tot_copy, cudaMemcpyHostToDevice, stream));
            ink_cuda_vision_gemm(t_mm2_w_->type, d_stage, d_b, d_proj_mid, d_proj_out + r0, n_out_tokens, cr, mm2_in, mm2_out, stream);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    out.hash      = h64;
    out.orig_w    = orig_w;
    out.orig_h    = orig_h;
    out.proc_w    = proc_w;
    out.proc_h    = proc_h;
    out.n_tokens  = n_out_tokens;
    out.n_embd    = mm2_out;
    out.embd.resize((size_t)n_out_tokens * mm2_out);
    INK_CUDA_CHECK(cudaMemcpy(out.embd.data(), d_proj_out, out.embd.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Free ALL temporary GPU memory immediately -> 0 MiB permanent VRAM footprint!
    cudaFree(d_proj_in);
    cudaFree(d_proj_mid);
    cudaFree(d_proj_out);
    if (own_stage) {
        cudaFree(d_stage);
        cudaFreeHost(h_stage);
    }

    out.encode_ms = now_ms_v() - t0;
    std::printf("[Vision] Encoded %dx%d -> %dx%d (%d patches -> %d LLM tokens) in %.1f ms (0 MiB permanent VRAM/RAM)\n",
                orig_w, orig_h, proc_w, proc_h, n_patches, n_out_tokens, out.encode_ms);
    std::fflush(stdout);

    cache_.push_back(out);
    auto cache_bytes = [&]() {
        size_t tot = 0;
        for (const auto& c : cache_) tot += c.embd.size() * sizeof(float);
        return tot;
    };
    while (cache_.size() > MAX_IMG_CACHE || (cache_.size() > 1 && cache_bytes() > MAX_IMG_CACHE_BYTES)) {
        cache_.pop_front();
    }
    return true;
}
