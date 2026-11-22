#include "gguf.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

namespace {

class BufferedReader {
public:
    BufferedReader(HANDLE h) : h_(h), buf_(1 << 20), pos_(0), len_(0), file_pos_(0) {}

    bool read_bytes(void* dst, size_t n) {
        uint8_t* out = static_cast<uint8_t*>(dst);
        while (n > 0) {
            if (pos_ == len_) {
                DWORD got = 0;
                if (!ReadFile(h_, buf_.data(), (DWORD)buf_.size(), &got, nullptr) || got == 0) {
                    return false;
                }
                pos_ = 0;
                len_ = got;
            }
            size_t avail = len_ - pos_;
            size_t take = (n < avail) ? n : avail;
            std::memcpy(out, buf_.data() + pos_, take);
            pos_ += take;
            file_pos_ += take;
            out += take;
            n -= take;
        }
        return true;
    }

    template <typename T>
    T read() {
        T v{};
        if (!read_bytes(&v, sizeof(T))) {
            throw std::runtime_error("GGUF unexpected EOF");
        }
        return v;
    }

    std::string read_str() {
        uint64_t n = read<uint64_t>();
        std::string s(n, '\0');
        if (n > 0 && !read_bytes(&s[0], (size_t)n)) {
            throw std::runtime_error("GGUF unexpected EOF reading string");
        }
        return s;
    }

    uint64_t tell() const { return file_pos_; }

private:
    HANDLE h_;
    std::vector<uint8_t> buf_;
    size_t pos_, len_;
    uint64_t file_pos_;
};

void read_value(BufferedReader& r, GGUFValueType type, GGUFMetaValue& out, bool keep_large_array) {
    out.type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   out.u64 = r.read<uint8_t>();  out.i64 = (int64_t)out.u64; break;
        case GGUF_TYPE_INT8:    out.i64 = r.read<int8_t>();   out.u64 = (uint64_t)out.i64; break;
        case GGUF_TYPE_UINT16:  out.u64 = r.read<uint16_t>(); out.i64 = (int64_t)out.u64; break;
        case GGUF_TYPE_INT16:   out.i64 = r.read<int16_t>();  out.u64 = (uint64_t)out.i64; break;
        case GGUF_TYPE_UINT32:  out.u64 = r.read<uint32_t>(); out.i64 = (int64_t)out.u64; break;
        case GGUF_TYPE_INT32:   out.i64 = r.read<int32_t>();  out.u64 = (uint64_t)out.i64; break;
        case GGUF_TYPE_UINT64:  out.u64 = r.read<uint64_t>(); out.i64 = (int64_t)out.u64; break;
        case GGUF_TYPE_INT64:   out.i64 = r.read<int64_t>();  out.u64 = (uint64_t)out.i64; break;
        case GGUF_TYPE_FLOAT32: out.f64 = r.read<float>(); break;
        case GGUF_TYPE_FLOAT64: out.f64 = r.read<double>(); break;
        case GGUF_TYPE_BOOL:    out.b   = r.read<uint8_t>() != 0; break;
        case GGUF_TYPE_STRING:  out.str = r.read_str(); break;
        case GGUF_TYPE_ARRAY: {
            GGUFValueType et = (GGUFValueType)r.read<uint32_t>();
            uint64_t n = r.read<uint64_t>();
            out.elem_type = et;
            if (et == GGUF_TYPE_BOOL) {
                out.arr_bool.resize((size_t)n);
                if (n > 0) r.read_bytes(out.arr_bool.data(), (size_t)n);
            } else if (et == GGUF_TYPE_INT32) {
                out.arr_i32.resize((size_t)n);
                if (n > 0) r.read_bytes(out.arr_i32.data(), (size_t)n * 4);
            } else if (et == GGUF_TYPE_UINT32) {
                out.arr_u32.resize((size_t)n);
                if (n > 0) r.read_bytes(out.arr_u32.data(), (size_t)n * 4);
            } else if (et == GGUF_TYPE_FLOAT32) {
                out.arr_f32.resize((size_t)n);
                if (n > 0) r.read_bytes(out.arr_f32.data(), (size_t)n * 4);
            } else if (et == GGUF_TYPE_STRING) {
                if (keep_large_array) {
                    out.arr_str.resize((size_t)n);
                    for (uint64_t i = 0; i < n; ++i) out.arr_str[(size_t)i] = r.read_str();
                } else {
                    for (uint64_t i = 0; i < n; ++i) (void)r.read_str();
                }
            } else {
                // Skip other array element types
                for (uint64_t i = 0; i < n; ++i) {
                    GGUFMetaValue dummy;
                    read_value(r, et, dummy, false);
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported GGUF value type");
    }
}

} // namespace

GGUFFile::~GGUFFile() { close(); }

void GGUFFile::close() {
    if (hfile_sync_ && hfile_sync_ != INVALID_HANDLE_VALUE) {
        CloseHandle((HANDLE)hfile_sync_);
        hfile_sync_ = nullptr;
    }
    if (hfile_async_ && hfile_async_ != INVALID_HANDLE_VALUE) {
        CloseHandle((HANDLE)hfile_async_);
        hfile_async_ = nullptr;
    }
}

bool GGUFFile::open(const std::string& path) {
    close();
    path_ = path;

    HANDLE hs = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (hs == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[GGUF] Failed to open '%s' (err=%lu)\n", path.c_str(), GetLastError());
        return false;
    }
    HANDLE ha = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (ha == INVALID_HANDLE_VALUE) {
        CloseHandle(hs);
        std::fprintf(stderr, "[GGUF] Failed to open async handle for '%s' (err=%lu)\n", path.c_str(), GetLastError());
        return false;
    }

    LARGE_INTEGER sz{};
    GetFileSizeEx(hs, &sz);
    file_size_ = (uint64_t)sz.QuadPart;
    hfile_sync_ = hs;
    hfile_async_ = ha;

    BufferedReader r(hs);
    uint32_t magic = r.read<uint32_t>();
    if (magic != 0x46554747u) { // "GGUF"
        std::fprintf(stderr, "[GGUF] Invalid magic 0x%08x\n", magic);
        close();
        return false;
    }
    uint32_t version = r.read<uint32_t>();
    if (version < 2 || version > 3) {
        std::fprintf(stderr, "[GGUF] Unsupported version %u\n", version);
        close();
        return false;
    }
    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv = r.read<uint64_t>();

    kv_.clear();
    kv_.reserve((size_t)n_kv * 2);
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.read_str();
        GGUFValueType vt = (GGUFValueType)r.read<uint32_t>();
        bool keep = (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges" ||
                     key == "tokenizer.ggml.token_type" ||
                     key.find("inkling2.") == 0 || key.find("moe_predictor.") == 0 ||
                     key.find("general.") == 0 || key.find("tokenizer.") == 0);
        GGUFMetaValue val;
        read_value(r, vt, val, keep);
        if (key == "general.alignment" && val.u64 > 0) {
            alignment_ = (uint32_t)val.u64;
        }
        kv_.emplace(std::move(key), std::move(val));
    }

    std::vector<GGUFTensorInfo> list;
    list.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGUFTensorInfo ti;
        ti.name = r.read_str();
        ti.n_dims = r.read<uint32_t>();
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            ti.ne[d] = r.read<uint64_t>();
            nelem *= ti.ne[d];
        }
        ti.type = r.read<uint32_t>();
        ti.offset = r.read<uint64_t>();
        size_t bs = ggml_type_block_size(ti.type);
        size_t bsz = ggml_type_byte_size(ti.type);
        ti.nbytes = (nelem / bs) * bsz;
        list.push_back(std::move(ti));
    }

    uint64_t cur = r.tell();
    uint64_t rem = cur % alignment_;
    data_offset_ = rem == 0 ? cur : (cur + alignment_ - rem);

    tensors_.clear();
    tensors_.reserve((size_t)n_tensors * 2);
    for (auto& ti : list) {
        ti.abs_offset = data_offset_ + ti.offset;
        tensors_.emplace(ti.name, std::move(ti));
    }
    return true;
}

bool GGUFFile::pread(uint64_t abs_offset, void* dst, size_t nbytes) const {
    if (!hfile_async_) return false;
    HANDLE h = (HANDLE)hfile_async_;
    uint8_t* out = static_cast<uint8_t*>(dst);
    // Use thread-local event for zero-allocation overlapped positional reads
    thread_local HANDLE hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);

    while (nbytes > 0) {
        DWORD chunk = (DWORD)std::min<size_t>(nbytes, 64u * 1024u * 1024u);
        OVERLAPPED ov{};
        ov.Offset = (DWORD)(abs_offset & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD)(abs_offset >> 32);
        ov.hEvent = hEvent;
        ResetEvent(hEvent);

        DWORD got = 0;
        BOOL ok = ReadFile(h, out, chunk, &got, &ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) {
                return false;
            }
            if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
                return false;
            }
        }
        if (got == 0) return false;
        out += got;
        abs_offset += got;
        nbytes -= got;
    }
    return true;
}

const GGUFTensorInfo* GGUFFile::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const GGUFTensorInfo& GGUFFile::require_tensor(const std::string& name) const {
    const auto* t = find_tensor(name);
    if (!t) throw std::runtime_error("Missing required GGUF tensor: " + name);
    return *t;
}

bool GGUFFile::has_kv(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

uint32_t GGUFFile::get_u32(const std::string& key, uint32_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : (uint32_t)it->second.u64;
}

int32_t GGUFFile::get_i32(const std::string& key, int32_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : (int32_t)it->second.i64;
}

float GGUFFile::get_f32(const std::string& key, float def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : (float)it->second.f64;
}

bool GGUFFile::get_bool(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.b;
}

std::string GGUFFile::get_str(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.str;
}

const std::vector<uint8_t>& GGUFFile::get_bool_arr(const std::string& key) const {
    static const std::vector<uint8_t> empty;
    auto it = kv_.find(key);
    return it == kv_.end() ? empty : it->second.arr_bool;
}

const std::vector<int32_t>& GGUFFile::get_i32_arr(const std::string& key) const {
    static const std::vector<int32_t> empty;
    auto it = kv_.find(key);
    return it == kv_.end() ? empty : it->second.arr_i32;
}

const std::vector<std::string>& GGUFFile::get_str_arr(const std::string& key) const {
    static const std::vector<std::string> empty;
    auto it = kv_.find(key);
    return it == kv_.end() ? empty : it->second.arr_str;
}
