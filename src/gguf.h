#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "quant_tables.h"

enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct GGUFMetaValue {
    GGUFValueType type = GGUF_TYPE_UINT32;
    GGUFValueType elem_type = GGUF_TYPE_UINT32;
    uint64_t u64 = 0;
    int64_t  i64 = 0;
    double   f64 = 0.0;
    bool     b   = false;
    std::string str;
    std::vector<uint8_t>     arr_bool;
    std::vector<int32_t>     arr_i32;
    std::vector<uint32_t>    arr_u32;
    std::vector<float>       arr_f32;
    std::vector<std::string> arr_str;
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint32_t type = 0;
    uint64_t offset = 0;      // relative to data_offset
    uint64_t abs_offset = 0;  // absolute byte offset in file
    uint64_t nbytes = 0;
};

class GGUFFile {
public:
    GGUFFile() = default;
    ~GGUFFile();
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    bool open(const std::string& path);
    void close();

    // Thread-safe positional read directly from disk (Windows OVERLAPPED ReadFile)
    bool pread(uint64_t abs_offset, void* dst, size_t nbytes) const;

    const GGUFTensorInfo* find_tensor(const std::string& name) const;
    const GGUFTensorInfo& require_tensor(const std::string& name) const;

    bool has_kv(const std::string& key) const;
    uint32_t get_u32(const std::string& key, uint32_t def = 0) const;
    int32_t  get_i32(const std::string& key, int32_t def = 0) const;
    float    get_f32(const std::string& key, float def = 0.0f) const;
    bool     get_bool(const std::string& key, bool def = false) const;
    std::string get_str(const std::string& key, const std::string& def = "") const;
    const std::vector<uint8_t>&     get_bool_arr(const std::string& key) const;
    const std::vector<int32_t>&     get_i32_arr(const std::string& key) const;
    const std::vector<float>&       get_f32_arr(const std::string& key) const;
    const std::vector<std::string>& get_str_arr(const std::string& key) const;

    uint64_t data_offset() const { return data_offset_; }
    uint64_t file_size() const { return file_size_; }
    const std::string& path() const { return path_; }
    const std::unordered_map<std::string, GGUFTensorInfo>& tensors() const { return tensors_; }

private:
    std::string path_;
    void* hfile_sync_ = nullptr;   // HANDLE
    void* hfile_async_ = nullptr;  // HANDLE with FILE_FLAG_OVERLAPPED
    uint64_t file_size_ = 0;
    uint64_t data_offset_ = 0;
    uint32_t alignment_ = 32;

    std::unordered_map<std::string, GGUFMetaValue> kv_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;
};
