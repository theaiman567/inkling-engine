#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "gguf.h"

struct ToolCallSpec {
    std::string id;
    std::string name;
    std::string arguments_json; // JSON object string, e.g. {"path":"main.cpp"}
};

struct ChatContentPart {
    bool is_image = false;
    std::string text;
    std::string image_url; // data:image/...;base64,... or local file path
};

struct ChatMessage {
    std::string role;              // "system", "user", "assistant", "tool"
    std::string content;
    std::vector<ChatContentPart> parts; // optional ordered text + image parts (for multimodal user/tool msgs)
    std::string name;              // optional tool or assistant name
    std::string tool_call_id;      // for role=="tool"
    std::string reasoning_content; // optional assistant thinking
    std::vector<ToolCallSpec> tool_calls;
};

class InkTokenizer {
public:
    bool load(const GGUFFile& gguf);

    std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
    std::string decode_token(int32_t id) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    // Format messages using Inkling-2's official chat & tool-calling template.
    // Prefix checkpoints (optional byte offsets):
    //   out_full_sys_prefix_bytes : end of tools + ALL leading system msgs + effort (fastest ~99% hit!)
    //   out_static_prefix_bytes   : trimmed before dynamic <env>/date blocks (survives env/file changes!)
    //   out_tools_prefix_bytes    : right after tool_declare (survives agent build<->plan switches!)
    std::string format_chat(
        const std::vector<ChatMessage>& msgs,
        const std::string& effort = "none",
        bool add_generation_prompt = true,
        const std::string& tools_canonical_json = "",
        size_t* out_static_prefix_bytes = nullptr,
        size_t* out_tools_prefix_bytes = nullptr,
        size_t* out_full_sys_prefix_bytes = nullptr) const;

    bool is_eog(int32_t id) const { return id == eos_id_ || id == pad_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t msg_model_id() const { return msg_model_id_; }
    int32_t end_msg_id() const { return end_msg_id_; }
    int32_t thinking_id() const { return thinking_id_; }
    int32_t text_id() const { return text_id_; }
    int32_t invoke_tool_json_id() const { return invoke_tool_json_id_; }
    int32_t invoke_tool_text_id() const { return invoke_tool_text_id_; }
    int32_t image_placeholder_id() const { return 200054; } // <|unused_200054|>
    int32_t vocab_size() const { return (int32_t)id_to_token_.size(); }

private:
    void bpe_encode_chunk(const std::string& chunk_utf8, std::vector<int32_t>& out) const;
    std::string bytes_to_gpt2(const std::string& raw) const;
    std::string gpt2_to_bytes(const std::string& tok) const;

    std::vector<std::string> id_to_token_;
    std::vector<int32_t>     token_type_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_;

    std::string byte_to_unicode_[256];
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;

    int32_t eos_id_ = 200006;
    int32_t pad_id_ = 199999;
    int32_t msg_model_id_ = 200001;
    int32_t end_msg_id_ = 200010;
    int32_t thinking_id_ = 200008;
    int32_t text_id_ = 200004;
    int32_t invoke_tool_json_id_ = 200049;
    int32_t invoke_tool_text_id_ = 200057;
};
