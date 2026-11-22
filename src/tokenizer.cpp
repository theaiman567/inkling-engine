#include "tokenizer.h"
#include <cstdio>
#include <algorithm>
#include <climits>

namespace {

std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back((char)cp);
    } else if (cp < 0x800) {
        s.push_back((char)(0xC0 | (cp >> 6)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back((char)(0xE0 | (cp >> 12)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        s.push_back((char)(0xF0 | (cp >> 18)));
        s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    }
    return s;
}

uint32_t decode_utf8(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { i += 1; return c; }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F);
        i += 2; return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F);
        i += 3; return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12)
                    | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F);
        i += 4; return cp;
    }
    i += 1; return c;
}

// Fast o200k_base-compatible pre-tokenizer splitting ASCII/UTF-8 into BPE chunks
inline bool is_ascii_letter(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool is_ascii_digit(unsigned char c)  { return (c >= '0' && c <= '9'); }
inline bool is_space_no_nl(unsigned char c)  { return c == ' ' || c == '\t'; }
inline bool is_nl(unsigned char c)           { return c == '\r' || c == '\n'; }

void split_o200k(const std::string& s, std::vector<std::string>& out) {
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t start = i;
        unsigned char c = (unsigned char)s[i];

        // 1. Optional single non-alnum/non-nl prefix (like leading space) + letters
        if (is_ascii_letter(c) || c >= 0x80) {
            i++;
            while (i < n && (is_ascii_letter((unsigned char)s[i]) || (unsigned char)s[i] >= 0x80)) i++;
            // Check contractions like 's, 't, 're, 've, 'm, 'll, 'd (keep separate next iter)
            out.push_back(s.substr(start, i - start));
            continue;
        }
        if (c == ' ' && i + 1 < n && (is_ascii_letter((unsigned char)s[i+1]) || (unsigned char)s[i+1] >= 0x80)) {
            i += 2;
            while (i < n && (is_ascii_letter((unsigned char)s[i]) || (unsigned char)s[i] >= 0x80)) i++;
            out.push_back(s.substr(start, i - start));
            continue;
        }
        // 2. Numbers: 1 to 3 digits (o200k_base chunks digits in groups of up to 3!)
        if (is_ascii_digit(c)) {
            int cnt = 0;
            while (i < n && is_ascii_digit((unsigned char)s[i]) && cnt < 3) { i++; cnt++; }
            out.push_back(s.substr(start, i - start));
            continue;
        }
        // 3. Newlines (optionally preceded by spaces)
        if (is_nl(c)) {
            while (i < n && is_nl((unsigned char)s[i])) i++;
            out.push_back(s.substr(start, i - start));
            continue;
        }
        // 4. Spaces: if followed by punctuation, include one space with punctuation (o200k rule),
        //    unless multiple spaces
        if (is_space_no_nl(c)) {
            size_t j = i;
            while (j < n && is_space_no_nl((unsigned char)s[j])) j++;
            if (j < n && is_nl((unsigned char)s[j])) {
                while (j < n && is_nl((unsigned char)s[j])) j++;
                out.push_back(s.substr(start, j - start));
                i = j;
                continue;
            }
            if (j == i + 1 && j < n && !is_ascii_digit((unsigned char)s[j])) {
                // Single space + punctuation run
                j++;
                while (j < n && !is_ascii_letter((unsigned char)s[j]) && !is_ascii_digit((unsigned char)s[j])
                       && !is_space_no_nl((unsigned char)s[j]) && !is_nl((unsigned char)s[j]) && (unsigned char)s[j] < 0x80) {
                    j++;
                }
                out.push_back(s.substr(start, j - start));
                i = j;
                continue;
            }
            if (j > i + 1 && j < n) {
                // Leave last space for next word!
                out.push_back(s.substr(start, (j - 1) - start));
                i = j - 1;
                continue;
            }
            out.push_back(s.substr(start, j - start));
            i = j;
            continue;
        }
        // 5. Apostrophe contractions ('s, 't, 're, 've, 'm, 'll, 'd)
        if (c == '\'' && i + 1 < n) {
            unsigned char c1 = (unsigned char)s[i+1];
            unsigned char c1l = (c1 >= 'A' && c1 <= 'Z') ? (c1 + 32) : c1;
            if (c1l == 's' || c1l == 't' || c1l == 'm' || c1l == 'd') {
                out.push_back(s.substr(i, 2)); i += 2; continue;
            }
            if (i + 2 < n) {
                unsigned char c2 = (unsigned char)s[i+2];
                unsigned char c2l = (c2 >= 'A' && c2 <= 'Z') ? (c2 + 32) : c2;
                if ((c1l == 'r' && c2l == 'e') || (c1l == 'v' && c2l == 'e') || (c1l == 'l' && c2l == 'l')) {
                    out.push_back(s.substr(i, 3)); i += 3; continue;
                }
            }
        }
        // 6. Punctuation run
        i++;
        while (i < n && !is_ascii_letter((unsigned char)s[i]) && !is_ascii_digit((unsigned char)s[i])
               && !is_space_no_nl((unsigned char)s[i]) && !is_nl((unsigned char)s[i]) && (unsigned char)s[i] < 0x80) {
            i++;
        }
        out.push_back(s.substr(start, i - start));
    }
}

} // namespace

bool InkTokenizer::load(const GGUFFile& gguf) {
    // Build GPT-2 byte <-> unicode bijection
    bool direct[256] = {};
    for (int b = '!'; b <= '~'; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    uint32_t extra = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? (uint32_t)b : extra++;
        byte_to_unicode_[b] = cp_to_utf8(cp);
        unicode_to_byte_[cp] = (uint8_t)b;
    }

    const auto& toks = gguf.get_str_arr("tokenizer.ggml.tokens");
    const auto& ttypes = gguf.get_i32_arr("tokenizer.ggml.token_type");
    const auto& merges = gguf.get_str_arr("tokenizer.ggml.merges");
    if (toks.empty()) {
        std::fprintf(stderr, "[Tokenizer] No tokenizer.ggml.tokens found in GGUF\n");
        return false;
    }

    id_to_token_ = toks;
    token_type_  = ttypes;
    token_to_id_.clear();
    token_to_id_.reserve(toks.size() * 2);
    special_tokens_.clear();

    for (int32_t i = 0; i < (int32_t)toks.size(); ++i) {
        token_to_id_[toks[i]] = i;
        if (i < (int32_t)ttypes.size() && ttypes[i] == 3) {
            special_tokens_.push_back({toks[i], i});
        }
    }
    // Sort special tokens longest-first for greedy matching
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    merge_rank_.clear();
    merge_rank_.reserve(merges.size() * 2);
    for (int32_t i = 0; i < (int32_t)merges.size(); ++i) {
        merge_rank_[merges[i]] = i;
    }

    eos_id_ = (int32_t)gguf.get_u32("tokenizer.ggml.eos_token_id", 200006);
    pad_id_ = (int32_t)gguf.get_u32("tokenizer.ggml.padding_token_id", 199999);

    auto find_sp = [&](const char* name, int32_t def) {
        auto it = token_to_id_.find(name);
        return it == token_to_id_.end() ? def : it->second;
    };
    msg_model_id_        = find_sp("<|message_model|>", 200001);
    end_msg_id_          = find_sp("<|end_message|>", 200010);
    thinking_id_         = find_sp("<|content_thinking|>", 200008);
    text_id_             = find_sp("<|content_text|>", 200004);
    invoke_tool_json_id_ = find_sp("<|content_invoke_tool_json|>", 200049);
    invoke_tool_text_id_ = find_sp("<|content_invoke_tool_text|>", 200057);

    std::printf("[Tokenizer] Loaded %zu tokens, %zu merges, %zu special tokens (eos=%d)\n",
                id_to_token_.size(), merge_rank_.size(), special_tokens_.size(), eos_id_);
    return true;
}

std::string InkTokenizer::bytes_to_gpt2(const std::string& raw) const {
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) out += byte_to_unicode_[c];
    return out;
}

std::string InkTokenizer::gpt2_to_bytes(const std::string& tok) const {
    std::string out;
    out.reserve(tok.size());
    size_t i = 0;
    while (i < tok.size()) {
        uint32_t cp = decode_utf8(tok, i);
        auto it = unicode_to_byte_.find(cp);
        if (it != unicode_to_byte_.end()) out.push_back((char)it->second);
        else out += cp_to_utf8(cp);
    }
    return out;
}

void InkTokenizer::bpe_encode_chunk(const std::string& chunk_raw, std::vector<int32_t>& out) const {
    std::string word = bytes_to_gpt2(chunk_raw);
    // Fast path: entire chunk is directly in vocab!
    auto direct_it = token_to_id_.find(word);
    if (direct_it != token_to_id_.end()) {
        out.push_back(direct_it->second);
        return;
    }

    // Split into UTF-8 codepoints (each represents 1 original byte)
    std::vector<std::string> syms;
    size_t i = 0;
    while (i < word.size()) {
        size_t prev = i;
        decode_utf8(word, i);
        syms.push_back(word.substr(prev, i - prev));
    }

    while (syms.size() >= 2) {
        int32_t best_rank = INT_MAX;
        size_t best_pos = (size_t)-1;
        for (size_t j = 0; j + 1 < syms.size(); ++j) {
            std::string key;
            key.reserve(syms[j].size() + 1 + syms[j+1].size());
            key += syms[j];
            key.push_back(' ');
            key += syms[j+1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = j;
            }
        }
        if (best_pos == (size_t)-1) break;
        syms[best_pos] += syms[best_pos + 1];
        syms.erase(syms.begin() + best_pos + 1);
    }

    for (const auto& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) out.push_back(it->second);
    }
}

std::vector<int32_t> InkTokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> ids;
    size_t pos = 0, n = text.size();

    while (pos < n) {
        // Check for special token match at pos (fast check: starts with "<|")
        if (parse_special && pos + 2 <= n && text[pos] == '<' && text[pos + 1] == '|') {
            bool matched = false;
            for (const auto& sp : special_tokens_) {
                if (pos + sp.first.size() <= n && text.compare(pos, sp.first.size(), sp.first) == 0) {
                    ids.push_back(sp.second);
                    pos += sp.first.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // Find next "<|" if parse_special
        size_t next_sp = parse_special ? text.find("<|", pos) : std::string::npos;
        if (next_sp == pos) next_sp = text.find("<|", pos + 2);
        size_t seg_end = (next_sp == std::string::npos) ? n : next_sp;

        std::vector<std::string> chunks;
        split_o200k(text.substr(pos, seg_end - pos), chunks);
        for (const auto& ch : chunks) {
            bpe_encode_chunk(ch, ids);
        }
        pos = seg_end;
    }
    return ids;
}

std::string InkTokenizer::decode_token(int32_t id) const {
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return "";
    if (id < (int32_t)token_type_.size() && token_type_[id] == 3) {
        return ""; // Control / special token hidden from display text
    }
    return gpt2_to_bytes(id_to_token_[id]);
}

std::string InkTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += decode_token(id);
    return out;
}

namespace {
std::string escape_json_min(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out.push_back((char)c);
    }
    return out;
}
} // namespace

std::string InkTokenizer::format_chat(
    const std::vector<ChatMessage>& msgs,
    const std::string& effort,
    bool add_generation_prompt,
    const std::string& tools_canonical_json,
    size_t* out_static_prefix_bytes,
    size_t* out_tools_prefix_bytes,
    size_t* out_full_sys_prefix_bytes) const
{
    std::string eff_l = effort;
    for (char& ch : eff_l) if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
    std::string eff_val = "0.99";
    if      (eff_l == "none" || eff_l == "0" || eff_l == "0.0") eff_val = "0";
    else if (eff_l == "minimal") eff_val = "0.1";
    else if (eff_l == "low")     eff_val = "0.2";
    else if (eff_l == "medium" || eff_l == "med") eff_val = "0.7";
    else if (eff_l == "high")    eff_val = "0.9";
    else if (eff_l == "xhigh" || eff_l == "max") eff_val = "0.99";
    else if (!eff_l.empty()) eff_val = eff_l;

    std::string s;
    s.reserve(4096);

    // 1. Official Inkling-2 tool declarations block
    size_t tools_prefix_end = 0;
    if (!tools_canonical_json.empty()) {
        s += "<|message_system|>tool_declare<|content_xml|>";
        s += tools_canonical_json;
        s += "<|end_message|>";
        tools_prefix_end = s.size();
    }

    bool effort_emitted = false;
    size_t static_prefix_end = 0;
    auto emit_effort = [&]() {
        s += "<|message_system|><|content_text|>Thinking effort level: " + eff_val + "<|end_message|>";
        effort_emitted = true;
        static_prefix_end = s.size();
    };

    for (size_t idx = 0; idx < msgs.size(); ++idx) {
        const auto& m = msgs[idx];
        if (!effort_emitted && m.role != "system") emit_effort();

        if (m.role == "system") {
            s += "<|message_system|><|content_text|>" + m.content + "<|end_message|>";
        } else if (m.role == "user") {
            s += "<|message_user|><|content_text|>" + m.content + "<|end_message|>";
        } else if (m.role == "tool") {
            std::string tname = m.name;
            if (tname.empty() && !m.tool_call_id.empty()) {
                for (size_t p = 0; p < idx; ++p) {
                    if (msgs[p].role == "assistant") {
                        for (const auto& tc : msgs[p].tool_calls) {
                            if (tc.id == m.tool_call_id && !tc.name.empty()) {
                                tname = tc.name;
                                break;
                            }
                        }
                    }
                    if (!tname.empty()) break;
                }
            }
            s += "<|message_tool|>" + tname + "<|content_text|>" + m.content + "<|end_message|>";
        } else if (m.role == "assistant") {
            std::string rtok = "<|message_model|>" + m.name;
            if (!m.reasoning_content.empty()) {
                s += rtok + "<|content_thinking|>" + m.reasoning_content + "<|end_message|>";
            }
            if (!m.content.empty() || (m.tool_calls.empty() && m.reasoning_content.empty())) {
                s += rtok + "<|content_text|>" + m.content + "<|end_message|>";
            }
            for (const auto& tc : m.tool_calls) {
                std::string args = tc.arguments_json.empty() ? "{}" : tc.arguments_json;
                s += "<|message_model|>" + tc.name + "<|content_invoke_tool_json|>{\"name\":\"" +
                     escape_json_min(tc.name) + "\",\"args\":" + args + "}<|end_message|>";
            }
            s += "<|content_model_end_sampling|>";
        }
    }

    if (!effort_emitted) emit_effort();
    size_t full_sys_prefix_end = static_prefix_end;

    // Trim static_prefix_end before dynamic environment / date / directory blocks (e.g. OpenCode <env>)
    // so even when <env> changes, brand-new sessions still reuse ~80%+ of the KV cache in ~28 ms!
    static const char* kDynamicMarkers[] = {
        "Here is some useful information about the environment",
        "<env>",
        "<files>",
        "<directories>",
        "Today's date",
        "Current date",
        "Current time",
        "Working directory:"
    };
    for (const char* mk : kDynamicMarkers) {
        size_t pos = s.find(mk, tools_prefix_end);
        if (pos != std::string::npos && pos < static_prefix_end) {
            static_prefix_end = pos;
        }
    }
    if (out_tools_prefix_bytes)    *out_tools_prefix_bytes    = tools_prefix_end;
    if (out_static_prefix_bytes)   *out_static_prefix_bytes   = static_prefix_end;
    if (out_full_sys_prefix_bytes) *out_full_sys_prefix_bytes = full_sys_prefix_end;

    if (add_generation_prompt) {
        if (tools_canonical_json.empty()) {
            s += "<|message_model|>";
            if (eff_val == "0") s += "<|content_text|>";
            else                s += "<|content_thinking|>";
        } else {
            // When tools are active, let the model choose <|content_thinking|>, <|content_text|>, or tool_name<|content_invoke_tool_json|>!
            s += "<|message_model|>";
            if (eff_val != "0") s += "<|content_thinking|>";
        }
    }
    return s;
}
