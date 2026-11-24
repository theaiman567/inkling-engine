#define _CRT_RAND_S
#include "server.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <list>
#include <unordered_map>
#include <functional>

#pragma comment(lib, "ws2_32.lib")

namespace {

// =========================================================================
// Minimal Self-Contained JSON AST, Parser & Canonical Serializer (Zero Deps!)
// =========================================================================
struct JsonVal {
    enum Type { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type = J_NULL;
    bool b = false;
    double num = 0.0;
    std::string num_raw;
    std::string str;
    std::vector<JsonVal> arr;
    std::map<std::string, JsonVal> obj; // std::map keeps keys sorted -> canonical Inkling-2 JSON!
    std::vector<std::string> keys;      // v4.1: original key insertion order (for template-faithful nested args)

    bool is_obj() const { return type == J_OBJ; }
    bool is_arr() const { return type == J_ARR; }
    bool is_str() const { return type == J_STR; }
    bool is_num() const { return type == J_NUM; }
    bool is_bool() const { return type == J_BOOL; }

    const JsonVal* get(const std::string& k) const {
        if (type != J_OBJ) return nullptr;
        auto it = obj.find(k);
        return (it == obj.end()) ? nullptr : &it->second;
    }
    std::string get_str(const std::string& k, const std::string& def = "") const {
        const auto* v = get(k);
        return (v && v->is_str()) ? v->str : def;
    }
    double get_num(const std::string& k, double def = 0.0) const {
        const auto* v = get(k);
        return (v && v->is_num()) ? v->num : def;
    }
    bool get_bool(const std::string& k, bool def = false) const {
        const auto* v = get(k);
        return (v && v->is_bool()) ? v->b : def;
    }
};

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

void json_dump(const JsonVal& v, std::string& out) {
    switch (v.type) {
        case JsonVal::J_NULL: out += "null"; break;
        case JsonVal::J_BOOL: out += v.b ? "true" : "false"; break;
        case JsonVal::J_NUM:
            if (!v.num_raw.empty()) out += v.num_raw;
            else {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.12g", v.num);
                out += buf;
            }
            break;
        case JsonVal::J_STR:
            out.push_back('"');
            out += json_escape(v.str);
            out.push_back('"');
            break;
        case JsonVal::J_ARR:
            out.push_back('[');
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) out.push_back(',');
                json_dump(v.arr[i], out);
            }
            out.push_back(']');
            break;
        case JsonVal::J_OBJ:
            out.push_back('{');
            {
                bool first = true;
                for (const auto& kv : v.obj) {
                    if (!first) out.push_back(',');
                    first = false;
                    out.push_back('"');
                    out += json_escape(kv.first);
                    out += "\":";
                    json_dump(kv.second, out);
                }
            }
            out.push_back('}');
            break;
    }
}

std::string json_stringify(const JsonVal& v) {
    std::string out;
    json_dump(v, out);
    return out;
}

// v4.1 (R2-01): render tool-call arguments exactly like the official Inkling-2 chat template:
//   top-level keys sorted (dictsort, case-sensitive) and every nested value via tojson(separators=(",",":"))
//   WITHOUT sort_keys, i.e. nested objects keep their original key order. std::map alone sorts every level.
void json_dump_template(const JsonVal& v, std::string& out, bool top_level) {
    if (v.type == JsonVal::J_ARR) {
        out.push_back('[');
        for (size_t i = 0; i < v.arr.size(); ++i) {
            if (i) out.push_back(',');
            json_dump_template(v.arr[i], out, false);
        }
        out.push_back(']');
        return;
    }
    if (v.type != JsonVal::J_OBJ) { json_dump(v, out); return; }
    out.push_back('{');
    bool first = true;
    auto emit_kv = [&](const std::string& k, const JsonVal& val) {
        if (!first) out.push_back(',');
        first = false;
        out.push_back('"');
        out += json_escape(k);
        out += "\":";
        json_dump_template(val, out, false);
    };
    if (top_level || v.keys.size() != v.obj.size()) {
        for (const auto& kv : v.obj) emit_kv(kv.first, kv.second);
    } else {
        for (const auto& k : v.keys) {
            auto it = v.obj.find(k);
            if (it != v.obj.end()) emit_kv(it->first, it->second);
        }
    }
    out.push_back('}');
}

std::string json_args_template(const JsonVal& v) {
    std::string out;
    json_dump_template(v, out, /*top_level=*/true);
    return out;
}

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s), i_(0), n_(s.size()) {}

    bool parse(JsonVal& out) {
        skip_ws();
        if (!parse_value(out, 0)) return false;
        return true;
    }

private:
    void skip_ws() {
        while (i_ < n_ && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\r' || s_[i_] == '\n')) ++i_;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out.push_back((char)cp);
        else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

    bool parse_hex4(uint32_t& val) {
        if (i_ + 4 > n_) return false;
        val = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            val <<= 4;
            if (c >= '0' && c <= '9') val |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') val |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= (uint32_t)(c - 'A' + 10);
            else return false;
        }
        return true;
    }

    bool parse_string(std::string& out) {
        if (i_ >= n_ || s_[i_] != '"') return false;
        ++i_;
        out.clear();
        while (i_ < n_) {
            char c = s_[i_++];
            if (c == '"') return true;
            if (c == '\\' && i_ < n_) {
                char e = s_[i_++];
                switch (e) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!parse_hex4(cp)) return false;
                        // Handle UTF-16 surrogate pairs
                        if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 6 <= n_ && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                            i_ += 2;
                            uint32_t lo = 0;
                            if (parse_hex4(lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parse_value(JsonVal& out, int depth) {
        if (depth > 64) return false;
        skip_ws();
        if (i_ >= n_) return false;
        char c = s_[i_];
        if (c == '"') {
            out.type = JsonVal::J_STR;
            return parse_string(out.str);
        }
        if (c == '{') {
            out.type = JsonVal::J_OBJ;
            ++i_;
            skip_ws();
            if (i_ < n_ && s_[i_] == '}') { ++i_; return true; }
            while (i_ < n_) {
                skip_ws();
                std::string key;
                if (!parse_string(key)) return false;
                skip_ws();
                if (i_ >= n_ || s_[i_] != ':') return false;
                ++i_;
                JsonVal val;
                if (!parse_value(val, depth + 1)) return false;
                if (out.obj.find(key) == out.obj.end()) out.keys.push_back(key);
                out.obj[std::move(key)] = std::move(val);
                skip_ws();
                if (i_ < n_ && s_[i_] == ',') { ++i_; continue; }
                if (i_ < n_ && s_[i_] == '}') { ++i_; return true; }
                return false;
            }
            return false;
        }
        if (c == '[') {
            out.type = JsonVal::J_ARR;
            ++i_;
            skip_ws();
            if (i_ < n_ && s_[i_] == ']') { ++i_; return true; }
            while (i_ < n_) {
                JsonVal elem;
                if (!parse_value(elem, depth + 1)) return false;
                out.arr.push_back(std::move(elem));
                skip_ws();
                if (i_ < n_ && s_[i_] == ',') { ++i_; continue; }
                if (i_ < n_ && s_[i_] == ']') { ++i_; return true; }
                return false;
            }
            return false;
        }
        if (s_.compare(i_, 4, "true") == 0)  { out.type = JsonVal::J_BOOL; out.b = true;  i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { out.type = JsonVal::J_BOOL; out.b = false; i_ += 5; return true; }
        if (s_.compare(i_, 4, "null") == 0)  { out.type = JsonVal::J_NULL; i_ += 4; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t start = i_;
            if (s_[i_] == '-') ++i_;
            while (i_ < n_ && (s_[i_] >= '0' && s_[i_] <= '9')) ++i_;
            if (i_ < n_ && s_[i_] == '.') { ++i_; while (i_ < n_ && (s_[i_] >= '0' && s_[i_] <= '9')) ++i_; }
            if (i_ < n_ && (s_[i_] == 'e' || s_[i_] == 'E')) {
                ++i_;
                if (i_ < n_ && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
                while (i_ < n_ && (s_[i_] >= '0' && s_[i_] <= '9')) ++i_;
            }
            out.type = JsonVal::J_NUM;
            out.num_raw = s_.substr(start, i_ - start);
            try { out.num = std::stod(out.num_raw); } catch (...) { out.num = 0.0; }
            return true;
        }
        return false;
    }

    const std::string& s_;
    size_t i_, n_;
};

enum OutState { ST_HEADER, ST_THINKING, ST_TEXT, ST_TOOL_JSON, ST_TOOL_TEXT };

// =========================================================================
// HTTP Helpers, Security (R2), UTF-8 Stream Splitter (R7) & Timeouts (R9)
// =========================================================================
std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string trim_ascii(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

std::string generate_csprng_hex_token(int n_bytes = 16) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve((size_t)n_bytes * 2);
    for (int i = 0; i < n_bytes; i += 4) {
        unsigned int r = 0;
        if (rand_s(&r) != 0) {
            r = (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ (0x9E3779B9u * (unsigned int)(i + 1));
        }
        for (int b = 0; b < 4 && (i + b) < n_bytes; ++b) {
            unsigned char v = (unsigned char)((r >> (b * 8)) & 0xFFu);
            out.push_back(kHex[v >> 4]);
            out.push_back(kHex[v & 0x0Fu]);
        }
    }
    return out;
}

bool const_time_eq(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff |= (unsigned char)(a[i] ^ b[i]);
    return diff == 0;
}

// DNS-rebinding protection (R2 / SR1-09): only accept localhost / loopback / explicit bind host!
bool is_allowed_host_header(const std::string& host_hdr, const std::string& bind_host) {
    std::string h = to_lower_ascii(trim_ascii(host_hdr));
    if (h.empty()) return true; // HTTP/1.0 or local minimal probes without Host header
    if (!h.empty() && h[0] == '[') {
        size_t rb = h.find(']');
        if (rb == std::string::npos) return false;
        h = h.substr(1, rb - 1);
    } else {
        size_t colon = h.find(':');
        if (colon != std::string::npos) h.resize(colon);
    }
    std::string bh = to_lower_ascii(bind_host);
    return (h == "127.0.0.1" || h == "localhost" || h == "::1" || h == bh);
}

// Holds incomplete trailing 1-3 UTF-8 bytes so every SSE delta is valid UTF-8 (R7 / SR1-12)!
struct Utf8StreamBuffer {
    std::string pending;

    std::string push(const std::string& chunk) {
        if (chunk.empty()) return "";
        pending += chunk;
        size_t n = pending.size();
        size_t scan_lo = (n > 4) ? (n - 4) : 0;
        size_t cut = n;
        for (size_t i = n; i > scan_lo; --i) {
            unsigned char b = (unsigned char)pending[i - 1];
            if ((b & 0xC0u) == 0x80u) continue; // continuation byte
            size_t need = 1;
            if      (b < 0x80u)            need = 1;
            else if ((b & 0xE0u) == 0xC0u) need = 2;
            else if ((b & 0xF0u) == 0xE0u) need = 3;
            else if ((b & 0xF8u) == 0xF0u) need = 4;
            else                           need = 1; // invalid lead byte: emit immediately
            size_t have = n - (i - 1);
            if (have < need) cut = i - 1;
            break;
        }
        if (cut == 0) return "";
        if (cut == n) {
            std::string out = std::move(pending);
            pending.clear();
            return out;
        }
        std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        return out;
    }

    std::string flush() {
        std::string out = std::move(pending);
        pending.clear();
        return out;
    }
};

// Order-insensitive canonical form (all levels sorted) used only to compare argument objects semantically.
std::string json_args_canonical(const std::string& args_json) {
    JsonVal v;
    JsonParser p(args_json);
    if (p.parse(v)) return json_stringify(v);
    return args_json;
}

std::string trim_ws(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

// =========================================================================
// v4.1 (R2-05): incremental scanner over the model's {"name":...,"args":{...}} tool payload.
// Yields the raw bytes of the top-level "args"/"arguments" object as they are generated, so a large
// file write streams to the client as OpenAI function.arguments deltas instead of minutes of silence.
// =========================================================================
struct ToolArgStreamer {
    enum Mode { SCAN, IN_ARGS, DONE, FALLBACK };
    Mode mode = SCAN;
    int depth = 0;
    bool in_str = false, esc = false;
    bool expect_key = false, reading_key = false, after_key = false, want_value = false;
    std::string key;
    std::string emitted; // everything yielded so far (== complete arguments text once mode == DONE)

    std::string feed(const std::string& piece) {
        std::string out;
        for (char c : piece) {
            if (mode == DONE || mode == FALLBACK) break;
            if (mode == IN_ARGS) {
                out.push_back(c);
                if (in_str) {
                    if (esc) esc = false;
                    else if (c == '\\') esc = true;
                    else if (c == '"') in_str = false;
                } else if (c == '"') {
                    in_str = true;
                } else if (c == '{' || c == '[') {
                    ++depth;
                } else if (c == '}' || c == ']') {
                    if (--depth <= 1) mode = DONE;
                }
                continue;
            }
            // SCAN: locate the top-level "args"/"arguments" key and the first byte of its value.
            if (in_str) {
                if (esc) { esc = false; if (reading_key) key.push_back(c); }
                else if (c == '\\') { esc = true; if (reading_key) key.push_back(c); }
                else if (c == '"') { in_str = false; if (reading_key) { reading_key = false; after_key = true; } }
                else if (reading_key) key.push_back(c);
                continue;
            }
            if (want_value) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
                want_value = false;
                if (c == '{') { mode = IN_ARGS; depth = 2; out.push_back(c); continue; }
                mode = FALLBACK; // string or scalar args: the end-of-call parser handles these
                break;
            }
            if (c == '"') {
                in_str = true;
                if (depth == 1 && expect_key) { reading_key = true; key.clear(); expect_key = false; }
                continue;
            }
            if (c == '{' || c == '[') { ++depth; if (depth == 1 && c == '{') expect_key = true; continue; }
            if (c == '}' || c == ']') { --depth; continue; }
            if (depth == 1 && c == ',') { expect_key = true; after_key = false; continue; }
            if (depth == 1 && c == ':' && after_key) {
                after_key = false;
                if (key == "args" || key == "arguments") want_value = true;
                continue;
            }
        }
        emitted += out;
        return out;
    }
};

// =========================================================================
// v4.1 (R2-01): token-exact replay of assistant turns.
// The engine reuses its KV cache only if the next prompt starts with EXACTLY the tokens it already
// holds. Re-rendering an assistant turn from text (re-serialised tool JSON, escapes, non-canonical
// BPE splits sampled at T=1) can differ by one token and throw away a 10K-token turn. So every turn
// this server generates is kept as its exact token sequence and spliced back when the client returns
// that turn unchanged (same tool_call ids, names, arguments, reasoning and text).
// =========================================================================
struct TurnRecord {
    std::vector<std::string> call_ids;
    std::vector<std::string> call_names;
    std::vector<std::string> call_args_canon;
    std::string reasoning; // trimmed
    std::string content;   // trimmed
    std::vector<int32_t> tokens; // generation prompt + generated tokens + closing tokens
};

class TurnRecordStore {
public:
    static constexpr size_t kMaxTokens  = 8u * 1024u * 1024u; // ~32 MB of token IDs
    static constexpr size_t kMaxRecords = 8192;

    void add(TurnRecord&& r) {
        std::string k = key_for(r.call_ids, r.reasoning, r.content);
        auto it = by_key_.find(k);
        if (it != by_key_.end()) {
            total_tokens_ -= it->second->tokens.size();
            recs_.erase(it->second);
            by_key_.erase(it);
        }
        total_tokens_ += r.tokens.size();
        recs_.push_back(std::move(r));
        by_key_[k] = std::prev(recs_.end());
        while (!recs_.empty() && (total_tokens_ > kMaxTokens || recs_.size() > kMaxRecords)) {
            auto& old = recs_.front();
            auto oit = by_key_.find(key_for(old.call_ids, old.reasoning, old.content));
            if (oit != by_key_.end() && oit->second == recs_.begin()) by_key_.erase(oit);
            total_tokens_ -= old.tokens.size();
            recs_.pop_front();
        }
    }

    const TurnRecord* match(const ChatMessage& m) const {
        if (m.role != "assistant") return nullptr;
        std::vector<std::string> ids;
        for (const auto& tc : m.tool_calls) ids.push_back(tc.id);
        const std::string rs = trim_ws(m.reasoning_content);
        const std::string ct = trim_ws(m.content);
        auto it = by_key_.find(key_for(ids, rs, ct));
        if (it == by_key_.end()) return nullptr;
        const TurnRecord& r = *it->second;
        if (r.call_ids != ids || r.reasoning != rs || r.content != ct) return nullptr;
        for (size_t i = 0; i < m.tool_calls.size(); ++i) {
            if (r.call_names[i] != m.tool_calls[i].name) return nullptr;
            if (r.call_args_canon[i] != json_args_canonical(m.tool_calls[i].arguments_json)) return nullptr;
        }
        return &r;
    }

    size_t size() const { return recs_.size(); }

private:
    static std::string key_for(const std::vector<std::string>& ids, const std::string& reasoning, const std::string& content) {
        if (!ids.empty()) return "c:" + ids[0];
        return "h:" + std::to_string(std::hash<std::string>{}(reasoning + '\x1f' + content));
    }
    std::list<TurnRecord> recs_;
    std::unordered_map<std::string, std::list<TurnRecord>::iterator> by_key_;
    size_t total_tokens_ = 0;
};

bool send_all(SOCKET sock, const std::string& data) {
    const char* p = data.data();
    size_t rem = data.size();
    while (rem > 0) {
        int sent = send(sock, p, (int)std::min<size_t>(rem, 65536), 0);
        if (sent <= 0) return false;
        p += sent;
        rem -= sent;
    }
    return true;
}

// Non-blocking <1 us Winsock check: immediately detects TCP FIN/RST if client aborted or timed out!
bool is_client_alive(SOCKET sock) {
    fd_set rfds, efds;
    FD_ZERO(&rfds);
    FD_ZERO(&efds);
    FD_SET(sock, &rfds);
    FD_SET(sock, &efds);
    timeval tv{0, 0};
    int rc = select(0, &rfds, nullptr, &efds, &tv);
    if (rc < 0) return false;
    if (rc > 0) {
        if (FD_ISSET(sock, &efds)) return false;
        if (FD_ISSET(sock, &rfds)) {
            char b = 0;
            int r = recv(sock, &b, 1, MSG_PEEK);
            if (r == 0) return false; // Peer cleanly closed TCP connection (FIN)
            if (r == SOCKET_ERROR && WSAGetLastError() != WSAEWOULDBLOCK) return false;
        }
    }
    return true;
}

const char* http_status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 503: return "Service Unavailable";
        default:  return "Error";
    }
}

void send_json(SOCKET sock, int code, const std::string& json_body, const std::string& cors_origin = "") {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << " " << http_status_text(code) << "\r\n"
        << "Content-Type: application/json\r\n";
    if (!cors_origin.empty()) {
        oss << "Access-Control-Allow-Origin: " << cors_origin << "\r\n"
            << "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
            << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            << "Vary: Origin\r\n";
    }
    oss << "Content-Length: " << json_body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << json_body;
    send_all(sock, oss.str());
}

bool decode_chunked(const std::string& raw, std::string& out) {
    out.clear();
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t crlf = raw.find("\r\n", pos);
        if (crlf == std::string::npos) return false;
        std::string hex = raw.substr(pos, crlf - pos);
        size_t semi = hex.find(';');
        if (semi != std::string::npos) hex.resize(semi);
        size_t chunk_sz = 0;
        try { chunk_sz = std::stoull(hex, nullptr, 16); } catch (...) { return false; }
        pos = crlf + 2;
        if (chunk_sz == 0) return true;
        if (pos + chunk_sz + 2 > raw.size()) return false;
        out.append(raw, pos, chunk_sz);
        pos += chunk_sz + 2;
    }
    return false;
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::string host_hdr;
    std::string auth_hdr;
    int error_code = 0;
};

bool read_http_request(SOCKET cli, HttpRequest& out) {
    out = HttpRequest{};
    // Socket recv/send timeouts (R9 / SR1-10): prevent stalled clients from ever hanging the server!
    DWORD rcv_to = 10000; // 10 seconds
    DWORD snd_to = 30000; // 30 seconds
    setsockopt(cli, SOL_SOCKET, SO_RCVTIMEO, (const char*)&rcv_to, sizeof(rcv_to));
    setsockopt(cli, SOL_SOCKET, SO_SNDTIMEO, (const char*)&snd_to, sizeof(snd_to));

    const auto t_start = std::chrono::steady_clock::now();
    std::string req;
    req.reserve(16384);
    char buf[16384];
    size_t header_end = std::string::npos;
    size_t content_len = 0;
    bool has_cl = false;
    bool is_chunked = false;
    bool sent_100 = false;

    while (true) {
        if (header_end != std::string::npos) {
            if (!is_chunked && req.size() >= header_end + 4 + content_len) break;
            if (is_chunked) {
                std::string decoded;
                if (decode_chunked(req.substr(header_end + 4), decoded)) {
                    out.body = std::move(decoded);
                    break;
                }
            }
        } else {
            double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (elapsed_s > 10.0) return false; // 10s header deadline (defeats slow-loris!)
        }

        int n = recv(cli, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, n);
        if (req.size() > 64ull * 1024 * 1024) { out.error_code = 400; return false; }

        if (header_end == std::string::npos) {
            header_end = req.find("\r\n\r\n");
            if (header_end == std::string::npos && req.size() > 65536) {
                out.error_code = 400;
                return false;
            }
            if (header_end != std::string::npos) {
                // Strict line-by-line header parser (R2 + SR1-11: rejects conflicting CL / Transfer-Encoding!)
                size_t line_start = req.find("\r\n");
                if (line_start == std::string::npos || line_start > header_end) {
                    out.error_code = 400;
                    return false;
                }
                size_t pos = line_start + 2;
                while (pos < header_end) {
                    size_t next_crlf = req.find("\r\n", pos);
                    if (next_crlf == std::string::npos || next_crlf > header_end) next_crlf = header_end;
                    std::string line = req.substr(pos, next_crlf - pos);
                    pos = next_crlf + 2;
                    size_t colon = line.find(':');
                    if (colon == std::string::npos) continue;
                    std::string k = to_lower_ascii(trim_ascii(line.substr(0, colon)));
                    std::string v = trim_ascii(line.substr(colon + 1));
                    if (k == "host") {
                        out.host_hdr = v;
                    } else if (k == "authorization") {
                        out.auth_hdr = v;
                    } else if (k == "transfer-encoding") {
                        if (to_lower_ascii(v).find("chunked") != std::string::npos) is_chunked = true;
                    } else if (k == "content-length") {
                        if (v.empty()) { out.error_code = 400; return false; }
                        for (char ch : v) {
                            if (ch < '0' || ch > '9') { out.error_code = 400; return false; }
                        }
                        size_t parsed_cl = 0;
                        try { parsed_cl = std::stoull(v); } catch (...) { out.error_code = 400; return false; }
                        if (has_cl && parsed_cl != content_len) { out.error_code = 400; return false; }
                        content_len = parsed_cl;
                        has_cl = true;
                    } else if (k == "expect") {
                        if (!sent_100 && to_lower_ascii(v).find("100-continue") != std::string::npos) {
                            send_all(cli, "HTTP/1.1 100 Continue\r\n\r\n");
                            sent_100 = true;
                        }
                    }
                }
                if (is_chunked && has_cl) {
                    out.error_code = 400; // RFC 7230: reject Transfer-Encoding + Content-Length smuggling!
                    return false;
                }
                if (has_cl && content_len > 64ull * 1024 * 1024) {
                    out.error_code = 400;
                    return false;
                }
            }
        }
    }

    if (header_end == std::string::npos) return false;
    std::string first_line = req.substr(0, req.find("\r\n"));
    {
        std::istringstream iss(first_line);
        iss >> out.method >> out.path;
    }
    if (out.method.empty() || out.path.empty()) { out.error_code = 400; return false; }
    size_t q = out.path.find('?');
    if (q != std::string::npos) out.path.resize(q);

    if (!is_chunked) {
        if (req.size() >= header_end + 4) {
            size_t avail = req.size() - (header_end + 4);
            if (avail < content_len) return false; // truncated/timed-out request body
            out.body = req.substr(header_end + 4, content_len);
        } else {
            if (content_len > 0) return false;
            out.body.clear();
        }
    }
    return true;
}

// =========================================================================
// OpenAI Request -> Inkling-2 Messages + Canonical Tool Specs Extractor
// =========================================================================
std::string extract_content_text(const JsonVal* cv) {
    if (!cv) return "";
    if (cv->is_str()) return cv->str;
    if (cv->is_arr()) {
        std::string out;
        for (const auto& part : cv->arr) {
            if (part.is_str()) {
                out += part.str;
            } else if (part.is_obj()) {
                std::string ptype = part.get_str("type", "text");
                if (ptype == "text" || ptype == "input_text") {
                    out += part.get_str("text", "");
                }
            }
        }
        return out;
    }
    return "";
}

std::string extract_image_url_from_part(const JsonVal& part) {
    if (!part.is_obj()) return "";
    std::string ptype = part.get_str("type", "");
    if (ptype == "image_url" || ptype == "input_image") {
        const auto* iu = part.get("image_url");
        if (iu && iu->is_obj()) return iu->get_str("url", "");
        if (iu && iu->is_str()) return iu->str;
        std::string u = part.get_str("url", "");
        if (!u.empty()) return u;
    } else if (ptype == "image") {
        std::string u = part.get_str("image", part.get_str("url", ""));
        if (!u.empty()) return u;
        if (const auto* src = part.get("source"); src && src->is_obj()) {
            std::string data = src->get_str("data", "");
            std::string mt   = src->get_str("media_type", "image/png");
            if (!data.empty()) return "data:" + mt + ";base64," + data;
        }
    }
    return "";
}

std::vector<ChatMessage> parse_openai_messages(const JsonVal& root, std::vector<std::string>* out_image_urls = nullptr) {
    std::vector<ChatMessage> out;
    const auto* arr = root.get("messages");
    if (!arr || !arr->is_arr()) return out;

    for (const auto& item : arr->arr) {
        if (!item.is_obj()) continue;
        ChatMessage m;
        m.role = item.get_str("role", "user");
        if (m.role == "developer") m.role = "system"; // OpenAI o-series alias -> system
        m.name = item.get_str("name", "");
        m.tool_call_id = item.get_str("tool_call_id", "");
        const auto* cv = item.get("content");
        m.content = extract_content_text(cv);
        // Inkling must get its previous thinking back (the official template re-inserts it into every
        // assistant turn). Accept the common client spellings so no harness silently drops it (R2-03b).
        m.reasoning_content = item.get_str("reasoning_content", "");
        if (m.reasoning_content.empty()) m.reasoning_content = item.get_str("reasoning", "");
        if (m.reasoning_content.empty()) m.reasoning_content = item.get_str("thinking", "");

        // Preserve ordered multimodal parts for user & tool messages when images are present
        if (cv && cv->is_arr() && (m.role == "user" || m.role == "tool")) {
            bool has_img = false;
            std::vector<ChatContentPart> parts;
            for (const auto& part : cv->arr) {
                if (part.is_str()) {
                    if (!part.str.empty()) {
                        ChatContentPart cp; cp.is_image = false; cp.text = part.str;
                        parts.push_back(std::move(cp));
                    }
                } else if (part.is_obj()) {
                    std::string img_url = extract_image_url_from_part(part);
                    if (!img_url.empty()) {
                        has_img = true;
                        ChatContentPart cp; cp.is_image = true; cp.image_url = img_url;
                        parts.push_back(std::move(cp));
                        if (out_image_urls) out_image_urls->push_back(std::move(img_url));
                    } else {
                        std::string ptype = part.get_str("type", "text");
                        if (ptype == "text" || ptype == "input_text") {
                            std::string t = part.get_str("text", "");
                            if (!t.empty()) {
                                ChatContentPart cp; cp.is_image = false; cp.text = std::move(t);
                                parts.push_back(std::move(cp));
                            }
                        }
                    }
                }
            }
            if (has_img) {
                m.parts = std::move(parts);
            }
        }

        if (const auto* tcs = item.get("tool_calls"); tcs && tcs->is_arr()) {
            for (const auto& tc : tcs->arr) {
                if (!tc.is_obj()) continue;
                ToolCallSpec spec;
                spec.id = tc.get_str("id", "");
                const auto* fn = tc.get("function");
                if (fn && fn->is_obj()) {
                    spec.name = fn->get_str("name", "");
                    const auto* av = fn->get("arguments");
                    if (av && av->is_str()) {
                        // Render like the official template: sorted top-level keys, nested values in original order.
                        JsonVal parsed_args;
                        JsonParser ap(av->str);
                        if (ap.parse(parsed_args) && parsed_args.is_obj()) {
                            spec.arguments_json = json_args_template(parsed_args);
                        } else {
                            spec.arguments_json = av->str.empty() ? "{}" : av->str;
                        }
                    } else if (av && av->is_obj()) {
                        spec.arguments_json = json_args_template(*av);
                    } else {
                        spec.arguments_json = "{}";
                    }
                }
                if (!spec.name.empty()) m.tool_calls.push_back(std::move(spec));
            }
        }
        out.push_back(std::move(m));
    }
    return out;
}

std::string build_canonical_tools_json(const JsonVal& root) {
    const auto* tarr = root.get("tools");
    if (!tarr || !tarr->is_arr() || tarr->arr.empty()) return "";

    JsonVal specs;
    specs.type = JsonVal::J_ARR;
    for (const auto& t : tarr->arr) {
        if (!t.is_obj()) continue;
        const JsonVal* fn = t.get("function");
        if (!fn || !fn->is_obj()) fn = &t;
        std::string name = fn->get_str("name", "");
        if (name.empty()) continue;

        JsonVal spec;
        spec.type = JsonVal::J_OBJ;
        {
            JsonVal d; d.type = JsonVal::J_STR; d.str = fn->get_str("description", "");
            spec.obj["description"] = std::move(d);
        }
        {
            JsonVal nm; nm.type = JsonVal::J_STR; nm.str = name;
            spec.obj["name"] = std::move(nm);
        }
        {
            const auto* params = fn->get("parameters");
            if (params && params->is_obj()) spec.obj["parameters"] = *params;
            else { JsonVal empty_obj; empty_obj.type = JsonVal::J_OBJ; spec.obj["parameters"] = std::move(empty_obj); }
        }
        {
            JsonVal tp; tp.type = JsonVal::J_STR; tp.str = t.get_str("type", "function");
            spec.obj["type"] = std::move(tp);
        }
        specs.arr.push_back(std::move(spec));
    }
    if (specs.arr.empty()) return "";
    return json_stringify(specs);
}

// Parse Inkling-2's {"name":"tool_name","args":{...}} tool invocation payload
bool parse_inkling_tool_invocation(const std::string& raw_json, const std::string& header_hint,
                                   std::string& out_name, std::string& out_args_json) {
    JsonVal v;
    JsonParser p(raw_json);
    if (p.parse(v) && v.is_obj()) {
        out_name = v.get_str("name", header_hint);
        const auto* av = v.get("args");
        if (!av) av = v.get("arguments");
        if (av && av->is_obj()) out_args_json = json_args_template(*av);
        else if (av && av->is_str()) out_args_json = av->str;
        else out_args_json = "{}";
        return !out_name.empty();
    }
    if (!header_hint.empty()) {
        out_name = header_hint;
        out_args_json = raw_json.empty() ? "{}" : raw_json;
        return true;
    }
    return false;
}

} // namespace

OpenAIServer::OpenAIServer(InklingEngine& engine, const ServerConfig& cfg)
    : engine_(engine), cfg_(cfg) {}

OpenAIServer::OpenAIServer(InklingEngine& engine, const std::string& host, int port, const std::string& default_effort)
    : engine_(engine) {
    cfg_.host = host;
    cfg_.port = port;
    cfg_.default_effort = default_effort;
}

bool OpenAIServer::run() {
    if (cfg_.require_auth && cfg_.api_key.empty()) {
        cfg_.api_key = generate_csprng_hex_token(16);
        std::printf("[Server] Generated CSPRNG session Bearer token: %s\n", cfg_.api_key.c_str());
    }

    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::fprintf(stderr, "[Server] WSAStartup failed\n");
        return false;
    }

    SOCKET srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (srv == INVALID_SOCKET) { WSACleanup(); return false; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)cfg_.port);
    inet_pton(AF_INET, cfg_.host.c_str(), &addr.sin_addr);

    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        std::fprintf(stderr, "[Server] Failed to bind %s:%d\n", cfg_.host.c_str(), cfg_.port);
        closesocket(srv);
        WSACleanup();
        return false;
    }
    listen(srv, 16);

    std::printf("[Server] OpenAI-compatible InkEngine v4.2 listening on http://%s:%d (effort=%s, vision=%s, auth=%s, cors=%s)\n",
                cfg_.host.c_str(), cfg_.port, cfg_.default_effort.c_str(),
                engine_.has_vision() ? "ON" : "OFF",
                cfg_.require_auth ? "Bearer" : "OFF",
                cfg_.cors_origin.empty() ? "OFF" : cfg_.cors_origin.c_str());
    std::printf("         Endpoints: /v1/chat/completions (vision+tools+thinking+SSE+KV-cache), /v1/completions, /v1/models, /health, /stats, /shutdown\n");
    std::fflush(stdout);

    std::atomic<uint64_t> req_counter{1};
    std::atomic<bool> stop_flag{false};
    std::atomic<int> active_conns{0};
    std::mutex gen_mu;
    TurnRecordStore turn_records;                              // guarded by gen_mu (R2-01)
    const std::string call_nonce = generate_csprng_hex_token(4); // tool_call ids stay unique across restarts

    auto handle_client = [&](SOCKET cli) {
        struct ConnGuard {
            SOCKET s;
            std::atomic<int>& cnt;
            ~ConnGuard() { closesocket(s); cnt.fetch_sub(1, std::memory_order_relaxed); }
        } guard{cli, active_conns};

        int nodelay = 1;
        setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));

        HttpRequest req;
        if (!read_http_request(cli, req)) {
            if (req.error_code == 400) {
                send_json(cli, 400, "{\"error\":{\"message\":\"Malformed HTTP request\"}}", cfg_.cors_origin);
            }
            return;
        }

        // Security (R2 / SR1-09): Block DNS rebinding via Host header validation
        if (!is_allowed_host_header(req.host_hdr, cfg_.host)) {
            send_json(cli, 403, "{\"error\":{\"message\":\"Forbidden Host header\"}}");
            return;
        }

        // Security (R2): Reject cross-origin preflights unless an explicit --cors-origin was configured
        if (req.method == "OPTIONS") {
            if (!cfg_.cors_origin.empty()) send_json(cli, 200, "{}", cfg_.cors_origin);
            else                           send_json(cli, 403, "{\"error\":{\"message\":\"CORS disabled\"}}");
            return;
        }

        // Security (R2): Require Authorization: Bearer <token> on ALL endpoints (including /shutdown & /stats)
        if (cfg_.require_auth) {
            bool auth_ok = false;
            std::string ah = trim_ascii(req.auth_hdr);
            if (ah.size() > 7 && to_lower_ascii(ah.substr(0, 7)) == "bearer ") {
                std::string provided = trim_ascii(ah.substr(7));
                auth_ok = const_time_eq(provided, cfg_.api_key);
            }
            if (!auth_ok) {
                send_json(cli, 401,
                    "{\"error\":{\"message\":\"Missing or invalid Authorization Bearer token\",\"type\":\"authentication_error\"}}",
                    cfg_.cors_origin);
                return;
            }
        }

        const std::string& method = req.method;
        const std::string& path   = req.path;
        const std::string& body   = req.body;

        // Lightweight endpoints NEVER wait behind active GPU generation or stalled sockets (R9)!
        if (path == "/health") {
            auto ks = engine_.kv_stats();
            char hb[384];
            std::snprintf(hb, sizeof(hb),
                "{\"status\":\"ok\",\"engine\":\"InkEngine\",\"version\":\"3.3\",\"model\":\"Inkling-2-Mini\","
                "\"max_ctx\":%d,\"effort\":\"%s\",\"vision\":%s,\"auth\":%s}",
                ks.max_ctx, json_escape(cfg_.default_effort).c_str(),
                engine_.has_vision() ? "true" : "false",
                cfg_.require_auth ? "true" : "false");
            send_json(cli, 200, hb, cfg_.cors_origin);
            return;
        }
        if (path == "/shutdown" && method == "POST") {
            send_json(cli, 200, "{\"status\":\"shutting_down\"}", cfg_.cors_origin);
            std::printf("[Server] Received authenticated /shutdown request. Stopping cleanly...\n");
            std::fflush(stdout);
            stop_flag.store(true, std::memory_order_release);
            closesocket(srv); // Wake main accept() immediately!
            return;
        }
        if (path == "/v1/models") {
            send_json(cli, 200,
                "{\"object\":\"list\",\"data\":[{\"id\":\"Inkling-2-Mini\",\"object\":\"model\",\"owned_by\":\"InkEngine\"}]}",
                cfg_.cors_origin);
            return;
        }
        if (path == "/stats") {
            auto es = engine_.expert_stats();
            auto ks = engine_.kv_stats();
            double hr = es.total_lookups ? (100.0 * es.vram_hits / es.total_lookups) : 0.0;
            double chr = es.core_lookups ? (100.0 * es.core_vram_hits / es.core_lookups) : 0.0;
            double ohr = es.outer_lookups ? (100.0 * es.outer_vram_hits / es.outer_lookups) : 0.0;
            char jbuf[1024];
            std::snprintf(jbuf, sizeof(jbuf),
                "{\"expert_vram_mb\":%.1f,\"expert_ram_mb\":%.1f,\"vram_hit_pct\":%.2f,"
                "\"core_hit_pct\":%.2f,\"outer_hit_pct\":%.2f,\"ssd_expert_reads\":%llu,"
                "\"dma_promotions\":%llu,\"kv_max_ctx\":%d,\"kv_cur_pos\":%d,"
                "\"kv_swa_vram_mb\":%.1f,\"kv_global_vram_mb\":%.1f,\"kv_quest_vram_mb\":%.1f,"
                "\"kv_ram_mb\":%.1f,\"kv_ssd_mb\":%.1f,\"kv_ssd_reads\":%llu,\"kv_ssd_writes\":%llu}",
                es.vram_mb, es.ram_mb, hr, chr, ohr,
                (unsigned long long)es.ssd_reads, (unsigned long long)es.dma_promotions,
                ks.max_ctx, ks.cur_pos, ks.swa_vram_mb, ks.global_vram_mb, ks.quest_summary_vram_mb,
                ks.ram_kv_mb, ks.ssd_kv_mb,
                (unsigned long long)ks.ssd_page_reads, (unsigned long long)ks.ssd_page_writes);
            send_json(cli, 200, jbuf, cfg_.cors_origin);
            return;
        }
        if (path != "/v1/chat/completions" && path != "/v1/completions") {
            send_json(cli, 404, "{\"error\":\"Not found\"}", cfg_.cors_origin);
            return;
        }

        JsonVal root;
        JsonParser jp(body);
        if (!jp.parse(root) || !root.is_obj()) {
            send_json(cli, 400, "{\"error\":{\"message\":\"Invalid JSON request body\"}}", cfg_.cors_origin);
            return;
        }

        bool is_chat = (path == "/v1/chat/completions");
        bool stream  = root.get_bool("stream", false);
        bool include_usage = false;
        if (const auto* so = root.get("stream_options"); so && so->is_obj()) {
            include_usage = so->get_bool("include_usage", false);
        }

        // Official Inkling-2-Mini recommended defaults: temp=1.0, top_p=1.0, top_k=0, min_p=0.0
        SamplingParams sp;
        sp.temperature = (float)root.get_num("temperature", 1.0);
        sp.top_p       = (float)root.get_num("top_p", 1.0);
        sp.top_k       = (int)root.get_num("top_k", 0);
        sp.min_p       = (float)root.get_num("min_p", 0.0);
        int max_tok    = (int)root.get_num("max_completion_tokens", root.get_num("max_tokens", 4096));
        sp.max_tokens  = std::max(1, std::min(max_tok, 32768));
        uint64_t def_seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                            ^ (0x9E3779B97F4A7C15ULL * (req_counter.load() + 1));
        if (def_seed == 0) def_seed = 42;
        sp.seed        = (root.get("seed") && root.get("seed")->is_num())
                            ? (uint64_t)root.get_num("seed", (double)def_seed)
                            : def_seed;

        std::string effort = root.get_str("reasoning_effort", cfg_.default_effort);
        std::string tools_json;
        std::string prompt_text;
        std::vector<std::string> image_urls;
        size_t static_prefix_bytes = 0;
        size_t tools_prefix_bytes = 0;
        size_t full_sys_prefix_bytes = 0;
        std::vector<ChatMessage> msgs;
        std::vector<std::pair<size_t, size_t>> msg_ranges;
        size_t gen_prompt_bytes = 0;

        if (is_chat) {
            msgs = parse_openai_messages(root, &image_urls);
            tools_json = build_canonical_tools_json(root);
            prompt_text = engine_.tokenizer().format_chat(msgs, effort, true, tools_json, &static_prefix_bytes, &tools_prefix_bytes,
                                                          &full_sys_prefix_bytes, &msg_ranges, &gen_prompt_bytes);
        } else {
            prompt_text = root.get_str("prompt", "");
        }

        // Serialize GPU/KV engine execution (while /health and /stats remain completely non-blocking!)
        std::lock_guard<std::mutex> glk(gen_mu);
        if (stop_flag.load(std::memory_order_acquire)) {
            send_json(cli, 503, "{\"error\":{\"message\":\"Server shutting down\"}}", cfg_.cors_origin);
            return;
        }

        std::vector<EncodedImage> encoded_imgs;
        MultimodalEmbeddings mm_embd;
        if (!image_urls.empty()) {
            if (!engine_.has_vision()) {
                send_json(cli, 400, "{\"error\":{\"message\":\"Vision encoder not loaded (place mmproj-F16.gguf next to the model GGUF or pass --mmproj)\"}}", cfg_.cors_origin);
                return;
            }
            encoded_imgs.reserve(image_urls.size());
            bool img_ok = true;
            for (size_t ii = 0; ii < image_urls.size(); ++ii) {
                EncodedImage enc;
                // Security (R2 / SR1-09): allow_local_files = false over HTTP (only data: base64 allowed!)
                if (!engine_.encode_image_input(image_urls[ii], enc, /*allow_local_files=*/false)) {
                    img_ok = false;
                    break;
                }
                encoded_imgs.push_back(std::move(enc));
            }
            if (!img_ok) {
                send_json(cli, 400, "{\"error\":{\"message\":\"Failed to decode input image (over HTTP only data:image/...;base64,... URIs are permitted)\"}}", cfg_.cors_origin);
                return;
            }
        }

        int static_prefix_tokens = 0;
        int tools_prefix_tokens = 0;
        int full_sys_prefix_tokens = 0;
        if (tools_prefix_bytes > 0 && tools_prefix_bytes < prompt_text.size()) {
            auto tp_ids = engine_.tokenizer().encode(prompt_text.substr(0, tools_prefix_bytes), true);
            tools_prefix_tokens = (int)tp_ids.size();
        }
        if (static_prefix_bytes > 0 && static_prefix_bytes < prompt_text.size()) {
            auto sp_ids = engine_.tokenizer().encode(prompt_text.substr(0, static_prefix_bytes), true);
            static_prefix_tokens = (int)sp_ids.size();
        }
        if (full_sys_prefix_bytes > 0 && full_sys_prefix_bytes < prompt_text.size()) {
            auto fp_ids = engine_.tokenizer().encode(prompt_text.substr(0, full_sys_prefix_bytes), true);
            full_sys_prefix_tokens = (int)fp_ids.size();
        }
        // R2-01: splice the exact generated tokens of every assistant turn we still hold, so the prompt
        // reproduces the KV-resident sequence token-for-token. Ranges start and end on special tokens,
        // and the tokenizer never merges across special tokens, so piecewise encoding is exact.
        std::vector<int32_t> raw_prompt_ids;
        int replayed_turns = 0, replay_candidates = 0;
        {
            size_t cur = 0;
            for (size_t mi = 0; mi < msgs.size() && mi < msg_ranges.size(); ++mi) {
                if (msgs[mi].role != "assistant") continue;
                ++replay_candidates;
                const TurnRecord* rec = turn_records.match(msgs[mi]);
                const auto rg = msg_ranges[mi];
                if (!rec || rg.first < cur || rg.second <= rg.first) continue;
                auto seg = engine_.tokenizer().encode(prompt_text.substr(cur, rg.first - cur), true);
                raw_prompt_ids.insert(raw_prompt_ids.end(), seg.begin(), seg.end());
                raw_prompt_ids.insert(raw_prompt_ids.end(), rec->tokens.begin(), rec->tokens.end());
                cur = rg.second;
                ++replayed_turns;
            }
            auto tail = engine_.tokenizer().encode(prompt_text.substr(cur), true);
            raw_prompt_ids.insert(raw_prompt_ids.end(), tail.begin(), tail.end());
        }
        std::vector<int32_t> prompt_ids;
        const MultimodalEmbeddings* mm_ptr = nullptr;
        if (!encoded_imgs.empty()) {
            InklingEngine::expand_multimodal_tokens(
                raw_prompt_ids, engine_.tokenizer().image_placeholder_id(), encoded_imgs, prompt_ids, mm_embd);
            mm_ptr = &mm_embd;
        } else {
            prompt_ids = std::move(raw_prompt_ids);
        }

        // Context overflow protection (R3 / SR1-02): return standard OpenAI context_length_exceeded so OpenCode compacts!
        int max_ctx = engine_.max_ctx();
        if ((int)prompt_ids.size() >= max_ctx) {
            char ebuf[384];
            std::snprintf(ebuf, sizeof(ebuf),
                "{\"error\":{\"message\":\"This model's maximum context length is %d tokens. However, your messages resulted in %zu tokens. Please reduce the length of the messages.\","
                "\"type\":\"invalid_request_error\",\"param\":\"messages\",\"code\":\"context_length_exceeded\"}}",
                max_ctx, prompt_ids.size());
            send_json(cli, 400, ebuf, cfg_.cors_origin);
            return;
        }
        int avail_gen = max_ctx - (int)prompt_ids.size();
        if (sp.max_tokens > avail_gen) sp.max_tokens = std::max(1, avail_gen);

        // v4.2 (R2-04) thinking budget: once a reply has thought for this many tokens, force <|end_message|>
        // so the model moves on to its answer or tool call instead of drafting whole files inside its thinking
        // until max_tokens runs out. Auto budget scales with effort and always leaves half of max_tokens to act.
        int think_budget = cfg_.think_budget;
        if (const auto* tb = root.get("thinking_budget"); tb && tb->is_num()) think_budget = (int)tb->num;
        else if (const auto* mt = root.get("max_thinking_tokens"); mt && mt->is_num()) think_budget = (int)mt->num;
        if (think_budget < 0) {
            std::string e = effort;
            for (char& ch : e) ch = (char)std::tolower((unsigned char)ch);
            double ev = 0.9;
            if      (e == "none" || e == "0" || e == "0.0") ev = 0.0;
            else if (e == "minimal") ev = 0.1;
            else if (e == "low") ev = 0.2;
            else if (e == "medium" || e == "med") ev = 0.7;
            else if (e == "high") ev = 0.9;
            else if (e == "xhigh" || e == "max") ev = 0.99;
            else { try { ev = std::stod(e); } catch (...) { ev = 0.9; } }
            think_budget = (ev <= 0.0) ? 1024 : (ev <= 0.1) ? 1024 : (ev <= 0.2) ? 2048 : (ev <= 0.7) ? 4096
                         : (ev <= 0.9) ? 8192 : 12288;
            think_budget = std::min(think_budget, std::max(256, sp.max_tokens / 2));
        }
        int think_tokens = 0;
        int think_force_at = think_budget;
        int think_forced = 0;

        uint64_t rid = req_counter.fetch_add(1);
        std::string chat_id = "chatcmpl-ink-" + std::to_string(rid);
        auto created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Inkling-2 output parser state machine: cleanly separates thinking, visible text, and tool invocations!
        bool starts_thinking = is_chat && (effort != "none" && effort != "0" && effort != "0.0");
        bool starts_text     = is_chat && !starts_thinking && tools_json.empty();
        OutState st = starts_thinking ? ST_THINKING : (starts_text ? ST_TEXT : ST_HEADER);

        std::string header_buf;
        std::string tool_buf;
        std::string full_text;
        std::string full_reasoning;
        Utf8StreamBuffer utf8_think;
        Utf8StreamBuffer utf8_text;
        std::vector<ToolCallSpec> emitted_tools;
        int tool_idx = 0;
        bool client_alive = true;
        // R2-05: live tool-call argument streaming state
        bool tool_streaming = false;
        std::string tool_stream_id, tool_stream_name;
        ToolArgStreamer arg_streamer;
        Utf8StreamBuffer utf8_args;
        // R2-10: decode-only expert-cache telemetry (snapshot taken at the first generated token)
        bool decode_started = false;
        ExpertCacheStats es_dec0{};
        auto make_call_id = [&](int idx) {
            return "call_ink_" + call_nonce + "_" + std::to_string(rid) + "_" + std::to_string(idx);
        };

        const auto& tok = engine_.tokenizer();
        const int32_t id_model = tok.msg_model_id();
        const int32_t id_think = tok.thinking_id();
        const int32_t id_text  = tok.text_id();
        const int32_t id_tjson = tok.invoke_tool_json_id();
        const int32_t id_ttext = tok.invoke_tool_text_id();
        const int32_t id_endm  = tok.end_msg_id();

        if (stream) {
            std::ostringstream hdr;
            hdr << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: text/event-stream\r\n"
                << "Cache-Control: no-cache\r\n";
            if (!cfg_.cors_origin.empty()) {
                hdr << "Access-Control-Allow-Origin: " << cfg_.cors_origin << "\r\n"
                    << "Vary: Origin\r\n";
            }
            hdr << "Connection: close\r\n\r\n";
            client_alive = send_all(cli, hdr.str());

            if (client_alive && is_chat) {
                // Opening role chunk (expected by OpenAI / Vercel AI SDK / OpenCode!)
                std::ostringstream first;
                first << "data: {\"id\":\"" << chat_id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
                      << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
                client_alive = send_all(cli, first.str());
            }
        }

        auto emit_sse_delta = [&](const std::string& delta_json) {
            if (!stream || !client_alive) return;
            std::ostringstream ev;
            if (is_chat) {
                ev << "data: {\"id\":\"" << chat_id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
                   << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"delta\":" << delta_json
                   << ",\"finish_reason\":null}]}\n\n";
            } else {
                ev << "data: {\"id\":\"" << chat_id << "\",\"object\":\"text_completion\",\"created\":" << created
                   << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"text\":" << delta_json
                   << ",\"finish_reason\":null}]}\n\n";
            }
            client_alive = send_all(cli, ev.str());
        };

        auto flush_utf8_buffers = [&]() {
            std::string rt = utf8_think.flush();
            if (!rt.empty()) emit_sse_delta("{\"reasoning_content\":\"" + json_escape(rt) + "\"}");
            std::string tx = utf8_text.flush();
            if (!tx.empty()) {
                if (is_chat) emit_sse_delta("{\"content\":\"" + json_escape(tx) + "\"}");
                else         emit_sse_delta("\"" + json_escape(tx) + "\"");
            }
        };

        // Real OpenAI SSE data-event heartbeat: resets client SSE chunkTimeout AND detects FIN/RST in <1 us!
        auto emit_sse_heartbeat = [&]() {
            if (!client_alive) return;
            if (!is_client_alive(cli)) { client_alive = false; return; }
            if (!stream) return;
            if (is_chat) emit_sse_delta("{}");
            else         client_alive = send_all(cli, ": keepalive\n\n");
        };

        auto emit_args_delta = [&](const std::string& frag) {
            if (frag.empty()) return;
            std::ostringstream d;
            d << "{\"tool_calls\":[{\"index\":" << tool_idx
              << ",\"function\":{\"arguments\":\"" << json_escape(frag) << "\"}}]}";
            emit_sse_delta(d.str());
        };

        // R2-05: open a streamed tool call as soon as the header names the tool, then forward the
        // raw "args" object bytes as they are generated (standard OpenAI chunked tool-call deltas).
        auto begin_tool_stream = [&]() {
            tool_streaming = false;
            if (!stream || !is_chat || st != ST_TOOL_JSON) return;
            std::string nm = trim_ws(header_buf);
            if (nm.empty()) return;
            tool_stream_id = make_call_id(tool_idx);
            tool_stream_name = nm;
            arg_streamer = ToolArgStreamer{};
            utf8_args = Utf8StreamBuffer{};
            std::ostringstream d;
            d << "{\"tool_calls\":[{\"index\":" << tool_idx
              << ",\"id\":\"" << tool_stream_id << "\",\"type\":\"function\",\"function\":{\"name\":\""
              << json_escape(nm) << "\",\"arguments\":\"\"}}]}";
            emit_sse_delta(d.str());
            tool_streaming = true;
        };

        auto flush_tool_call = [&]() {
            std::string tname, targs;
            const bool parsed = parse_inkling_tool_invocation(tool_buf, header_buf, tname, targs);
            if (tool_streaming) {
                emit_args_delta(utf8_args.flush());
                std::string sent = arg_streamer.emitted;
                if (arg_streamer.mode != ToolArgStreamer::DONE && sent.empty()) {
                    // The args object never started streaming (e.g. string-valued args): send it whole now.
                    sent = parsed ? targs : "{}";
                    emit_args_delta(sent);
                }
                ToolCallSpec tc;
                tc.id = tool_stream_id;
                tc.name = tool_stream_name;
                tc.arguments_json = sent;
                emitted_tools.push_back(std::move(tc));
                tool_idx++;
                tool_streaming = false;
            } else if (parsed) {
                ToolCallSpec tc;
                tc.id = make_call_id(tool_idx);
                tc.name = tname;
                tc.arguments_json = targs;
                if (stream && is_chat) {
                    std::ostringstream d;
                    d << "{\"tool_calls\":[{\"index\":" << tool_idx
                      << ",\"id\":\"" << tc.id << "\",\"type\":\"function\",\"function\":{\"name\":\""
                      << json_escape(tc.name) << "\",\"arguments\":\"" << json_escape(tc.arguments_json) << "\"}}]}";
                    emit_sse_delta(d.str());
                }
                emitted_tools.push_back(std::move(tc));
                tool_idx++;
            }
            tool_buf.clear();
            header_buf.clear();
        };

        std::printf("[HTTP] Incoming %s (id=%llu): prompt=%zu tok (full_sys=%d, static=%d, tools=%d, imgs=%zu) | stream=%d | effort=%s | has_tools=%d | T=%.2f p=%.2f k=%d mp=%.2f\n",
                    path.c_str(), (unsigned long long)rid, prompt_ids.size(), full_sys_prefix_tokens, static_prefix_tokens, tools_prefix_tokens,
                    encoded_imgs.size(), stream ? 1 : 0, effort.c_str(), tools_json.empty() ? 0 : 1,
                    sp.temperature, sp.top_p, sp.top_k, sp.min_p);
        std::fflush(stdout);

        int silent_toks = 0;
        int poll_toks = 0;

        auto res = engine_.generate(
            prompt_ids, sp,
            [&](int32_t tok_id, const std::string& piece) -> bool {
                if (!decode_started) { decode_started = true; es_dec0 = engine_.expert_stats(); }
                if (!client_alive) return false;
                if ((++poll_toks & 7) == 0 && !is_client_alive(cli)) {
                    client_alive = false;
                    return false;
                }
                if (!is_chat) {
                    full_text += piece;
                    if (stream && !piece.empty()) {
                        std::string valid = utf8_text.push(piece);
                        if (!valid.empty()) emit_sse_delta("\"" + json_escape(valid) + "\"");
                    }
                    return client_alive;
                }

                if (tok_id == id_model) {
                    flush_utf8_buffers();
                    st = ST_HEADER;
                    header_buf.clear();
                    return client_alive;
                }
                if (tok_id == id_think) {
                    flush_utf8_buffers();
                    st = ST_THINKING;
                    return client_alive;
                }
                if (tok_id == id_text) {
                    flush_utf8_buffers();
                    st = ST_TEXT;
                    return client_alive;
                }
                if (tok_id == id_tjson) {
                    flush_utf8_buffers();
                    st = ST_TOOL_JSON;
                    tool_buf.clear();
                    silent_toks = 0;
                    begin_tool_stream();
                    if (!tool_streaming) emit_sse_heartbeat();
                    return client_alive;
                }
                if (tok_id == id_ttext) {
                    flush_utf8_buffers();
                    st = ST_TOOL_TEXT;
                    tool_buf.clear();
                    silent_toks = 0;
                    emit_sse_heartbeat();
                    return client_alive;
                }
                if (tok_id == id_endm) {
                    flush_utf8_buffers();
                    if (st == ST_TOOL_JSON || st == ST_TOOL_TEXT) {
                        flush_tool_call();
                        silent_toks = 0;
                    }
                    st = ST_HEADER;
                    header_buf.clear();
                    return client_alive;
                }

                if (piece.empty()) return client_alive;

                if (st == ST_HEADER) {
                    header_buf += piece;
                    if (++silent_toks >= 16) { silent_toks = 0; emit_sse_heartbeat(); }
                } else if (st == ST_THINKING) {
                    full_reasoning += piece;
                    silent_toks = 0;
                    if (think_budget > 0 && ++think_tokens >= think_force_at) {
                        engine_.force_next_token(id_endm);
                        think_forced++;
                        think_force_at = think_tokens + 1024; // if it re-opens thinking, close it again later
                    }
                    if (stream) {
                        std::string valid = utf8_think.push(piece);
                        if (!valid.empty()) emit_sse_delta("{\"reasoning_content\":\"" + json_escape(valid) + "\"}");
                    }
                } else if (st == ST_TEXT) {
                    full_text += piece;
                    silent_toks = 0;
                    if (stream) {
                        std::string valid = utf8_text.push(piece);
                        if (!valid.empty()) emit_sse_delta("{\"content\":\"" + json_escape(valid) + "\"}");
                    }
                } else if (st == ST_TOOL_JSON || st == ST_TOOL_TEXT) {
                    tool_buf += piece;
                    if (tool_streaming && st == ST_TOOL_JSON) {
                        std::string frag = utf8_args.push(arg_streamer.feed(piece));
                        if (!frag.empty()) { emit_args_delta(frag); silent_toks = 0; }
                    }
                    // Keep SSE stream alive while buffering (or before the args object starts streaming)
                    if (++silent_toks >= 16) { silent_toks = 0; emit_sse_heartbeat(); }
                }
                return client_alive;
            },
            false, // enable_profile
            true,  // reuse_kv_prefix (incremental KV + full Global/SWA/ShortConv RAM checkpoint!)
            static_prefix_tokens,
            [&](int done, int total) -> bool {
                emit_sse_heartbeat();
                if (!client_alive) {
                    std::printf("[HTTP] Client disconnected mid-prefill (id=%llu at %d/%d tokens) - aborting immediately\n",
                                (unsigned long long)rid, done, total);
                    std::fflush(stdout);
                    return false;
                }
                if (total >= 256 && ((done % 256) == 0 || done == total)) {
                    std::printf("[HTTP] Prefill id=%llu: %d / %d tokens\n", (unsigned long long)rid, done, total);
                    std::fflush(stdout);
                }
                return client_alive;
            },
            tools_prefix_tokens,
            full_sys_prefix_tokens,
            mm_ptr);

        if (res.io_error) {
            std::string emsg = res.error_msg.empty() ? "Internal KV/expert storage I/O error" : res.error_msg;
            std::fprintf(stderr, "[HTTP] %s I/O ERROR (id=%llu): %s\n", path.c_str(), (unsigned long long)rid, emsg.c_str());
            if (!stream) {
                send_json(cli, 500, "{\"error\":{\"message\":\"" + json_escape(emsg) + "\",\"type\":\"server_error\",\"code\":\"storage_io_error\"}}", cfg_.cors_origin);
            } else if (client_alive) {
                std::ostringstream err_evt;
                err_evt << "data: {\"error\":{\"message\":\"" << json_escape(emsg) << "\",\"type\":\"server_error\",\"code\":\"storage_io_error\"}}\n\n"
                        << "data: [DONE]\n\n";
                send_all(cli, err_evt.str());
            }
            return;
        }

        if (client_alive) {
            flush_utf8_buffers();
            if ((st == ST_TOOL_JSON || st == ST_TOOL_TEXT) && (!tool_buf.empty() || tool_streaming)) {
                flush_tool_call();
            }
        }

        std::string finish_reason = !emitted_tools.empty()
            ? "tool_calls"
            : ((res.hit_max_ctx || res.gen_tokens >= sp.max_tokens) ? "length" : "stop");
        double tps = res.decode_ms > 0 ? (res.gen_tokens * 1000.0 / res.decode_ms) : 0.0;
        int new_prompt = res.prompt_tokens - res.cached_tokens;
        double ptps = (res.prefill_ms > 0 && new_prompt > 0) ? (new_prompt * 1000.0 / res.prefill_ms) : 0.0;

        // R2-01: remember this turn's exact tokens so the next request can replay them token-for-token.
        bool recorded_turn = false;
        if (is_chat && client_alive && !res.io_error && res.gen_tokens > 0) {
            std::vector<int32_t> closing;
            const bool hit_length = res.hit_max_ctx || res.gen_tokens >= sp.max_tokens;
            if (res.stop_token >= 0) {
                closing.push_back(res.stop_token);
            } else if (hit_length && (st == ST_THINKING || st == ST_TEXT)) {
                // Cut mid-block: the template closes it with <|end_message|> + end-of-turn.
                closing.push_back(id_endm);
                closing.push_back(tok.eos_id());
            } else if (hit_length && st == ST_HEADER && res.tokens.back() == id_endm) {
                closing.push_back(tok.eos_id());
            }
            if (!closing.empty()) {
                TurnRecord rec;
                for (const auto& tc : emitted_tools) {
                    rec.call_ids.push_back(tc.id);
                    rec.call_names.push_back(tc.name);
                    rec.call_args_canon.push_back(json_args_canonical(tc.arguments_json));
                }
                rec.reasoning = trim_ws(full_reasoning);
                rec.content = trim_ws(full_text);
                rec.tokens = tok.encode(prompt_text.substr(std::min(gen_prompt_bytes, prompt_text.size())), true);
                rec.tokens.insert(rec.tokens.end(), res.tokens.begin(), res.tokens.end());
                rec.tokens.insert(rec.tokens.end(), closing.begin(), closing.end());
                turn_records.add(std::move(rec));
                recorded_turn = true;
            }
        }

        // R2-10: per-request decode telemetry (decode-only expert hit rate, scalar vs speculative step cost)
        std::string telem;
        {
            char tb[512];
            ExpertCacheStats es1 = engine_.expert_stats();
            auto pct = [](uint64_t hit, uint64_t tot) { return tot ? 100.0 * (double)hit / (double)tot : 0.0; };
            const uint64_t dl = es1.total_lookups - es_dec0.total_lookups;
            const double scalar_ms_tok = res.scalar_steps ? res.scalar_ms / res.scalar_steps : 0.0;
            const double spec_ms_step  = res.spec_steps ? res.spec_ms / res.spec_steps : 0.0;
            const double spec_tok_step = res.spec_steps ? (double)(res.spec_accepted + res.spec_steps) / res.spec_steps : 0.0;
            const double spec_gain = (spec_ms_step > 0 && scalar_ms_tok > 0) ? scalar_ms_tok * spec_tok_step / spec_ms_step : 0.0;
            std::snprintf(tb, sizeof(tb),
                "\n         decode: scalar %.1f ms/tok x%d | spec %.1f ms/step x%d (%.2f tok/step, gain %.2fx) | "
                "draft cost %.2f | think %d tok (budget %d%s) | "
                "expert VRAM hit %.1f%% (core %.1f%%, outer %.1f%%) | replayed %d/%d turns%s",
                scalar_ms_tok, res.scalar_steps, spec_ms_step, res.spec_steps, spec_tok_step, spec_gain,
                res.spec_delta, think_tokens, think_budget, think_forced ? ", forced end" : "",
                decode_started && dl ? pct(es1.vram_hits - es_dec0.vram_hits, dl) : 0.0,
                pct(es1.core_vram_hits - es_dec0.core_vram_hits, es1.core_lookups - es_dec0.core_lookups),
                pct(es1.outer_vram_hits - es_dec0.outer_vram_hits, es1.outer_lookups - es_dec0.outer_lookups),
                replayed_turns, replay_candidates, recorded_turn ? " | turn recorded" : "");
            telem = tb;
        }

        if (stream) {
            if (client_alive) {
                std::ostringstream done;
                if (is_chat) {
                    done << "data: {\"id\":\"" << chat_id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
                         << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\""
                         << finish_reason << "\"}]}\n\n";
                    if (include_usage) {
                        done << "data: {\"id\":\"" << chat_id << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
                             << ",\"model\":\"Inkling-2-Mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":" << res.prompt_tokens
                             << ",\"completion_tokens\":" << res.gen_tokens
                             << ",\"total_tokens\":" << (res.prompt_tokens + res.gen_tokens)
                             << ",\"prompt_tokens_details\":{\"cached_tokens\":" << res.cached_tokens << "}}}\n\n";
                    }
                    done << "data: [DONE]\n\n";
                } else {
                    done << "data: {\"id\":\"" << chat_id << "\",\"object\":\"text_completion\",\"created\":" << created
                         << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\""
                         << finish_reason << "\"}]}\n\n"
                         << "data: [DONE]\n\n";
                }
                send_all(cli, done.str());
            }
            std::printf("[HTTP] %s stream: prompt=%d (cached=%d, new=%d @ %.1f tok/s) | gen=%d (%.2f tok/s) | spec=%d/%d | tools=%zu%s%s\n",
                        path.c_str(), res.prompt_tokens, res.cached_tokens, new_prompt, ptps,
                        res.gen_tokens, tps, res.spec_accepted, res.spec_drafted, emitted_tools.size(),
                        client_alive ? "" : " | ABORTED(client_closed)", telem.c_str());
            std::fflush(stdout);
        } else {
            std::ostringstream j;
            if (is_chat) {
                j << "{\"id\":\"" << chat_id << "\",\"object\":\"chat.completion\",\"created\":" << created
                  << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
                if (full_text.empty() && !emitted_tools.empty()) j << "null";
                else j << "\"" << json_escape(full_text) << "\"";
                if (!full_reasoning.empty()) {
                    j << ",\"reasoning_content\":\"" << json_escape(full_reasoning) << "\"";
                }
                if (!emitted_tools.empty()) {
                    j << ",\"tool_calls\":[";
                    for (size_t i = 0; i < emitted_tools.size(); ++i) {
                        if (i) j << ",";
                        j << "{\"id\":\"" << emitted_tools[i].id << "\",\"type\":\"function\",\"function\":{\"name\":\""
                          << json_escape(emitted_tools[i].name) << "\",\"arguments\":\""
                          << json_escape(emitted_tools[i].arguments_json) << "\"}}";
                    }
                    j << "]";
                }
                j << "},\"finish_reason\":\"" << finish_reason << "\"}],"
                  << "\"usage\":{\"prompt_tokens\":" << res.prompt_tokens
                  << ",\"completion_tokens\":" << res.gen_tokens
                  << ",\"total_tokens\":" << (res.prompt_tokens + res.gen_tokens)
                  << ",\"prompt_tokens_details\":{\"cached_tokens\":" << res.cached_tokens << "}},"
                  << "\"timings\":{\"prefill_ms\":" << res.prefill_ms
                  << ",\"decode_ms\":" << res.decode_ms
                  << ",\"tokens_per_second\":" << tps << "}}";
            } else {
                j << "{\"id\":\"" << chat_id << "\",\"object\":\"text_completion\",\"created\":" << created
                  << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"text\":\""
                  << json_escape(full_text) << "\",\"finish_reason\":\"" << finish_reason << "\"}],"
                  << "\"timings\":{\"tokens_per_second\":" << tps << "}}";
            }
            send_json(cli, 200, j.str(), cfg_.cors_origin);
            std::printf("[HTTP] %s: prompt=%d (cached=%d, new=%d @ %.1f tok/s) | gen=%d (%.2f tok/s) | spec=%d/%d | tools=%zu%s\n",
                        path.c_str(), res.prompt_tokens, res.cached_tokens, new_prompt, ptps,
                        res.gen_tokens, tps, res.spec_accepted, res.spec_drafted, emitted_tools.size(), telem.c_str());
            std::fflush(stdout);
        }
    };

    while (!stop_flag.load(std::memory_order_acquire)) {
        SOCKET cli = accept(srv, nullptr, nullptr);
        if (cli == INVALID_SOCKET) {
            if (stop_flag.load(std::memory_order_acquire)) break;
            continue;
        }
        if (active_conns.load(std::memory_order_relaxed) >= 16) {
            send_json(cli, 503, "{\"error\":{\"message\":\"Too many concurrent connections\"}}", cfg_.cors_origin);
            closesocket(cli);
            continue;
        }
        active_conns.fetch_add(1, std::memory_order_relaxed);
        std::thread(handle_client, cli).detach();
    }

    // Wait cleanly for any in-flight GPU work before tearing down CUDA context
    {
        std::lock_guard<std::mutex> glk(gen_mu);
    }
    for (int w = 0; w < 50 && active_conns.load(std::memory_order_relaxed) > 0; ++w) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (srv != INVALID_SOCKET) closesocket(srv);
    WSACleanup();
    return true;
}
