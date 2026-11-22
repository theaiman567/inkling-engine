#include "server.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <atomic>

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

// =========================================================================
// HTTP Helpers (supports Content-Length, Chunked Transfer, Expect: 100-continue)
// =========================================================================
std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

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

void send_json(SOCKET sock, int code, const std::string& json_body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << (code == 200 ? " OK" : (code == 400 ? " Bad Request" : " Error")) << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Content-Length: " << json_body.size() << "\r\n"
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

bool read_http_request(SOCKET cli, std::string& method, std::string& path, std::string& body) {
    std::string req;
    req.reserve(16384);
    char buf[16384];
    size_t header_end = std::string::npos;
    size_t content_len = 0;
    bool is_chunked = false;
    bool sent_100 = false;

    while (true) {
        if (header_end != std::string::npos) {
            if (!is_chunked && req.size() >= header_end + 4 + content_len) break;
            if (is_chunked) {
                std::string decoded;
                if (decode_chunked(req.substr(header_end + 4), decoded)) {
                    body = std::move(decoded);
                    break;
                }
            }
        }
        int n = recv(cli, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, n);
        if (req.size() > 64ull * 1024 * 1024) return false; // 64 MB safety cap

        if (header_end == std::string::npos) {
            header_end = req.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                std::string hdr_l = to_lower_ascii(req.substr(0, header_end));
                if (hdr_l.find("transfer-encoding: chunked") != std::string::npos) {
                    is_chunked = true;
                }
                size_t cl = hdr_l.find("content-length:");
                if (cl != std::string::npos) {
                    content_len = (size_t)std::strtoull(hdr_l.c_str() + cl + 15, nullptr, 10);
                }
                if (!sent_100 && hdr_l.find("expect: 100-continue") != std::string::npos) {
                    send_all(cli, "HTTP/1.1 100 Continue\r\n\r\n");
                    sent_100 = true;
                }
            }
        }
    }

    if (header_end == std::string::npos) return false;
    std::string first_line = req.substr(0, req.find("\r\n"));
    {
        std::istringstream iss(first_line);
        iss >> method >> path;
    }
    // Strip query string from path
    size_t q = path.find('?');
    if (q != std::string::npos) path.resize(q);

    if (!is_chunked) {
        if (req.size() >= header_end + 4) {
            size_t avail = req.size() - (header_end + 4);
            body = req.substr(header_end + 4, std::min(avail, content_len));
        } else {
            body.clear();
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

std::vector<ChatMessage> parse_openai_messages(const JsonVal& root) {
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
        m.content = extract_content_text(item.get("content"));
        m.reasoning_content = item.get_str("reasoning_content", "");

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
                        // Canonicalize JSON string arguments into sorted-key JSON object!
                        JsonVal parsed_args;
                        JsonParser ap(av->str);
                        if (ap.parse(parsed_args) && parsed_args.is_obj()) {
                            spec.arguments_json = json_stringify(parsed_args);
                        } else {
                            spec.arguments_json = av->str.empty() ? "{}" : av->str;
                        }
                    } else if (av && av->is_obj()) {
                        spec.arguments_json = json_stringify(*av);
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
        if (av && av->is_obj()) out_args_json = json_stringify(*av);
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

OpenAIServer::OpenAIServer(InklingEngine& engine, const std::string& host, int port, const std::string& default_effort)
    : engine_(engine), host_(host), port_(port), default_effort_(default_effort) {}

bool OpenAIServer::run() {
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
    addr.sin_port = htons((u_short)port_);
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        std::fprintf(stderr, "[Server] Failed to bind %s:%d\n", host_.c_str(), port_);
        closesocket(srv);
        WSACleanup();
        return false;
    }
    listen(srv, 16);

    std::printf("[Server] OpenAI-compatible InkEngine listening on http://%s:%d (default effort=%s)\n",
                host_.c_str(), port_, default_effort_.c_str());
    std::printf("         Endpoints: /v1/chat/completions (tools+thinking+SSE+KV-cache), /v1/completions, /v1/models, /health, /stats, /shutdown\n");
    std::fflush(stdout);

    std::atomic<uint64_t> req_counter{1};

    while (true) {
        SOCKET cli = accept(srv, nullptr, nullptr);
        if (cli == INVALID_SOCKET) continue;

        int nodelay = 1;
        setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));

        std::string method, path, body;
        if (!read_http_request(cli, method, path, body)) {
            closesocket(cli);
            continue;
        }

        if (method == "OPTIONS") {
            send_json(cli, 200, "{}");
            closesocket(cli);
            continue;
        }

        if (path == "/health") {
            auto ks = engine_.kv_stats();
            char hb[256];
            std::snprintf(hb, sizeof(hb), "{\"status\":\"ok\",\"engine\":\"InkEngine\",\"model\":\"Inkling-2-Mini\",\"max_ctx\":%d,\"effort\":\"%s\"}",
                          ks.max_ctx, json_escape(default_effort_).c_str());
            send_json(cli, 200, hb);
        } else if (path == "/shutdown" && method == "POST") {
            send_json(cli, 200, "{\"status\":\"shutting_down\"}");
            closesocket(cli);
            std::printf("[Server] Received /shutdown request. Stopping cleanly...\n");
            std::fflush(stdout);
            break;
        } else if (path == "/v1/models") {
            send_json(cli, 200,
                "{\"object\":\"list\",\"data\":[{\"id\":\"Inkling-2-Mini\",\"object\":\"model\",\"owned_by\":\"InkEngine\"}]}");
        } else if (path == "/stats") {
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
            send_json(cli, 200, jbuf);
        } else if (path == "/v1/chat/completions" || path == "/v1/completions") {
            JsonVal root;
            JsonParser jp(body);
            if (!jp.parse(root) || !root.is_obj()) {
                send_json(cli, 400, "{\"error\":{\"message\":\"Invalid JSON request body\"}}");
                closesocket(cli);
                continue;
            }

            bool is_chat = (path == "/v1/chat/completions");
            bool stream  = root.get_bool("stream", false);
            bool include_usage = false;
            if (const auto* so = root.get("stream_options"); so && so->is_obj()) {
                include_usage = so->get_bool("include_usage", false);
            }

            SamplingParams sp;
            sp.temperature = (float)root.get_num("temperature", 0.6);
            sp.top_p       = (float)root.get_num("top_p", 0.95);
            sp.top_k       = (int)root.get_num("top_k", 40);
            sp.min_p       = (float)root.get_num("min_p", 0.05);
            int max_tok    = (int)root.get_num("max_completion_tokens", root.get_num("max_tokens", 4096));
            sp.max_tokens  = std::max(1, std::min(max_tok, 32768));
            sp.seed        = (uint64_t)root.get_num("seed", (double)(42 + req_counter.load()));

            std::string effort = root.get_str("reasoning_effort", default_effort_);
            std::string tools_json;
            std::string prompt_text;
            size_t static_prefix_bytes = 0;
            size_t tools_prefix_bytes = 0;
            size_t full_sys_prefix_bytes = 0;

            if (is_chat) {
                auto msgs = parse_openai_messages(root);
                tools_json = build_canonical_tools_json(root);
                prompt_text = engine_.tokenizer().format_chat(msgs, effort, true, tools_json, &static_prefix_bytes, &tools_prefix_bytes, &full_sys_prefix_bytes);
            } else {
                prompt_text = root.get_str("prompt", "");
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
            auto prompt_ids = engine_.tokenizer().encode(prompt_text, true);

            uint64_t rid = req_counter.fetch_add(1);
            std::string chat_id = "chatcmpl-ink-" + std::to_string(rid);
            auto created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            // Inkling-2 output parser state machine: cleanly separates thinking, visible text, and tool invocations!
            enum OutState { ST_HEADER, ST_THINKING, ST_TEXT, ST_TOOL_JSON, ST_TOOL_TEXT };
            bool starts_thinking = is_chat && (effort != "none" && effort != "0" && effort != "0.0");
            bool starts_text     = is_chat && !starts_thinking && tools_json.empty();
            OutState st = starts_thinking ? ST_THINKING : (starts_text ? ST_TEXT : ST_HEADER);

            std::string header_buf;
            std::string tool_buf;
            std::string full_text;
            std::string full_reasoning;
            std::vector<ToolCallSpec> emitted_tools;
            int tool_idx = 0;
            bool client_alive = true;

            const auto& tok = engine_.tokenizer();
            const int32_t id_model = tok.msg_model_id();
            const int32_t id_think = tok.thinking_id();
            const int32_t id_text  = tok.text_id();
            const int32_t id_tjson = tok.invoke_tool_json_id();
            const int32_t id_ttext = tok.invoke_tool_text_id();
            const int32_t id_endm  = tok.end_msg_id();

            if (stream) {
                std::string hdr =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Connection: close\r\n\r\n";
                client_alive = send_all(cli, hdr);

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

            auto flush_tool_call = [&]() {
                std::string tname, targs;
                if (parse_inkling_tool_invocation(tool_buf, header_buf, tname, targs)) {
                    ToolCallSpec tc;
                    tc.id = "call_ink_" + std::to_string(rid) + "_" + std::to_string(tool_idx);
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

            std::printf("[HTTP] Incoming %s (id=%llu): prompt=%zu tok (full_sys=%d, static=%d, tools=%d) | stream=%d | effort=%s | has_tools=%d\n",
                        path.c_str(), (unsigned long long)rid, prompt_ids.size(), full_sys_prefix_tokens, static_prefix_tokens, tools_prefix_tokens,
                        stream ? 1 : 0, effort.c_str(), tools_json.empty() ? 0 : 1);
            std::fflush(stdout);

            auto res = engine_.generate(
                prompt_ids, sp,
                [&](int32_t tok_id, const std::string& piece) -> bool {
                    if (!client_alive) return false;
                    if (!is_chat) {
                        full_text += piece;
                        if (stream && !piece.empty()) emit_sse_delta("\"" + json_escape(piece) + "\"");
                        return client_alive;
                    }

                    if (tok_id == id_model) {
                        st = ST_HEADER;
                        header_buf.clear();
                        return true;
                    }
                    if (tok_id == id_think) {
                        st = ST_THINKING;
                        return true;
                    }
                    if (tok_id == id_text) {
                        st = ST_TEXT;
                        return true;
                    }
                    if (tok_id == id_tjson) {
                        st = ST_TOOL_JSON;
                        tool_buf.clear();
                        return true;
                    }
                    if (tok_id == id_ttext) {
                        st = ST_TOOL_TEXT;
                        tool_buf.clear();
                        return true;
                    }
                    if (tok_id == id_endm) {
                        if (st == ST_TOOL_JSON || st == ST_TOOL_TEXT) {
                            flush_tool_call();
                        }
                        st = ST_HEADER;
                        header_buf.clear();
                        return true;
                    }

                    if (piece.empty()) return true;

                    if (st == ST_HEADER) {
                        header_buf += piece;
                    } else if (st == ST_THINKING) {
                        full_reasoning += piece;
                        emit_sse_delta("{\"reasoning_content\":\"" + json_escape(piece) + "\"}");
                    } else if (st == ST_TEXT) {
                        full_text += piece;
                        emit_sse_delta("{\"content\":\"" + json_escape(piece) + "\"}");
                    } else if (st == ST_TOOL_JSON || st == ST_TOOL_TEXT) {
                        tool_buf += piece;
                    }
                    return client_alive;
                },
                false, // enable_profile
                true,  // reuse_kv_prefix (incremental KV + full Global/SWA/ShortConv RAM checkpoint!)
                static_prefix_tokens,
                [&](int done, int total) -> bool {
                    if (stream && client_alive) {
                        client_alive = send_all(cli, ": keepalive\n\n");
                    }
                    if (total >= 256 && ((done % 256) == 0 || done == total)) {
                        std::printf("[HTTP] Prefill id=%llu: %d / %d tokens\n", (unsigned long long)rid, done, total);
                        std::fflush(stdout);
                    }
                    return client_alive;
                },
                tools_prefix_tokens,
                full_sys_prefix_tokens);

            if ((st == ST_TOOL_JSON || st == ST_TOOL_TEXT) && !tool_buf.empty()) {
                flush_tool_call();
            }

            std::string finish_reason = !emitted_tools.empty() ? "tool_calls" : "stop";
            double tps = res.decode_ms > 0 ? (res.gen_tokens * 1000.0 / res.decode_ms) : 0.0;
            int new_prompt = res.prompt_tokens - res.cached_tokens;
            double ptps = (res.prefill_ms > 0 && new_prompt > 0) ? (new_prompt * 1000.0 / res.prefill_ms) : 0.0;

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
                             << ",\"model\":\"Inkling-2-Mini\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\n"
                             << "data: [DONE]\n\n";
                    }
                    send_all(cli, done.str());
                }
                std::printf("[HTTP] %s stream: prompt=%d (cached=%d, new=%d @ %.1f tok/s) | gen=%d (%.2f tok/s) | spec=%d/%d | tools=%zu\n",
                            path.c_str(), res.prompt_tokens, res.cached_tokens, new_prompt, ptps,
                            res.gen_tokens, tps, res.spec_accepted, res.spec_drafted, emitted_tools.size());
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
                      << json_escape(full_text) << "\",\"finish_reason\":\"stop\"}],"
                      << "\"timings\":{\"tokens_per_second\":" << tps << "}}";
                }
                send_json(cli, 200, j.str());
                std::printf("[HTTP] %s: prompt=%d (cached=%d, new=%d @ %.1f tok/s) | gen=%d (%.2f tok/s) | spec=%d/%d | tools=%zu\n",
                            path.c_str(), res.prompt_tokens, res.cached_tokens, new_prompt, ptps,
                            res.gen_tokens, tps, res.spec_accepted, res.spec_drafted, emitted_tools.size());
                std::fflush(stdout);
            }
        } else {
            send_json(cli, 404, "{\"error\":\"Not found\"}");
        }

        closesocket(cli);
    }

    closesocket(srv);
    WSACleanup();
    return true;
}
