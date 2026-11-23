#pragma once
#include <string>
#include "model.h"

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string default_effort = "max";
    std::string api_key;      // Empty + require_auth=true -> auto-generates random 32-hex CSPRNG bearer token
    bool require_auth = true; // Require Authorization: Bearer <api_key> on all endpoints
    std::string cors_origin;  // Empty = NO CORS headers (blocks browser cross-origin attacks!); set explicit origin if needed
};

class OpenAIServer {
public:
    OpenAIServer(InklingEngine& engine, const ServerConfig& cfg);
    OpenAIServer(InklingEngine& engine, const std::string& host, int port, const std::string& default_effort = "max");
    bool run(); // blocking loop until Ctrl-C or authenticated /shutdown
private:
    InklingEngine& engine_;
    ServerConfig cfg_;
};
