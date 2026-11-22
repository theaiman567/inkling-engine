#pragma once
#include <string>
#include "model.h"

class OpenAIServer {
public:
    OpenAIServer(InklingEngine& engine, const std::string& host, int port, const std::string& default_effort = "max");
    bool run(); // blocking loop until Ctrl-C or /shutdown
private:
    InklingEngine& engine_;
    std::string host_;
    int port_;
    std::string default_effort_;
};
