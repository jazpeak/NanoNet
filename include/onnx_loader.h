#pragma once
#include <string>
#include "graph.h"

class ONNXLoader {
public:
    static Graph load(const std::string& path);
};
