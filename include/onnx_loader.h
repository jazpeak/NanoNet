#pragma once
#include <string>
#include "graph.h"
using namespace std;

class ONNXLoader {
public:
    static Graph load(const string& path);
};
