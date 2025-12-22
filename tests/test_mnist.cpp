#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
//using namespace std;

#include "onnx_loader.h"

// ---------- helpers ----------

std::vector<float> load_row(const std::string& path, int row, int cols) {
    std::ifstream in(path);
    std::vector<float> v(cols);
    for (int i = 0; i <= row; ++i) {
        for (int j = 0; j < cols; ++j) {
            in >> v[j];
        }
    }
    return v;
}

int load_label(const std::string& path, int row) {
    std::ifstream in(path);
    int y = 0;
    for (int i = 0; i <= row; ++i) {
        in >> y;
    }
    return y;
}

int argmax(const Tensor& T) {
    int idx = 0;
    float best = T.at(0,0);
    for (int i = 1; i < T.cols(); ++i) {
        if (T.at(0,i) > best) {
            best = T.at(0,i);
            idx = i;
        }
    }
    return idx;
}

// ---------- main test ----------

int main() {
    ONNXLoader loader;
    Graph g = loader.load("mnist_mlp.onnx");

    // ===============================
    // PART 1: single-image test
    // ===============================
    {
        auto x    = load_row("mnist_X.txt", 0, 28*28);
        auto ref  = load_row("mnist_logits_ref.txt", 0, 10);
        int label = load_label("mnist_Y.txt", 0);

        Tensor& X = g.tensors.at("X");
        for (int i = 0; i < 28*28; ++i)
            X.data()[i] = x[i];

        g.run();

        Tensor& Y = g.tensors.at("Y");

        float max_err = 0.f;
        for (int i = 0; i < 10; ++i) {
            max_err = std::max(
                max_err,
                std::abs(Y.at(0,i) - ref[i])
            );
        }

        std::cout << "=== Single image ===\n";
        std::cout << "Max logit error: " << max_err << "\n";
        std::cout << "Pred: " << argmax(Y)
                  << "  True: " << label << "\n\n";
    }

    // ===============================
    // PART 2: batch accuracy test
    // ===============================
    {
        int correct = 0;
        int N = 100;

        for (int i = 0; i < N; ++i) {
            auto x = load_row("mnist_X.txt", i, 28*28);
            int label = load_label("mnist_Y.txt", i);

            Tensor& X = g.tensors.at("X");
            for (int j = 0; j < 28*28; ++j)
                X.data()[j] = x[j];

            g.run();

            if (argmax(g.tensors.at("Y")) == label)
                correct++;
        }

        std::cout << "=== Batch test ===\n";
        std::cout << "Accuracy over " << N << ": "
                  << (100.0 * correct / N) << "%\n";
    }

    return 0;
}
