#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"
#include "model_loader.h"
#include "tensor.h"
#include "math_engine.h"

int main() {
    std::cout << blue;
    std::cout << "======================================\n";
    std::cout << "🚀 Arm-LLM-Core Initialized \n";
    std::cout << "⚙️ Target Architecture: ARM Apple Silicon \n";
    std::cout << "======================================\n";
    std::cout << reset;

    int dim = 4096;

    ModelLoader loader("../dummy_weights.bin");

    float * raw_weights = loader.get_data();

    Tensor weights(raw_weights, dim, dim);
    print_tensor_shape("W-q(Query Weights)", weights);

    std::vector<float> x_data(dim, 1.0f);
    std::vector<float> y_data(dim, 0.0f);

    Tensor x(x_data.data(), dim, 1);
    Tensor y(y_data.data(), dim, 1);

    print_tensor_shape("x (input)", x);
    print_tensor_shape("y (output)", y);

    std::cout << "\n starting math engine.. \n";
    std::cout << "\n⚡ Waking up the Silicon (Cold Start/Page Faults)...\n";
    matmul_forward(weights, x, y); // Run 1: Forces the OS to load the SSD data into RAM

    std::cout << "🔥 Running Benchmark (10 iterations)...\n";

    // Reset output to zero before the real test
    std::fill(y_data.begin(), y_data.end(), 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    int iterations = 10;
    for (int i = 0; i < iterations; i++)
    {
        // We have to reset y to 0 each time, otherwise we just keep adding to infinity
        std::fill(y_data.begin(), y_data.end(), 0.0f);
        matmul_forward(weights, x, y);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = end - start;

    std::cout << "✅ Math Complete!\n";
    std::cout << "⏱️ Average Execution Time: " << (total_time.count() / iterations) << " ms\n";

    // The math proof:
    // We have a row of 4096 numbers (all 0.01). We multiply by a vector of 1.0s.
    // 4096 * 0.01 = 40.96.
    std::cout << "📊 Output y[0]: " << y.data[0] << " (Expected: ~40.96)\n\n";
    return 0;
}