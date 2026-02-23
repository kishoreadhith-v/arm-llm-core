#include <iostream>
#include "utils.h"
#include "model_loader.h"
#include "tensor.h"

int main() {
    std::cout << blue;
    std::cout << "======================================\n";
    std::cout << "🚀 Arm-LLM-Core Initialized \n";
    std::cout << "⚙️ Target Architecture: ARM Apple Silicon \n";
    std::cout << "======================================\n";
    std::cout << reset;

    ModelLoader loader("../dummy_weights.bin");

    float * raw_weights = loader.get_data();

    Tensor weights(raw_weights, 4096, 4096);

    print_tensor_shape("W-q(Query Weights)", weights);

    std::cout << "First weight from loaded file - " << weights.data[0] << "\n";
    
    return 0;
}