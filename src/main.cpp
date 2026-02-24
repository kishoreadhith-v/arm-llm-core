#include <iostream>
#include <vector>

#include "tensor.h"
#include "transformer.h"

int main()
{
    std::cout << "======================================\n";
    std::cout << "🧭 ARM-LLM-Core: RoPE Test \n";
    std::cout << "======================================\n";

    int head_dim = 4; // A tiny attention head of size 4

    // 1. Create a dummy vector for a word: [1.0, 1.0, 1.0, 1.0]
    std::vector<float> x_data = {1.0f, 1.0f, 1.0f, 1.0f};
    Tensor x(x_data.data(), head_dim, 1);

    std::cout << "Position 0 (Original Word):\n";
    std::cout << "x[0], x[1] (Fast Gear): " << x.data[0] << ", " << x.data[1] << "\n";
    std::cout << "x[2], x[3] (Slow Gear): " << x.data[2] << ", " << x.data[3] << "\n\n";

    // 2. Move the word to Position 1 (the second word in a sentence)
    // This will trigger the trigonometry to rotate the coordinates.
    apply_rope(x, 1, head_dim);

    std::cout << "Position 1 (Rotated Word):\n";
    std::cout << "x[0], x[1] (Fast Gear): " << x.data[0] << ", " << x.data[1] << "\n";
    std::cout << "x[2], x[3] (Slow Gear): " << x.data[2] << ", " << x.data[3] << "\n\n";

    return 0;
}