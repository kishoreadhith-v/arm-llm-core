#pragma once

#include <arm_neon.h>
#include <iostream>
#include "tensor.h"
#include "utils.h"

// Forward pass matrix multiplication - GEMV
// y = W * x
inline void matmul_forward(Tensor &y, const Tensor &x, const Tensor &W)
{

    if (W.cols != x.rows)
    {
        std::cout << "\n💥 DIMENSION COLLISION DETECTED 💥\n";
        std::cout << "Weight Matrix (W): " << W.rows << " rows, " << W.cols << " cols\n";
        std::cout << "Input Vector (x): " << x.rows << " rows, " << x.cols << " cols\n";
        std::cout << "Output Vector (y): " << y.rows << " rows, " << y.cols << " cols\n";
    }

    // check dimensions
    CHECK(W.cols == x.rows, "Matrix - Vector shape mismatch - W.cols must be equal to x.rows");
    CHECK(W.rows == y.rows, "Matrix - Output shape mismatch - W.cols must be equal to y.rows");

    std::memset(y.data, 0, y.rows * y.cols * sizeof(float));
    
    for (int i = 0; i < W.rows; i++)
    {
        float32x4_t vec_sum = vdupq_n_f32(0.0f);

        for (int j = 0; j < W.cols; j += 4)
        {
            int index = (i * W.cols) + j;

            float32x4_t w_vec = vld1q_f32(&W.data[index]);
            float32x4_t x_vec = vld1q_f32(&x.data[j]);

            vec_sum = vmlaq_f32(vec_sum, w_vec, x_vec);
        }

        float final_sum = vaddvq_f32(vec_sum);

        y.data[i] = final_sum;
    }
}