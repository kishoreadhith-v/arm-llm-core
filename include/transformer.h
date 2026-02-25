#pragma once

#include <cmath>
#include "tensor.h"
#include "utils.h"
#include "math_engine.h"

inline void rmsnorm(Tensor &x, const Tensor &weight, float eps = 1e-5f) {
    // check dimensions
    CHECK(weight.rows == x.rows && weight.cols== x.cols, "RMS weight - vector shape mismatch");

    int size = x.size();
    float ss = 0.0f;

    for (int i = 0; i < size; i++)
    {
        ss += x.data[i] * x.data[i];
    }
    
    float rms = std::sqrt((ss / size) + eps);
    
    for (int i = 0; i < size; i++)
    {
        x.data[i] = (x.data[i] / rms) * weight.data[i];
    }
}

inline void apply_rope(Tensor& x, int pos, int head_dim) {
    // calc no of heads in given vector
    int num_heads = x.size() / head_dim;

    // each attention head
    for (int h = 0; h < num_heads; h++)
    {
        int head_offset = h * head_dim;

        // each pair of (x, y)
        for (int i = 0; i < head_dim; i += 2)
        {
            float freq = 1.0f / pow(10000.0f, (float)i / head_dim);

            float val = pos * freq;

            float sin_val = std::sin(val);
            float cos_val = std::cos(val);

            float x0 = x.data[head_offset + i];
            float x1 = x.data[head_offset + i + 1];

            float new_x0 = (x0 * cos_val) - (x1 * sin_val);
            float new_x1 = (x0 * sin_val) + (x1 * cos_val);

            x.data[head_offset + i] = new_x0;
            x.data[head_offset + i + 1] = new_x1;
        }
    }
}

inline void softmax(Tensor& x) {
    int size = x.size();

    float max_val = x.data[0];
    for (int i = 0; i < size; i++)
    {
        if (x.data[i] > max_val)
        {
            max_val = x.data[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x.data[i] = std::exp(x.data[i] - max_val);
        sum += x.data[i];
    }
    
    for (int i = 0; i < size; i++)
    {
        x.data[i] = x.data[i] / sum;
    }
}

inline void prepare_qkv(Tensor& x, Tensor& q, Tensor& k, Tensor& v , Tensor& Wq, Tensor& Wk, Tensor& Wv, int pos, int head_dim) {
    matmul_forward(Wq, x, q);
    matmul_forward(Wk, x, k);
    matmul_forward(Wv, x, v);

    apply_rope(q, pos, head_dim);
    apply_rope(k, pos, head_dim);
}

inline void calculate_attention_scores(const Tensor& q, const float* k_cache, float* scores, int current_pos, int head_dim) {
    float scale = 1.0f / std::sqrt((float)head_dim);

    for (int i = 0; i < current_pos; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < q.size(); j++)
        {
            sum += k_cache[(i * head_dim) + j] * q.data[j];
        }
        scores[i] = sum * scale;
    }
}