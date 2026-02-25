#pragma once

#include <cstring>
#include "tensor.h"
#include "utils.h"

struct KVCache
{
    int max_seq_len;
    int kv_dim;
    int current_pos;

    float* k_cache;
    float* v_cache;

    KVCache(int max_len, int dim) {
        max_seq_len = max_len;
        kv_dim = dim;
        current_pos = 0;

        k_cache = new float[max_seq_len * kv_dim]();
        v_cache = new float[max_seq_len * kv_dim]();
    }

    ~KVCache() {
        delete[] k_cache;
        delete[] v_cache;
    }

    void save_kv(const Tensor& k, const Tensor& v) {
        CHECK(current_pos < max_seq_len, "KV Cache Overflow! Context limit reached.");
        CHECK(k.size() == kv_dim && v.size() == kv_dim, "KV dimension mismatch!");

        int offset = current_pos * kv_dim;

        std::memcpy(k_cache + offset, k.data, kv_dim * sizeof(float));

        std::memcpy(v_cache + offset, v.data, kv_dim * sizeof(float));

        current_pos++;
    }
};
