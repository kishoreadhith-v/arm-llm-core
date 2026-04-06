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

    KVCache() : max_seq_len(0), kv_dim(0), k_cache(nullptr), v_cache(nullptr) {}

    void init(int seq_len, int dim)
    {
        max_seq_len = seq_len;
        kv_dim = dim;
        // Allocate the zeroed-out heap memory
        k_cache = new float[seq_len * dim]();
        v_cache = new float[seq_len * dim]();
    }

    ~KVCache()
    {
        if (k_cache)
            delete[] k_cache;
        if (v_cache)
            delete[] v_cache;
    }

    void save_kv(const Tensor &k, const Tensor &v, int pos)
    {
        CHECK(pos < max_seq_len, "KV Cache Overflow! Context limit reached.");
        CHECK(k.size() == kv_dim && v.size() == kv_dim, "KV dimension mismatch!");

        int offset = pos * kv_dim; // Safely lock to the exact token generation step
        std::memcpy(k_cache + offset, k.data, kv_dim * sizeof(float));
        std::memcpy(v_cache + offset, v.data, kv_dim * sizeof(float));
    }
};
