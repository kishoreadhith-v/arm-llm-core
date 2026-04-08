#pragma once

#include <cmath>
#include "tensor.h"
#include "utils.h"
#include "math_engine.h"
#include "kvcache.h"

inline void rmsnorm(Tensor &x, const Tensor &weight, float eps = 1e-5f)
{
    // check dimensions
    CHECK(weight.rows == x.rows && weight.cols == x.cols, "RMS weight - vector shape mismatch");

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

inline void apply_rope(Tensor &x, int pos, int head_dim)
{
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

inline void softmax(Tensor &x)
{
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

inline void silu(Tensor &x)
{
    for (int i = 0; i < x.size(); i++)
    {
        float val = x.data[i];
        float sig = 1.0f / (1.0f + std::exp(-val));
        x.data[i] = val * sig;
    }
}

inline void prepare_qkv(Tensor &x, Tensor &q, Tensor &k, Tensor &v, Tensor &Wq, Tensor &Wk, Tensor &Wv, int pos, int head_dim)
{
    matmul_forward(q, x, Wq);
    matmul_forward(k, x, Wk);
    matmul_forward(v, x, Wv);

    apply_rope(q, pos, head_dim);
    apply_rope(k, pos, head_dim);
}

inline void calculate_attention_scores(const Tensor &q, const float *k_cache, float *scores, int current_pos, int head_dim, int kv_dim, int head_offset)
{
    float scale = 1.0f / std::sqrt((float)head_dim);

    for (int i = 0; i < current_pos; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < q.size(); j++)
        {
            sum += k_cache[(i * kv_dim) + head_offset + j] * q.data[j];
        }
        scores[i] = sum * scale;

        if (std::isnan(scores[i]))
            scores[i] = -10000.0f;
        if (scores[i] < -10000.0f)
            scores[i] = -10000.0f;
    }
}

// multiply attention percentagesagainst values to extract meaning
inline void compute_attention_output(const float *scores, const float *v_cache, Tensor &out, int current_pos, int head_dim, int kv_dim, int head_offset)
{
    for (int i = 0; i < out.size(); i++)
    {
        out.data[i] = 0.0f;
    }

    for (int i = 0; i < current_pos; i++)
    {
        float prob = scores[i];

        for (int j = 0; j < head_dim; j++)
        {
            out.data[j] += v_cache[(i * kv_dim) + head_offset + j] * prob;
        }
    }
}

struct AttentionBlock
{
    int num_heads;
    int head_dim;

    KVCache cache;

    // weights for this layer
    Tensor Wq, Wk, Wv, Wo;

    // forward pass
    void forward(Tensor &x, int pos)
    {
        float *q_data = new float[x.size()]();
        float *k_data = new float[x.size()]();
        float *v_data = new float[x.size()]();
        float *meaning_data = new float[x.size()]();
        float *output_data = new float[x.size()]();

        Tensor q(q_data, x.size(), 1);
        Tensor k(k_data, x.size(), 1);
        Tensor v(v_data, x.size(), 1);
        Tensor final_meaning(meaning_data, x.size(), 1);
        Tensor final_output(output_data, x.size(), 1);

        prepare_qkv(x, q, k, v, Wq, Wk, Wv, pos, head_dim);

        cache.save_kv(k, v, pos);

        int kv_dim = num_heads * head_dim;

        for (int h = 0; h < num_heads; h++)
        {
            int offset = h * head_dim;

            Tensor head_q(q.data + offset, head_dim, 1);
            Tensor head_out(final_meaning.data + offset, head_dim, 1);

            float *scores = new float[pos + 1];

            calculate_attention_scores(head_q, cache.k_cache, scores, pos + 1, head_dim, kv_dim, offset);
            Tensor scores_tensor(scores, pos + 1, 1);
            softmax(scores_tensor);
            compute_attention_output(scores, cache.v_cache, head_out, pos + 1, head_dim, kv_dim, offset);

            delete[] scores;
        }

        matmul_forward(final_output, final_meaning, Wo);
        std::memcpy(x.data, final_output.data, x.size() * sizeof(float));

        std::memcpy(x.data, final_output.data, x.size() * sizeof(float));

        delete[] q_data;
        delete[] k_data;
        delete[] v_data;
        delete[] meaning_data;
        delete[] output_data;
    }
};

struct FeedForwardBlock
{
    int hidden_dim;

    Tensor W_gate, W_up, W_down;

    void forward(Tensor &x)
    {
        float *gate_data = new float[hidden_dim]();
        float *up_data = new float[hidden_dim]();
        float *merged_data = new float[hidden_dim]();
        float *output_data = new float[x.size()](); // Compresses back to normal size

        Tensor gate(gate_data, hidden_dim, 1);
        Tensor up(up_data, hidden_dim, 1);
        Tensor merged(merged_data, hidden_dim, 1);
        Tensor final_output(output_data, x.size(), 1);

        matmul_forward(gate, x, W_gate);
        silu(gate);

        matmul_forward(up, x, W_up);

        for (int i = 0; i < hidden_dim; i++)
        {
            merged.data[i] = gate.data[i] * up.data[i];
        }

        matmul_forward(final_output, merged, W_down);

        std::memcpy(x.data, final_output.data, x.size() * sizeof(float));

        delete[] gate_data;
        delete[] up_data;
        delete[] merged_data;
        delete[] output_data;
    }
};

struct TransformerLayer
{
    AttentionBlock attention;
    FeedForwardBlock ffn;

    Tensor attention_norm_weight;
    Tensor ffn_norm_weight;

    void forward(Tensor& x, int pos) {
        // Part 1 - attention

        // save a copy of the data
        float* h_data = new float[x.size()]();
        std::memcpy(h_data, x.data, x.size() * sizeof(float));
        Tensor h(h_data, x.size(), 1);

        // prenorm and multihead attention
        rmsnorm(x, attention_norm_weight);
        attention.forward(x, pos);

        // residual connection: x + attention(rmsnorm(x))
        for (int i = 0; i < x.size(); i++)
        {
            x.data[i] += h.data[i];
        }
        
        // Part 2 - feed forward

        std::memcpy(h_data, x.data, x.size() * sizeof(float));

        rmsnorm(x, ffn_norm_weight);
        ffn.forward(x);

        // residual connection: x + ffn(rmsnorm(x))
        for (int i = 0; i < x.size(); i++)
        {
            x.data[i] += h.data[i];
        }

        delete[] h_data;
    }
};

struct LLaMa
{
    int num_layers;
    int vocab_size;
    int dim;

    Tensor tok_embeddings;      // token ids -> vectors lookup
    TransformerLayer* layers;   // array of all layers
    Tensor final_norm_weight;   // final rmsnorm weight
    Tensor output_weight;       // weight for classifier(vectors -> token)

    void load_weights(float* ptr) {
        tok_embeddings = Tensor(ptr, vocab_size, dim);
        ptr += (vocab_size * dim);

        // 2. Attention Norms (ALL LAYERS AT ONCE)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].attention_norm_weight = Tensor(ptr, dim, 1);
            ptr += dim;
        }

        // 3. Wq (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].attention.Wq = Tensor(ptr, dim, dim);
            ptr += (dim * dim);
        }

        // 4. Wk (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].attention.Wk = Tensor(ptr, dim, dim);
            ptr += (dim * dim);
        }

        // 5. Wv (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].attention.Wv = Tensor(ptr, dim, dim);
            ptr += (dim * dim);
        }

        // 6. Wo (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].attention.Wo = Tensor(ptr, dim, dim);
            ptr += (dim * dim);
        }

        // 7. FFN Norms (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].ffn_norm_weight = Tensor(ptr, dim, 1);
            ptr += dim;
        }

        // 8. FFN Gate (w1) (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].ffn.W_gate = Tensor(ptr, layers[i].ffn.hidden_dim, dim);
            ptr += (dim * layers[i].ffn.hidden_dim);
        }

        // 9. FFN Down (w2) (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].ffn.W_down = Tensor(ptr, dim, layers[i].ffn.hidden_dim);
            ptr += (layers[i].ffn.hidden_dim * dim);
        }

        // 10. FFN Up (w3) (ALL LAYERS)
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].ffn.W_up = Tensor(ptr, layers[i].ffn.hidden_dim, dim);
            ptr += (dim * layers[i].ffn.hidden_dim);
        }

        // 11. Final Norm
        final_norm_weight = Tensor(ptr, dim, 1);
        ptr += dim;

        // 12. Output Classifier (Weight Tying!)
        output_weight = Tensor(ptr, vocab_size, dim);
    }

    // Master forward pass
    // token_id -> logits
    void forward(int token_id, int pos, Tensor& logits) {
        // embedding layer
        float* x_data = new float[dim]();
        std::memcpy(x_data, tok_embeddings.data + (token_id * dim), dim * sizeof(float));
        Tensor x(x_data, dim, 1);

        // layer stack
        for (int i = 0; i < num_layers; i++)
        {
            layers[i].forward(x, pos);
        }
    
        // final norm
        rmsnorm(x, final_norm_weight);

        // classification (logits)
        matmul_forward(logits, x, output_weight);

        delete[] x_data;
    }
};
