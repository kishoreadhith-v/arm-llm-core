#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "transformer.h"
#include "kvcache.h"
#include "model_loader.h"
#include "tokenizer.h"

int sample(Tensor &logits, float temperature, int vocab_size)
{
    for (int i = 0; i < vocab_size; i++)
    {
        if (std::isnan(logits.data[i]) || std::isinf(logits.data[i]))
        {
            std::cout << "\n💥 FATAL: Neural Network Exploded to NaN at logit " << i << "! 💥\n";
            exit(1);
        }
    }

    int max_index = 0;
    float max_val = logits.data[0];
    for (int i = 0; i < vocab_size; i++)
    {
        if (logits.data[i] > max_val) {
            max_val = logits.data[i];
            max_index = i;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++)
    {
        logits.data[i] = std::exp((logits.data[i] - max_val) / temperature);
        sum += logits.data[i];
    }

    float r = ((float)rand() / RAND_MAX) * sum;
    float current_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++)
    {
        current_sum += logits.data[i];
        if (r <= current_sum)
        {
            return i;
        }
    }
    return max_index;
}

// decoder
int argmax(const Tensor &logits, int vocab_size)
{
    int max_index = 0;
    float max_val = logits.data[0];

    for (int i = 0; i < vocab_size; i++)
    {
        if (logits.data[i] > max_val)
        {
            max_val = logits.data[i];
            max_index = i;
        }
    }
    
    return max_index;
}

int main(){
    std::cout << "[Start] Arm-LLM-Core...\n";

    int dim = 288;
    int hidden_dim = 768;
    int num_layers = 6;
    int num_heads = 6;
    int vocab_size = 32000;
    int max_seq_len = 256;
    int EOS_TOKEN = 2;

    LLaMa model;
    model.num_layers = num_layers;
    model.vocab_size = vocab_size;
    model.dim = dim;
    model.layers = new TransformerLayer[num_layers];

    for (int i = 0; i < num_layers; i++)
    {
        model.layers[i].attention.num_heads = num_heads;
        model.layers[i].attention.head_dim = dim / num_heads;
        model.layers[i].ffn.hidden_dim = hidden_dim;

        model.layers[i].attention.cache.init(max_seq_len, dim);
    }

    ModelLoader loader("../stories15M.bin");
    // Skip the 256-byte header/ metadata
    float *weights_ptr = loader.get_data() + (256 / sizeof(float));
    model.load_weights(weights_ptr);
    std::cout << "✅ Model routed safely." << std::endl;

    Tokenizer tokenizer("../tokenizer.bin", vocab_size);
    std::cout << "✅ Tokenizer loaded safely." << std::endl;

    Tensor logits(new float[vocab_size](), vocab_size, 1);

    std::srand(std::time(nullptr));

    float temperature = 0.6f; // Standard for TinyStories
    int next_token = 1; 
    int pos = 0;

    std::cout << "Generating text: \n";

    while (pos < 256)
    {
        std::memset(logits.data, 0, vocab_size * sizeof(float));

        model.forward(next_token, pos, logits);

        if (temperature <= 0.001f)
        {
            next_token = argmax(logits, vocab_size);
        }
        else
        {
            next_token = sample(logits, temperature, vocab_size);
        }

        if (next_token == EOS_TOKEN)
        {
            std::cout << "\n[Model finished generating naturally]" << std::endl;
            break;
        }

        tokenizer.print_word(next_token);
        // tokenizer.print_word(30999);
        std::cout << std::flush;
        pos += 1;
    }
    
    std::cout << "\n" << "Generation complete: \n";

    delete[] logits.data;
    delete[] model.layers;
    return 0;
}