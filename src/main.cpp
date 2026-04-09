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
        if (logits.data[i] > max_val)
        {
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

int main()
{
    std::cout << "[Start] Arm-LLM-Core...\n";

    int dim = 2048;
    int hidden_dim = 5632;
    int num_layers = 22;
    int num_heads = 32;
    int head_dim = dim / num_heads; // MUST BE 64
    int vocab_size = 32000;
    int max_seq_len = 1024;

    LLaMa model;
    model.num_layers = num_layers;
    model.vocab_size = vocab_size;
    model.dim = dim;
    model.layers = new TransformerLayer[num_layers];

    for (int i = 0; i < num_layers; i++)
    {
        model.layers[i].attention.num_heads = num_heads;
        model.layers[i].attention.head_dim = head_dim;
        model.layers[i].ffn.hidden_dim = hidden_dim;

        model.layers[i].attention.cache.init(max_seq_len, num_heads * head_dim);
    }

    ModelLoader loader("../tinyllama_1B.bin");
    // Skip the 256-byte header/ metadata
    float *weights_ptr = loader.get_data();
    model.load_weights(weights_ptr);
    std::cout << "[*] Model routed safely." << std::endl;

    Tokenizer tokenizer("../tokenizer.bin", vocab_size);
    std::cout << "[*] Tokenizer loaded safely." << std::endl;

    Tensor logits(new float[vocab_size](), vocab_size, 1);

    std::srand(std::time(nullptr));

    std::vector<int> prompt_tokens = {1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 8444, 14137, 319, 29902, 29889, 2, 13, 29966, 29989, 1792, 29989, 29958, 13, 6113, 263, 2560, 4544, 1813, 5934, 15043, 2787, 29901, 2, 13, 29966, 29989, 465, 22137, 29989, 29958, 13, 28956, 1420, 13, 29966, 29991, 21300, 3472, 29958, 13, 29966, 1420, 29958, 13, 29966, 2813, 29958, 13};
    int pos = 0;

        // We stop 1 token early so we can use the final token to kick off generation
    std::cout << "Ingesting prompt...\n";
    for (size_t i = 0; i < prompt_tokens.size() - 1; i++)
    {
        std::cout << "\rProcessing token " << (i + 1) << " of " << (prompt_tokens.size() - 1) << std::flush;
        std::memset(logits.data, 0, vocab_size * sizeof(float));
        model.forward(prompt_tokens[i], pos, logits);
        pos++;
    }
    std::cout << "\n";
    // log the prompt used
    std::cout << "\nPrompt:\n";
    std::cout << "<|system|>\nYou are a helpful coding AI.</s>\n";
    std::cout << "<|user|>\nWrite a simple HTML page saying Hello World:</s>\n";
    std::cout << "<|assistant|>\n```html\n<!DOCTYPE html>\n<html>\n<head>\n";

    int next_token = prompt_tokens.back(); // Token 3694 (" named")
    float temperature = 0.2f;              // Back to 0.8! With a prompt, the model is confident enough for creativity.
    int EOS_TOKEN = 2;

    // set terminal o/p green for ai
    std::cout << "\033[1;32m";

    while (pos < 800)
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
            std::cout << "\n[App Generation Complete]" << std::endl;
            break;
        }

        tokenizer.print_word(next_token);
        std::cout << std::flush;
        pos += 1;
    }

    // reset terminal color
    std::cout << "\033[0m";

    std::cout << "\n"
              << "Generation complete: \n";

    delete[] logits.data;
    delete[] model.layers;
    return 0;
}