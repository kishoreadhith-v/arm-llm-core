#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "utils.h"

struct Tokenizer
{
    char **vocab;
    int vocab_size;

    Tokenizer(const char *filepath, int size)
    {
        vocab_size = size;
        vocab = new char *[vocab_size];

        FILE *file = fopen(filepath, "rb");
        if (!file)
        {
            std::cout << "[ERROR]: Failed to open " << filepath << std::endl;
            exit(1);
        }

        int max_token_length;
        if (fread(&max_token_length, sizeof(int), 1, file) != 1)
            exit(1);
        std::cout << "   [Tokenizer] Max token length read: " << max_token_length << std::endl;

        for (int i = 0; i < vocab_size; i++)
        {
            float score;
            if (fread(&score, sizeof(float), 1, file) != 1)
                break;
            if (i == 0)
                std::cout << "   [Token 0] Score read: " << score << std::endl;

            int len;
            if (fread(&len, sizeof(int), 1, file) != 1)
                break;
            if (i == 0)
                std::cout << "   [Token 0] Len read: " << len << std::endl;

            // Catch zero-length tokens or corrupted lengths before they allocate!
            if (len < 0 || len > max_token_length)
            {
                std::cout << "💥 ILLEGAL LENGTH AT TOKEN " << i << ": " << len << std::endl;
                exit(1);
            }

            vocab[i] = new char[len + 1];
            if (i == 0)
                std::cout << "   [Token 0] Memory allocated." << std::endl;

            // If length is 0, we don't fread, we just null terminate.
            if (len > 0)
            {
                if (fread(vocab[i], len, 1, file) != 1)
                {
                    std::cout << "💥 FREAD FAILED ON STRING FOR TOKEN " << i << std::endl;
                    break;
                }
            }
            if (i == 0)
                std::cout << "   [Token 0] String read." << std::endl;

            vocab[i][len] = '\0';
            if (i == 0)
                std::cout << "   [Token 0] Null terminated safely!" << std::endl;

            if (i == 31999)
                std::cout << "   [Tokenizer] Final token 31999 loaded!" << std::endl;
        }

        fclose(file);
    }

    ~Tokenizer()
    {
        for (int i = 0; i < vocab_size; i++)
        {
            delete[] vocab[i];
        }
        delete[] vocab;
    }

    void print_word(int token_id)
    {
        const char *word = vocab[token_id];

        if(std::strcmp(word, "<0x0A>") == 0)
        {
            std::cout << '\n';
        }
        else if (std::strcmp(word, "<0x09>") == 0)
        {
            std::cout << '\t';
        }
        else if (std::strcmp(word, "<0x20>") == 0)
        {
            std::cout << ' ';
        }
        else if (std::strcmp(word, "<s>") == 0 || std::strcmp(word, "</s>") == 0)
        {
            // drop control tokens
        }
        else
        {
            // Print normal words
            std::cout << word;
        }

        std::flush(std::cout);
    }
};
