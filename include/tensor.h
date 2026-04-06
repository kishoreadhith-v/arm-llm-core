#pragma once

#include <iostream>

struct Tensor
{
    float* data;
    int rows;
    int cols;

    Tensor() : data(nullptr), rows(0), cols(0) {}

    Tensor(float* d, int r, int c) {
        data = d;
        rows = r;
        cols = c;
    }

    int size() const {
        return (rows * cols);
    }
};

void print_tensor_shape(const char* name, Tensor& t) {
    std::cout << "📊 Tensor " << name << " | Shape: ["
              << t.rows << " x " << t.cols << "] | Total Elements: "
              << t.size() << "\n";
}