#pragma once

#include <vector>
#include <iostream>
#include <unordered_map>
#include <loader.hpp>
#include <ops.hpp>
#include <limits>
#include <iomanip>
#include <cmath>
#include <tensor.hpp>

class Model {
private:
    // Borrowed weights from the Loader's map
    const std::unordered_map<std::string, Tensor>& weight_map;

    static constexpr int n_layer     = 12;   // number of transformer layers
    static constexpr int n_head      = 12;   // number of attention heads
    static constexpr int n_embd      = 768;   // embedding size
    static constexpr int n_positions = 1024; // maximum sequence length
    static constexpr int n_vocab     = 50257; // number of tokens

    Tensor embed(const std::vector<int>& tokens) const; // token and position embeddings 
    Tensor attention(Tensor& input);

public:
    // Initialize loaded weights
    Model(const std::unordered_map<std::string, Tensor>& weight_map) : weight_map(weight_map) {}

    // Ren the network on a token sequence
    Tensor forward(const std::vector<int>& tokens);
};