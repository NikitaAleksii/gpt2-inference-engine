#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "tensor.hpp"

class Model {
private:
    // Borrowed weights from the Loader's map (Loader must outlive the Model)
    const std::unordered_map<std::string, Tensor>& weight_map;

    static constexpr int n_layer     = 12;    // number of transformer layers
    static constexpr int n_head      = 12;    // number of attention heads
    static constexpr int n_embd      = 768;   // embedding size
    static constexpr int n_positions = 1024;  // maximum sequence length
    static constexpr int n_vocab     = 50257; // vocabulary size

    // Fetches a per-layer weight by name, e.g. weight(0, "attn.c_attn.weight")
    const Tensor& weight(int layer, const std::string& name) const {
        return weight_map.at("h." + std::to_string(layer) + "." + name);
    }

    Tensor embed(const std::vector<int>& tokens) const; // token + position embeddings
    Tensor attention(Tensor& input);
    Tensor mlp(Tensor& input);

public:
    // Bind to the loaded weights (no copy)
    Model(const std::unordered_map<std::string, Tensor>& weight_map) : weight_map(weight_map) {}

    // Run the network on a token sequence
    Tensor forward(const std::vector<int>& tokens);
};