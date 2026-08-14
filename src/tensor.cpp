#include "tensor.hpp"

// Creates a tensor with newly allocated memory used for activations
Tensor::Tensor(std::vector<int>& shape) {
    // Save shape of the tensor
    this->shape  = shape;

    compute_strides();

    // Allocate memory for a new tensor
    this->buffer      = new float[compute_size()];
    this->owns_buffer = true;
}

// Creates a tensor with pre-assigned memory used for weights
Tensor::Tensor(std::vector<int>& shape, void* data, size_t byte_offset) {
    // Save shape of the tensor
    this->shape  = shape;

    compute_strides();

    // Point to pre-allocated data
    char* base        = reinterpret_cast<char*>(data);
    this->buffer      = reinterpret_cast<float*>(base + byte_offset);
    this->owns_buffer = false;
}
