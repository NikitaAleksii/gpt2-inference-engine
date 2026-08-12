#include <iostream>
#include "tensor.hpp"

Tensor::Tensor() {

}

Tensor::Tensor(std::vector<int>& shape) {
    // Stride is equal to the number of columns in a newly allocated tensor
    this->stride = shape[1];

    // Allocate memory for a new tensor
    this->buffer      = new float[shape[0] * shape[1]];
    this->owns_buffer = true;

    // Save shape of the tensor
    this->shape  = move(shape);
}

Tensor::Tensor(std::vector<int>& shape, void* data, size_t byte_offset) {
    // Stride is equal to the number of columns in a pre-allocated data array
    this->stride = shape[1];

    // Point to pre-allocated data
    char* base        = reinterpret_cast<char*>(data);
    this->buffer      = reinterpret_cast<float*>(base + byte_offset);
    this->owns_buffer = false;

    // Save shape of the tensor
    this->shape  = shape;
}
