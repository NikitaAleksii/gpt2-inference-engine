#include <iostream>
#include <mdspan>
#include <cmath>
#include <numbers>
#include "tensor.hpp"

// Constants
constexpr float EPSILON = 1e-5f;

float gelu(const float x); 
void matmul(Tensor& A, Tensor& B, Tensor& out); 
void add(Tensor& A, Tensor& B, Tensor& out);
void layernorm(Tensor& A, const Tensor& bias, const Tensor& gain, Tensor& out);
void gelu(const Tensor& A, Tensor& out);
void softmax(const Tensor& A, Tensor& out);