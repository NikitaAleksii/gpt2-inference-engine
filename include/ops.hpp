#pragma once

#include <iostream>
#include <mdspan>
#include <cmath>
#include <numbers>
#include "tensor.hpp"

// Constants
constexpr float EPSILON = 1e-5f;

float gelu(const float x); 
void matmul(const Tensor& A, const Tensor& B, Tensor& out); 
void add(const Tensor& A, const Tensor& B, Tensor& out);
void layernorm(const Tensor& A, const Tensor& bias, const Tensor& gain, Tensor& out);
void gelu(const Tensor& A, Tensor& out);
void softmax(const Tensor& A, Tensor& out);
void transpose(const Tensor& A, Tensor& out);