#include "ops.hpp"

#include <mdspan>
#include <cmath>
#include <numbers>
#include <stdexcept>

// mdspan views over a tensor's flat buffer
static auto view2d(Tensor& t)       { return std::mdspan(t.data(), t.shape[0], t.shape[1]); }
static auto view2d(const Tensor& t) { return std::mdspan(t.data(), t.shape[0], t.shape[1]); }
static auto view1d(const Tensor& t) { return std::mdspan(t.data(), t.shape[0]); }

// Matrix product: out[M, N] = A[M, K] @ B[K, N]
void matmul(const Tensor& A, const Tensor& B, Tensor& out) {
    auto A_view   = view2d(A);
    auto B_view   = view2d(B);
    auto out_view = view2d(out);

    const int M = A_view.extent(0);
    const int K = A_view.extent(1);
    const int N = B_view.extent(1);

    // Clear the destination
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            out_view[r, c] = 0.0f;
        }
    }

    // i-k-j order keeps the inner loop unit-stride over B and out (cache-friendly)
    for (int r = 0; r < M; r++) {
        for (int i = 0; i < K; i++) {
            const float A_c = A_view[r, i];
            for (int c = 0; c < N; c++) {
                out_view[r, c] += A_c * B_view[i, c];
            }
        }
    }
}

// Element-wise sum: out = A + B
void add(const Tensor& A, const Tensor& B, Tensor& out) {
    if (A.compute_size() != B.compute_size() || out.compute_size() != A.compute_size()) {
        throw std::runtime_error(std::string("Sizes don't match"));
    }

    size_t total_elements = A.compute_size();
    for (size_t i = 0; i < total_elements; i++) {
        out.data()[i] = A.data()[i] + B.data()[i];
    }
}

// Layer normalization over the last dimension, per row
void layernorm(const Tensor& A, const Tensor& bias, const Tensor& gain, Tensor& out) {
    if (A.compute_size() != out.compute_size()) {
        throw std::runtime_error(std::string("Sizes don't match"));
    }

    auto A_view    = view2d(A);
    auto bias_view = view1d(bias);
    auto gain_view = view1d(gain);
    auto out_view  = view2d(out);

    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t r = 0; r < M; r++) {
        // Mean
        float sum = 0.0f;
        for (size_t c = 0; c < N; c++) {
            sum += A_view[r, c];
        }
        float mean = sum / N;

        // Variance
        sum = 0;
        for (size_t c = 0; c < N; c++) {
            sum += (A_view[r, c] - mean) * (A_view[r, c] - mean);
        }
        float variance = sum / N;

        // Normalize, scale, shift
        float den = std::sqrt(variance + EPSILON);
        for (size_t c = 0; c < N; c++) {
            float num = A_view[r, c] - mean;
            out_view[r, c] = (num / den) * gain_view[c] + bias_view[c];
        }
    }
}

// GELU (tanh approximation, as used by GPT-2)
float gelu(const float x) {
    static const float k = std::sqrt(2.0f / std::numbers::pi_v<float>);
    float cube = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * cube)));
}

void gelu(const Tensor& A, Tensor& out) {
    auto A_view   = view2d(A);
    auto out_view = view2d(out);
    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t r = 0; r < M; r++) {
        for (size_t c = 0; c < N; c++) {
            out_view[r, c] = gelu(A_view[r, c]);
        }
    }
}

// Row-wise softmax (numerically stable via max subtraction)
void softmax(const Tensor& A, Tensor& out) {
    auto A_view   = view2d(A);
    auto out_view = view2d(out);
    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t r = 0; r < M; r++) {
        // Row max, subtracted to keep exp() from overflowing
        float m = A_view[r, 0];
        for (size_t c = 1; c < N; c++) {
            if (A_view[r, c] > m)
                m = A_view[r, c];
        }

        // Accumulate denominator
        float sum = 0.0f;
        for (size_t c = 0; c < N; c++) {
            float e = std::exp(A_view[r, c] - m);
            out_view[r, c] = e;
            sum += e;
        }

        // Normalize
        for (size_t c = 0; c < N; c++) {
            out_view[r, c] /= sum;
        }
    }
}

// 2-D transpose: out[j, i] = A[i, j]
void transpose(const Tensor& A, Tensor& out) {
    auto A_view   = view2d(A);
    auto out_view = view2d(out);

    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            out_view[j, i] = A_view[i, j];
        }
    }
}