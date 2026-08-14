#include "ops.hpp"

// Computes the dot product of two tensors `A` and `B` and outputs results into tensor `out`
void matmul(const Tensor& A, const Tensor& B, Tensor& out) {
    // Represent flat arrays into matrix-like form
    std::mdspan A_view(A.data(), A.shape[0], A.shape[1]);
    std::mdspan B_view(B.data(), B.shape[0], B.shape[1]);
    std::mdspan out_view(out.data(), out.shape[0], out.shape[1]);

    // Get dimensions of the final tensor
    const int M = A_view.extent(0);
    const int K = A_view.extent(1);
    const int N = B_view.extent(1);
    
    // Clear out the destination
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            out_view[r, c] = 0.0f;
        }
    }

    // Compute the dot product
    for (int r = 0; r < M; r++) {
        for (int i = 0; i < K; i++) {
            const float A_c = A_view[r, i];
            for (int c = 0; c < N; c++) {
                out_view[r, c] += A_c * B_view[i, c];
            }
        }
    }
}

// Adds two tensors `A` and `B` and outputs the result in `out`
void add(const Tensor&A, const Tensor& B, Tensor& out) {
    if (A.compute_size() != B.compute_size() || out.compute_size() != A.compute_size()) {
        throw std::runtime_error(std::string("Sizes don't match"));
    }

    size_t total_elements = A.compute_size();
    for (size_t i = 0; i < total_elements; i++) {
        out.data()[i] = A.data()[i] + B.data()[i]; 
    }
}

// Normalizes layer
void layernorm(const Tensor& A, const Tensor& bias, const Tensor& gain, Tensor& out) {
    if (A.compute_size() != out.compute_size()) {
        throw std::runtime_error(std::string("Sizes don't match"));
    }

    // Represent flat arrays into matrix-like form
    std::mdspan A_view(A.data(), A.shape[0], A.shape[1]);
    std::mdspan bias_view(bias.data(), bias.shape[0]);
    std::mdspan gain_view(gain.data(), gain.shape[0]);
    std::mdspan out_view(out.data(), out.shape[0], out.shape[1]);

    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t r = 0; r < M; r++) {
        // Compute mean
        float sum = 0.0f;
        for (size_t c = 0; c < N; c++) {
            sum += A_view[r, c];
        }
        float mean = sum / N;

        // Calculate varience
        sum = 0;
        for (size_t c = 0; c < N; c++) {
            sum += (A_view[r, c] - mean) * (A_view[r, c] - mean);
        }
        float varience = sum / N;

        // Normalize
        float den = std::sqrt(varience + EPSILON);
        for (size_t c = 0; c < N; c++) {
            float num = A_view[r, c] - mean; 
            out_view[r, c] = (num / den) * gain_view[c] + bias_view[c];
        }
    }
}

// Implements gelu function
float gelu(const float x) {
    float cube = x * x * x;
    float pi_2 = std::sqrt(2 / std::numbers::pi);
    return 0.5f * x * (1.0f + std::tanh(pi_2 * (x + 0.044715f * cube)));
}

void gelu(const Tensor& A, Tensor& out) {
    // Represent flat arrays into matrix-like form 
    std::mdspan A_view(A.data(), A.shape[0], A.shape[1]);
    std::mdspan out_view(out.data(), out.shape[0], out.shape[1]);
    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t r = 0; r < M; r++) {
        for (size_t c = 0; c < N; c++) {
            out_view[r, c] = gelu(A_view[r, c]);
        }
    }
}

// Implements softmax
void softmax(const Tensor& A, Tensor& out) {
    std::mdspan A_view(A.data(), A.shape[0], A.shape[1]);
    std::mdspan out_view(out.data(), out.shape[0], out.shape[1]);
    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t r = 0; r < M; r++) {
        // Stabilize exponents since they because infinity after crossing a certain threshold
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

// Transpose operation
void transpose(const Tensor& A, Tensor& out) {
    std::mdspan A_view(A.data(), A.shape[0], A.shape[1]);
    std::mdspan out_view(out.data(), out.shape[0], out.shape[1]);

    const size_t M = A_view.extent(0);
    const size_t N = A_view.extent(1);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            out_view[j, i] = A_view[i, j];
        }
    }
}