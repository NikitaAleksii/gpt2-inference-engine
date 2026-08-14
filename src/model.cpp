#include "model.hpp"

#include "ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

Tensor Model::embed(const std::vector<int>& tokens) const {
    const int T = static_cast<int>(tokens.size());

    // Token and position embedding tables
    const Tensor& wte = weight_map.at("wte.weight");
    const Tensor& wpe = weight_map.at("wpe.weight");

    std::vector<int> shape = {T, n_embd};
    Tensor x(shape);

    // Sum token and position embeddings
    for (int i = 0; i < T; i++) {
        if (tokens[i] < 0 || tokens[i] >= n_vocab) {
            throw std::runtime_error(std::string("Token id out of range"));
        }
        for (int c = 0; c < n_embd; c++) {
            x.at({i, c}) = wte.at({tokens[i], c}) + wpe.at({i, c});
        }
    }
    return x;
}

// Multi-head self-attention sub-layer over an input of shape [T, 768].
Tensor Model::attention(Tensor& input) {
    // Project the input into concatenated queries (Q), keys (K), and values (V)
    const Tensor& c_attn_weight = weight(0, "attn.c_attn.weight");
    const Tensor& c_attn_bias   = weight(0, "attn.c_attn.bias");

    const int T = input.shape[0];
    std::vector<int> qkv_shape = {T, c_attn_weight.shape[1]};

    Tensor qkv(qkv_shape);                                      // [T, 2304]
    matmul(input, c_attn_weight, qkv);

    // Split into Q, K, V and add the projection bias
    std::vector<int> query_shape = {T, n_embd};
    Tensor Q(query_shape), K(query_shape), V(query_shape);      // [T, 768]
    for (int i = 0; i < T; i++) {
        for (int c = 0; c < n_embd; c++) {
            Q.at({i, c}) = qkv.at({i, c})             + c_attn_bias.at({c});
            K.at({i, c}) = qkv.at({i, c + n_embd})     + c_attn_bias.at({c + n_embd});
            V.at({i, c}) = qkv.at({i, c + 2 * n_embd}) + c_attn_bias.at({c + 2 * n_embd});
        }
    }

    // Split each of Q, K, V into per-head [T, head_dim] tensors
    std::vector<Tensor> Q_heads, K_heads, V_heads;

    const int head_dim = n_embd / n_head;
    std::vector<int> head_shape = {T, head_dim};

    for (int i = 0; i < n_head; i++) {
        Tensor Q_head(head_shape), K_head(head_shape), V_head(head_shape); // [T, 64]

        for (int r = 0; r < T; r++) {
            for (int c = 0; c < head_dim; c++) {
                const int g = i * head_dim + c;
                Q_head.at({r, c}) = Q.at({r, g});
                K_head.at({r, c}) = K.at({r, g});
                V_head.at({r, c}) = V.at({r, g});
            }
        }
        Q_heads.push_back(Q_head);
        K_heads.push_back(K_head);
        V_heads.push_back(V_head);
    }

    // Per-head scaled dot-product attention, written back into concat
    Tensor concat(query_shape);             // [T, 768]
    std::vector<int> scores_shape = {T, T};
    for (int i = 0; i < n_head; i++) {
        Tensor scores(scores_shape);        // [T, T]

        // scores = Q @ K^T
        std::vector<int> kt_shape = {head_dim, T};
        Tensor K_head_T(kt_shape);          // [64, T]
        transpose(K_heads[i], K_head_T);
        matmul(Q_heads[i], K_head_T, scores);

        // Scale, then mask future positions with -infinity (causal attention)
        for (int r = 0; r < T; r++) {
            for (int c = 0; c < T; c++) {
                scores.at({r, c}) = scores.at({r, c}) / std::sqrt(head_dim);
                if (c > r) {
                    scores.at({r, c}) = -std::numeric_limits<float>::infinity();
                }
            }
        }

        Tensor softmaxed_scores(scores_shape);      // [T, T]
        softmax(scores, softmaxed_scores);

        // Weighted sum of the values for this head
        Tensor head_out(head_shape);                // [T, 64]
        matmul(softmaxed_scores, V_heads[i], head_out);

        // Place this head's output into its column block of concat
        for (int r = 0; r < T; r++) {
            for (int c = 0; c < head_dim; c++) {
                concat.at({r, i * head_dim + c}) = head_out.at({r, c});
            }
        }
    }

    // Output projection: out = concat @ c_proj.weight + c_proj.bias
    Tensor out(query_shape);                                    // [T, 768]
    const Tensor& c_proj_bias = weight(0, "attn.c_proj.bias");  // [768]
    matmul(concat, weight(0, "attn.c_proj.weight"), out);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < n_embd; c++)
            out.at({t, c}) += c_proj_bias.at({c});
    return out;
}

// Position-wise feed-forward sub-layer: ln_2 -> c_fc -> gelu -> c_proj.
Tensor Model::mlp(Tensor& input) {
    const int T = input.shape[0];

    // Pre-MLP layer norm
    Tensor normalized(input.shape);
    layernorm(input, weight(0, "ln_2.bias"), weight(0, "ln_2.weight"), normalized);

    // Expansion to 4 * n_embd
    const Tensor& c_fc_weight = weight(0, "mlp.c_fc.weight");
    std::vector<int> exp_shape = {T, c_fc_weight.shape[1]};
    Tensor expanded(exp_shape);
    matmul(normalized, c_fc_weight, expanded);

    const Tensor& c_fc_bias = weight(0, "mlp.c_fc.bias");
    for (int r = 0; r < T; r++) {
        for (int c = 0; c < exp_shape[1]; c++) {
            expanded.at({r, c}) += c_fc_bias.at({c});
        }
    }

    // GELU activation
    Tensor gelued(exp_shape);
    gelu(expanded, gelued);

    // Projection back to n_embd
    const Tensor& c_proj_weight = weight(0, "mlp.c_proj.weight");
    std::vector<int> out_shape = {T, n_embd};
    Tensor projected(out_shape);
    matmul(gelued, c_proj_weight, projected);

    const Tensor& c_proj_bias = weight(0, "mlp.c_proj.bias");
    for (int r = 0; r < T; r++) {
        for (int c = 0; c < n_embd; c++) {
            projected.at({r, c}) += c_proj_bias.at({c});
        }
    }

    return projected;
}

Tensor Model::forward(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error(std::string("Tokens are empty"));
    }
    if (tokens.size() > n_positions) {
        throw std::runtime_error(std::string("Sequence is longer than context"));
    }

    Tensor input = embed(tokens);
    Tensor normalized(input.shape);

    // Pre-attention layer norm
    layernorm(input, weight(0, "ln_1.bias"), weight(0, "ln_1.weight"), normalized);

    // Attention sub-layer with residual: x = x + attn(ln_1(x))
    Tensor attn = attention(normalized);
    Tensor attn_out(input.shape);
    add(attn, input, attn_out);

    // MLP sub-layer with residual: x = x + mlp(ln_2(x))
    Tensor mlp_res = mlp(attn_out);
    Tensor mlp_out(input.shape);
    add(mlp_res, attn_out, mlp_out);

    return mlp_out;
}