#include "model.hpp"

Tensor Model::embed(const std::vector<int>& tokens) const {
    const int T = tokens.size();

    // Fetch token and position embeddings from weight map
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

// Applies attention mechanism on an input tensor with the dimension [T, 768].
// It allows to look at the entire sentence and weigh the importance of each word with relation to other words
Tensor Model::attention(Tensor& input) {
    // Convolutional layer to project the input tensor into queries (Q), keys (K), and values (V) 
    const Tensor& c_attn_weight = weight_map.at("h.0.attn.c_attn.weight");
    const Tensor& c_attn_bias   = weight_map.at("h.0.attn.c_attn.bias");

    const int T                 = input.shape[0];
    std::vector<int> mult_shape = {T, c_attn_weight.shape[1]};

    // Calculate dot product
    Tensor multipled_tensor(mult_shape);                        // [T, 2304]
    matmul(input, c_attn_weight, multipled_tensor);
    
    // Divide that tensor into Q, K, and V and apply bias
    std::vector<int> query_shape = {T, n_embd};
    Tensor Q(query_shape), K(query_shape), V(query_shape);      // [T, 768]
    for (int i = 0; i < T; i++) {
        for (int c = 0; c < n_embd; c++) {
            Q.at({i, c}) = multipled_tensor.at({i, c}) + c_attn_bias.at({c});
            K.at({i, c}) = multipled_tensor.at({i, c + n_embd}) + c_attn_bias.at({c + n_embd});
            V.at({i, c}) = multipled_tensor.at({i, c + 2 * n_embd}) + c_attn_bias.at({c + 2 * n_embd});
        }
    }

    // Split into heads
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

    // Attention mechanism
    Tensor concat (query_shape);            // [T, 768]
    std::vector<int> scores_shape = {T, T};
    for (int i = 0; i < n_head; i++) {
        Tensor scores(scores_shape);        // [T, T]

        // Calculate dot product betweet Q and K
        std::vector<int> kt_shape = {head_dim, T};
        Tensor K_head_T(kt_shape);          // [64, T]
        transpose(K_heads[i], K_head_T);
        matmul(Q_heads[i], K_head_T, scores);

        // Normalize scores tensor and set unreachable entries to -infinity
        for (int r = 0; r < T; r++) {
            for (int c = 0; c < T; c++) {
                scores.at({r, c}) = scores.at({r, c}) / std::sqrt(head_dim);
                if (c > r) {
                    scores.at({r, c}) = -std::numeric_limits<float>::infinity();
                }
            }
        }

        // Apply softmax
        Tensor softmaxed_scores(scores_shape);      // [T, T]
        softmax(scores, softmaxed_scores);

        // Head output
        Tensor head_out(head_shape);                // [T, 64]
        matmul(softmaxed_scores, V_heads[i], head_out);

        // Concat the head output
        for (int r = 0; r < T; r++) {
            for (int c = 0; c < head_dim; c++) {
                concat.at({r, i*head_dim + c}) = head_out.at({r, c}); 
            }
        }
    }
    
    Tensor out(query_shape);                                           // [T, 768]
    const Tensor& c_proj_bias = weight_map.at("h.0.attn.c_proj.bias"); // [768]
    matmul(concat, weight_map.at("h.0.attn.c_proj.weight"), out);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < n_embd; c++)
            out.at({t, c}) += c_proj_bias.at({c});
    return out;
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

    // Normalize
    layernorm(input, weight_map.at("h.0.ln_1.bias"), weight_map.at("h.0.ln_1.weight"), normalized);

    // Apply attention
    Tensor attn_out = attention(normalized);

    // Residual add
    Tensor out(input.shape);
    add(attn_out, input, out);

    return out;
}