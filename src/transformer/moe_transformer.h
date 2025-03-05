//Transformer model with Mixture of Experts implementation
// This is a C++ implementation of the DeepSeek-MoE architecture, featuring:
// - 28 transformer layers (1 dense + 27 MoE)
// - Distributed execution with tensor parallelism
// - Mixture of Experts with 64 routed experts and 2 shared experts
// - Top-6 expert routing per token
// - Rotary positional embeddings

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <functional>
#include <cmath>
#include <algorithm>

// X namespace for tensor operations
namespace X {
    // Basic tensor class definition
    class Tensor {
    public:
        Tensor() = default;
        Tensor(const std::vector<int64_t>& shape) {}
        Tensor(const Tensor& other) = default;

        std::vector<int64_t> sizes() const { return {}; }
        Tensor& zero_() { return *this; }
        int64_t size(int dim) const { return 0; }

        // Helper to extract a single value
        template<typename T>
        T item() const { return T(); }

        // Dummy implementation to make code compile
        bool defined() const { return true; }

        // Operator overloads for index access
        Tensor operator[](int idx) const { return Tensor(); }
        Tensor& operator[](int idx) { return *this; }

        static Tensor zeros_like(const Tensor& x) { return Tensor(); }
        static Tensor ones(const std::vector<int64_t>& shape) { return Tensor(shape); }
        static Tensor full(const std::vector<int64_t>& shape, float value) { return Tensor(shape); }
        static Tensor zeros(const std::vector<int64_t>& shape) { return Tensor(shape); }
    };

    // Tensor operations - declared but implemented with dummy functionality
    Tensor matmul(const Tensor& a, const Tensor& b) { return Tensor(); }
    Tensor add(const Tensor& a, const Tensor& b) { return Tensor(); }
    Tensor mul(const Tensor& a, const Tensor& b) { return Tensor(); }

    // Activation functions
    Tensor softmax(const Tensor& x, int dim) { return Tensor(); }
    Tensor silu(const Tensor& x) { return Tensor(); }

    // Normalization
    Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-6) { return Tensor(); }

    // Tensor manipulation
    Tensor split(const Tensor& x, int chunks, int dim) { return Tensor(); }
    Tensor cat(const std::vector<Tensor>& tensors, int dim) { return Tensor(); }
    Tensor view(const Tensor& x, const std::vector<int64_t>& shape) { return Tensor(); }
    Tensor transpose(const Tensor& x, int dim1, int dim2) { return Tensor(); }
    Tensor squeeze(const Tensor& x, int dim) { return Tensor(); }

    // Masking and indexing operations
    Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
        // Dummy implementation: selects elements from x or y based on condition
        Tensor result;
        // In a real implementation: result[i] = condition[i] ? x[i] : y[i]
        return result;
    }

    Tensor masked_select(const Tensor& x, const Tensor& mask) { return Tensor(); }

    Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
        // Dummy implementation: fills elements in x with value where mask is true
        Tensor result = x;
        // In a real implementation: for each i where mask[i] is true, result[i] = value
        return result;
    }

    Tensor index_select(const Tensor& x, int dim, const Tensor& indices) {
        // Dummy implementation: selects slices from x along dimension dim according to indices
        Tensor result;
        // In a real implementation: result would contain x slices at positions specified by indices
        return result;
    }

    Tensor gather(const Tensor& x, int dim, const Tensor& index) { return Tensor(); }

    // Scatter operation - adds src values to specified locations in x
    Tensor scatter_add(const Tensor& x, int dim, const Tensor& index, const Tensor& src) {
        // Dummy implementation: adds src values to x at positions specified by index
        Tensor result = x;
        // In a real implementation: 
        // for each i, j: result[i, index[i, j], k] += src[i, j, k]
        return result;
    }

    // Comparison operations
    Tensor eq(const Tensor& x, const Tensor& y) { return Tensor(); }
    Tensor eq(const Tensor& x, int value) { return Tensor(); }
    Tensor lt(const Tensor& x, int value) { return Tensor(); }
    Tensor ge(const Tensor& x, int value) { return Tensor(); }
    Tensor gt(const Tensor& x, float value) { return Tensor(); }
    Tensor logical_or(const Tensor& x, const Tensor& y) { return Tensor(); }

    // Returns values and indices of the k largest elements
    std::pair<Tensor, Tensor> topk(const Tensor& x, int k, int dim) {
        // Dummy implementation: returns k largest elements and their indices
        Tensor values, indices;
        // In a real implementation, this would perform a partial sort
        return { values, indices };
    }

    Tensor nonzero(const Tensor& x) { return Tensor(); }
    Tensor select(const Tensor& x, int dim, int index) { return Tensor(); }
    Tensor slice(const Tensor& x, int dim, int start, int end) { return Tensor(); }
    Tensor sub(const Tensor& x, int value) { return Tensor(); }
    Tensor unsqueeze(const Tensor& x, int dim) { return Tensor(); }

    // Reduction operations
    template<typename T>
    T sum(const Tensor& x, int dim, bool keepdim = false) { return T(); }

    // Special operations
    Tensor embedding(const Tensor& indices, const Tensor& weight) {
        // Dummy implementation: looks up embedding vectors for each index
        Tensor result;
        // In a real implementation: for each i, result[i] = weight[indices[i]]
        return result;
    }

    Tensor apply_rotary_emb(const Tensor& x, const Tensor& freqs) {
        // Dummy implementation: applies rotary position embeddings
        Tensor result = x;
        // In a real implementation, this would apply complex phase rotations
        return result;
    }

    // Parallelism operations
    Tensor all_reduce(const Tensor& x) {
        // Dummy implementation: simulates a sum across all processes
        return x; // In a distributed setting, this would sum x across all ranks
    }

    Tensor all_gather(const Tensor& x, int dim) {
        // Dummy implementation: collects x from all processes
        return x; // In a distributed setting, this would concatenate x from all ranks
    }

    void scatter_tensor(Tensor& dst, const Tensor& src, int dim, int rank, int world_size) {
        // Dummy implementation: distributes src across processes
        // In a distributed setting, this would split src and send pieces to different ranks
    }

    // Environment functions
    int get_rank() { return 0; }
    int get_world_size() { return 1; }
}

// Garnet namespace for model implementation
namespace Garnet {
    namespace Transformer {

        // Configuration for the model
        struct ModelConfig {
            int vocab_size = 102400;         // Size of vocabulary (number of unique tokens)
            int hidden_size = 2048;          // Model dimension for embeddings and layer inputs/outputs
            int intermediate_size = 10944;   // Dimension used in dense MLP layers
            int moe_intermediate_size = 1408; // Dimension used in expert networks
            int num_hidden_layers = 28;      // Total number of transformer blocks
            int first_k_dense_replace = 1;   // Number of dense layers (non-MoE)
            int num_attention_heads = 16;    // Number of attention heads in multi-head attention
            int num_key_value_heads = 16;    // Number of key/value heads (for grouped-query attention)
            int n_routed_experts = 64;       // Number of experts in MoE layers
            int n_shared_experts = 2;        // Number of shared experts (always active)
            int num_experts_per_tok = 6;     // Number of experts to route each token to (top-k)
            float rms_norm_eps = 1e-6;       // Epsilon for RMSNorm stability
            float rope_theta = 10000.0;      // Base for rotary positional embedding
            bool use_cache = true;           // Whether to use KV cache for inference
            int max_seq_len = 4096;          // Maximum sequence length supported
        };

        // Base module class - all model components extend this
        class Module {
        public:
            virtual ~Module() = default;
            virtual X::Tensor forward(const X::Tensor& input) = 0;
        };

        // RMSNorm layer - normalizes inputs using RMS normalization
        // Less sensitive to outliers than LayerNorm, used in all modern transformers
        class RMSNorm : public Module {
        private:
            X::Tensor mWeight;  // Learnable scale parameter
            float mEps;         // Small constant for numerical stability

        public:
            RMSNorm(int size, float eps = 1e-6) : mEps(eps) {
                mWeight = X::Tensor::ones({ size });
            }

            X::Tensor forward(const X::Tensor& input) override {
                // RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight
                return X::rms_norm(input, mWeight, mEps);
            }
        };

        // Base class for parallelized linear layers
        class Linear : public Module {
        protected:
            X::Tensor mWeight;   // Weight matrix
            X::Tensor mBias;     // Optional bias vector
            int mInFeatures;     // Input dimension
            int mOutFeatures;    // Output dimension
            bool mHasBias;       // Whether to use bias

        public:
            Linear(int in_features, int out_features, bool bias = false)
                : mInFeatures(in_features), mOutFeatures(out_features), mHasBias(bias) {
                mWeight = X::Tensor({ out_features, in_features });
                if (mHasBias) {
                    mBias = X::Tensor({ out_features });
                }
            }

            virtual X::Tensor forward(const X::Tensor& input) override {
                // Linear transform: y = xW^T + b
                X::Tensor output = X::matmul(input, X::transpose(mWeight, 0, 1));
                if (mHasBias) {
                    output = X::add(output, mBias);
                }
                return output;
            }
        };

        // Column-parallel linear layer - distributes output features across GPUs
        // Used for expanding dimensions (e.g., hidden→intermediate in MLP)
        class ColumnParallelLinear : public Linear {
        private:
            int mWorldSize;   // Number of GPUs/processes
            int mRank;        // Current GPU/process ID

        public:
            ColumnParallelLinear(int in_features, int out_features, bool bias = false)
                : Linear(in_features, out_features / X::get_world_size(), bias),
                mWorldSize(X::get_world_size()),
                mRank(X::get_rank()) {
                // Weight and bias are partitioned along output dimension
                // Each GPU owns out_features/world_size output features
            }

            X::Tensor forward(const X::Tensor& input) override {
                // Each GPU computes its portion of the output, no all_reduce needed
                return Linear::forward(input);
            }
        };

        // Row-parallel linear layer - distributes input features across GPUs
        // Used for reducing dimensions (e.g., intermediate→hidden in MLP)
        class RowParallelLinear : public Linear {
        private:
            int mWorldSize;   // Number of GPUs/processes
            int mRank;        // Current GPU/process ID

        public:
            RowParallelLinear(int in_features, int out_features, bool bias = false)
                : Linear(in_features / X::get_world_size(), out_features, bias),
                mWorldSize(X::get_world_size()),
                mRank(X::get_rank()) {
                // Weight is partitioned along input dimension
                // Each GPU owns in_features/world_size input features
            }

            X::Tensor forward(const X::Tensor& input) override {
                // First compute partial result on this GPU
                X::Tensor output = Linear::forward(input);

                // All-reduce to combine partial results from all GPUs
                output = X::all_reduce(output);

                // Apply bias after all-reduce if needed
                // Note: bias is only applied on one GPU to avoid duplication
                return output;
            }
        };

        // Parallel embedding layer - distributes vocabulary across GPUs
        class ParallelEmbedding : public Module {
        private:
            int mVocabSize;        // Total vocabulary size
            int mEmbeddingDim;     // Embedding dimension
            int mWorldSize;        // Number of GPUs/processes
            int mRank;             // Current GPU/process ID
            X::Tensor mWeight;     // Embedding table for this GPU's vocab portion
            int mLocalVocabSize;   // Size of vocabulary portion on this GPU
            int mVocabStartIdx;    // Starting index of this GPU's vocab portion

        public:
            ParallelEmbedding(int vocab_size, int embedding_dim)
                : mVocabSize(vocab_size), mEmbeddingDim(embedding_dim),
                mWorldSize(X::get_world_size()), mRank(X::get_rank()) {
                // Each GPU handles a subset of vocabulary
                mLocalVocabSize = mVocabSize / mWorldSize;
                mVocabStartIdx = mRank * mLocalVocabSize;
                mWeight = X::Tensor({ mLocalVocabSize, mEmbeddingDim });
            }

            X::Tensor forward(const X::Tensor& input) override {
                // Identify tokens that belong to this GPU's vocabulary range
                X::Tensor below_range = X::lt(input, mVocabStartIdx);
                X::Tensor above_range = X::ge(input, mVocabStartIdx + mLocalVocabSize);
                X::Tensor out_of_range = X::logical_or(below_range, above_range);

                // Adjust indices to be relative to this GPU's vocabulary portion
                X::Tensor local_indices = X::sub(input, mVocabStartIdx);

                // Zero out indices for tokens not in this GPU's range (to avoid out-of-bounds)
                local_indices = X::where(out_of_range, X::zeros_like(local_indices), local_indices);

                // Look up embeddings from local table
                X::Tensor local_output = X::embedding(local_indices, mWeight);

                // Zero out embeddings for tokens not in this GPU's range
                local_output = X::where(out_of_range.unsqueeze(-1), X::zeros_like(local_output), local_output);

                // Sum results across all GPUs (tokens not in a GPU's range are zeroed)
                X::Tensor output = X::all_reduce(local_output);

                return output;
            }
        };

        // Rotary Positional Embedding implementation (RoPE)
        // Applies position-dependent rotation to attention heads
        // Advantages: extrapolates better to unseen sequence lengths
        class RotaryPositionalEmbedding {
        private:
            X::Tensor mFreqsCis;   // Precomputed complex rotation values
            int mMaxSeqLen;        // Maximum sequence length
            int mRotaryDim;        // Dimension to apply rotations to

        public:
            RotaryPositionalEmbedding(int dim, int max_seq_len, float theta = 10000.0f)
                : mMaxSeqLen(max_seq_len), mRotaryDim(dim) {
                // Precompute frequency table for rotary embeddings
                // θ_i = 10000^(-2(i-1)/d) for i in [1,2,...,d/2]

                // For each position, we compute complex rotations e^(iθ_pos)
                // where θ_pos = pos · θ_i for each frequency i

                X::Tensor freqs = X::Tensor({ max_seq_len, dim / 2 });
                // In real implementation, fill with values: freqs[pos, i] = pos / (theta^(2i/d))

                // Store as complex numbers for efficient rotation
                mFreqsCis = X::Tensor({ max_seq_len, dim / 2 });
                // In real implementation: mFreqsCis[pos, i] = e^(iθ_pos_i) = cos(θ_pos_i) + i·sin(θ_pos_i)
            }

            X::Tensor apply(const X::Tensor& x, int offset = 0) {
                // Get sequence length from input tensor
                int seq_len = x.size(1);

                // Extract rotation values for current positions
                X::Tensor freqs_for_seq = X::slice(mFreqsCis, 0, offset, offset + seq_len);

                // Reshape x for complex number calculation
                // Original: [batch, seq, head, dim]
                // Reshaped: [batch, seq, head, dim/2, 2] to treat as complex numbers
                std::vector<int64_t> shape = x.sizes();
                X::Tensor x_reshaped = X::view(x, { shape[0], shape[1], shape[2], shape[3] / 2, 2 });

                // Apply rotary embeddings via complex multiplication
                // For each position pos and dimension i:
                // [real, imag] = [cos(θ) · real - sin(θ) · imag, cos(θ) · imag + sin(θ) · real]
                X::Tensor rotated = X::apply_rotary_emb(x_reshaped, freqs_for_seq);

                // Reshape back to original shape
                return X::view(rotated, shape);
            }
        };

        // Multi-head attention implementation with row/column parallelism
        class MultiHeadAttention : public Module {
        private:
            int mHiddenSize;           // Model dimension
            int mNumHeads;             // Total number of attention heads
            int mHeadDim;              // Dimension per attention head
            float mSoftmaxScale;       // Scale for attention scores (1/√dim)
            int mLocalHeads;           // Number of heads on this GPU

            // Projection matrices
            ColumnParallelLinear mQProj;  // Query projection
            ColumnParallelLinear mKProj;  // Key projection
            ColumnParallelLinear mVProj;  // Value projection
            RowParallelLinear mOProj;     // Output projection

            RotaryPositionalEmbedding mRotaryEmb;  // Positional embedding
            bool mUseCache;                        // Whether to cache KV for generation
            std::unordered_map<std::string, X::Tensor> mCache;  // KV cache storage

        public:
            MultiHeadAttention(const ModelConfig& config)
                : mHiddenSize(config.hidden_size),
                mNumHeads(config.num_attention_heads),
                mHeadDim(config.hidden_size / config.num_attention_heads),
                mSoftmaxScale(1.0f / std::sqrt(static_cast<float>(mHeadDim))),
                mLocalHeads(config.num_attention_heads / X::get_world_size()),
                mQProj(config.hidden_size, config.hidden_size),
                mKProj(config.hidden_size, config.hidden_size),
                mVProj(config.hidden_size, config.hidden_size),
                mOProj(config.hidden_size, config.hidden_size),
                mRotaryEmb(mHeadDim, config.max_seq_len, config.rope_theta),
                mUseCache(config.use_cache) {
            }

            X::Tensor forward(const X::Tensor& x,
                int start_pos = 0,
                const X::Tensor& attention_mask = X::Tensor()) {
                // Get batch size and sequence length
                int batch_size = x.size(0);
                int seq_len = x.size(1);

                // 1. Compute query, key, value projections
                X::Tensor q = mQProj.forward(x); // [batch, seq, hidden]
                X::Tensor k = mKProj.forward(x); // [batch, seq, hidden]
                X::Tensor v = mVProj.forward(x); // [batch, seq, hidden]

                // 2. Reshape for multi-head attention
                // From [batch, seq, hidden] to [batch, seq, heads, head_dim]
                q = X::view(q, { batch_size, seq_len, mLocalHeads, mHeadDim });
                k = X::view(k, { batch_size, seq_len, mLocalHeads, mHeadDim });
                v = X::view(v, { batch_size, seq_len, mLocalHeads, mHeadDim });

                // 3. Apply rotary positional embeddings
                q = mRotaryEmb.apply(q, start_pos);
                k = mRotaryEmb.apply(k, start_pos);

                // 4. Handle KV cache for autoregressive generation
                if (mUseCache) {
                    if (mCache.find("k") != mCache.end() && mCache.find("v") != mCache.end()) {
                        // For generation, concatenate new keys/values with cached ones
                        X::Tensor k_cache = mCache["k"];
                        X::Tensor v_cache = mCache["v"];

                        k = X::cat({ k_cache, k }, 1); // Concat along sequence dimension
                        v = X::cat({ v_cache, v }, 1);
                    }

                    // Update cache with new keys/values
                    mCache["k"] = k;
                    mCache["v"] = v;
                }

                // 5. Transpose tensors for efficient attention computation
                q = X::transpose(q, 1, 2); // [batch, head, seq_q, dim]
                k = X::transpose(k, 1, 2); // [batch, head, seq_k, dim]
                v = X::transpose(v, 1, 2); // [batch, head, seq_v, dim]

                // 6. Compute scaled dot-product attention
                // First transpose k for matrix multiplication: [batch, head, dim, seq_k]
                k = X::transpose(k, 2, 3);

                // Compute attention scores: [batch, head, seq_q, seq_k]
                X::Tensor scores = X::matmul(q, k);

                // Scale scores by 1/√d_k
                scores = X::mul(scores, mSoftmaxScale);

                // 7. Apply attention mask for causal/padding masking
                if (attention_mask.defined()) {
                    // Add mask (where mask=-inf, softmax→0)
                    scores = X::add(scores, attention_mask);
                }

                // 8. Apply softmax to get attention weights
                X::Tensor attn_weights = X::softmax(scores, -1);

                // 9. Apply attention weights to values
                X::Tensor context = X::matmul(attn_weights, v); // [batch, head, seq_q, dim]

                // 10. Transpose back to original ordering
                context = X::transpose(context, 1, 2); // [batch, seq, head, dim]

                // 11. Reshape and apply output projection
                context = X::view(context, { batch_size, seq_len, mLocalHeads * mHeadDim });
                return mOProj.forward(context);
            }
        };

        // Dense feed-forward network with SwiGLU activation
        // Implements: FFN(x) = (SiLU(xW1) ⊗ xW3)W2
        class FeedForward : public Module {
        private:
            ColumnParallelLinear mGateProj;  // W1 projection (produces gating values)
            ColumnParallelLinear mUpProj;    // W3 projection (produces values to be gated)
            RowParallelLinear mDownProj;     // W2 projection (final projection)

        public:
            FeedForward(int hidden_size, int intermediate_size)
                : mGateProj(hidden_size, intermediate_size),
                mUpProj(hidden_size, intermediate_size),
                mDownProj(intermediate_size, hidden_size) {
            }

            X::Tensor forward(const X::Tensor& x) override {
                // 1. Compute gating activation: SiLU(xW1)
                X::Tensor gate = X::silu(mGateProj.forward(x));

                // 2. Compute values to be gated: xW3
                X::Tensor up = mUpProj.forward(x);

                // 3. Apply element-wise multiplication (gating)
                X::Tensor activation = X::mul(gate, up);

                // 4. Apply final projection
                X::Tensor output = mDownProj.forward(activation);

                return output;
            }
        };

        // Expert network for MoE - same structure as FeedForward but specialized for experts
        class ExpertNetwork : public Module {
        private:
            X::Tensor mW1; // gate projection weights
            X::Tensor mW2; // down projection weights
            X::Tensor mW3; // up projection weights
            int mHiddenSize;       // Input/output dimension
            int mIntermediateSize; // Intermediate dimension

        public:
            ExpertNetwork(int hidden_size, int intermediate_size)
                : mHiddenSize(hidden_size), mIntermediateSize(intermediate_size) {
                // Initialize expert weights
                mW1 = X::Tensor({ intermediate_size, hidden_size });
                mW2 = X::Tensor({ hidden_size, intermediate_size });
                mW3 = X::Tensor({ intermediate_size, hidden_size });
            }

            X::Tensor forward(const X::Tensor& x) override {
                // Same computation as FeedForward, but with direct matrix operations

                // 1. Compute gate projection and activation
                X::Tensor gate = X::matmul(x, X::transpose(mW1, 0, 1));
                gate = X::silu(gate);

                // 2. Compute up projection
                X::Tensor up = X::matmul(x, X::transpose(mW3, 0, 1));

                // 3. Apply gating
                X::Tensor intermediate = X::mul(gate, up);

                // 4. Compute down projection
                X::Tensor output = X::matmul(intermediate, X::transpose(mW2, 0, 1));

                return output;
            }
        };

        // Routing gate for MoE - determines which experts to use for each token
        class RoutingGate : public Module {
        private:
            X::Tensor mWeight;   // Routing matrix
            int mHiddenSize;     // Input dimension
            int mNumExperts;     // Total number of experts
            int mTopK;           // Number of experts to select per token

        public:
            RoutingGate(int hidden_size, int num_experts, int top_k)
                : mHiddenSize(hidden_size), mNumExperts(num_experts), mTopK(top_k) {
                // Weights for computing token-to-expert routing scores
                mWeight = X::Tensor({ num_experts, hidden_size });
            }

            std::pair<X::Tensor, X::Tensor> forward(const X::Tensor& x) {
                // 1. Compute routing scores: dot product between token representations and expert embeddings
                X::Tensor scores = X::matmul(x, X::transpose(mWeight, 0, 1)); // [batch*seq, num_experts]

                // 2. Apply softmax to get routing probabilities
                X::Tensor probs = X::softmax(scores, -1);

                // 3. Select top-k experts for each token
                // Note: In actual implementation, this would ensure at least k experts have non-zero
                // probability to avoid potential numerical issues
                std::pair<X::Tensor, X::Tensor> topk_result = X::topk(probs, mTopK, -1);
                X::Tensor topk_probs = topk_result.first;      // Values [batch*seq, k]
                X::Tensor topk_indices = topk_result.second;   // Indices [batch*seq, k]

                return { topk_probs, topk_indices };
            }
        };

        // Mixture of Experts implementation - core of the sparse architecture
        class MixtureOfExperts : public Module {
        private:
            int mHiddenSize;       // Model dimension
            int mNumExperts;       // Total number of experts
            int mExpertsPerToken;  // Number of experts to route each token to
            int mWorldSize;        // Number of GPUs/processes
            int mRank;             // Current GPU/process ID
            int mLocalExperts;     // Number of experts on this GPU
            int mExpertsStartIdx;  // Starting expert index on this GPU

            RoutingGate mGate;     // Expert selection mechanism
            std::vector<std::unique_ptr<ExpertNetwork>> mExperts;  // Expert networks
            FeedForward mSharedExperts;  // Shared experts (always active)

        public:
            MixtureOfExperts(const ModelConfig& config)
                : mHiddenSize(config.hidden_size),
                mNumExperts(config.n_routed_experts),
                mExpertsPerToken(config.num_experts_per_tok),
                mWorldSize(X::get_world_size()),
                mRank(X::get_rank()),
                mLocalExperts(config.n_routed_experts / X::get_world_size()),
                mExpertsStartIdx(mRank* mLocalExperts),
                mGate(config.hidden_size, config.n_routed_experts, config.num_experts_per_tok),
                mSharedExperts(config.hidden_size, config.n_shared_experts* config.moe_intermediate_size) {

                // Initialize expert networks - each GPU handles a subset of experts
                for (int i = 0; i < mNumExperts; ++i) {
                    if (i >= mExpertsStartIdx && i < mExpertsStartIdx + mLocalExperts) {
                        // Create experts owned by this GPU
                        mExperts.push_back(std::make_unique<ExpertNetwork>(
                            config.hidden_size, config.moe_intermediate_size));
                    }
                    else {
                        // Placeholder for experts not on this GPU
                        mExperts.push_back(nullptr);
                    }
                }
            }

            X::Tensor forward(const X::Tensor& x) override {
                // Get original shape for later reshaping
                int batch_size = x.size(0);
                int seq_len = x.size(1);

                // 1. Reshape input for routing: [batch, seq, hidden] -> [batch*seq, hidden]
                X::Tensor x_reshaped = X::view(x, { -1, mHiddenSize });

                // 2. Get routing probabilities and expert indices
                auto [probs, indices] = mGate.forward(x_reshaped);
                // probs: [batch*seq, k] - probability for each selected expert
                // indices: [batch*seq, k] - indices of selected experts

                // 3. Create output tensor initialized with zeros
                X::Tensor output = X::zeros_like(x_reshaped);

                // 4. For load balancing statistics
                std::vector<int> token_counts(mNumExperts, 0);

                // 5. Process each expert locally (only the ones assigned to this GPU)
                for (int expert_idx = mExpertsStartIdx; expert_idx < mExpertsStartIdx + mLocalExperts; ++expert_idx) {
                    // Loop through each expert position (1st choice, 2nd choice, etc.)
                    for (int k = 0; k < mExpertsPerToken; ++k) {
                        // Find tokens that selected this expert as their k-th choice
                        X::Tensor selected_mask = X::eq(indices.select(1, k), expert_idx);

                        // Only process if any tokens selected this expert
                        if (X::sum(selected_mask, 0).item<int>() > 0) {
                            // Get indices of tokens that selected this expert
                            X::Tensor token_indices = X::nonzero(selected_mask).squeeze(1);

                            // Get the corresponding input vectors
                            X::Tensor expert_inputs = X::index_select(x_reshaped, 0, token_indices);

                            // Get the corresponding routing weights
                            X::Tensor expert_probs = X::index_select(probs.select(1, k), 0, token_indices);

                            // Process inputs with this expert
                            X::Tensor expert_output = mExperts[expert_idx]->forward(expert_inputs);

                            // Scale outputs by routing probabilities
                            expert_output = X::mul(expert_output, expert_probs.unsqueeze(1));

                            // Add to output tensor using scatter addition
                            // This efficiently handles the case where multiple experts contribute to the same token
                            for (int i = 0; i < token_indices.size(0); ++i) {
                                int token_idx = token_indices[i].item<int>();
                                output[token_idx] = X::add(output[token_idx], expert_output[i]);
                            }

                            // Update token count for this expert (for load balancing analysis)
                            token_counts[expert_idx] += token_indices.size(0);
                        }
                    }
                }

                // 6. Process with shared experts - these are applied to all tokens
                X::Tensor shared_output = mSharedExperts.forward(x_reshaped);

                // 7. Add shared experts output to routed experts output
                output = X::add(output, shared_output);

                // 8. All-reduce to combine expert outputs from all GPUs
                output = X::all_reduce(output);

                // 9. Reshape back to original shape: [batch*seq, hidden] -> [batch, seq, hidden]
                output = X::view(output, { batch_size, seq_len, mHiddenSize });

                return output;
            }
        };

        // Dense transformer block - used for first layer
        class DenseBlock : public Module {
        private:
            MultiHeadAttention mAttention;  // Self-attention mechanism
            FeedForward mFeedForward;       // Feed-forward network
            RMSNorm mAttnNorm;              // Pre-attention normalization
            RMSNorm mFfnNorm;               // Pre-feedforward normalization

        public:
            DenseBlock(const ModelConfig& config)
                : mAttention(config),
                mFeedForward(config.hidden_size, config.intermediate_size),
                mAttnNorm(config.hidden_size, config.rms_norm_eps),
                mFfnNorm(config.hidden_size, config.rms_norm_eps) {
            }

            X::Tensor forward(const X::Tensor& x, int start_pos = 0,
                const X::Tensor& attention_mask = X::Tensor()) override {
                // 1. Apply pre-attention normalization
                X::Tensor attn_input = mAttnNorm.forward(x);

                // 2. Apply self-attention
                X::Tensor attn_output = mAttention.forward(attn_input, start_pos, attention_mask);

                // 3. Add residual connection
                X::Tensor hidden_states = X::add(x, attn_output);

                // 4. Apply pre-feedforward normalization
                X::Tensor ffn_input = mFfnNorm.forward(hidden_states);

                // 5. Apply feed-forward network
                X::Tensor ffn_output = mFeedForward.forward(ffn_input);

                // 6. Add second residual connection
                X::Tensor output = X::add(hidden_states, ffn_output);

                return output;
            }
        };

        // MoE transformer block - used for layers 1-27
        class MoEBlock : public Module {
        private:
            MultiHeadAttention mAttention;  // Self-attention mechanism
            MixtureOfExperts mFeedForward;  // MoE feed-forward network
            RMSNorm mAttnNorm;              // Pre-attention normalization
            RMSNorm mFfnNorm;               // Pre-feedforward normalization

        public:
            MoEBlock(const ModelConfig& config)
                : mAttention(config),
                mFeedForward(config),
                mAttnNorm(config.hidden_size, config.rms_norm_eps),
                mFfnNorm(config.hidden_size, config.rms_norm_eps) {
            }

            X::Tensor forward(const X::Tensor& x, int start_pos = 0,
                const X::Tensor& attention_mask = X::Tensor()) override {
                // Structure identical to DenseBlock, but using MoE instead of standard FFN

                // 1. Apply pre-attention normalization
                X::Tensor attn_input = mAttnNorm.forward(x);

                // 2. Apply self-attention
                X::Tensor attn_output = mAttention.forward(attn_input, start_pos, attention_mask);

                // 3. Add residual connection
                X::Tensor hidden_states = X::add(x, attn_output);

                // 4. Apply pre-feedforward normalization
                X::Tensor ffn_input = mFfnNorm.forward(hidden_states);

                // 5. Apply MoE feed-forward network
                X::Tensor ffn_output = mFeedForward.forward(ffn_input);

                // 6. Add second residual connection
                X::Tensor output = X::add(hidden_states, ffn_output);

                return output;
            }
        };

        // Main transformer model
        class DeepSeekTransformer : public Module {
        private:
            ModelConfig mConfig;               // Model configuration
            ParallelEmbedding mEmbedding;      // Token embedding layer
            std::vector<std::unique_ptr<Module>> mLayers;  // Transformer layers
            RMSNorm mFinalNorm;                // Final layer normalization
            ColumnParallelLinear mLmHead;      // Output projection to vocabulary

        public:
            DeepSeekTransformer(const ModelConfig& config)
                : mConfig(config),
                mEmbedding(config.vocab_size, config.hidden_size),
                mFinalNorm(config.hidden_size, config.rms_norm_eps),
                mLmHead(config.hidden_size, config.vocab_size) {

                // Initialize layers - first layer is dense, rest are MoE
                for (int i = 0; i < config.num_hidden_layers; ++i) {
                    if (i < config.first_k_dense_replace) {
                        mLayers.push_back(std::make_unique<DenseBlock>(config));
                    }
                    else {
                        mLayers.push_back(std::make_unique<MoEBlock>(config));
                    }
                }
            }

            X::Tensor forward(const X::Tensor& input_ids,
                int start_pos = 0,
                bool return_logits = true) override {
                // 1. Convert token IDs to embeddings
                X::Tensor hidden_states = mEmbedding.forward(input_ids);

                // 2. Create causal attention mask if sequence length > 1
                // The mask prevents tokens from attending to future tokens
                X::Tensor attention_mask;
                if (input_ids.size(1) > 1) {
                    int seq_len = input_ids.size(1);
                    // Create a mask filled with negative infinity
                    attention_mask = X::full({ seq_len, seq_len }, -std::numeric_limits<float>::infinity());

                    // Create lower triangular pattern (set to 0 to allow attention)
                    // In an actual implementation, this would use a more efficient tensor operation
                    for (int i = 0; i < seq_len; ++i) {
                        for (int j = 0; j <= i; ++j) {
                            // Set lower triangle to 0 (meaning attention is allowed)
                            // attention_mask[i, j] = 0.0
                        }
                    }

                    // Reshape for broadcasting with attention scores
                    attention_mask = X::view(attention_mask, { 1, 1, seq_len, seq_len });
                }

                // 3. Apply all transformer layers sequentially
                for (auto& layer : mLayers) {
                    hidden_states = layer->forward(hidden_states, start_pos, attention_mask);
                }

                // 4. Apply final layer normalization
                hidden_states = mFinalNorm.forward(hidden_states);

                // 5. Return logits if requested (for training or generation)
                if (return_logits) {
                    X::Tensor last_hidden;

                    if (input_ids.size(1) == 1) {
                        // For single-token generation, use the whole tensor
                        last_hidden = hidden_states;
                    }
                    else {
                        // For multi-token input, we usually only need the last token's output
                        // This is used for next-token prediction in autoregressive generation
                        last_hidden = hidden_states.select(1, hidden_states.size(1) - 1).unsqueeze(1);
                    }

                    // Project to vocabulary logits
                    X::Tensor logits = mLmHead.forward(last_hidden);

                    // If distributed, gather logits from all ranks
                    if (X::get_world_size() > 1) {
                        std::vector<X::Tensor> all_logits(X::get_world_size());

                        // Initialize tensors for gathering
                        for (int i = 0; i < X::get_world_size(); ++i) {
                            all_logits[i] = X::Tensor();
                        }

                        // This would be implemented as a collective operation to gather
                        // distributed parts of the logits from all GPUs
                        X::Tensor full_logits = X::cat(all_logits, -1);
                        return full_logits;
                    }
                    return logits;
                }

                // Return hidden states if logits not requested (e.g., for feature extraction)
                return hidden_states;
            }
        };
    }// namespace Transformer
} // namespace Garnet