//Tranformer model with Mixture of Experts implementation

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <functional>
#include <cmath>

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

        // Dummy implementation to make code compile
        bool defined() const { return true; }

        static Tensor zeros_like(const Tensor& x) { return Tensor(); }
        static Tensor ones(const std::vector<int64_t>& shape) { return Tensor(shape); }
        static Tensor full(const std::vector<int64_t>& shape, float value) { return Tensor(shape); }
        static Tensor zeros(const std::vector<int64_t>& shape) { return Tensor(shape); }
    };

    // Tensor operations - declared but assumed to be implemented elsewhere
    Tensor matmul(const Tensor& a, const Tensor& b) { return Tensor(); }
    Tensor add(const Tensor& a, const Tensor& b) { return Tensor(); }
    Tensor mul(const Tensor& a, const Tensor& b) { return Tensor(); }
    Tensor softmax(const Tensor& x, int dim) { return Tensor(); }
    Tensor silu(const Tensor& x) { return Tensor(); }
    Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-6) { return Tensor(); }
    Tensor split(const Tensor& x, int chunks, int dim) { return Tensor(); }
    Tensor cat(const std::vector<Tensor>& tensors, int dim) { return Tensor(); }
    Tensor view(const Tensor& x, const std::vector<int64_t>& shape) { return Tensor(); }
    Tensor transpose(const Tensor& x, int dim1, int dim2) { return Tensor(); }

    // Masking and indexing operations
    Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) { return Tensor(); }
    Tensor masked_select(const Tensor& x, const Tensor& mask) { return Tensor(); }
    Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) { return Tensor(); }
    Tensor index_select(const Tensor& x, int dim, const Tensor& indices) { return Tensor(); }
    Tensor gather(const Tensor& x, int dim, const Tensor& index) { return Tensor(); }
    Tensor scatter(const Tensor& x, int dim, const Tensor& index, const Tensor& src) { return Tensor(); }

    // Comparison operations
    Tensor eq(const Tensor& x, const Tensor& y) { return Tensor(); }
    Tensor eq(const Tensor& x, int value) { return Tensor(); }
    Tensor lt(const Tensor& x, int value) { return Tensor(); }
    Tensor ge(const Tensor& x, int value) { return Tensor(); }
    Tensor gt(const Tensor& x, float value) { return Tensor(); }
    Tensor logical_or(const Tensor& x, const Tensor& y) { return Tensor(); }
    Tensor topk(const Tensor& x, int k, int dim) { return Tensor(); }
    Tensor nonzero(const Tensor& x) { return Tensor(); }
    Tensor select(const Tensor& x, int dim, int index) { return Tensor(); }
    Tensor slice(const Tensor& x, int dim, int start, int end) { return Tensor(); }
    Tensor sub(const Tensor& x, int value) { return Tensor(); }
    Tensor unsqueeze(const Tensor& x, int dim) { return Tensor(); }

    // Reduction operations
    Tensor sum(const Tensor& x, int dim, bool keepdim = false) { return Tensor(); }

    // Special operations
    Tensor embedding(const Tensor& indices, const Tensor& weight) { return Tensor(); }
    Tensor apply_rotary_emb(const Tensor& x, const Tensor& freqs) { return Tensor(); }

    // Parallelism operations
    Tensor all_reduce(const Tensor& x) { return Tensor(); }
    Tensor all_gather(const Tensor& x, int dim) { return Tensor(); }
    void scatter_tensor(Tensor& dst, const Tensor& src, int dim, int rank, int world_size) {}

    // Environment functions
    int get_rank() { return 0; }
    int get_world_size() { return 1; }
}

// Garnet namespace for model implementation
namespace Garnet {
    namespace Transformer {

        // Configuration for the model
        struct ModelConfig {
            int vocab_size = 102400;
            int hidden_size = 2048;
            int intermediate_size = 10944;
            int moe_intermediate_size = 1408;
            int num_hidden_layers = 28;
            int first_k_dense_replace = 1;
            int num_attention_heads = 16;
            int num_key_value_heads = 16;
            int n_routed_experts = 64;
            int n_shared_experts = 2;
            int num_experts_per_tok = 6;
            float rms_norm_eps = 1e-6;
            float rope_theta = 10000.0;
            bool use_cache = true;
            int max_seq_len = 4096;
        };

        // Base module class
        class Module {
        public:
            virtual ~Module() = default;
            virtual X::Tensor forward(const X::Tensor& input) = 0;
        };

        // RMSNorm layer
        class RMSNorm : public Module {
        private:
            X::Tensor mWeight;
            float mEps;

        public:
            RMSNorm(int size, float eps = 1e-6) : mEps(eps) {
                mWeight = X::Tensor::ones({ size });
            }

            X::Tensor forward(const X::Tensor& input) override {
                return X::rms_norm(input, mWeight, mEps);
            }
        };

        // Base class for parallelized linear layers
        class Linear : public Module {
        protected:
            X::Tensor mWeight;
            X::Tensor mBias;
            int mInFeatures;
            int mOutFeatures;
            bool mHasBias;

        public:
            Linear(int in_features, int out_features, bool bias = false)
                : mInFeatures(in_features), mOutFeatures(out_features), mHasBias(bias) {
                mWeight = X::Tensor({ out_features, in_features });
                if (mHasBias) {
                    mBias = X::Tensor({ out_features });
                }
            }

            virtual X::Tensor forward(const X::Tensor& input) override {
                X::Tensor output = X::matmul(input, X::transpose(mWeight, 0, 1));
                if (mHasBias) {
                    output = X::add(output, mBias);
                }
                return output;
            }
        };

        // Column-parallel linear layer
        class ColumnParallelLinear : public Linear {
        private:
            int mWorldSize;
            int mRank;

        public:
            ColumnParallelLinear(int in_features, int out_features, bool bias = false)
                : Linear(in_features, out_features / X::get_world_size(), bias),
                mWorldSize(X::get_world_size()),
                mRank(X::get_rank()) {
                // Weight and bias are partitioned along output dimension
            }

            X::Tensor forward(const X::Tensor& input) override {
                return Linear::forward(input);
                // No all_reduce needed - each GPU computes a portion of the output
            }
        };

        // Row-parallel linear layer
        class RowParallelLinear : public Linear {
        private:
            int mWorldSize;
            int mRank;

        public:
            RowParallelLinear(int in_features, int out_features, bool bias = false)
                : Linear(in_features / X::get_world_size(), out_features, bias),
                mWorldSize(X::get_world_size()),
                mRank(X::get_rank()) {
                // Weight is partitioned along input dimension
            }

            X::Tensor forward(const X::Tensor& input) override {
                X::Tensor output = Linear::forward(input);
                // All-reduce across GPUs to get complete output
                output = X::all_reduce(output);
                return output;
            }
        };

        // Parallel embedding layer
        class ParallelEmbedding : public Module {
        private:
            int mVocabSize;
            int mEmbeddingDim;
            int mWorldSize;
            int mRank;
            X::Tensor mWeight;
            int mLocalVocabSize;
            int mVocabStartIdx;

        public:
            ParallelEmbedding(int vocab_size, int embedding_dim)
                : mVocabSize(vocab_size), mEmbeddingDim(embedding_dim),
                mWorldSize(X::get_world_size()), mRank(X::get_rank()) {
                mLocalVocabSize = mVocabSize / mWorldSize;
                mVocabStartIdx = mRank * mLocalVocabSize;
                mWeight = X::Tensor({ mLocalVocabSize, mEmbeddingDim });
            }

            X::Tensor forward(const X::Tensor& input) override {
                // Create mask for tokens in this rank's vocabulary range
                X::Tensor below_range = X::lt(input, mVocabStartIdx);
                X::Tensor above_range = X::ge(input, mVocabStartIdx + mLocalVocabSize);
                X::Tensor out_of_range = X::logical_or(below_range, above_range);

                // Adjust indices for local vocabulary
                X::Tensor local_indices = X::sub(input, mVocabStartIdx);
                local_indices = X::where(out_of_range, X::zeros_like(local_indices), local_indices);

                // Look up embeddings
                X::Tensor local_output = X::embedding(local_indices, mWeight);

                // Zero out embeddings for tokens not in this rank's range
                local_output = X::where(out_of_range.unsqueeze(-1), X::zeros_like(local_output), local_output);

                // Sum results across all ranks
                X::Tensor output = X::all_reduce(local_output);

                return output;
            }
        };

        // Rotary Positional Embedding implementation
        class RotaryPositionalEmbedding {
        private:
            X::Tensor mFreqsCis;
            int mMaxSeqLen;
            int mRotaryDim;

        public:
            RotaryPositionalEmbedding(int dim, int max_seq_len, float theta = 10000.0f)
                : mMaxSeqLen(max_seq_len), mRotaryDim(dim) {
                // Precompute frequency table for rotary embeddings
                X::Tensor freqs = X::Tensor({ max_seq_len, dim / 2 });
                // This would be populated with proper values in actual implementation
                mFreqsCis = X::Tensor({ max_seq_len, dim / 2 });
            }

            X::Tensor apply(const X::Tensor& x, int offset = 0) {
                // Get sequence length from input tensor
                int seq_len = x.size(1);

                // Extract freqs for current positions
                X::Tensor freqs_for_seq = X::slice(mFreqsCis, 0, offset, offset + seq_len);

                // Reshape x for complex number calculation
                std::vector<int64_t> shape = x.sizes();
                X::Tensor x_reshaped = X::view(x, { shape[0], shape[1], shape[2], shape[3] / 2, 2 });

                // Apply rotary embeddings
                // Implementation depends on complex number support in X::Tensor
                X::Tensor rotated = X::apply_rotary_emb(x_reshaped, freqs_for_seq);

                // Reshape back to original shape
                return X::view(rotated, shape);
            }
        };

        // Multi-head attention implementation with row/column parallelism
        class MultiHeadAttention : public Module {
        private:
            int mHiddenSize;
            int mNumHeads;
            int mHeadDim;
            float mSoftmaxScale;
            int mLocalHeads;

            ColumnParallelLinear mQProj;
            ColumnParallelLinear mKProj;
            ColumnParallelLinear mVProj;
            RowParallelLinear mOProj;

            RotaryPositionalEmbedding mRotaryEmb;
            bool mUseCache;
            std::unordered_map<std::string, X::Tensor> mCache;

        public:
            MultiHeadAttention(const ModelConfig& config)
                : mHiddenSize(config.hidden_size),
                mNumHeads(config.num_attention_heads),
                mHeadDim(config.hidden_size / config.num_attention_heads),
                mSoftmaxScale(1.0f / std::sqrt(mHeadDim)),
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

                // Compute query, key, value projections
                X::Tensor q = mQProj.forward(x);
                X::Tensor k = mKProj.forward(x);
                X::Tensor v = mVProj.forward(x);

                // Reshape for multi-head attention
                q = X::view(q, { batch_size, seq_len, mLocalHeads, mHeadDim });
                k = X::view(k, { batch_size, seq_len, mLocalHeads, mHeadDim });
                v = X::view(v, { batch_size, seq_len, mLocalHeads, mHeadDim });

                // Apply rotary embeddings
                q = mRotaryEmb.apply(q, start_pos);
                k = mRotaryEmb.apply(k, start_pos);

                // Handle KV cache for generation
                if (mUseCache) {
                    if (mCache.find("k") != mCache.end() && mCache.find("v") != mCache.end()) {
                        // Concatenate with cached keys and values
                        X::Tensor k_cache = mCache["k"];
                        X::Tensor v_cache = mCache["v"];

                        k = X::cat({ k_cache, k }, 1); // Concat along sequence dimension
                        v = X::cat({ v_cache, v }, 1);
                    }

                    // Update cache
                    mCache["k"] = k;
                    mCache["v"] = v;
                }

                // Transpose q, k, v for attention computation
                q = X::transpose(q, 1, 2); // [batch, head, seq, dim]
                k = X::transpose(k, 1, 2); // [batch, head, seq, dim]
                v = X::transpose(v, 1, 2); // [batch, head, seq, dim]

                // Compute attention scores
                k = X::transpose(k, 2, 3); // [batch, head, dim, seq]
                X::Tensor scores = X::matmul(q, k); // [batch, head, seq_q, seq_k]
                scores = X::mul(scores, mSoftmaxScale);

                // Apply attention mask if provided
                if (attention_mask.defined()) {
                    scores = X::add(scores, attention_mask);
                }

                // Apply softmax
                X::Tensor attn_weights = X::softmax(scores, -1);

                // Apply attention to values
                X::Tensor context = X::matmul(attn_weights, v); // [batch, head, seq_q, dim]

                // Transpose back
                context = X::transpose(context, 1, 2); // [batch, seq, head, dim]

                // Reshape and apply output projection
                context = X::view(context, { batch_size, seq_len, mLocalHeads * mHeadDim });
                return mOProj.forward(context);
            }
        };

        // Dense feed-forward network
        class FeedForward : public Module {
        private:
            ColumnParallelLinear mGateProj;
            ColumnParallelLinear mUpProj;
            RowParallelLinear mDownProj;

        public:
            FeedForward(int hidden_size, int intermediate_size)
                : mGateProj(hidden_size, intermediate_size),
                mUpProj(hidden_size, intermediate_size),
                mDownProj(intermediate_size, hidden_size) {
            }

            X::Tensor forward(const X::Tensor& x) override {
                X::Tensor gate = X::silu(mGateProj.forward(x));
                X::Tensor up = mUpProj.forward(x);
                X::Tensor activation = X::mul(gate, up);
                X::Tensor output = mDownProj.forward(activation);
                return output;
            }
        };

        // Expert network for MoE
        class ExpertNetwork : public Module {
        private:
            X::Tensor mW1; // gate projection
            X::Tensor mW2; // down projection
            X::Tensor mW3; // up projection
            int mHiddenSize;
            int mIntermediateSize;

        public:
            ExpertNetwork(int hidden_size, int intermediate_size)
                : mHiddenSize(hidden_size), mIntermediateSize(intermediate_size) {
                mW1 = X::Tensor({ intermediate_size, hidden_size });
                mW2 = X::Tensor({ hidden_size, intermediate_size });
                mW3 = X::Tensor({ intermediate_size, hidden_size });
            }

            X::Tensor forward(const X::Tensor& x) override {
                // Compute gate projection
                X::Tensor gate = X::matmul(x, X::transpose(mW1, 0, 1));
                gate = X::silu(gate);

                // Compute up projection
                X::Tensor up = X::matmul(x, X::transpose(mW3, 0, 1));

                // Compute intermediate activation
                X::Tensor intermediate = X::mul(gate, up);

                // Compute down projection
                X::Tensor output = X::matmul(intermediate, X::transpose(mW2, 0, 1));

                return output;
            }
        };

        // Routing gate for MoE
        class RoutingGate : public Module {
        private:
            X::Tensor mWeight;
            int mHiddenSize;
            int mNumExperts;
            int mTopK;

        public:
            RoutingGate(int hidden_size, int num_experts, int top_k)
                : mHiddenSize(hidden_size), mNumExperts(num_experts), mTopK(top_k) {
                mWeight = X::Tensor({ num_experts, hidden_size });
            }

            std::pair<X::Tensor, X::Tensor> forward(const X::Tensor& x) {
                // Compute routing scores
                X::Tensor scores = X::matmul(x, X::transpose(mWeight, 0, 1));

                // Apply softmax
                X::Tensor probs = X::softmax(scores, -1);

                // Select top-k experts
                X::Tensor topk_result = X::topk(probs, mTopK, -1);
                X::Tensor topk_probs = std::get<0>(topk_result);     // Values
                X::Tensor topk_indices = std::get<1>(topk_result);   // Indices

                return { topk_probs, topk_indices };
            }
        };

        // Mixture of Experts implementation
        class MixtureOfExperts : public Module {
        private:
            int mHiddenSize;
            int mNumExperts;
            int mExpertsPerToken;
            int mWorldSize;
            int mRank;
            int mLocalExperts;
            int mExpertsStartIdx;

            RoutingGate mGate;
            std::vector<std::unique_ptr<ExpertNetwork>> mExperts;
            FeedForward mSharedExperts;

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

                // Initialize expert networks
                for (int i = 0; i < mNumExperts; ++i) {
                    if (i >= mExpertsStartIdx && i < mExpertsStartIdx + mLocalExperts) {
                        mExperts.push_back(std::make_unique<ExpertNetwork>(
                            config.hidden_size, config.moe_intermediate_size));
                    }
                    else {
                        // Placeholder for experts not on this rank
                        mExperts.push_back(nullptr);
                    }
                }
            }

            X::Tensor forward(const X::Tensor& x) override {
                // Original shape
                int batch_size = x.size(0);
                int seq_len = x.size(1);

                // Reshape input for routing
                X::Tensor x_reshaped = X::view(x, { -1, mHiddenSize });

                // Get routing probabilities and expert indices
                auto [probs, indices] = mGate.forward(x_reshaped);

                // Create output tensor initialized with zeros
                X::Tensor output = X::zeros_like(x_reshaped);

                // Count tokens per expert for load balancing statistics
                std::vector<int> token_counts(mNumExperts, 0);

                // Process each expert locally
                for (int expert_idx = mExpertsStartIdx; expert_idx < mExpertsStartIdx + mLocalExperts; ++expert_idx) {
                    // Find all tokens assigned to this expert in the batch
                    for (int k = 0; k < mExpertsPerToken; ++k) {
                        // Create mask where this expert was selected as the k-th expert
                        X::Tensor selected_mask = X::eq(indices.select(1, k), expert_idx);

                        if (X::sum(selected_mask, 0).item<int>() > 0) {
                            // Get the token indices where this expert was selected
                            X::Tensor token_indices = X::nonzero(selected_mask).squeeze(1);

                            // Get the corresponding input vectors
                            X::Tensor expert_inputs = X::index_select(x_reshaped, 0, token_indices);

                            // Get the corresponding routing weights
                            X::Tensor expert_probs = X::index_select(probs.select(1, k), 0, token_indices);

                            // Process inputs with this expert
                            X::Tensor expert_output = mExperts[expert_idx]->forward(expert_inputs);

                            // Scale outputs by routing probabilities
                            expert_output = X::mul(expert_output, expert_probs.unsqueeze(1));

                            // Add to output tensor using scatter
                            for (int i = 0; i < token_indices.size(0); ++i) {
                                int token_idx = token_indices[i].item<int>();
                                output[token_idx] = X::add(output[token_idx], expert_output[i]);
                            }

                            // Update token count for this expert
                            token_counts[expert_idx] += token_indices.size(0);
                        }
                    }
                }

                // Process with shared experts
                X::Tensor shared_output = mSharedExperts.forward(x_reshaped);

                // Add shared experts output to routed experts output
                output = X::add(output, shared_output);

                // All-reduce to combine expert outputs from all ranks
                output = X::all_reduce(output);

                // Reshape back to original shape
                output = X::view(output, { batch_size, seq_len, mHiddenSize });

                return output;
            }
        };

        // Dense transformer block
        class DenseBlock : public Module {
        private:
            MultiHeadAttention mAttention;
            FeedForward mFeedForward;
            RMSNorm mAttnNorm;
            RMSNorm mFfnNorm;

        public:
            DenseBlock(const ModelConfig& config)
                : mAttention(config),
                mFeedForward(config.hidden_size, config.intermediate_size),
                mAttnNorm(config.hidden_size, config.rms_norm_eps),
                mFfnNorm(config.hidden_size, config.rms_norm_eps) {
            }

            X::Tensor forward(const X::Tensor& x, int start_pos = 0,
                const X::Tensor& attention_mask = X::Tensor()) override {
                // Apply attention
                X::Tensor attn_input = mAttnNorm.forward(x);
                X::Tensor attn_output = mAttention.forward(attn_input, start_pos, attention_mask);
                X::Tensor hidden_states = X::add(x, attn_output);

                // Apply feed-forward
                X::Tensor ffn_input = mFfnNorm.forward(hidden_states);
                X::Tensor ffn_output = mFeedForward.forward(ffn_input);
                X::Tensor output = X::add(hidden_states, ffn_output);

                return output;
            }
        };

        // MoE transformer block
        class MoEBlock : public Module {
        private:
            MultiHeadAttention mAttention;
            MixtureOfExperts mFeedForward;
            RMSNorm mAttnNorm;
            RMSNorm mFfnNorm;

        public:
            MoEBlock(const ModelConfig& config)
                : mAttention(config),
                mFeedForward(config),
                mAttnNorm(config.hidden_size, config.rms_norm_eps),
                mFfnNorm(config.hidden_size, config.rms_norm_eps) {
            }

            X::Tensor forward(const X::Tensor& x, int start_pos = 0,
                const X::Tensor& attention_mask = X::Tensor()) override {
                // Apply attention
                X::Tensor attn_input = mAttnNorm.forward(x);
                X::Tensor attn_output = mAttention.forward(attn_input, start_pos, attention_mask);
                X::Tensor hidden_states = X::add(x, attn_output);

                // Apply MoE feed-forward
                X::Tensor ffn_input = mFfnNorm.forward(hidden_states);
                X::Tensor ffn_output = mFeedForward.forward(ffn_input);
                X::Tensor output = X::add(hidden_states, ffn_output);

                return output;
            }
        };

        // Main transformer model
        class DeepSeekTransformer : public Module {
        private:
            ModelConfig mConfig;
            ParallelEmbedding mEmbedding;
            std::vector<std::unique_ptr<Module>> mLayers;
            RMSNorm mFinalNorm;
            ColumnParallelLinear mLmHead;

        public:
            DeepSeekTransformer(const ModelConfig& config)
                : mConfig(config),
                mEmbedding(config.vocab_size, config.hidden_size),
                mFinalNorm(config.hidden_size, config.rms_norm_eps),
                mLmHead(config.hidden_size, config.vocab_size) {

                // Initialize layers - first k layers are dense, rest are MoE
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
                // Get embeddings
                X::Tensor hidden_states = mEmbedding.forward(input_ids);

                // Create causal mask if sequence length > 1
                X::Tensor attention_mask;
                if (input_ids.size(1) > 1) {
                    int seq_len = input_ids.size(1);
                    attention_mask = X::full({ seq_len, seq_len }, -std::numeric_limits<float>::infinity());

                    // Fill the lower triangular part with zeros (allows attention to previous tokens)
                    for (int i = 0; i < seq_len; ++i) {
                        for (int j = 0; j <= i; ++j) {
                            // attention_mask[i, j] = 0.0
                            // This would be done with a more efficient tensor operation in practice
                        }
                    }

                    // Expand dimensions for broadcasting with attention scores
                    attention_mask = X::view(attention_mask, { 1, 1, seq_len, seq_len });
                }

                // Apply transformer layers
                for (auto& layer : mLayers) {
                    hidden_states = layer->forward(hidden_states, start_pos, attention_mask);
                }

                // Apply final normalization
                hidden_states = mFinalNorm.forward(hidden_states);

                if (return_logits) {
                    X::Tensor last_hidden;

                    if (input_ids.size(1) == 1) {
                        // For generation, we only need the last token
                        last_hidden = hidden_states;
                    }
                    else {
                        // For multiple tokens, we need the last token of each sequence
                        last_hidden = hidden_states.select(1, hidden_states.size(1) - 1).unsqueeze(1);
                    }

                    // Project to vocabulary
                    X::Tensor logits = mLmHead.forward(last_hidden);

                    // If distributed, gather logits from all ranks
                    if (X::get_world_size() > 1) {
                        std::vector<X::Tensor> all_logits(X::get_world_size());
                        for (int i = 0; i < X::get_world_size(); ++i) {
                            all_logits[i] = X::Tensor();
                        }
                        // This would be a collective operation to gather distributed parts
                        X::Tensor full_logits = X::cat(all_logits, -1);
                        return full_logits;
                    }
                    return logits;
                }

                return hidden_states;
            }
        };
	}// namespace Transformer
} // namespace Garnet