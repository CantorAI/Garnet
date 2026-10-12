// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <utility>
namespace Garnet {
class MetalMatrix {
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit MetalMatrix(std::shared_ptr<Impl> impl): impl_(std::move(impl)) {}
    friend class MetalContext;
public:
    MetalMatrix() = default;
};
// C++ interface keeps Objective-C ownership out of model-facing headers.
class MetalContext {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void EncodePointwise(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&,
        unsigned operation, float epsilon, float offset);
    void EncodeStateKernel(unsigned operation,const std::vector<MetalMatrix>&,
        const MetalMatrix&,const void* parameters,size_t parameterBytes,size_t threads);
public:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    std::string DeviceName() const;
    static std::string AvailableDeviceName();
    MetalMatrix Upload(const std::vector<float>& values, size_t rows, size_t columns);
    MetalMatrix Allocate(size_t rows, size_t columns);
    // Synchronous host update into an existing allocation, between command batches.
    void Update(const MetalMatrix&,const std::vector<float>& values);
    void Update(const MetalMatrix&,const float* values,size_t count);
    void Begin();
    void EncodeMatMul(const MetalMatrix& left, const MetalMatrix& right,
        const MetalMatrix& output, bool transposeRight = false);
    void EncodeRMSNorm(const MetalMatrix& input, const MetalMatrix& weight,
        const MetalMatrix& output, float epsilon, float weightOffset = 1.0f);
    void EncodeSiluMul(const MetalMatrix& gate, const MetalMatrix& up,
        const MetalMatrix& output);
    void EncodeAdd(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    void EncodeMaskRows(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    // Packed rows retain explicit caller-owned states after the output segment.
    void EncodeCausalConv(const MetalMatrix& token,const MetalMatrix& weight,
        const MetalMatrix& state,const MetalMatrix& active,const MetalMatrix& packed,size_t kernel);
    void EncodeGatedDelta(const MetalMatrix& query,const MetalMatrix& key,const MetalMatrix& value,
        const MetalMatrix& decay,const MetalMatrix& beta,const MetalMatrix& state,
        const MetalMatrix& active,const MetalMatrix& packed,size_t heads,size_t keyDim,size_t valueDim);
    void EncodeSliceGroups(const MetalMatrix&,const MetalMatrix&,size_t groups,size_t sourceWidth,size_t start,size_t count);
    void EncodeLinearPrepare(const MetalMatrix& conv,const MetalMatrix& a,const MetalMatrix& b,
        const MetalMatrix& aLog,const MetalMatrix& dtBias,const MetalMatrix& output,
        size_t keyHeads,size_t valueHeads,size_t keyDim,size_t valueDim,size_t kernel);
    void EncodeGatedNorm(const MetalMatrix& core,const MetalMatrix& z,const MetalMatrix& norm,
        const MetalMatrix& output,size_t heads,size_t valueDim,float epsilon);
    void EncodeConcat(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,bool includeThird=true);
    void EncodeNormRope(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t heads,size_t dim,size_t rotary,float epsilon);
    void EncodeKVUpdate(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t kvHeads,size_t dim,size_t capacity);
    void EncodeAttentionScores(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t heads,size_t kvHeads,size_t dim,size_t capacity);
    void EncodeAttentionContext(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        size_t heads,size_t kvHeads,size_t dim,size_t capacity);
    void SubmitAndWait();
    // Fixed-grid image operators; projections continue to use MPS matmul.
    void EncodeVision(unsigned operation,const std::vector<MetalMatrix>& inputs,
        const MetalMatrix& output,size_t tokens,size_t hidden,size_t heads,size_t mergeUnit);
    void Abort();
    std::vector<float> Download(const MetalMatrix& matrix) const;
    // Initial synchronous FP32 correctness primitive. Not the final scheduler:
    // Qwen execution must retain buffers and encode multiple ops per submission.
    std::vector<float> MatMul(const std::vector<float>& left,
        const std::vector<float>& right, size_t rows, size_t inner,
        size_t columns, bool transposeRight = false);
};
}
