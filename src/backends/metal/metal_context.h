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
public:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    std::string DeviceName() const;
    MetalMatrix Upload(const std::vector<float>& values, size_t rows, size_t columns);
    MetalMatrix Allocate(size_t rows, size_t columns);
    void Begin();
    void EncodeMatMul(const MetalMatrix& left, const MetalMatrix& right,
        const MetalMatrix& output, bool transposeRight = false);
    void EncodeRMSNorm(const MetalMatrix& input, const MetalMatrix& weight,
        const MetalMatrix& output, float epsilon, float weightOffset = 1.0f);
    void EncodeSiluMul(const MetalMatrix& gate, const MetalMatrix& up,
        const MetalMatrix& output);
    void EncodeAdd(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    void EncodeMaskRows(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    void SubmitAndWait();
    void Abort();
    std::vector<float> Download(const MetalMatrix& matrix) const;
    // Initial synchronous FP32 correctness primitive. Not the final scheduler:
    // Qwen execution must retain buffers and encode multiple ops per submission.
    std::vector<float> MatMul(const std::vector<float>& left,
        const std::vector<float>& right, size_t rows, size_t inner,
        size_t columns, bool transposeRight = false);
};
}
