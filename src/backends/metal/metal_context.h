// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace Garnet {
// C++ interface keeps Objective-C ownership out of model-facing headers.
class MetalContext {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    std::string DeviceName() const;
    // Initial synchronous FP32 correctness primitive. Not the final scheduler:
    // Qwen execution must retain buffers and encode multiple ops per submission.
    std::vector<float> MatMul(const std::vector<float>& left,
        const std::vector<float>& right, size_t rows, size_t inner,
        size_t columns, bool transposeRight = false);
};
}
