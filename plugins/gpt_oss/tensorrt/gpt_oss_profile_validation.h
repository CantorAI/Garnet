// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <NvInferRuntime.h>
#include <cstdint>
#include <initializer_list>

namespace Garnet::GptOssProfileValidation {

inline bool validProfileRange(const nvinfer1::DynamicPluginTensorDesc& input) {
    const auto& minimum = input.min;
    const auto& maximum = input.max;
    const auto& descriptor = input.desc.dims;
    if (minimum.nbDims <= 0 || minimum.nbDims != maximum.nbDims ||
        descriptor.nbDims != minimum.nbDims) return false;
    for (int dimension = 0; dimension < minimum.nbDims; ++dimension) {
        if (minimum.d[dimension] <= 0 || minimum.d[dimension] > maximum.d[dimension] ||
            (descriptor.d[dimension] != -1 &&
                (descriptor.d[dimension] != minimum.d[dimension] ||
                    minimum.d[dimension] != maximum.d[dimension]))) return false;
    }
    return true;
}

inline bool profileShape(const nvinfer1::DynamicPluginTensorDesc& input,
    std::initializer_list<int64_t> expected) {
    if (!validProfileRange(input) || input.min.nbDims != int(expected.size())) return false;
    int dimension = 0;
    for (int64_t extent : expected) {
        const int64_t minimum = input.min.d[dimension];
        const int64_t maximum = input.max.d[dimension];
        const int64_t declared = input.desc.dims.d[dimension];
        if (extent == -1) {
            if (declared != -1 && (minimum != declared || maximum != declared)) return false;
        } else if (minimum != extent || maximum != extent ||
            (declared != -1 && declared != extent)) {
            return false;
        }
        ++dimension;
    }
    return true;
}

inline bool profileAxisEqual(const nvinfer1::DynamicPluginTensorDesc& left, int leftAxis,
    const nvinfer1::DynamicPluginTensorDesc& right, int rightAxis) {
    return validProfileRange(left) && validProfileRange(right) &&
        leftAxis >= 0 && leftAxis < left.min.nbDims &&
        rightAxis >= 0 && rightAxis < right.min.nbDims &&
        left.min.d[leftAxis] == right.min.d[rightAxis] &&
        left.max.d[leftAxis] == right.max.d[rightAxis];
}

inline bool proportionalPageRange(const nvinfer1::DynamicPluginTensorDesc& pages,
    const nvinfer1::DynamicPluginTensorDesc& activations) {
    if (!validProfileRange(pages) || !validProfileRange(activations) || pages.min.nbDims != 5 ||
        activations.min.nbDims != 3 ||
        !profileShape(pages, {pages.min.d[0], -1, pages.min.d[2], pages.min.d[3], pages.min.d[4]}))
        return false;
    const int64_t minBatch = activations.min.d[0], maxBatch = activations.max.d[0];
    const int64_t minPages = pages.min.d[1], maxPages = pages.max.d[1];
    return minBatch > 0 && maxBatch >= minBatch && minPages > 0 && maxPages >= minPages &&
        minPages % minBatch == 0 && maxPages % maxBatch == 0 &&
        minPages / minBatch == maxPages / maxBatch;
}

} // namespace Garnet::GptOssProfileValidation
