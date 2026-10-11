// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_profile_validation.h"
#include <cassert>
#include <initializer_list>

using nvinfer1::Dims;
using nvinfer1::DynamicPluginTensorDesc;
using Garnet::GptOssProfileValidation::profileAxisEqual;
using Garnet::GptOssProfileValidation::profileShape;
using Garnet::GptOssProfileValidation::proportionalPageRange;
using Garnet::GptOssProfileValidation::validProfileRange;

namespace {
Dims dims(std::initializer_list<int64_t> values) {
    Dims result{};
    result.nbDims = static_cast<int>(values.size());
    int index = 0;
    for (int64_t value : values) result.d[index++] = value;
    return result;
}
DynamicPluginTensorDesc profile(std::initializer_list<int64_t> declared,
    std::initializer_list<int64_t> minimum, std::initializer_list<int64_t> maximum) {
    DynamicPluginTensorDesc result{};
    result.desc.dims = dims(declared);
    result.min = dims(minimum);
    result.max = dims(maximum);
    // TensorRT V2 build-phase callbacks may leave the newer opt field empty.
    return result;
}
}

int main() {
    auto activations = profile({-1, 16, 2560}, {1, 16, 2560}, {224, 16, 2560});
    assert(validProfileRange(activations));
    assert(profileShape(activations, {-1, 16, 2560}));

    auto pages = profile({18, -1, 16, 4, 64}, {18, 9, 16, 4, 64},
        {18, 2016, 16, 4, 64});
    assert(validProfileRange(pages));
    assert(proportionalPageRange(pages, activations));

    auto mismatchedPages = pages;
    mismatchedPages.max.d[1] = 2015;
    assert(!proportionalPageRange(mismatchedPages, activations));

    auto fixed = profile({8, 16, 2560}, {8, 16, 2560}, {8, 16, 2560});
    assert(validProfileRange(fixed));
    assert(profileShape(fixed, {8, 16, 2560}));
    assert(!profileShape(fixed, {-1, 16, 2560}));

    auto invertedBounds = activations;
    invertedBounds.max.d[0] = 0;
    assert(!validProfileRange(invertedBounds));
    auto undeclaredDynamic = activations;
    undeclaredDynamic.desc.dims.d[0] = 1;
    assert(!validProfileRange(undeclaredDynamic));
    auto incompatibleBatch = profile({-1}, {1}, {223});
    assert(!profileAxisEqual(activations, 0, incompatibleBatch, 0));
}
