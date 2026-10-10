// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <string>
namespace Garnet {
bool ParseGptOssPeerGroupOptions(const char*, size_t, size_t,
    std::array<size_t,2>&, size_t&, int&);
std::string GptOssPeerGroupDescription(const std::array<size_t,2>&,
    size_t capacity, int ctas, size_t ownedBytes, size_t mappedBytes,
    bool gridSignals=false, int threadsPerCta=256);
}
