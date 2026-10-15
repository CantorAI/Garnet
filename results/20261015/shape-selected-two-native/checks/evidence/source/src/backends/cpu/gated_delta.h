// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <string>
namespace Garnet {
// One recurrent step. Layouts: Q/K [B,H,K], V/out [B,H,V],
// gates [B,H], state [B,H,K,V]. State is caller-owned per request/layer.
// Q/K are L2-normalized here; queries are additionally scaled by sqrt(K).
bool GatedDeltaStep(const float* query, const float* key, const float* value,
    const float* logDecay, const float* beta, const unsigned char* active,
    std::size_t batch, std::size_t heads, std::size_t keyDim,
    std::size_t valueDim, float* state, float* output, std::string& error);
}
