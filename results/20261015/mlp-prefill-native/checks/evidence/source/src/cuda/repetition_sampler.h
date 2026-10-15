// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

cudaError_t sampleRepetitionFP32(const float* logits, const unsigned char* seen,
    float penalty, long long* token, float* score, int vocab, cudaStream_t stream);
cudaError_t sampleRepetitionBF16(const __nv_bfloat16* logits, const unsigned char* seen,
    float penalty, long long* token, float* score, int vocab, cudaStream_t stream);
