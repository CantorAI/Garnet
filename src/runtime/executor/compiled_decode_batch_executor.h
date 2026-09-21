// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compiled_model_runtime.h"
#include "continuous_batch_scheduler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Garnet
{
    struct DecodeBatchExecutionResult
    {
        std::vector<std::uint64_t> requestIds;
        std::vector<long long> tokenIds;
        double executionMilliseconds = 0.0;
    };

    // Packs backend-neutral scheduler plans into stable-address XLang tensors
    // and invokes a compiled model runtime. The runtime may be TensorRT today
    // or another backend implementing the same compiled graph contract later.
    class CompiledDecodeBatchExecutor
    {
        std::shared_ptr<CompiledModelRuntime> m_runtime;
        std::shared_ptr<PagedKVPool> m_kvPool;
        int m_bucketSize = 0;
        int m_maxLogicalPages = 0;
        int m_positionComponents = 3;

        X::Value m_inputIds;
        X::Value m_positionIds;
        X::Value m_keyPages;
        X::Value m_valuePages;
        X::Value m_pageTable;
        X::Value m_contextLengths;
        X::Value m_slotPositions;
        X::Value m_activeMask;
        X::Value m_inputs;

        bool Upload(X::Value tensor, const void* source, std::size_t bytes);

    public:
        bool Initialize(
            X3PackageHost* host,
            std::shared_ptr<CompiledModelRuntime> runtime,
            std::shared_ptr<PagedKVPool> kvPool,
            int bucketSize,
            int maxLogicalPages,
            std::string& errorMessage,
            int positionComponents = 3);

        bool Execute(
            const BatchPlan& plan,
            DecodeBatchExecutionResult& result,
            std::string& errorMessage);
    };
}
