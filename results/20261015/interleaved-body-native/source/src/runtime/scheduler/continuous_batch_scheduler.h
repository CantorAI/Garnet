// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_kv_pool.h"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace Garnet
{
    enum class RequestStage
    {
        Queued,
        Prefill,
        Decode,
        Finished,
        Cancelled,
    };

    struct GenerationRequest
    {
        std::uint64_t id = 0;
        int promptTokens = 0;
        int maxNewTokens = 0;
        int priority = 0;
        bool hasVision = false;
        int positionDelta = 0;
    };

    struct ScheduledRequest
    {
        std::uint64_t id = 0;
        int tokenId = 0;
        int contextLength = 0;
        int slotPosition = 0;
        int promptTokens = 0;
        int generatedTokens = 0;
        int positionDelta = 0;
        std::vector<int> pageTable;
    };

    struct BatchPlan
    {
        RequestStage stage = RequestStage::Queued;
        int bucketSize = 0;
        int scheduledTokens = 0;
        std::vector<ScheduledRequest> requests;
        std::vector<int> activeMask;
    };

    struct SchedulerTick
    {
        std::vector<BatchPlan> batches;
    };

    struct ContinuousBatchSchedulerConfig
    {
        std::vector<int> decodeBuckets{1, 2, 4, 8};
        int maxPrefillBatch = 4;
        int maxPrefillTokensPerTick = 2048;
        int maxActiveRequests = 256;
    };

    // Backend-neutral policy engine. It creates tensor-ready batch metadata;
    // a GPU executor binds those tensors to the compiled graph entrypoint.
    class ContinuousBatchScheduler
    {
        struct State
        {
            GenerationRequest request;
            RequestStage stage = RequestStage::Queued;
            std::uint64_t enqueueOrder = 0;
            int generatedTokens = 0;
            int nextTokenId = 0;
        };

        ContinuousBatchSchedulerConfig m_config;
        std::unordered_map<std::uint64_t, State> m_requests;
        std::deque<std::uint64_t> m_prefillQueue;
        std::vector<std::uint64_t> m_decode;
        std::uint64_t m_nextRequestId = 1;
        std::uint64_t m_enqueueClock = 0;

        int DecodeBucket(int activeCount) const;
        static bool HigherPriority(const State* left, const State* right);

    public:
        explicit ContinuousBatchScheduler(
            ContinuousBatchSchedulerConfig config = {});

        std::uint64_t Submit(GenerationRequest request);
        bool Cancel(std::uint64_t requestId, PagedKVPool& kvPool);
        SchedulerTick Schedule(PagedKVPool& kvPool);
        bool CompletePrefill(
            std::uint64_t requestId,
            int firstTokenId,
            PagedKVPool& kvPool);
        bool CompleteDecode(
            std::uint64_t requestId,
            int nextTokenId,
            bool finished,
            PagedKVPool& kvPool);

        RequestStage Stage(std::uint64_t requestId) const;
        int ActiveRequestCount() const;
        int QueuedRequestCount() const;
    };
}
