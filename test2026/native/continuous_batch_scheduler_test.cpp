// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batch_scheduler.h"

#include <iostream>
#include <stdexcept>

static void Require(bool condition)
{
    if (!condition) throw std::runtime_error("scheduler lifecycle check failed");
}

int main()
{
    Garnet::PagedKVPool pool;
    Garnet::PagedKVPoolConfig poolConfig;
    poolConfig.totalPages = 32;
    poolConfig.pageSize = 16;
    poolConfig.numLayers = 2;
    poolConfig.numKVHeads = 2;
    poolConfig.headDim = 16;
    Require(pool.Initialize(poolConfig));

    Garnet::ContinuousBatchSchedulerConfig schedulerConfig;
    schedulerConfig.decodeBuckets = {1, 2, 4, 8};
    schedulerConfig.maxPrefillBatch = 4;
    schedulerConfig.maxPrefillTokensPerTick = 256;
    Garnet::ContinuousBatchScheduler scheduler(schedulerConfig);

    Garnet::GenerationRequest first;
    first.promptTokens = 33;
    first.maxNewTokens = 8;
    first.priority = 1;
    const auto firstId = scheduler.Submit(first);

    Garnet::GenerationRequest second;
    second.promptTokens = 17;
    second.maxNewTokens = 8;
    second.priority = 2;
    second.positionDelta = -3;
    const auto secondId = scheduler.Submit(second);
    Require(firstId != 0 && secondId != 0);

    auto tick = scheduler.Schedule(pool);
    Require(tick.batches.size() == 1);
    Require(tick.batches[0].stage == Garnet::RequestStage::Prefill);
    Require(tick.batches[0].requests.size() == 2);
    Require(scheduler.CompletePrefill(firstId, 101, pool));
    Require(scheduler.CompletePrefill(secondId, 202, pool));

    tick = scheduler.Schedule(pool);
    Require(!tick.batches.empty());
    const auto& decode = tick.batches[0];
    Require(decode.stage == Garnet::RequestStage::Decode);
    Require(decode.bucketSize == 2);
    Require(decode.requests.size() == 2);
    // Priority controls row order without changing per-request page ownership.
    Require(decode.requests[0].id == secondId);
    Require(decode.requests[1].id == firstId);
    Require(decode.requests[0].contextLength == 18);
    Require(decode.requests[0].slotPosition == 17);
    Require(decode.requests[0].positionDelta == -3);
    Require(decode.requests[1].contextLength == 34);
    Require(decode.requests[1].slotPosition == 33);

    Require(scheduler.CompleteDecode(secondId, 203, true, pool));
    Garnet::GenerationRequest third;
    third.promptTokens = 16;
    third.maxNewTokens = 4;
    const auto thirdId = scheduler.Submit(third);
    tick = scheduler.Schedule(pool);
    Require(tick.batches.size() == 2);
    Require(tick.batches[0].stage == Garnet::RequestStage::Decode);
    Require(tick.batches[0].requests.size() == 1);
    Require(tick.batches[1].stage == Garnet::RequestStage::Prefill);
    Require(tick.batches[1].requests[0].id == thirdId);

    Require(scheduler.Cancel(firstId, pool));
    Require(pool.Stats().sequenceCount == 1);
    Require(scheduler.Cancel(thirdId, pool));
    Require(pool.Stats().usedPages == 0);
    std::cout << "continuous batching and global paged KV lifecycle passed\n";
    return 0;
}
