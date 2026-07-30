#include "continuous_batch_scheduler.h"

#include <cassert>
#include <iostream>

int main()
{
    Garnet::PagedKVPool pool;
    Garnet::PagedKVPoolConfig poolConfig;
    poolConfig.totalPages = 32;
    poolConfig.pageSize = 16;
    poolConfig.numLayers = 2;
    poolConfig.numKVHeads = 2;
    poolConfig.headDim = 16;
    assert(pool.Initialize(poolConfig));

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
    assert(firstId != 0 && secondId != 0);

    auto tick = scheduler.Schedule(pool);
    assert(tick.batches.size() == 1);
    assert(tick.batches[0].stage == Garnet::RequestStage::Prefill);
    assert(tick.batches[0].requests.size() == 2);
    assert(scheduler.CompletePrefill(firstId, 101, pool));
    assert(scheduler.CompletePrefill(secondId, 202, pool));

    tick = scheduler.Schedule(pool);
    assert(!tick.batches.empty());
    const auto& decode = tick.batches[0];
    assert(decode.stage == Garnet::RequestStage::Decode);
    assert(decode.bucketSize == 2);
    assert(decode.requests.size() == 2);
    // Priority controls row order without changing per-request page ownership.
    assert(decode.requests[0].id == secondId);
    assert(decode.requests[1].id == firstId);
    assert(decode.requests[0].contextLength == 18);
    assert(decode.requests[0].slotPosition == 17);
    assert(decode.requests[0].positionDelta == -3);
    assert(decode.requests[1].contextLength == 34);
    assert(decode.requests[1].slotPosition == 33);

    assert(scheduler.CompleteDecode(secondId, 203, true, pool));
    Garnet::GenerationRequest third;
    third.promptTokens = 16;
    third.maxNewTokens = 4;
    const auto thirdId = scheduler.Submit(third);
    tick = scheduler.Schedule(pool);
    assert(tick.batches.size() == 2);
    assert(tick.batches[0].stage == Garnet::RequestStage::Decode);
    assert(tick.batches[0].requests.size() == 1);
    assert(tick.batches[1].stage == Garnet::RequestStage::Prefill);
    assert(tick.batches[1].requests[0].id == thirdId);

    assert(scheduler.Cancel(firstId, pool));
    assert(pool.Stats().sequenceCount == 1);
    assert(scheduler.Cancel(thirdId, pool));
    assert(pool.Stats().usedPages == 0);
    std::cout << "continuous batching and global paged KV lifecycle passed\n";
    return 0;
}
