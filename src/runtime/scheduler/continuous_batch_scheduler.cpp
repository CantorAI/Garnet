#include "continuous_batch_scheduler.h"

#include <algorithm>
#include <utility>

namespace Garnet
{
    ContinuousBatchScheduler::ContinuousBatchScheduler(
        ContinuousBatchSchedulerConfig config)
        : m_config(std::move(config))
    {
        std::sort(m_config.decodeBuckets.begin(), m_config.decodeBuckets.end());
        m_config.decodeBuckets.erase(
            std::unique(
                m_config.decodeBuckets.begin(),
                m_config.decodeBuckets.end()),
            m_config.decodeBuckets.end());
        if (m_config.decodeBuckets.empty()) m_config.decodeBuckets = {1};
    }

    std::uint64_t ContinuousBatchScheduler::Submit(GenerationRequest request)
    {
        if (request.promptTokens <= 0 || request.maxNewTokens <= 0 ||
            ActiveRequestCount() >= m_config.maxActiveRequests) {
            return 0;
        }
        if (request.id == 0) request.id = m_nextRequestId++;
        if (m_requests.find(request.id) != m_requests.end()) return 0;

        State state;
        state.request = request;
        state.stage = RequestStage::Queued;
        state.enqueueOrder = m_enqueueClock++;
        m_requests.emplace(request.id, state);
        m_prefillQueue.push_back(request.id);
        return request.id;
    }

    bool ContinuousBatchScheduler::Cancel(
        std::uint64_t requestId,
        PagedKVPool& kvPool)
    {
        auto found = m_requests.find(requestId);
        if (found == m_requests.end() ||
            found->second.stage == RequestStage::Finished ||
            found->second.stage == RequestStage::Cancelled) {
            return false;
        }
        found->second.stage = RequestStage::Cancelled;
        kvPool.Release(requestId);
        m_decode.erase(
            std::remove(m_decode.begin(), m_decode.end(), requestId),
            m_decode.end());
        return true;
    }

    int ContinuousBatchScheduler::DecodeBucket(int activeCount) const
    {
        for (const int bucket : m_config.decodeBuckets) {
            if (bucket >= activeCount) return bucket;
        }
        return m_config.decodeBuckets.back();
    }

    bool ContinuousBatchScheduler::HigherPriority(
        const State* left,
        const State* right)
    {
        if (left->request.priority != right->request.priority) {
            return left->request.priority > right->request.priority;
        }
        return left->enqueueOrder < right->enqueueOrder;
    }

    SchedulerTick ContinuousBatchScheduler::Schedule(PagedKVPool& kvPool)
    {
        SchedulerTick tick;

        std::vector<State*> decodeCandidates;
        for (const std::uint64_t id : m_decode) {
            auto found = m_requests.find(id);
            if (found != m_requests.end() &&
                found->second.stage == RequestStage::Decode) {
                decodeCandidates.push_back(&found->second);
            }
        }
        std::stable_sort(
            decodeCandidates.begin(),
            decodeCandidates.end(),
            HigherPriority);
        const int maxDecodeBatch = m_config.decodeBuckets.back();
        if (static_cast<int>(decodeCandidates.size()) > maxDecodeBatch) {
            decodeCandidates.resize(maxDecodeBatch);
        }
        if (!decodeCandidates.empty()) {
            BatchPlan decode;
            decode.stage = RequestStage::Decode;
            decode.bucketSize =
                DecodeBucket(static_cast<int>(decodeCandidates.size()));
            decode.scheduledTokens = static_cast<int>(decodeCandidates.size());
            decode.activeMask.assign(decode.bucketSize, 0);
            for (State* state : decodeCandidates) {
                const int contextLength =
                    state->request.promptTokens + state->generatedTokens;
                if (!kvPool.EnsureCapacity(state->request.id, contextLength + 1)) {
                    continue;
                }
                ScheduledRequest item;
                item.id = state->request.id;
                item.tokenId = state->nextTokenId;
                item.contextLength = contextLength;
                // The decode graph writes the input token at slot N and then
                // attends over N + 1 tokens.
                item.slotPosition = contextLength - 1;
                item.promptTokens = state->request.promptTokens;
                item.generatedTokens = state->generatedTokens;
                item.positionDelta = state->request.positionDelta;
                item.pageTable = kvPool.PageTable(state->request.id);
                decode.activeMask[decode.requests.size()] = 1;
                decode.requests.push_back(std::move(item));
            }
            if (!decode.requests.empty()) tick.batches.push_back(std::move(decode));
        }

        BatchPlan prefill;
        prefill.stage = RequestStage::Prefill;
        prefill.bucketSize = m_config.maxPrefillBatch;
        prefill.activeMask.assign(prefill.bucketSize, 0);
        int examined = static_cast<int>(m_prefillQueue.size());
        while (examined-- > 0 &&
               static_cast<int>(prefill.requests.size()) <
                   m_config.maxPrefillBatch) {
            const std::uint64_t id = m_prefillQueue.front();
            m_prefillQueue.pop_front();
            auto found = m_requests.find(id);
            if (found == m_requests.end() ||
                found->second.stage != RequestStage::Queued) {
                continue;
            }
            State& state = found->second;
            if (prefill.scheduledTokens + state.request.promptTokens >
                    m_config.maxPrefillTokensPerTick &&
                !prefill.requests.empty()) {
                m_prefillQueue.push_back(id);
                continue;
            }
            if (!kvPool.Reserve(id, state.request.promptTokens + 1)) {
                m_prefillQueue.push_back(id);
                continue;
            }
            state.stage = RequestStage::Prefill;
            ScheduledRequest item;
            item.id = id;
            item.promptTokens = state.request.promptTokens;
            item.contextLength = state.request.promptTokens;
            item.slotPosition = 0;
            item.positionDelta = state.request.positionDelta;
            item.pageTable = kvPool.PageTable(id);
            prefill.activeMask[prefill.requests.size()] = 1;
            prefill.scheduledTokens += state.request.promptTokens;
            prefill.requests.push_back(std::move(item));
        }
        if (!prefill.requests.empty()) tick.batches.push_back(std::move(prefill));
        return tick;
    }

    bool ContinuousBatchScheduler::CompletePrefill(
        std::uint64_t requestId,
        int firstTokenId,
        PagedKVPool& kvPool)
    {
        auto found = m_requests.find(requestId);
        if (found == m_requests.end() ||
            found->second.stage != RequestStage::Prefill ||
            !kvPool.CommitLength(
                requestId, found->second.request.promptTokens)) {
            return false;
        }
        State& state = found->second;
        state.stage = RequestStage::Decode;
        state.nextTokenId = firstTokenId;
        state.generatedTokens = 1;
        m_decode.push_back(requestId);
        return true;
    }

    bool ContinuousBatchScheduler::CompleteDecode(
        std::uint64_t requestId,
        int nextTokenId,
        bool finished,
        PagedKVPool& kvPool)
    {
        auto found = m_requests.find(requestId);
        if (found == m_requests.end() ||
            found->second.stage != RequestStage::Decode) {
            return false;
        }
        State& state = found->second;
        ++state.generatedTokens;
        state.nextTokenId = nextTokenId;
        finished = finished ||
            state.generatedTokens >= state.request.maxNewTokens;
        if (finished) {
            state.stage = RequestStage::Finished;
            kvPool.Release(requestId);
            m_decode.erase(
                std::remove(m_decode.begin(), m_decode.end(), requestId),
                m_decode.end());
            return true;
        }
        return kvPool.CommitLength(
            requestId,
            state.request.promptTokens + state.generatedTokens - 1);
    }

    RequestStage ContinuousBatchScheduler::Stage(
        std::uint64_t requestId) const
    {
        const auto found = m_requests.find(requestId);
        return found == m_requests.end()
            ? RequestStage::Cancelled
            : found->second.stage;
    }

    int ContinuousBatchScheduler::ActiveRequestCount() const
    {
        int count = 0;
        for (const auto& item : m_requests) {
            if (item.second.stage != RequestStage::Finished &&
                item.second.stage != RequestStage::Cancelled) {
                ++count;
            }
        }
        return count;
    }

    int ContinuousBatchScheduler::QueuedRequestCount() const
    {
        int count = 0;
        for (const auto& item : m_requests) {
            if (item.second.stage == RequestStage::Queued) ++count;
        }
        return count;
    }
}
