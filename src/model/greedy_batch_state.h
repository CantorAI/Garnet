// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace Garnet {
// Host-only serial greedy feedback. GPU launch arguments must copy Current()
// by value; a pointer into this object's storage must never escape a call.
class GreedyBatchState {
    mutable std::mutex mutex_;
    std::size_t batch_, outputs_, step_ = 0;
    bool poisoned_ = false, released_ = false;
    std::vector<std::int32_t> history_, current_, pending_;
    void Live() const {
        if (released_) throw std::invalid_argument("greedy batch state released");
        if (poisoned_) throw std::invalid_argument("greedy batch state poisoned");
    }
public:
    static constexpr std::size_t MaximumBatch = 512, MaximumOutputs = 2048;
    struct Status { std::size_t batch, outputs, step, ownedBytes; bool poisoned, released; };
    GreedyBatchState(std::size_t batch, std::size_t outputs):batch_(batch),outputs_(outputs) {
        if (!batch || batch > MaximumBatch || !outputs || outputs > MaximumOutputs)
            throw std::invalid_argument("greedy batch shape outside bounded host storage");
        history_.resize(batch * outputs); current_.resize(batch); pending_.resize(batch);
    }
    GreedyBatchState(const GreedyBatchState&) = delete;
    GreedyBatchState& operator=(const GreedyBatchState&) = delete;
    void Reset() {
        std::lock_guard<std::mutex> hold(mutex_);
        if (released_) throw std::invalid_argument("greedy batch state released");
        if (step_ && step_ != outputs_ && !poisoned_)
            throw std::invalid_argument("cannot reset incomplete greedy batch");
        step_ = 0; poisoned_ = false;
        std::fill(history_.begin(), history_.end(), 0);
        std::fill(current_.begin(), current_.end(), 0);
        std::fill(pending_.begin(), pending_.end(), 0);
    }
    std::vector<std::int32_t> Consume(const float* pairs, std::size_t count) {
        std::lock_guard<std::mutex> hold(mutex_); Live();
        if (!pairs || count != batch_ * 4 || step_ >= outputs_) {
            poisoned_ = true;
            throw std::invalid_argument("greedy batch candidate count or step invalid");
        }
        // Validate every ID before committing any row. Preserve the existing
        // CPU comparator, including its NaN branch and minimum-ID exact tie.
        for (std::size_t row = 0; row < batch_; ++row) {
            const float aId = pairs[row * 4 + 1], bId = pairs[row * 4 + 3];
            if (!std::isfinite(aId) || !std::isfinite(bId) ||
                !(0 <= aId && aId < 16777216 && 0 <= bId && bId < 16777216) ||
                std::trunc(aId) != aId || std::trunc(bId) != bId) {
                poisoned_ = true;
                throw std::invalid_argument("greedy batch IDs must be exact nonnegative FP32 integers");
            }
            const float a = pairs[row * 4], b = pairs[row * 4 + 2];
            const bool takeB = b > a || (b == a && bId < aId);
            pending_[row] = static_cast<std::int32_t>(takeB ? bId : aId);
        }
        current_.swap(pending_);
        for (std::size_t row = 0; row < batch_; ++row)
            history_[row * outputs_ + step_] = current_[row];
        ++step_;
        return current_; // One independent by-value feedback payload for both rank workers.
    }
    std::vector<std::int32_t> Current(std::size_t expectedStep) const {
        std::lock_guard<std::mutex> hold(mutex_); Live();
        // The current ID from the final requested token is still a valid
        // forward result even though no further decode step may consume it.
        if (!step_ || step_ > outputs_ || expectedStep != step_)
            throw std::invalid_argument("greedy feedback step is stale or complete");
        return current_; // Independent launch payload; no borrowed host pointer.
    }
    std::vector<std::int32_t> History() const {
        std::lock_guard<std::mutex> hold(mutex_); Live();
        if (step_ != outputs_) throw std::invalid_argument("greedy batch history incomplete");
        return history_;
    }
    Status GetStatus() const {
        std::lock_guard<std::mutex> hold(mutex_);
        return {batch_, outputs_, step_, (history_.capacity() + current_.capacity() +
            pending_.capacity()) * sizeof(std::int32_t), poisoned_, released_};
    }
    void Release() {
        std::lock_guard<std::mutex> hold(mutex_);
        std::vector<std::int32_t>().swap(history_);
        std::vector<std::int32_t>().swap(current_);
        std::vector<std::int32_t>().swap(pending_);
        released_ = true;
    }
};
}
