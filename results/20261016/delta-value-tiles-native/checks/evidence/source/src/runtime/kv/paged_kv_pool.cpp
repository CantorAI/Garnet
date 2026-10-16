// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "paged_kv_pool.h"

#include <cuda_runtime.h>

#include <limits>

namespace Garnet
{
    PagedKVPool::~PagedKVPool()
    {
        Reset();
    }

    int PagedKVPool::PagesForTokens(int tokenCount) const
    {
        if (tokenCount <= 0) return 0;
        return (tokenCount + m_config.pageSize - 1) / m_config.pageSize;
    }

    bool PagedKVPool::Initialize(const PagedKVPoolConfig& config)
    {
        Reset();
        if (config.totalPages <= 0 || config.pageSize <= 0 ||
            config.numLayers <= 0 || config.numKVHeads <= 0 ||
            config.headDim <= 0 || config.elementBytes <= 0 ||
            config.deviceId < 0) {
            return false;
        }

        const std::size_t elements =
            static_cast<std::size_t>(config.totalPages) *
            static_cast<std::size_t>(config.pageSize) *
            static_cast<std::size_t>(config.numLayers) *
            static_cast<std::size_t>(config.numKVHeads) *
            static_cast<std::size_t>(config.headDim);
        if (elements >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(config.elementBytes)) {
            return false;
        }

        m_config = config;
        m_bytesPerArena = elements * static_cast<std::size_t>(config.elementBytes);
        if (cudaSetDevice(config.deviceId) != cudaSuccess ||
            cudaMalloc(&m_keyPages, m_bytesPerArena) != cudaSuccess ||
            cudaMalloc(&m_valuePages, m_bytesPerArena) != cudaSuccess) {
            Reset();
            return false;
        }

        m_freePages.clear();
        for (int page = 0; page < config.totalPages; ++page) {
            m_freePages.push_back(page);
        }
        return true;
    }

    void PagedKVPool::Reset()
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        if (m_keyPages) cudaFree(m_keyPages);
        if (m_valuePages) cudaFree(m_valuePages);
        m_keyPages = nullptr;
        m_valuePages = nullptr;
        m_bytesPerArena = 0;
        m_freePages.clear();
        m_sequences.clear();
        m_config = {};
    }

    bool PagedKVPool::EnsureCapacityLocked(
        Sequence& sequence,
        int tokenCapacity)
    {
        const int requiredPages = PagesForTokens(tokenCapacity);
        if (requiredPages <= static_cast<int>(sequence.pageTable.size())) {
            return true;
        }
        const int additional =
            requiredPages - static_cast<int>(sequence.pageTable.size());
        if (additional > static_cast<int>(m_freePages.size())) return false;
        for (int index = 0; index < additional; ++index) {
            sequence.pageTable.push_back(m_freePages.front());
            m_freePages.pop_front();
        }
        return true;
    }

    bool PagedKVPool::Reserve(std::uint64_t sequenceId, int tokenCapacity)
    {
        if (sequenceId == 0 || tokenCapacity < 0 || !m_keyPages) return false;
        std::lock_guard<std::mutex> guard(m_mutex);
        if (m_sequences.find(sequenceId) != m_sequences.end()) return false;
        Sequence sequence;
        if (!EnsureCapacityLocked(sequence, tokenCapacity)) return false;
        m_sequences.emplace(sequenceId, std::move(sequence));
        return true;
    }

    bool PagedKVPool::EnsureCapacity(
        std::uint64_t sequenceId,
        int tokenCapacity)
    {
        if (tokenCapacity < 0) return false;
        std::lock_guard<std::mutex> guard(m_mutex);
        auto found = m_sequences.find(sequenceId);
        return found != m_sequences.end() &&
            EnsureCapacityLocked(found->second, tokenCapacity);
    }

    bool PagedKVPool::CommitLength(
        std::uint64_t sequenceId,
        int logicalLength)
    {
        if (logicalLength < 0) return false;
        std::lock_guard<std::mutex> guard(m_mutex);
        auto found = m_sequences.find(sequenceId);
        if (found == m_sequences.end() ||
            !EnsureCapacityLocked(found->second, logicalLength)) {
            return false;
        }
        found->second.logicalLength = logicalLength;
        return true;
    }

    bool PagedKVPool::Release(std::uint64_t sequenceId)
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        auto found = m_sequences.find(sequenceId);
        if (found == m_sequences.end()) return false;
        for (const int page : found->second.pageTable) {
            m_freePages.push_back(page);
        }
        m_sequences.erase(found);
        return true;
    }

    int PagedKVPool::LogicalLength(std::uint64_t sequenceId) const
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        const auto found = m_sequences.find(sequenceId);
        return found == m_sequences.end() ? -1 : found->second.logicalLength;
    }

    std::vector<int> PagedKVPool::PageTable(std::uint64_t sequenceId) const
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        const auto found = m_sequences.find(sequenceId);
        return found == m_sequences.end()
            ? std::vector<int>{}
            : found->second.pageTable;
    }

    PagedKVPoolStats PagedKVPool::Stats() const
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        PagedKVPoolStats stats;
        stats.totalPages = m_config.totalPages;
        stats.freePages = static_cast<int>(m_freePages.size());
        stats.usedPages = stats.totalPages - stats.freePages;
        stats.sequenceCount = static_cast<int>(m_sequences.size());
        stats.bytesPerArena = m_bytesPerArena;
        return stats;
    }
}
