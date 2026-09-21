// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "q4q8_linear.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace Garnet
{
    namespace
    {
        class WorkerPool
        {
        public:
            static WorkerPool& Get()
            {
                static WorkerPool value;
                return value;
            }

            void Run(
                std::size_t count,
                int requested,
                const std::function<void(std::size_t, std::size_t)>& job)
            {
                const int threads = std::max(
                    1,
                    std::min({requested, static_cast<int>(m_workers.size()) + 1,
                              static_cast<int>(count)}));
                if (threads == 1) {
                    job(0, count);
                    return;
                }
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_job = &job;
                    m_count = count;
                    m_threads = threads;
                    m_pending = threads - 1;
                    ++m_generation;
                }
                m_start.notify_all();
                Slice(0, threads, count, job);
                std::unique_lock<std::mutex> lock(m_mutex);
                m_done.wait(lock, [&]() { return m_pending == 0; });
                m_job = nullptr;
            }

            ~WorkerPool()
            {
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_stop = true;
                    ++m_generation;
                }
                m_start.notify_all();
                for (auto& worker : m_workers) worker.join();
            }

        private:
            WorkerPool()
            {
                const int count = static_cast<int>(std::max(
                    1U, std::min(16U, std::thread::hardware_concurrency())));
                m_workers.reserve(static_cast<std::size_t>(count - 1));
                for (int index = 1; index < count; ++index) {
                    m_workers.emplace_back([this, index]() { Loop(index); });
                }
            }

            static void Slice(
                int index,
                int threads,
                std::size_t count,
                const std::function<void(std::size_t, std::size_t)>& job)
            {
                job(count * static_cast<std::size_t>(index) / threads,
                    count * static_cast<std::size_t>(index + 1) / threads);
            }

            void Loop(int index)
            {
                std::uint64_t seen = 0;
                for (;;) {
                    const std::function<void(std::size_t, std::size_t)>* job;
                    std::size_t count;
                    int threads;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_start.wait(lock, [&]() {
                            return m_stop || m_generation != seen;
                        });
                        if (m_stop) return;
                        seen = m_generation;
                        job = m_job;
                        count = m_count;
                        threads = m_threads;
                    }
                    if (index >= threads || !job) continue;
                    Slice(index, threads, count, *job);
                    {
                        std::lock_guard<std::mutex> guard(m_mutex);
                        if (--m_pending == 0) m_done.notify_one();
                    }
                }
            }

            std::mutex m_mutex;
            std::condition_variable m_start;
            std::condition_variable m_done;
            std::vector<std::thread> m_workers;
            const std::function<void(std::size_t, std::size_t)>* m_job = nullptr;
            std::size_t m_count = 0;
            int m_threads = 1;
            int m_pending = 0;
            std::uint64_t m_generation = 0;
            bool m_stop = false;
        };

        int HorizontalSum(__m256i value)
        {
            __m128i sum = _mm_add_epi32(
                _mm256_castsi256_si128(value),
                _mm256_extracti128_si256(value, 1));
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            return _mm_cvtsi128_si32(sum);
        }

        int DotGroup(
            const std::uint8_t* packed,
            const std::int8_t* input,
            int groupSize)
        {
            const __m128i mask = _mm_set1_epi8(0x0F);
            const __m256i ones = _mm256_set1_epi16(1);
            __m256i total = _mm256_setzero_si256();
            for (int offset = 0; offset < groupSize; offset += 32) {
                const __m128i source = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(packed));
                const __m128i low = _mm_and_si128(source, mask);
                const __m128i high = _mm_and_si128(
                    _mm_srli_epi16(source, 4), mask);
                __m256i weights = _mm256_castsi128_si256(low);
                weights = _mm256_inserti128_si256(weights, high, 1);
                const __m256i activation = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(input + offset));
                total = _mm256_add_epi32(
                    total,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(weights, activation), ones));
                packed += 16;
            }
            return HorizontalSum(total);
        }

        std::uint8_t Nibble(
            const std::vector<std::uint8_t>& values,
            std::size_t index)
        {
            return static_cast<std::uint8_t>(
                (values[index / 2] >> ((index & 1U) * 4U)) & 0x0FU);
        }
    }

    bool Q4Q8Linear::AVX2Supported()
    {
#if defined(_MSC_VER)
        int registers[4] = {};
        __cpuid(registers, 0);
        if (registers[0] < 7) return false;
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 5)) != 0;
#else
        return __builtin_cpu_supports("avx2");
#endif
    }

    bool Q4Q8Linear::Prepare(
        const QuantizedWeight& weight,
        std::string& errorMessage)
    {
        if (weight.shape.size() != 2 || weight.shape[0] <= 0 ||
            weight.shape[1] <= 0 || weight.blockSize <= 0 ||
            weight.blockSize % 32 != 0 ||
            weight.shape[1] % weight.blockSize != 0) {
            errorMessage = "Q4Q8 requires rank-two INT4 weights and a 32-aligned group";
            return false;
        }
        m_rows = static_cast<int>(weight.shape[0]);
        m_columns = static_cast<int>(weight.shape[1]);
        m_groupSize = weight.blockSize;
        const std::size_t elements =
            static_cast<std::size_t>(m_rows) * m_columns;
        const std::size_t groups =
            static_cast<std::size_t>(m_rows) * m_columns / m_groupSize;
        if (weight.packedValues.size() != elements / 2 ||
            weight.fp16Scales.size() != groups) {
            errorMessage = "Q4Q8 canonical weight storage is inconsistent";
            return false;
        }
        m_weights.resize(elements / 2);
        m_scales.resize(weight.fp16Scales.size());
        std::transform(
            weight.fp16Scales.begin(), weight.fp16Scales.end(),
            m_scales.begin(), [](std::uint16_t scale) {
                return _mm_cvtss_f32(
                    _mm_cvtph_ps(_mm_cvtsi32_si128(scale)));
            });
        for (int row = 0; row < m_rows; ++row) {
            for (int offset = 0; offset < m_columns; offset += 32) {
                const std::size_t logical =
                    static_cast<std::size_t>(row) * m_columns + offset;
                const std::size_t destination = logical / 2;
                for (int lane = 0; lane < 16; ++lane) {
                    const std::uint8_t low =
                        Nibble(weight.packedValues, logical + lane) ^ 8U;
                    const std::uint8_t high =
                        Nibble(weight.packedValues, logical + lane + 16) ^ 8U;
                    m_weights[destination + lane] =
                        static_cast<std::uint8_t>(low | (high << 4));
                }
            }
        }
        m_weightData = m_weights.data();
        m_scaleData = m_scales.data();
        errorMessage.clear();
        return true;
    }

    bool Q4Q8Linear::BindPackedView(
        int rows,
        int columns,
        int groupSize,
        const std::uint8_t* weights,
        std::size_t weightBytes,
        const float* scales,
        std::size_t scaleCount,
        std::string& errorMessage)
    {
        if (rows <= 0 || columns <= 0 || groupSize <= 0 ||
            groupSize % 32 != 0 || columns % groupSize != 0 ||
            !weights || !scales ||
            weightBytes != static_cast<std::size_t>(rows) * columns / 2 ||
            scaleCount != static_cast<std::size_t>(rows) * columns /
                groupSize) {
            errorMessage = "Q4Q8 persisted packed view is invalid";
            return false;
        }
        m_rows = rows;
        m_columns = columns;
        m_groupSize = groupSize;
        m_weights.clear();
        m_scales.clear();
        m_weightData = weights;
        m_scaleData = scales;
        errorMessage.clear();
        return true;
    }

    bool Q4Q8Linear::Forward(
        const float* input,
        float* output,
        int threadCount,
        std::string& errorMessage) const
    {
        if (!input || !output || m_rows <= 0 ||
            !m_weightData || !m_scaleData || !AVX2Supported()) {
            errorMessage = "Q4Q8 operator is unprepared or AVX2 is unavailable";
            return false;
        }
        const int groups = m_columns / m_groupSize;
        struct Scratch
        {
            std::vector<std::int8_t> quantized;
            std::vector<float> scales;
            std::vector<int> sums;
        };
        thread_local Scratch scratch;
        scratch.quantized.resize(static_cast<std::size_t>(m_columns));
        scratch.scales.resize(static_cast<std::size_t>(groups));
        scratch.sums.resize(static_cast<std::size_t>(groups));
        std::int8_t* const quantized = scratch.quantized.data();
        float* const activationScales = scratch.scales.data();
        int* const activationSums = scratch.sums.data();
        for (int group = 0; group < groups; ++group) {
            const int base = group * m_groupSize;
            float maximum = 0.0F;
            for (int index = 0; index < m_groupSize; ++index) {
                maximum = std::max(maximum, std::abs(input[base + index]));
            }
            const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
            activationScales[group] = scale;
            const float inverse = 1.0F / scale;
            int sum = 0;
            for (int index = 0; index < m_groupSize; ++index) {
                const int value = std::clamp<int>(
                    static_cast<int>(std::nearbyint(input[base + index] * inverse)),
                    -127, 127);
                quantized[base + index] =
                    static_cast<std::int8_t>(value);
                sum += value;
            }
            activationSums[group] = sum;
        }
        const auto job = [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row) {
                float value = 0.0F;
                for (int group = 0; group < groups; ++group) {
                    const std::size_t logical =
                        row * static_cast<std::size_t>(m_columns) +
                        static_cast<std::size_t>(group * m_groupSize);
                    int dot = DotGroup(
                        m_weightData + logical / 2,
                        quantized + group * m_groupSize,
                        m_groupSize);
                    dot -= 8 * activationSums[group];
                    value += static_cast<float>(dot) *
                        activationScales[group] *
                        m_scaleData[
                            row * static_cast<std::size_t>(groups) + group];
                }
                output[row] = value;
            }
        };
        WorkerPool::Get().Run(static_cast<std::size_t>(m_rows), threadCount, job);
        errorMessage.clear();
        return true;
    }

    std::size_t Q4Q8Linear::PackedBytes() const
    {
        return PackedWeightBytes() + PackedScaleCount() * sizeof(float);
    }

    std::size_t Q4Q8Linear::PackedWeightBytes() const
    {
        return m_rows > 0 && m_columns > 0
            ? static_cast<std::size_t>(m_rows) * m_columns / 2
            : 0;
    }

    std::size_t Q4Q8Linear::PackedScaleCount() const
    {
        return m_rows > 0 && m_columns > 0 && m_groupSize > 0
            ? static_cast<std::size_t>(m_rows) * m_columns / m_groupSize
            : 0;
    }
}
