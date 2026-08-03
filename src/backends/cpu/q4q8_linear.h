#pragma once

#include "weight_quantization.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Garnet
{
    // Garnet-owned decode GEMV implementation for a lowered TensorGraph
    // linear node. It is an operator only: it has no model, KV, or tokenizer
    // ownership.
    class Q4Q8Linear
    {
    public:
        bool Prepare(const QuantizedWeight& weight, std::string& errorMessage);
        bool BindPackedView(
            int rows,
            int columns,
            int groupSize,
            const std::uint8_t* weights,
            std::size_t weightBytes,
            const float* scales,
            std::size_t scaleCount,
            std::string& errorMessage);
        bool Forward(
            const float* input,
            float* output,
            int threadCount,
            std::string& errorMessage) const;

        int Rows() const { return m_rows; }
        int Columns() const { return m_columns; }
        int GroupSize() const { return m_groupSize; }
        std::size_t PackedBytes() const;
        const std::uint8_t* PackedWeightData() const { return m_weightData; }
        const float* PackedScaleData() const { return m_scaleData; }
        std::size_t PackedWeightBytes() const;
        std::size_t PackedScaleCount() const;
        static bool AVX2Supported();

    private:
        int m_rows = 0;
        int m_columns = 0;
        int m_groupSize = 0;
        std::vector<std::uint8_t> m_weights;
        std::vector<float> m_scales;
        const std::uint8_t* m_weightData = nullptr;
        const float* m_scaleData = nullptr;
    };
}
