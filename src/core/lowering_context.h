#pragma once

#include "xlang3/xlang3.h"

#include <string>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace Garnet
{
    inline bool IsTensor(const X::Value& value) {
        return X::Tensor::IsTensor(value);
    }

    inline uint64_t TensorId(const X::Value& value) {
        return X::Tensor(value).Info().id;
    }

    inline int TensorDimension(const X::Tensor& tensor, uint32_t index) {
        const auto info = tensor.Info();
        if (info.rank == UINT32_MAX || index >= info.rank ||
            info.shape[index] < 0 || info.shape[index] > INT_MAX)
            throw std::runtime_error("backend tensor dimension is out of range");
        return static_cast<int>(info.shape[index]);
    }

    inline bool IsContiguousTensor(const X::Tensor& tensor) {
        const auto info = tensor.Info();
        if (info.rank == UINT32_MAX) return false;
        uint64_t stride;
        switch (info.dtype) {
            case X3_TENSOR_FLOAT16: case X3_TENSOR_BFLOAT16: stride = 2; break;
            case X3_TENSOR_FLOAT32: case X3_TENSOR_INT32: stride = 4; break;
            case X3_TENSOR_FLOAT64: case X3_TENSOR_INT64: stride = 8; break;
            default: return false;
        }
        for (uint32_t i = info.rank; i-- > 0;) {
            if (info.shape[i] == 0) return true;
            if (info.shape[i] < 0 || (info.shape[i] > 1 &&
                    (info.strides[i] < 0 || static_cast<uint64_t>(info.strides[i]) != stride)) ||
                stride > UINT64_MAX / static_cast<uint64_t>(info.shape[i]))
                return false;
            stride *= info.shape[i];
        }
        return true;
    }

    inline std::pair<std::string, X::Value>* FindKeyword(X::KWARGS& values, const char* name) {
        auto found = std::find_if(values.begin(), values.end(),
            [name](const auto& entry) { return entry.first == name; });
        return found == values.end() ? nullptr : &*found;
    }

    inline std::vector<int64_t> IntegerAttribute(X::KWARGS& values, const char* name) {
        const auto* entry = FindKeyword(values, name);
        if (!entry || (!entry->second.IsList() &&
                x3_value_object_kind(entry->second.raw()) != X3_OBJECT_KIND_TUPLE))
            throw std::runtime_error(std::string("expected integer sequence: ") + name);
        std::vector<int64_t> result;
        result.reserve(entry->second.Size());
        for (uint64_t i = 0; i < entry->second.Size(); ++i) {
            auto item = entry->second.Get(i);
            if (item.raw().tag != X3_TAG_INT64)
                throw std::runtime_error(std::string("expected integer in ") + name);
            result.push_back(item.ToLongLong());
        }
        return result;
    }

    // Backend-neutral target used while replaying an XLang TensorGraph.
    // Model expressions never select or include a concrete runtime backend.
    class ILoweringContext
    {
    public:
        virtual ~ILoweringContext() = default;

        virtual X::Value HandleBinaryOp(
            const std::string& opName,
            X::Value graph,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value input1,
            X::Value input2,
            X::Value output) = 0;

        virtual X::Value HandleUnaryOp(
            const std::string& opName,
            X::Value graph,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value input,
            X::Value output) = 0;

        virtual X::Value HandleBranchBegin(
            const std::string& condition,
            int branchType,
            unsigned long long flowId,
            int branchId) = 0;

        virtual X::Value HandleBranchEnd() = 0;
    };

    // Replay the captured DAG, never call the Python forward function again.
    bool ReplayTensorGraph(const X::Value& graph, ILoweringContext& context,
        std::string& error);

    extern thread_local ILoweringContext* g_loweringContext;

    class ScopedLoweringContext
    {
        ILoweringContext* m_previous = nullptr;

    public:
        explicit ScopedLoweringContext(ILoweringContext& context);
        ~ScopedLoweringContext();

        ScopedLoweringContext(const ScopedLoweringContext&) = delete;
        ScopedLoweringContext& operator=(const ScopedLoweringContext&) = delete;
    };
}
