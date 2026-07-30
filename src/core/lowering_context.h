#pragma once

#include "xlang.h"

#include <string>

namespace Garnet
{
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
