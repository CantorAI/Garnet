// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compiled_graph_capture.h"
#include "lowering_context.h"
#include "safetensors_index.h"
#include "xlang3/xlang3.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Garnet
{
    // Lowers the backend-neutral XLang TensorGraph directly to an OpenVINO
    // model. Model Python (.py) files never import or select OpenVINO.
    class OpenVINOBuilder : public ILoweringContext
    {
    public:
        bool AnalyzeCapturedGraph(
            X::Value graph,
            X::Value forwardFunction,
            X::ARGS& graphArguments,
            std::vector<CapturedTensorOperation>& operations,
            std::string& errorMessage);

        bool BuildCapturedGraph(
            X::Value graph,
            X::Value forwardFunction,
            X::ARGS& graphArguments,
            X::ARGS& symbolicInputs,
            const SafeTensorsIndex* weightIndex,
            const std::string& enginePath,
            const std::string& precision,
            std::string& errorMessage);

        bool PrepareCapturedEngine(
            const std::string& enginePath,
            const SafeTensorsIndex* weightIndex,
            std::string& errorMessage);

        X::Value RunCapturedEngine(
            const std::string& enginePath,
            X::Value inputs,
            const SafeTensorsIndex* weightIndex,
            X::Value reusableOutput,
            bool resetState,
            std::uint64_t sessionId,
            std::string& errorMessage);

        static void ReleaseCachedExecutions(const std::string& cacheRoot);
        static void ReleaseCachedSession(
            const std::string& enginePath,
            std::uint64_t sessionId);
        static bool IsAvailable();

        X::Value HandleBinaryOp(
            const std::string& opName,
            X::Value graph,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value input1,
            X::Value input2,
            X::Value output) override;

        X::Value HandleUnaryOp(
            const std::string& opName,
            X::Value graph,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value input,
            X::Value output) override;

        X::Value HandleBranchBegin(
            const std::string& condition,
            int branchType,
            unsigned long long flowId,
            int branchId) override;
        X::Value HandleBranchEnd() override;

    private:
        bool m_analysisActive = false;
        bool m_loweringActive = false;
        std::vector<CapturedTensorOperation> m_analyzedOperations;
        std::string m_error;

#if defined(GARNET_WITH_OPENVINO)
        struct Implementation;
        Implementation* m_implementation = nullptr;
#endif
    };
}
