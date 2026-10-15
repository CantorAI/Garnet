// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "lowering_context.h"
#include "graph_capture.h"

namespace Garnet
{
    bool ReplayTensorGraph(const X::Value& graph, ILoweringContext& context,
        std::string& error)
    {
        try {
            TensorGraphCapture capture(graph);
            for (const auto& operation : capture.Operations()) {
                if (operation.provider.empty()) continue;
                if (operation.provider != "garnet" && operation.provider != "cpu")
                    throw std::runtime_error("unsupported tensor provider: " + operation.provider);
                X::KWARGS keywords;
                if (operation.attributes.IsDict()) {
                    keywords.reserve(operation.attributes.Size());
                    for (uint64_t i = 0; i < operation.attributes.Size(); ++i) {
                        X::Value key, value;
                        if (!operation.attributes.DictEntry(i, key, value) || !key.IsString())
                            throw std::runtime_error("invalid tensor operation attributes");
                        keywords.emplace_back(key.ToString(), std::move(value));
                    }
                }
                X::ARGS params;
                X::Value result;
                if (operation.operandCount == 1 && !operation.inputs.empty())
                    result = context.HandleUnaryOp(operation.name, graph, params, keywords,
                        operation.inputs[0], operation.output);
                else if (operation.operandCount == 2 && operation.inputs.size() >= 2)
                    result = context.HandleBinaryOp(operation.name, graph, params, keywords,
                        operation.inputs[0], operation.inputs[1], operation.output);
                else
                    throw std::runtime_error("invalid operand count for " + operation.name);
                if (!result.IsValid())
                    throw std::runtime_error("backend lowering failed for " + operation.name);
            }
            error.clear();
            return true;
        } catch (const std::exception& failure) {
            error = failure.what();
            return false;
        }
    }

    thread_local ILoweringContext* g_loweringContext = nullptr;

    ScopedLoweringContext::ScopedLoweringContext(ILoweringContext& context)
        : m_previous(g_loweringContext)
    {
        g_loweringContext = &context;
    }

    ScopedLoweringContext::~ScopedLoweringContext()
    {
        g_loweringContext = m_previous;
    }
}
