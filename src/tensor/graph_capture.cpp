#include "graph_capture.h"
#include <stdexcept>

namespace Garnet {

X::Value RegisterTensorOperator(X3PackageHost* host, uint32_t arity) {
    if (!host || !host->runtime || (arity != 1 && arity != 2))
        throw std::invalid_argument("Garnet tensor operator requires a host and unary/binary arity");
    X3TensorOperatorDef definition{};
    definition.size = sizeof(definition);
    definition.provider = "garnet";
    definition.name = arity == 1 ? "unary_op" : "binary_op";
    definition.arity = arity;
    // Cache updates and binding operations must retain program order.
    definition.flags = X3_TENSOR_ORDERED;
    X3Value result = x3_value_invalid();
    if (x3_tensor_register_operator(host->runtime, &definition, &result) != X3_STATUS_OK)
        throw std::runtime_error(host->runtime_last_error(host->runtime));
    return X::Value(host, result, false);
}

TensorGraphCapture::TensorGraphCapture(const X::Value& graph) : graph_(graph) {
    if (!graph_.host() || !graph_.runtime())
        throw std::invalid_argument("Garnet graph capture requires a runtime graph");
    X3Value outputs = x3_value_invalid();
    if (x3_tensor_graph_outputs(graph_.runtime(), graph_.raw(), &outputs) != X3_STATUS_OK)
        throw std::runtime_error(graph_.host()->runtime_last_error(graph_.runtime()));
    outputs_ = X::Value(graph_.host(), outputs, false);
    if (x3_tensor_graph_replay(graph_.runtime(), graph_.raw(), Visit, this) != X3_STATUS_OK)
        throw std::runtime_error(error_.empty()
            ? graph_.host()->runtime_last_error(graph_.runtime()) : error_);
}

X3Status TensorGraphCapture::Visit(X3Runtime* runtime, void* context,
    const X3TensorOperation* operation) noexcept {
    auto& capture = *static_cast<TensorGraphCapture*>(context);
    try {
        if (runtime != capture.graph_.runtime() || !operation ||
            operation->size != sizeof(*operation) || !operation->name ||
            !operation->provider || operation->operand_count > operation->input_count ||
            (operation->input_count && !operation->inputs))
            throw std::runtime_error("invalid Garnet graph replay operation");
        GraphOperation node;
        node.id = operation->id;
        node.provider = operation->provider;
        node.name = operation->name;
        node.operandCount = operation->operand_count;
        node.flags = operation->flags;
        auto* host = capture.graph_.host();
        node.inputs.reserve(operation->input_count);
        for (uint32_t index = 0; index < operation->input_count; ++index)
            node.inputs.emplace_back(host, operation->inputs[index], true);
        node.attributes = X::Value(host, operation->attributes, true);
        node.regions = X::Value(host, operation->regions, true);
        node.output = X::Value(host, operation->output, true);
        capture.operations_.push_back(std::move(node));
        return X3_STATUS_OK;
    } catch (const std::exception& error) {
        try { capture.error_ = error.what(); } catch (...) {}
    } catch (...) {
        try { capture.error_ = "unknown Garnet graph replay error"; } catch (...) {}
    }
    return X3_STATUS_ERROR;
}

}
