#pragma once

#include "xlang3/xlang3.h"
#include <string>
#include <vector>

namespace Garnet {

// Owns references, not copies of tensor payloads. Expression metadata remains
// unknown until the backend infers it; it must not be copied from operand zero.
struct GraphOperation {
    uint64_t id = 0;
    std::string provider;
    std::string name;
    uint32_t operandCount = 0;
    uint32_t flags = 0;
    std::vector<X::Value> inputs;
    X::Value attributes;
    X::Value regions;
    X::Value output;
};

class TensorGraphCapture {
public:
    explicit TensorGraphCapture(const X::Value& graph);
    const std::vector<GraphOperation>& Operations() const { return operations_; }
    const X::Value& Graph() const { return graph_; }
    const X::Value& Outputs() const { return outputs_; }

private:
    static X3Status Visit(X3Runtime*, void*, const X3TensorOperation*) noexcept;
    X::Value graph_;
    X::Value outputs_;
    std::vector<GraphOperation> operations_;
    std::string error_;
};

// Both factories are lazy. A backend consumes their graph through capture;
// registration never installs an eager CPU/CUDA execution path.
X::Value RegisterTensorOperator(X3PackageHost* host, uint32_t arity);

}
