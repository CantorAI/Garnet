#pragma once

#include "xlang.h"
#include <string>

namespace Garnet {
    class ITRTContext {
    public:
        virtual ~ITRTContext() {}
        virtual X::Value HandleBinaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input1, X::Value input2, X::Value output) = 0;
        virtual X::Value HandleUnaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value output) = 0;
        virtual X::Value HandleBranchBegin(const std::string& condition, int branchType, unsigned long long flowId, int branchId) = 0;
        virtual X::Value HandleBranchEnd() = 0;
    };
}
