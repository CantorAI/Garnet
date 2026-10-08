// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "operator_execution_bridge.h"
#include "execution_stream.h"
#include "xlang3/xlang3.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
namespace Garnet {
class OperatorExecutionOwner {
    const GarnetOperatorExecutionServices* services_;
    void* owner_;
public:
    OperatorExecutionOwner(const GarnetOperatorExecutionServices*,void*);
    ~OperatorExecutionOwner();
    OperatorExecutionOwner(const OperatorExecutionOwner&)=delete;
    OperatorExecutionOwner& operator=(const OperatorExecutionOwner&)=delete;
    const GarnetOperatorExecutionServices* Services()const{return services_;}
    void* Identity()const{return owner_;}
};
std::shared_ptr<OperatorExecutionOwner> CurrentOperatorExecutionOwner();
int CurrentOperatorExecutionPhase();
int CurrentOperatorExecutionRank();
class OperatorExecutionScope {
    std::shared_ptr<OperatorExecutionOwner> owner_;
    int rank_=-1;
    bool active_=false;
public:
    OperatorExecutionScope(const X::Value& payload,int rank,int phase,
        const std::string& backend,const std::string& executionPlan);
    OperatorExecutionScope(const OperatorExecutionScope&)=delete;
    OperatorExecutionScope& operator=(const OperatorExecutionScope&)=delete;
    void Finish();
    ~OperatorExecutionScope();
};
}
