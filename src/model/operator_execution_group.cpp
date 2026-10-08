// SPDX-License-Identifier: Apache-2.0
#include "operator_execution_group.h"
#include "operator_plugins.h"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <cstring>
namespace Garnet {
namespace {
struct Current { std::shared_ptr<OperatorExecutionOwner> owner;cudaStream_t stream=nullptr;int rank=-1,phase=-1; };
thread_local Current current;
void Validate(const GarnetOperatorExecutionPayload* p){
    if(!p || p->abi!=1 || p->size!=sizeof(*p) || !p->owner || !p->services)
        throw std::invalid_argument("operator execution payload ABI mismatch");
    const auto* s=p->services;
    if(s->abi!=1 || s->size!=sizeof(*s) || !s->plugin_id || !*s->plugin_id || !s->backend ||
       !s->ranks || !s->phases || !s->retain || !s->release || !s->bind_phase || !s->enter || !s->leave || !s->status_json)
        throw std::invalid_argument("operator execution services ABI mismatch");
    std::string error;
    if(OperatorPluginSymbol(s->plugin_id,GarnetOperatorExecutionSymbol,error)!=s)
        throw std::invalid_argument("operator execution provider differs from bound native module: "+error);
}
}
OperatorExecutionOwner::OperatorExecutionOwner(const GarnetOperatorExecutionServices* s,void* p):services_(s),owner_(p){s->retain(p);}
OperatorExecutionOwner::~OperatorExecutionOwner(){services_->release(owner_);}
std::shared_ptr<OperatorExecutionOwner> CurrentOperatorExecutionOwner(){return current.owner;}
int CurrentOperatorExecutionPhase(){return current.phase;}
int CurrentOperatorExecutionRank(){return current.rank;}
OperatorExecutionScope::OperatorExecutionScope(const X::Value& value,int rank,int phase,
        const std::string& backend,const std::string& executionPlan){
    if(current.owner)throw std::invalid_argument("nested operator execution group is unsupported");
    auto* host=value.host();
    if(!host || !host->instance_get_native_data)throw std::invalid_argument("native operator execution group required");
    const auto* p=static_cast<const GarnetOperatorExecutionPayload*>(host->instance_get_native_data(value.raw(),GarnetOperatorExecutionType));
    Validate(p);const auto* s=p->services;
    if(backend!=s->backend || rank<0 || uint32_t(rank)>=s->ranks || phase<0 || uint32_t(phase)>=s->phases)
        throw std::invalid_argument("operator execution backend/rank/phase mismatch");
    const auto plan=nlohmann::json::parse(executionPlan);bool required=false;
    for(const auto& plugin:plan.value("operator_plugins",nlohmann::json::array()))
        if(plugin.at("id")==s->plugin_id && plugin.at("backend")==backend)required=true;
    if(!required)throw std::invalid_argument("operator execution provider is not required by this compiled model");
    owner_=std::make_shared<OperatorExecutionOwner>(s,p->owner);rank_=rank;
    void* stream=nullptr;const int code=s->enter(p->owner,uint32_t(phase),uint32_t(rank),&stream);
    if(code || !stream){
        if(!code)s->leave(p->owner,uint32_t(rank));
        throw std::runtime_error("operator execution enter failed: "+std::to_string(code));
    }
    current.owner=owner_;current.stream=static_cast<cudaStream_t>(stream);current.rank=rank;current.phase=phase;active_=true;
    detail::operatorExecutionStream=current.stream;
}
void OperatorExecutionScope::Finish(){
    if(!active_)return;
    const auto* s=owner_->Services();const int code=s->leave(owner_->Identity(),uint32_t(rank_));
    detail::operatorExecutionStream=nullptr;current=Current{};active_=false;
    if(code)throw std::runtime_error("operator execution completion/fault check failed: "+std::to_string(code));
}
OperatorExecutionScope::~OperatorExecutionScope(){
    if(!active_)return;
    try{Finish();}catch(const std::exception& e){std::fprintf(stderr,"OPERATOR_EXECUTION_RETIRE_FAILED: %s\n",e.what());std::terminate();}
}
}
