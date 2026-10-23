// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_tensor_io.h"
#include "tensor_helper.h"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <climits>
#include <vector>
namespace Garnet {
X::Value CUDATensorIO::GroupWeights(X::Value values){
    std::lock_guard<std::mutex> lock(mutex_);
    if(!table_.IsValid()||!values.IsList()||values.Size()<2||values.Size()>8)
        throw X::Error("Grouped projections require an open checkpoint and2..8 names");
    std::vector<std::string> names;names.reserve(values.Size());
    std::unordered_set<std::string> unique;
    nlohmann::json ordered=nlohmann::json::array();
    for(uint64_t i=0;i<values.Size();++i){
        auto name=values.Get(i).ToString();
        if((name.rfind("model.language_model.layers.",0)!=0&&name.rfind("mtp.",0)!=0)||!unique.insert(name).second)
            throw X::Error("Grouped projection names must be unique text-layer/MTP matrices");
        names.push_back(name);ordered.push_back(name);
    }
    const auto key=ordered.dump();
    auto cached=projectionWeightGroups_.find(key);if(cached!=projectionWeightGroups_.end())return cached->second;
    int64_t rows=0,columns=0;size_t bytes=0;
    for(const auto& name:names){
        if(projectionWeights_.count(name))throw X::Error("Group projections before loading any member; overlapping groups are unsupported");
        auto* e=index_.Find(name);
        if(!e||e->dataType!="BF16"||e->shape.size()!=2||e->shape[0]<1||e->shape[1]<1||
            e->shape[0]>32768||e->shape[1]>16384||uint64_t(e->shape[0])*uint64_t(e->shape[1])*2!=e->dataSize)
            throw X::Error("Grouped projection requires valid bounded BF16 matrices");
        if(columns&&columns!=e->shape[1])throw X::Error("Grouped projection inner dimensions differ");
        columns=e->shape[1];rows+=e->shape[0];
        if(rows>32768||e->dataSize>(512ULL<<20)-bytes)throw X::Error("Grouped projection exceeds32768rows/512MiB");
        bytes+=size_t(e->dataSize);
    }
    // Validate and reserve bookkeeping before device allocation. Member views
    // deliberately share one readonly storage, including its provider fences.
    projectionWeights_.reserve(projectionWeights_.size()+names.size());
    projectionWeightGroupMembers_.reserve(projectionWeightGroupMembers_.size()+names.size());
    projectionWeightGroups_.reserve(projectionWeightGroups_.size()+1);
    auto owner=std::make_shared<X::Tensor>(TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,{rows,columns}));
    {
        auto use=TensorHelper::AcquireGPU(*owner,X3_TENSOR_WRITE);size_t offset=0;
        for(const auto& name:names){
            auto* e=index_.Find(name);SafeTensorsMappedFile mapping;std::string error;
            if(!mapping.Open(e->filePath,error))throw X::Error(error);
            auto* data=mapping.DataAt(e->dataOffset,e->dataSize);if(!data)throw X::Error("Grouped projection checkpoint span invalid");
            auto status=cudaMemcpyAsync(static_cast<char*>(owner->Info().data)+offset,data,size_t(e->dataSize),cudaMemcpyHostToDevice,CurrentExecutionStream());
            if(status!=cudaSuccess)throw X::Error(cudaGetErrorString(status));
            // The mmap is cold-load staging; retire its copy before unmapping.
            status=cudaStreamSynchronize(CurrentExecutionStream());if(status!=cudaSuccess)throw X::Error(cudaGetErrorString(status));
            offset+=size_t(e->dataSize);
        }
        use.Finish();
    }
    auto info=owner->Info();info.readonly=1;
    X::Tensor joined=TensorHelper::WrapBorrowedGPU(Host(),info,owner);
    std::vector<X::Tensor> members;members.reserve(names.size());uint64_t offset=0;
    for(const auto& name:names){
        auto* e=index_.Find(name);members.push_back(joined.View(e->shape,{columns*2,2},offset));offset+=e->dataSize;
    }
    try{
        projectionWeightGroups_.emplace(key,joined);
        for(size_t i=0;i<names.size();++i){projectionWeights_.emplace(names[i],members[i]);projectionWeightGroupMembers_.insert(names[i]);}
    }catch(...){
        for(const auto& name:names){projectionWeights_.erase(name);projectionWeightGroupMembers_.erase(name);}
        projectionWeightGroups_.erase(key);throw;
    }
    return joined;
}
}
#endif
