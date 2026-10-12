// SPDX-License-Identifier: Apache-2.0
#include "compiled_graph.h"
#include "graph_capture.h"
#include "lowering_context.h"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <algorithm>
namespace Garnet {
size_t MetalCompiledGraph::Allocate(size_t rows,size_t columns) {
    auto id=slots_.size();slots_.push_back({gpu_->Allocate(rows,columns),rows,columns});return id;
}
size_t MetalCompiledGraph::Weight(const std::string& name) {
    auto found=weights_.find(name);if(found!=weights_.end())return found->second;
    auto entry=index_.Find(name);
    if(!entry||entry->shape.empty()||entry->shape.size()>2)
        throw std::runtime_error("Metal matrix weight missing/invalid: "+name);
    for(auto dim:entry->shape)if(dim<=0)throw std::runtime_error("Invalid Metal checkpoint dimension");
    const size_t rows=entry->shape.size()==1?1:entry->shape[0],columns=entry->shape.back();
    if(rows>std::numeric_limits<size_t>::max()/columns || rows*columns>std::numeric_limits<size_t>::max()/4)
        throw std::runtime_error("Metal checkpoint matrix size overflow");
    SafeTensorsMappedFile file;std::string error;
    if(!file.Open(entry->filePath,error))throw std::runtime_error(error);
    std::vector<float> data(rows*columns);
    auto raw=file.DataAt(entry->dataOffset,entry->dataSize);
    if(!raw)throw std::runtime_error("Metal checkpoint data is out of file bounds");
    if(entry->dataType=="BF16") {
        if(entry->dataSize!=data.size()*2)throw std::runtime_error("Invalid BF16 weight length");
        for(size_t i=0;i<data.size();++i){uint16_t half;std::memcpy(&half,static_cast<const char*>(raw)+2*i,2);
            uint32_t bits=uint32_t(half)<<16;std::memcpy(&data[i],&bits,4);}
    }else if(entry->dataType=="F32") {
        if(entry->dataSize!=data.size()*4)throw std::runtime_error("Invalid FP32 weight length");
        std::memcpy(data.data(),raw,entry->dataSize);
    }else throw std::runtime_error("Unsupported initial Metal weight dtype: "+entry->dataType);
    const auto id=slots_.size();slots_.push_back({gpu_->Upload(data,rows,columns),rows,columns});
    weights_[name]=id;return id;
}
void MetalCompiledGraph::Clear() {
    ready_=false;ops_.clear();inputs_.clear();inputShapes_.clear();tensors_.clear();weights_.clear();slots_.clear();gpu_.reset();
}
void MetalCompiledGraph::Close() {std::lock_guard<std::mutex> lock(mutex_);Clear();}
bool MetalCompiledGraph::Build(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);Clear();
    auto start=std::chrono::steady_clock::now();
    try {
        if(graph.runtime()!=Host()->runtime||symbols.runtime()!=Host()->runtime||!symbols.IsList())
            throw std::runtime_error("Metal graph inputs must belong to this runtime");
        std::string error;if(!index_.Open(checkpoint,error))throw std::runtime_error(error);
        gpu_=std::make_unique<MetalContext>();
        for(uint64_t i=0;i<symbols.Size();++i) {
            X::Tensor symbol(symbols.Get(i));auto info=symbol.Info();
            if(!info.symbolic||(info.rank!=1&&info.rank!=2)||info.dtype!=X3_TENSOR_FLOAT32||
                info.shape[0]<=0||(info.rank==2&&info.shape[1]<=0))
                throw std::runtime_error("Metal graph input must be symbolic FP32 vector or matrix");
            inputShapes_.emplace_back(info.shape,info.shape+info.rank);
            const auto id=Allocate(info.shape[0],info.rank==2?info.shape[1]:1);inputs_.push_back(id);
            if(!tensors_.emplace(info.id,id).second)throw std::runtime_error("Duplicate Metal symbolic input");
        }
        TensorGraphCapture capture(graph);
        auto lookup=[&](const X::Value& value){auto found=tensors_.find(TensorId(value));
            if(found==tensors_.end())throw std::runtime_error("Unknown Metal graph operand");return found->second;};
        auto matmul=[&](size_t a,size_t b){
            if(slots_[a].columns!=slots_[b].columns)throw std::runtime_error("Metal checkpoint projection shape mismatch");
            auto out=Allocate(slots_[a].rows,slots_[b].rows);ops_.push_back({Kind::MatMul,a,b,out,0,0,true});return out;};
        for(const auto& node:capture.Operations()) {
            if(node.provider.empty())continue;
            if(node.provider!="garnet"||node.inputs.empty()||node.operandCount<1||node.operandCount>2)
                throw std::runtime_error("Unsupported initial Metal graph operation: "+node.name);
            auto a=lookup(node.inputs[0]);size_t out;
            if(node.operandCount==2) {
                auto b=lookup(node.inputs[1]);
                const bool mask=node.name=="qwen35_mask_rows";
                if(node.name!="add"&&!mask)throw std::runtime_error("Metal binary lowering not implemented: "+node.name);
                if(slots_[a].rows!=slots_[b].rows||slots_[b].columns!=(mask?1:slots_[a].columns))
                    throw std::runtime_error("Metal binary shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({mask?Kind::Mask:Kind::Add,a,b,out});
                tensors_[TensorId(node.output)]=out;continue;
            }
            if(node.name=="qwen35_rms_norm") {
                auto weight=Weight(node.attributes.Get("weight_name").ToString());
                if(slots_[weight].rows!=1||slots_[weight].columns!=slots_[a].columns)
                    throw std::runtime_error("Metal normalization shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({Kind::Norm,a,weight,out,float(node.attributes.Get("eps").ToDouble()),1,false});
            }else if(node.name=="qwen35_mlp") {
                auto prefix=node.attributes.Get("weight_prefix").ToString();
                auto gate=matmul(a,Weight(prefix+".gate_proj.weight"));
                auto up=matmul(a,Weight(prefix+".up_proj.weight"));
                if(slots_[gate].columns!=slots_[up].columns)throw std::runtime_error("Metal gate/up shape mismatch");
                auto activated=Allocate(slots_[gate].rows,slots_[gate].columns);
                ops_.push_back({Kind::Silu,gate,up,activated});
                out=matmul(activated,Weight(prefix+".down_proj.weight"));
            }else if(node.name=="qwen35_tied_head")out=matmul(a,Weight("model.language_model.embed_tokens.weight"));
            else throw std::runtime_error("Metal lowering not implemented: "+node.name);
            tensors_[TensorId(node.output)]=out;
        }
        auto output=capture.Outputs();
        if((output.IsList()||x3_value_object_kind(output.raw())==X3_OBJECT_KIND_TUPLE)&&output.Size()==1)
            output=output.Get(uint64_t(0));
        if(!IsTensor(output))throw std::runtime_error("Initial Metal graph requires one tensor output");
        output_=lookup(output);ready_=true;
        compileMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return true;
    }catch(...){Clear();throw;}
}
X::Value MetalCompiledGraph::Run(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!ready_||inputs.runtime()!=Host()->runtime||!inputs.IsList()||inputs.Size()!=inputs_.size())
        throw std::runtime_error("Metal graph not ready or input count invalid");
    auto start=std::chrono::steady_clock::now();
    for(size_t i=0;i<inputs_.size();++i) {
        X::Tensor tensor(inputs.Get(i));auto info=tensor.Info();auto& slot=slots_[inputs_[i]];
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=inputShapes_[i].size()||
            !std::equal(inputShapes_[i].begin(),inputShapes_[i].end(),info.shape)||!info.data||!IsContiguousTensor(tensor))
            throw std::runtime_error("Metal runtime input shape/storage mismatch");
        auto lease=tensor.Acquire();auto data=static_cast<const float*>(info.data);
        slot.matrix=gpu_->Upload(std::vector<float>(data,data+slot.rows*slot.columns),slot.rows,slot.columns);
    }
    gpu_->Begin();
    try {
        for(const auto& op:ops_) {
            auto a=slots_[op.a].matrix,b=slots_[op.b].matrix,out=slots_[op.out].matrix;
            if(op.kind==Kind::MatMul)gpu_->EncodeMatMul(a,b,out,op.transpose);
            else if(op.kind==Kind::Norm)gpu_->EncodeRMSNorm(a,b,out,op.epsilon,op.offset);
            else if(op.kind==Kind::Silu)gpu_->EncodeSiluMul(a,b,out);
            else if(op.kind==Kind::Add)gpu_->EncodeAdd(a,b,out);
            else gpu_->EncodeMaskRows(a,b,out);
        }
        gpu_->SubmitAndWait();
    }catch(...){gpu_->Abort();throw;}
    auto result=gpu_->Download(slots_[output_].matrix);auto& output=slots_[output_];
    inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{int64_t(output.rows),int64_t(output.columns)},result.data(),result.size()*4);
}
}
