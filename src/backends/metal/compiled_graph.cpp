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
    auto id=slots_.size();slots_.push_back({gpu_->Allocate(rows,columns),rows,columns,{int64_t(rows),int64_t(columns)}});return id;
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
    const auto id=slots_.size();slots_.push_back({gpu_->Upload(data,rows,columns),rows,columns,
        std::vector<int64_t>(entry->shape.begin(),entry->shape.end())});
    weights_[name]=id;return id;
}
void MetalCompiledGraph::Clear() {
    ready_=false;ops_.clear();inputs_.clear();inputShapes_.clear();tensors_.clear();weights_.clear();bindings_.clear();slots_.clear();gpu_.reset();
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
            if(!info.symbolic||info.rank<1||info.rank>4||info.dtype!=X3_TENSOR_FLOAT32)
                throw std::runtime_error("Metal graph input must be symbolic FP32 rank one through four");
            size_t columns=1;
            for(size_t axis=0;axis<info.rank;++axis) {
                if(info.shape[axis]<=0)throw std::runtime_error("Metal graph input dimensions must be positive");
                if(axis) {
                    if(uint64_t(info.shape[axis])>std::numeric_limits<size_t>::max()/columns)
                        throw std::runtime_error("Metal graph input dimensions overflow");
                    columns*=info.shape[axis];
                }
            }
            inputShapes_.emplace_back(info.shape,info.shape+info.rank);
            const auto id=Allocate(info.shape[0],columns);inputs_.push_back(id);slots_[id].shape=inputShapes_.back();
            if(!tensors_.emplace(info.id,id).second)throw std::runtime_error("Duplicate Metal symbolic input");
        }
        TensorGraphCapture capture(graph);
        auto lookup=[&](const X::Value& value){auto found=tensors_.find(TensorId(value));
            if(found==tensors_.end())throw std::runtime_error("Unknown Metal graph operand");
            return found->second;};
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
                if(node.name.rfind("qwen35_conv_bind_",0)==0||node.name.rfind("qwen35_delta_bind_",0)==0) {
                    static const std::vector<std::string> allowed={"qwen35_conv_bind_weight","qwen35_conv_bind_state",
                        "qwen35_conv_bind_active","qwen35_delta_bind_key","qwen35_delta_bind_value",
                        "qwen35_delta_bind_decay","qwen35_delta_bind_beta","qwen35_delta_bind_state","qwen35_delta_bind_active"};
                    if(std::find(allowed.begin(),allowed.end(),node.name)==allowed.end())
                        throw std::runtime_error("Unknown Metal attention binding: "+node.name);
                    auto bound=bindings_[TensorId(node.inputs[0])];bound[node.name]=b;
                    bindings_[TensorId(node.output)]=std::move(bound);tensors_[TensorId(node.output)]=a;continue;
                }
                const bool mask=node.name=="qwen35_mask_rows";
                if(node.name!="add"&&!mask)throw std::runtime_error("Metal binary lowering not implemented: "+node.name);
                if(slots_[a].rows!=slots_[b].rows||slots_[b].columns!=(mask?1:slots_[a].columns))
                    throw std::runtime_error("Metal binary shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({mask?Kind::Mask:Kind::Add,a,b,out});
                tensors_[TensorId(node.output)]=out;continue;
            }
            if(node.name=="qwen35_conv_output"||node.name=="qwen35_conv_state"||
                node.name=="qwen35_delta_output"||node.name=="qwen35_delta_state") {
                const auto shape=slots_[a].shape;
                if(shape.size()!=3)throw std::runtime_error("Metal attention unpack requires rank three packed input");
                const bool delta=node.name.rfind("qwen35_delta_",0)==0,wantState=node.name.find("_state")!=std::string::npos;
                size_t start=0,count=1;std::vector<int64_t> outShape;
                if(delta) {
                    const auto kd=node.attributes.Get("key_dim").ToInt64(),vd=node.attributes.Get("value_dim").ToInt64();
                    if(kd<=0||vd<=0||uint64_t(kd)>=std::numeric_limits<size_t>::max()/uint64_t(vd)||
                        uint64_t(shape[2])!=(uint64_t(kd)+1)*uint64_t(vd))throw std::runtime_error("Metal delta unpack dimensions invalid");
                    start=wantState?vd:0;count=wantState?size_t(kd)*size_t(vd):size_t(vd);
                    outShape=wantState?std::vector<int64_t>{shape[0],shape[1],kd,vd}:std::vector<int64_t>{shape[0],shape[1],vd};
                }else {
                    const auto kernel=node.attributes.Get("kernel_size").ToInt64();
                    if(kernel<=0||uint64_t(kernel)==std::numeric_limits<size_t>::max()||uint64_t(shape[2])!=uint64_t(kernel)+1)
                        throw std::runtime_error("Metal convolution unpack dimensions invalid");
                    start=wantState?1:0;count=wantState?kernel:1;
                    outShape=wantState?std::vector<int64_t>{shape[0],shape[1],kernel}:std::vector<int64_t>{shape[0],shape[1]};
                }
                out=Allocate(slots_[a].rows,size_t(shape[1])*count);slots_[out].shape=std::move(outShape);
                Op op{Kind::Slice,a,a,out};op.dimensions={size_t(shape[1]),size_t(shape[2]),start,count};ops_.push_back(std::move(op));
            }else if(node.name=="qwen35_conv_step"||node.name=="qwen35_delta_step") {
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const std::string& name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error("Metal attention missing binding: "+name);
                    return it->second;};
                const auto shape=slots_[a].shape;
                if(node.name=="qwen35_conv_step") {
                    auto w=require("qwen35_conv_bind_weight"),s=require("qwen35_conv_bind_state"),active=require("qwen35_conv_bind_active");
                    const auto ss=slots_[s].shape;
                    if(shape.size()!=2||ss.size()!=3||ss[0]!=shape[0]||ss[1]!=shape[1]||
                        slots_[w].shape!=std::vector<int64_t>{shape[1],ss[2]}||slots_[active].shape!=std::vector<int64_t>{shape[0]})
                        throw std::runtime_error("Metal causal convolution logical shape mismatch");
                    const size_t kernel=ss[2];
                    if(slots_[s].columns>std::numeric_limits<size_t>::max()-slots_[a].columns)
                        throw std::runtime_error("Metal convolution packed shape overflow");
                    out=Allocate(slots_[a].rows,slots_[a].columns+slots_[s].columns);
                    slots_[out].shape={shape[0],shape[1],int64_t(kernel+1)};
                    Op op{Kind::Conv,a,w,out};op.arguments={a,w,s,active};op.dimensions={kernel};ops_.push_back(std::move(op));
                }else {
                    auto k=require("qwen35_delta_bind_key"),v=require("qwen35_delta_bind_value"),g=require("qwen35_delta_bind_decay"),
                        beta=require("qwen35_delta_bind_beta"),s=require("qwen35_delta_bind_state"),active=require("qwen35_delta_bind_active");
                    const auto vs=slots_[v].shape;
                    if(shape.size()!=3||vs.size()!=3||vs[0]!=shape[0]||vs[1]!=shape[1]||slots_[k].shape!=shape||
                        slots_[s].shape!=std::vector<int64_t>{shape[0],shape[1],shape[2],vs[2]}||
                        slots_[g].shape!=std::vector<int64_t>{shape[0],shape[1]}||slots_[beta].shape!=slots_[g].shape||
                        slots_[active].shape!=std::vector<int64_t>{shape[0]})throw std::runtime_error("Metal gated delta logical shape mismatch");
                    if(slots_[s].columns>std::numeric_limits<size_t>::max()-slots_[v].columns)
                        throw std::runtime_error("Metal delta packed shape overflow");
                    out=Allocate(slots_[a].rows,slots_[v].columns+slots_[s].columns);
                    slots_[out].shape={shape[0],shape[1],int64_t((size_t(shape[2])+1)*size_t(vs[2]))};
                    Op op{Kind::Delta,a,k,out};op.arguments={a,k,v,g,beta,s,active};
                    op.dimensions={size_t(shape[1]),size_t(shape[2]),size_t(vs[2])};ops_.push_back(std::move(op));
                }
            }else if(node.name=="qwen35_rms_norm") {
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
            else if(op.kind==Kind::Mask)gpu_->EncodeMaskRows(a,b,out);
            else if(op.kind==Kind::Conv) {
                auto& args=op.arguments;gpu_->EncodeCausalConv(slots_[args[0]].matrix,slots_[args[1]].matrix,
                    slots_[args[2]].matrix,slots_[args[3]].matrix,out,op.dimensions[0]);
            }else if(op.kind==Kind::Delta) {
                auto& args=op.arguments;gpu_->EncodeGatedDelta(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    slots_[args[3]].matrix,slots_[args[4]].matrix,slots_[args[5]].matrix,slots_[args[6]].matrix,out,
                    op.dimensions[0],op.dimensions[1],op.dimensions[2]);
            }else {
                gpu_->EncodeSliceGroups(a,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
            }
        }
        gpu_->SubmitAndWait();
    }catch(...){gpu_->Abort();throw;}
    auto result=gpu_->Download(slots_[output_].matrix);auto& output=slots_[output_];
    inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,output.shape,result.data(),result.size()*4);
}
}
