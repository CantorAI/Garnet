// SPDX-License-Identifier: Apache-2.0
#include "compiled_graph.h"
#include "graph_capture.h"
#include "lowering_context.h"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cmath>
namespace Garnet {
size_t MetalCompiledGraph::Allocate(size_t rows,size_t columns) {
    auto id=slots_.size();slots_.push_back({gpu_->Allocate(rows,columns),rows,columns,{int64_t(rows),int64_t(columns)}});return id;
}
size_t MetalCompiledGraph::Weight(const std::string& name) {
    auto found=weights_.find(name);if(found!=weights_.end())return found->second;
    auto entry=index_.Find(name);
    if(!entry||entry->shape.empty()||entry->shape.size()>3||
        (entry->shape.size()==3&&entry->shape[1]!=1))
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
        auto slice=[&](size_t input,size_t groups,size_t sourceWidth,size_t start,size_t count,
            std::vector<int64_t> shape=std::vector<int64_t>{}) {
            if(!groups||!count||start>sourceWidth||count>sourceWidth-start||groups>std::numeric_limits<size_t>::max()/sourceWidth||
                slots_[input].columns!=groups*sourceWidth)throw std::runtime_error("Metal graph slice shape mismatch");
            auto out=Allocate(slots_[input].rows,groups*count);
            if(!shape.empty())slots_[out].shape=std::move(shape);
            Op op{Kind::Slice,input,input,out};op.dimensions={groups,sourceWidth,start,count};ops_.push_back(std::move(op));return out;
        };
        for(const auto& node:capture.Operations()) {
            if(node.provider.empty())continue;
            if(node.provider!="garnet"||node.inputs.empty()||node.operandCount<1||node.operandCount>2)
                throw std::runtime_error("Unsupported initial Metal graph operation: "+node.name);
            auto a=lookup(node.inputs[0]);size_t out;
            if(node.operandCount==2) {
                auto b=lookup(node.inputs[1]);
                if(node.name.rfind("qwen35_conv_bind_",0)==0||node.name.rfind("qwen35_delta_bind_",0)==0||node.name.rfind("qwen35_attention_bind_",0)==0) {
                    static const std::vector<std::string> allowed={"qwen35_conv_bind_weight","qwen35_conv_bind_state",
                        "qwen35_conv_bind_active","qwen35_delta_bind_key","qwen35_delta_bind_value",
                        "qwen35_delta_bind_decay","qwen35_delta_bind_beta","qwen35_delta_bind_state","qwen35_delta_bind_active",
                        "qwen35_attention_bind_key","qwen35_attention_bind_value","qwen35_attention_bind_positions",
                        "qwen35_attention_bind_slots","qwen35_attention_bind_active"};
                    if(std::find(allowed.begin(),allowed.end(),node.name)==allowed.end())
                        throw std::runtime_error("Unknown Metal attention binding: "+node.name);
                    auto bound=bindings_[TensorId(node.inputs[0])];bound[node.name]=b;
                    bindings_[TensorId(node.output)]=std::move(bound);tensors_[TensorId(node.output)]=a;continue;
                }
                if(node.name=="qwen35_replace_packed_hidden") {
                    auto hiddenDim=node.attributes.Get("hidden_dim").ToInt64();
                    if(hiddenDim<=0||slots_[a].rows!=slots_[b].rows||slots_[b].columns!=uint64_t(hiddenDim)||
                        slots_[a].columns<=uint64_t(hiddenDim))throw std::runtime_error("Metal packed hidden replacement shape mismatch");
                    const auto packedWidth=slots_[a].columns;
                    auto tail=slice(a,1,packedWidth,hiddenDim,packedWidth-hiddenDim);
                    out=Allocate(slots_[a].rows,packedWidth);
                    Op op{Kind::Concat,b,tail,out};op.arguments={b,tail,tail};op.transpose=false;ops_.push_back(std::move(op));
                    tensors_[TensorId(node.output)]=out;continue;
                }
                const bool mask=node.name=="qwen35_mask_rows";
                if(node.name!="add"&&!mask)throw std::runtime_error("Metal binary lowering not implemented: "+node.name);
                if(slots_[a].rows!=slots_[b].rows||slots_[b].columns!=(mask?1:slots_[a].columns))
                    throw std::runtime_error("Metal binary shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({mask?Kind::Mask:Kind::Add,a,b,out});
                tensors_[TensorId(node.output)]=out;continue;
            }
            if(node.name=="qwen35_attention_step") {
                auto integer=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal attention dimension: ")+name);
                    return size_t(v);};
                const auto heads=integer("num_heads",128),kv=integer("num_kv_heads",128),dim=integer("head_dim",4096),rotary=integer("rotary_dim",4096);
                const auto theta=node.attributes.Get("rope_theta").ToDouble(),epsilon=node.attributes.Get("eps").ToDouble();
                auto t=node.attributes.Get("mrope_t").ToInt64(),h=node.attributes.Get("mrope_h").ToInt64(),w=node.attributes.Get("mrope_w").ToInt64();
                if(heads%kv||rotary%2||rotary>dim||t<0||h<0||w<0||t>int64_t(rotary/2)||h>int64_t(rotary/2)||w>int64_t(rotary/2)||
                    uint64_t(t+h+w)!=rotary/2||!(theta>0)||!std::isfinite(theta)||!(epsilon>0)||!std::isfinite(float(epsilon)))
                    throw std::runtime_error("Metal attention RoPE configuration invalid");
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const char* name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error(std::string("Metal attention missing binding: ")+name);
                    return it->second;};
                auto key=require("qwen35_attention_bind_key"),value=require("qwen35_attention_bind_value"),positions=require("qwen35_attention_bind_positions"),
                    slot=require("qwen35_attention_bind_slots"),active=require("qwen35_attention_bind_active");
                const auto hs=slots_[a].shape,cs=slots_[key].shape;
                if(hs.size()!=2||cs.size()!=4||cs[0]!=hs[0]||uint64_t(cs[1])!=kv||cs[2]<=0||cs[2]>8192||uint64_t(cs[3])!=dim||
                    slots_[value].shape!=cs||slots_[positions].shape!=std::vector<int64_t>{hs[0],3}||
                    slots_[slot].shape!=std::vector<int64_t>{hs[0]}||slots_[active].shape!=slots_[slot].shape)
                    throw std::runtime_error("Metal full attention state shapes invalid");
                // Control tensors must be explicit host inputs for pre-dispatch bounds checks.
                for(auto control:{positions,slot,active})if(std::find(inputs_.begin(),inputs_.end(),control)==inputs_.end())
                    throw std::runtime_error("Metal attention control tensors must be graph inputs");
                const size_t batch=hs[0],capacity=cs[2],cacheWidth=kv*capacity*dim;
                auto prefix=node.attributes.Get("weight_prefix").ToString();
                auto qg=matmul(a,Weight(prefix+".q_proj.weight")),kp=matmul(a,Weight(prefix+".k_proj.weight")),vp=matmul(a,Weight(prefix+".v_proj.weight"));
                auto qw=Weight(prefix+".q_norm.weight"),kw=Weight(prefix+".k_norm.weight");
                if(slots_[qg].columns!=2*heads*dim||slots_[kp].columns!=kv*dim||slots_[vp].columns!=kv*dim||
                    slots_[qw].rows!=1||slots_[qw].columns!=dim||slots_[kw].rows!=1||slots_[kw].columns!=dim)
                    throw std::runtime_error("Metal full attention checkpoint dimensions mismatch");
                auto q=slice(qg,heads,2*dim,0,dim),gate=slice(qg,heads,2*dim,dim,dim);
                std::vector<float> frequencies(rotary);
                for(size_t i=0;i<rotary/2;++i) {
                    frequencies[i]=float(std::pow(theta,-2.0*double(i)/double(rotary)));
                    frequencies[rotary/2+i]=(i%3==1&&i<size_t(h)*3)?1.0f:((i%3==2&&i<size_t(w)*3)?2.0f:0.0f);
                }
                auto freq=slots_.size();slots_.push_back({gpu_->Upload(frequencies,2,rotary/2),2,rotary/2,{2,int64_t(rotary/2)}});
                auto qr=Allocate(batch,heads*dim),kr=Allocate(batch,kv*dim);
                Op qrope{Kind::NormRope,q,qw,qr,float(epsilon)};qrope.arguments={q,qw,positions,freq};qrope.dimensions={heads,dim,rotary};ops_.push_back(std::move(qrope));
                Op krope{Kind::NormRope,kp,kw,kr,float(epsilon)};krope.arguments={kp,kw,positions,freq};krope.dimensions={kv,dim,rotary};ops_.push_back(std::move(krope));
                auto nextKey=Allocate(batch,cacheWidth),nextValue=Allocate(batch,cacheWidth);
                for(auto pair:{std::make_pair(kr,std::make_pair(key,nextKey)),std::make_pair(vp,std::make_pair(value,nextValue))}) {
                    Op update{Kind::KVUpdate,pair.first,pair.second.first,pair.second.second};
                    update.arguments={pair.first,pair.second.first,slot,active,positions};update.dimensions={kv,dim,capacity};ops_.push_back(std::move(update));
                }
                auto probabilities=Allocate(batch,heads*capacity);
                Op scores{Kind::Scores,qr,nextKey,probabilities};scores.arguments={qr,nextKey,slot,active};scores.dimensions={heads,kv,dim,capacity};ops_.push_back(std::move(scores));
                auto context=Allocate(batch,heads*dim);
                Op attend{Kind::AttentionContext,probabilities,nextValue,context};attend.arguments={probabilities,nextValue,gate};attend.dimensions={heads,kv,dim,capacity};ops_.push_back(std::move(attend));
                auto hidden=matmul(context,Weight(prefix+".o_proj.weight"));
                if(slots_[hidden].columns!=uint64_t(hs[1]))throw std::runtime_error("Metal attention output projection shape mismatch");
                out=Allocate(batch,size_t(hs[1])+2*cacheWidth);
                Op pack{Kind::Concat,hidden,nextKey,out};pack.arguments={hidden,nextKey,nextValue};pack.transpose=true;ops_.push_back(std::move(pack));
            }else if(node.name=="qwen35_attention_output"||node.name=="qwen35_attention_key_cache"||node.name=="qwen35_attention_value_cache") {
                auto dimension=[&](const char* name,size_t limit){auto n=node.attributes.Get(name).ToInt64();
                    if(n<=0||uint64_t(n)>limit)throw std::runtime_error("Invalid Metal attention unpack dimension");
                    return size_t(n);};
                auto hidden=dimension("hidden_dim",65536),kv=dimension("num_kv_heads",128),dim=dimension("head_dim",4096),capacity=dimension("capacity",8192);
                const size_t cache=kv*dim*capacity,width=hidden+2*cache;const auto batch=int64_t(slots_[a].rows);
                if(slots_[a].columns!=width)throw std::runtime_error("Metal attention packed shape mismatch");
                if(node.name=="qwen35_attention_output")out=slice(a,1,width,0,hidden,{batch,int64_t(hidden)});
                else out=slice(a,1,width,hidden+(node.name=="qwen35_attention_value_cache"?cache:0),cache,
                    {batch,int64_t(kv),int64_t(capacity),int64_t(dim)});
            }else if(node.name=="qwen35_linear_step") {
                auto integer=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal linear dimension: ")+name);
                    return size_t(v);};
                const auto kh=integer("key_heads",128),vh=integer("value_heads",128),kd=integer("key_dim",4096),vd=integer("value_dim",4096);
                if(vh%kh)throw std::runtime_error("Metal linear head grouping mismatch");
                const auto hiddenShape=slots_[a].shape;
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const char* name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error(std::string("Metal linear missing binding: ")+name);
                    return it->second;};
                auto convState=require("qwen35_conv_bind_state"),state=require("qwen35_delta_bind_state"),active=require("qwen35_conv_bind_active");
                const auto cs=slots_[convState].shape;
                const size_t channels=2*kh*kd+vh*vd;
                if(hiddenShape.size()!=2||cs.size()!=3||cs[0]!=hiddenShape[0]||uint64_t(cs[1])!=channels||cs[2]<=0||cs[2]>64||
                    slots_[state].shape!=std::vector<int64_t>{hiddenShape[0],int64_t(vh),int64_t(kd),int64_t(vd)}||
                    slots_[active].shape!=std::vector<int64_t>{hiddenShape[0]})throw std::runtime_error("Metal linear state logical shape mismatch");
                const auto batch=size_t(hiddenShape[0]),kernel=size_t(cs[2]);
                const auto prefix=node.attributes.Get("weight_prefix").ToString();
                const auto epsilon=float(node.attributes.Get("eps").ToDouble());
                if(!(epsilon>0)||!std::isfinite(epsilon))throw std::runtime_error("Invalid Metal linear epsilon");
                auto qkv=matmul(a,Weight(prefix+".in_proj_qkv.weight"));
                auto z=matmul(a,Weight(prefix+".in_proj_z.weight"));
                auto ap=matmul(a,Weight(prefix+".in_proj_a.weight")),bp=matmul(a,Weight(prefix+".in_proj_b.weight"));
                auto convWeight=Weight(prefix+".conv1d.weight"),aLog=Weight(prefix+".A_log"),dt=Weight(prefix+".dt_bias");
                if(slots_[qkv].columns!=channels||slots_[z].columns!=vh*vd||slots_[ap].columns!=vh||slots_[bp].columns!=vh||
                    slots_[convWeight].rows!=channels||slots_[convWeight].columns!=kernel||slots_[aLog].columns!=vh||slots_[dt].columns!=vh)
                    throw std::runtime_error("Metal linear checkpoint projection shape mismatch");
                auto convPacked=Allocate(batch,channels*(kernel+1));
                Op conv{Kind::Conv,qkv,convWeight,convPacked};conv.arguments={qkv,convWeight,convState,active};conv.dimensions={kernel};ops_.push_back(std::move(conv));
                const size_t qw=vh*kd,vw=vh*vd,pw=2*qw+vw+2*vh;
                auto prepared=Allocate(batch,pw);
                Op prepare{Kind::Prepare,convPacked,ap,prepared};prepare.arguments={convPacked,ap,bp,aLog,dt};
                prepare.dimensions={kh,vh,kd,vd,kernel};ops_.push_back(std::move(prepare));
                auto q=slice(prepared,1,pw,0,qw),k=slice(prepared,1,pw,qw,qw),v=slice(prepared,1,pw,2*qw,vw),
                    g=slice(prepared,1,pw,2*qw+vw,vh),beta=slice(prepared,1,pw,2*qw+vw+vh,vh);
                auto deltaPacked=Allocate(batch,vw*(kd+1));
                Op delta{Kind::Delta,q,k,deltaPacked};delta.arguments={q,k,v,g,beta,state,active};delta.dimensions={vh,kd,vd};ops_.push_back(std::move(delta));
                auto core=slice(deltaPacked,vh,(kd+1)*vd,0,vd);
                auto gated=Allocate(batch,vw),norm=Weight(prefix+".norm.weight");
                if(slots_[norm].rows!=1||slots_[norm].columns!=vd)throw std::runtime_error("Metal linear norm weight shape mismatch");
                Op gate{Kind::GateNorm,core,z,gated,epsilon};gate.arguments={core,z,norm};gate.dimensions={vh,vd};ops_.push_back(std::move(gate));
                auto hidden=matmul(gated,Weight(prefix+".out_proj.weight"));
                if(slots_[hidden].columns!=slots_[a].columns)throw std::runtime_error("Metal linear output projection shape mismatch");
                auto nextConv=slice(convPacked,channels,kernel+1,1,kernel),nextState=slice(deltaPacked,vh,(kd+1)*vd,vd,kd*vd);
                out=Allocate(batch,slots_[hidden].columns+slots_[nextConv].columns+slots_[nextState].columns);
                Op pack{Kind::Concat,hidden,nextConv,out};pack.arguments={hidden,nextConv,nextState};pack.transpose=true;ops_.push_back(std::move(pack));
            }else if(node.name=="qwen35_linear_output"||node.name=="qwen35_linear_conv_state"||node.name=="qwen35_linear_recurrent_state") {
                auto dim=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error("Invalid Metal linear unpack dimension");
                    return size_t(v);};
                const auto h=dim("hidden_dim",65536),kh=dim("key_heads",128),vh=dim("value_heads",128),kd=dim("key_dim",4096),
                    vd=dim("value_dim",4096),kernel=dim("kernel_size",64),channels=2*kh*kd+vh*vd,conv=channels*kernel,state=vh*kd*vd;
                if(vh%kh||slots_[a].columns!=h+conv+state)throw std::runtime_error("Metal linear packed dimensions mismatch");
                const auto batch=int64_t(slots_[a].rows);
                if(node.name=="qwen35_linear_output")out=slice(a,1,h+conv+state,0,h,{batch,int64_t(h)});
                else if(node.name=="qwen35_linear_conv_state")out=slice(a,1,h+conv+state,h,conv,{batch,int64_t(channels),int64_t(kernel)});
                else out=slice(a,1,h+conv+state,h+conv,state,{batch,int64_t(vh),int64_t(kd),int64_t(vd)});
            }else if(node.name=="qwen35_conv_output"||node.name=="qwen35_conv_state"||
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
    std::unordered_map<size_t,std::vector<float>> hostInputs;
    for(size_t i=0;i<inputs_.size();++i) {
        X::Tensor tensor(inputs.Get(i));auto info=tensor.Info();auto& slot=slots_[inputs_[i]];
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=inputShapes_[i].size()||
            !std::equal(inputShapes_[i].begin(),inputShapes_[i].end(),info.shape)||!info.data||!IsContiguousTensor(tensor))
            throw std::runtime_error("Metal runtime input shape/storage mismatch");
        auto lease=tensor.Acquire();auto data=static_cast<const float*>(info.data);
        hostInputs[inputs_[i]]=std::vector<float>(data,data+slot.rows*slot.columns);
        slot.matrix=gpu_->Upload(hostInputs[inputs_[i]],slot.rows,slot.columns);
    }
    for(const auto& op:ops_)if(op.kind==Kind::KVUpdate) {
        const auto& positions=hostInputs.at(op.arguments[4]);const auto& slots=hostInputs.at(op.arguments[2]);
        const auto& active=hostInputs.at(op.arguments[3]);
        for(size_t row=0;row<active.size();++row) {
            if(!std::isfinite(active[row])||(active[row]!=0&&active[row]!=1)||!std::isfinite(slots[row])||slots[row]<0||
                std::floor(slots[row])!=slots[row]||(active[row]!=0&&slots[row]>=op.dimensions[2]))
                throw std::runtime_error("Metal attention active row/slot invalid");
            for(size_t axis=0;axis<3;++axis)if(!std::isfinite(positions[row*3+axis]))
                throw std::runtime_error("Metal RoPE position must be finite");
        }
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
            }else if(op.kind==Kind::Prepare) {
                auto& args=op.arguments;gpu_->EncodeLinearPrepare(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    slots_[args[3]].matrix,slots_[args[4]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]);
            }else if(op.kind==Kind::GateNorm) {
                auto& args=op.arguments;gpu_->EncodeGatedNorm(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    out,op.dimensions[0],op.dimensions[1],op.epsilon);
            }else if(op.kind==Kind::Concat) {
                auto& args=op.arguments;gpu_->EncodeConcat(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,out,op.transpose);
            }else if(op.kind==Kind::NormRope) {
                auto& args=op.arguments;gpu_->EncodeNormRope(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.epsilon);
            }else if(op.kind==Kind::KVUpdate) {
                auto& args=op.arguments;gpu_->EncodeKVUpdate(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2]);
            }else if(op.kind==Kind::Scores) {
                auto& args=op.arguments;gpu_->EncodeAttentionScores(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
            }else if(op.kind==Kind::AttentionContext) {
                auto& args=op.arguments;gpu_->EncodeAttentionContext(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
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
