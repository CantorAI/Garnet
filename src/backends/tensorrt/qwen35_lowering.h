// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <climits>
#include <cmath>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include "bf16_projection_plugin.h"
#include "qwen35_delta_plugin.h"
#include "qwen35_attention_core_plugin.h"
#include "qwen35_f32_barrier_plugin.h"

namespace Garnet {
// Primitive TensorRT lowering for Qwen3.5. Large projection constants retain
// checkpoint BF16 storage; recurrent state, normalization and reductions are
// FP32. State is explicit and per request, never a hidden engine singleton.
class Qwen35TRTOps {
    using T=nvinfer1::ITensor;
    nvinfer1::INetworkDefinition& n;
    std::deque<std::vector<float>>& constants;
    std::function<T*(const std::string&)> lookup;
    std::vector<nvinfer1::IPluginV2*>& plugins;
    bool projectionWorkspace_=false;
    std::function<T*(T*,T*)> nativeProjection_;
    static T* checked(nvinfer1::ILayer* l) {
        if(!l||!l->getOutput(0))throw std::runtime_error("Qwen3.5 TensorRT layer creation failed");
        return l->getOutput(0);
    }
public:
    void SetNativeProjection(std::function<T*(T*,T*)> projection){nativeProjection_=std::move(projection);}
    void SetProjectionWorkspace(int bytes){
        if(bytes!=0&&bytes!=16384)throw std::invalid_argument("Qwen3.5 projection workspace requires0 or16384bytes");
        projectionWorkspace_=bytes!=0;
    }
    Qwen35TRTOps(nvinfer1::INetworkDefinition& network,std::deque<std::vector<float>>& storage,
        std::function<T*(const std::string&)> weights,std::vector<nvinfer1::IPluginV2*>& owned):n(network),constants(storage),lookup(std::move(weights)),plugins(owned) {}
    static nvinfer1::Dims dims(const std::vector<int>& shape) {
        if(shape.size()>nvinfer1::Dims::MAX_DIMS)throw std::invalid_argument("Qwen3.5 rank exceeds TensorRT limit");
        nvinfer1::Dims d{};d.nbDims=int(shape.size());
        for(int i=0;i<d.nbDims;++i)d.d[i]=shape[i];return d;
    }
    static std::vector<int> shape(T* t) {
        if(!t)throw std::invalid_argument("Qwen3.5 missing tensor");
        auto d=t->getDimensions();std::vector<int> s;
        for(int i=0;i<d.nbDims;++i){
            if(d.d[i]<=0||d.d[i]>INT_MAX)throw std::invalid_argument("Qwen3.5 CUDA graph requires bounded static dimensions");
            s.push_back(int(d.d[i]));
        }
        return s;
    }
    static void require(T* t,const std::vector<int>& s) {
        if(shape(t)!=s)throw std::invalid_argument("Qwen3.5 tensor shape mismatch");
    }
    T* constant(std::vector<float> data,const std::vector<int>& s) {
        size_t count=1;for(int d:s){if(d<=0||count>SIZE_MAX/size_t(d))throw std::invalid_argument("Qwen3.5 invalid constant shape");count*=d;}
        if(count!=data.size())throw std::invalid_argument("Qwen3.5 constant size mismatch");
        constants.push_back(std::move(data));auto& v=constants.back();
        return checked(n.addConstant(dims(s),{nvinfer1::DataType::kFLOAT,v.data(),int64_t(v.size())}));
    }
    T* scalar(float v,int rank) {return constant({v},std::vector<int>(rank,1));}
    T* cast(T* x,nvinfer1::DataType type=nvinfer1::DataType::kFLOAT) {
        if(!x)throw std::invalid_argument("Qwen3.5 missing cast input");
        return x->getType()==type?x:checked(n.addCast(*x,type));
    }
    T* reshape(T* x,const std::vector<int>& s) {
        auto* l=n.addShuffle(*x);if(!l)throw std::runtime_error("Qwen3.5 reshape failed");
        l->setZeroIsPlaceholder(false);l->setReshapeDimensions(dims(s));return checked(l);
    }
    T* rank(T* x,int target) {
        auto s=shape(x);if(int(s.size())>target)throw std::invalid_argument("Qwen3.5 broadcast rank mismatch");
        s.insert(s.begin(),target-int(s.size()),1);return reshape(x,s);
    }
    T* ew(T* a,T* b,nvinfer1::ElementWiseOperation op) {
        int r=std::max(shape(a).size(),shape(b).size());a=rank(a,r);b=rank(b,r);
        return checked(n.addElementWise(*a,*b,op));
    }
    T* add(T* a,T* b){return ew(a,b,nvinfer1::ElementWiseOperation::kSUM);}
    T* sub(T* a,T* b){return ew(a,b,nvinfer1::ElementWiseOperation::kSUB);}
    T* mul(T* a,T* b){return ew(a,b,nvinfer1::ElementWiseOperation::kPROD);}
    T* div(T* a,T* b){return ew(a,b,nvinfer1::ElementWiseOperation::kDIV);}
    T* unary(T* x,nvinfer1::UnaryOperation op){return checked(n.addUnary(*x,op));}
    T* sigmoid(T* x){return checked(n.addActivation(*x,nvinfer1::ActivationType::kSIGMOID));}
    T* silu(T* x){return mul(x,sigmoid(x));}
    T* reduce(T* x,int axis,bool keep,bool mean=false) {
        int r=shape(x).size();if(axis<0)axis+=r;
        if(axis<0||axis>=r)throw std::invalid_argument("Qwen3.5 reduction axis out of range");
        return checked(n.addReduce(*x,mean?nvinfer1::ReduceOperation::kAVG:nvinfer1::ReduceOperation::kSUM,1U<<axis,keep));
    }
    T* slice(T* x,int axis,int start,int count) {
        auto s=shape(x);if(axis<0)axis+=int(s.size());
        if(axis<0||axis>=int(s.size())||start<0||count<=0||start+count>s[axis])throw std::invalid_argument("Qwen3.5 slice out of range");
        std::vector<int> begin(s.size(),0),stride(s.size(),1);begin[axis]=start;s[axis]=count;
        return checked(n.addSlice(*x,dims(begin),dims(s),dims(stride)));
    }
    T* cat(std::vector<T*> values,int axis) {
        auto* l=n.addConcatenation(values.data(),int(values.size()));
        if(!l)throw std::runtime_error("Qwen3.5 concatenation failed");l->setAxis(axis);return checked(l);
    }
    T* f32Barrier(T* input){
        auto s=shape(input);
        if(input->getType()!=nvinfer1::DataType::kFLOAT||s.size()!=3||s[0]>8||s[1]>128||s[2]>16384)throw std::invalid_argument("FP32 layer barrier profile invalid");
        if(!EnsureQwen35F32BarrierPluginRegistered())throw std::runtime_error("FP32 layer barrier registration failed");
        auto* plugin=new Qwen35F32BarrierPlugin;plugins.push_back(plugin);T* inputs[]={input};return checked(n.addPluginV2(inputs,1,*plugin));
    }
    T* select(T* condition,T* yes,T* no) {
        int r=std::max(shape(yes).size(),shape(no).size());
        condition=rank(condition,r);yes=rank(yes,r);no=rank(no,r);
        return checked(n.addSelect(*cast(condition,nvinfer1::DataType::kBOOL),*yes,*no));
    }
    T* restoreKVPrefix(T* current,T* backup,T* spans) {
        auto s=shape(current);
        if(s.size()!=4||s[0]>8||s[1]>64||s[2]<32||s[2]>8192||s[3]>256||
            current->getType()!=nvinfer1::DataType::kFLOAT||!backup||!spans||
            backup->getType()!=nvinfer1::DataType::kFLOAT||spans->getType()!=nvinfer1::DataType::kFLOAT)
            throw std::invalid_argument("Qwen3.5 prefix cache profile/type invalid");
        require(backup,s);require(spans,{s[0],2});
        auto* start=reshape(slice(spans,1,0,1),{s[0],1,1,1});
        auto* count=reshape(slice(spans,1,1,1),{s[0],1,1,1});
        std::vector<float> positions(s[2]);for(int i=0;i<s[2];++i)positions[i]=float(i);
        auto* index=constant(std::move(positions),{1,1,s[2],1});
        auto* after=ew(index,sub(start,scalar(1,4)),nvinfer1::ElementWiseOperation::kGREATER);
        auto* before=ew(index,add(start,count),nvinfer1::ElementWiseOperation::kLESS);
        return select(ew(after,before,nvinfer1::ElementWiseOperation::kAND),current,backup);
    }
    T* weight(const std::string& name) {
        T* t=lookup(name);if(!t)throw std::invalid_argument("Qwen3.5 missing checkpoint weight: "+name);return t;
    }
    T* project(T* x,T* w) {
        if(nativeProjection_)if(auto* result=nativeProjection_(x,w))return cast(result);
        auto xs=shape(x),ws=shape(w);
        if(ws.size()!=2||xs.back()!=ws[1])throw std::invalid_argument("Qwen3.5 projection shape mismatch");
        auto type=w->getType();
        if(type==nvinfer1::DataType::kBF16) {
            if(!EnsureBF16ProjectionPluginRegistered())throw std::runtime_error("BF16 projection registration failed");
            auto* plugin=new BF16ProjectionPlugin(projectionWorkspace_);plugins.push_back(plugin);
            int rows=1;for(int i=0;i<int(xs.size())-1;++i){
                if(rows>INT_MAX/xs[i])throw std::invalid_argument("Qwen3.5 projection row count overflows");
                rows*=xs[i];
            }
            T* inputs[]={reshape(cast(x,type),{rows,ws[1]}),w};
            auto* result=checked(n.addPluginV2(inputs,2,*plugin));xs.back()=ws[0];return cast(reshape(result,xs));
        }
        auto* value=checked(n.addMatrixMultiply(*cast(x,type),nvinfer1::MatrixOperation::kNONE,
            *rank(w,int(xs.size())),nvinfer1::MatrixOperation::kTRANSPOSE));
        return cast(value);
    }
    T* rms(T* x,T* scale,float epsilon,bool offset) {
        x=cast(x);const int r=shape(x).size();
        auto* denominator=unary(add(reduce(mul(x,x),-1,true,true),scalar(epsilon,r)),nvinfer1::UnaryOperation::kSQRT);
        scale=cast(scale);if(offset)scale=add(scale,scalar(1,shape(scale).size()));
        return mul(div(x,denominator),scale);
    }
    T* norm(T* x,const std::string& name,float eps){return rms(x,weight(name),eps,true);}
    T* mlp(T* x,const std::string& prefix) {
        return project(mul(silu(project(x,weight(prefix+".gate_proj.weight"))),
            project(x,weight(prefix+".up_proj.weight"))),weight(prefix+".down_proj.weight"));
    }
    T* maskRows(T* x,T* active) {
        auto s=shape(x);require(active,{s[0]});
        return select(reshape(active,{s[0],1}),x,scalar(0,s.size()));
    }
    T* transpose(T* x,const std::vector<int>& order) {
        if(order.size()!=shape(x).size())throw std::invalid_argument("Qwen3.5 transpose rank mismatch");
        nvinfer1::Permutation p{};for(int i=0;i<int(order.size());++i)p.order[i]=order[i];
        auto* l=n.addShuffle(*x);if(!l)throw std::runtime_error("Qwen3.5 transpose failed");l->setFirstTranspose(p);return checked(l);
    }
    T* matmul(T* a,T* b,bool transposeB=false) {
        return checked(n.addMatrixMultiply(*a,nvinfer1::MatrixOperation::kNONE,*b,
            transposeB?nvinfer1::MatrixOperation::kTRANSPOSE:nvinfer1::MatrixOperation::kNONE));
    }
    T* softmax(T* x,int axis) {
        int r=shape(x).size();if(axis<0)axis+=r;
        auto* l=n.addSoftMax(*x);if(!l)throw std::runtime_error("Qwen3.5 softmax failed");l->setAxes(1U<<axis);return checked(l);
    }
    T* attentionProjected(T* qg,T* projectedK,T* projectedV,T* key,T* value,T* positions,T* slots,T* active,
        const std::string& prefix,int heads,int kv,int dim,int rotary,
        const std::array<int,3>& sections,float theta,float eps) {
        auto qs=shape(qg),ks=shape(key);
        if(qs.size()!=3||ks.size()!=4||heads<=0||kv<=0||heads%kv||dim<=0||rotary<=0||rotary>dim||rotary%2||
            sections[0]<0||sections[1]<0||sections[2]<0||sections[0]+sections[1]+sections[2]!=rotary/2||!(theta>0)||!(eps>0))
            throw std::invalid_argument("Qwen3.5 full attention dimensions invalid");
        const int batch=qs[0],capacity=ks[2];
        require(qg,{batch,heads,2*dim});require(projectedK,{batch,kv,dim});require(projectedV,{batch,kv,dim});
        require(key,{batch,kv,capacity,dim});require(value,shape(key));require(positions,{batch,3});require(slots,{batch});require(active,{batch});
        auto* q=rms(slice(qg,-1,0,dim),weight(prefix+".q_norm.weight"),eps,true);
        auto* gate=reshape(slice(qg,-1,dim,dim),{batch,heads*dim});
        auto* k=rms(projectedK,weight(prefix+".k_norm.weight"),eps,true);
        auto* v=projectedV;
        std::vector<float> axes,frequencies;
        for(int i=0;i<rotary/2;++i) {
            int axis=0;if(i%3==1&&i<sections[1]*3)axis=1;else if(i%3==2&&i<sections[2]*3)axis=2;
            axes.push_back(float(axis));frequencies.push_back(float(std::pow(double(theta),-2.0*i/rotary)));
        }
        auto* indices=cast(constant(axes,{rotary/2}),nvinfer1::DataType::kINT32);
        auto* pos=checked(n.addGather(*positions,*indices,1));
        auto* angle=mul(pos,constant(frequencies,{1,rotary/2}));
        angle=reshape(cat({angle,angle},1),{batch,1,rotary});
        auto* cosine=unary(angle,nvinfer1::UnaryOperation::kCOS);
        auto* sine=unary(angle,nvinfer1::UnaryOperation::kSIN);
        auto rotate=[&](T* input)->T* {
            auto* part=slice(input,-1,0,rotary);
            auto* half=cat({unary(slice(part,-1,rotary/2,rotary/2),nvinfer1::UnaryOperation::kNEG),slice(part,-1,0,rotary/2)},2);
            auto* out=add(mul(part,cosine),mul(half,sine));
            return rotary==dim?out:cat({out,slice(input,-1,rotary,dim-rotary)},2);
        };
        q=rotate(q);k=rotate(k);
        std::vector<float> range(capacity);for(int i=0;i<capacity;++i)range[i]=float(i);
        auto* slots2=reshape(slots,{batch,1});auto* range2=constant(range,{1,capacity});
        auto* equal=ew(slots2,range2,nvinfer1::ElementWiseOperation::kEQUAL);
        auto* write=ew(equal,cast(reshape(active,{batch,1}),nvinfer1::DataType::kBOOL),nvinfer1::ElementWiseOperation::kAND);
        T* nextK=select(reshape(write,{batch,1,capacity,1}),reshape(k,{batch,kv,1,dim}),key);
        T* nextV=select(reshape(write,{batch,1,capacity,1}),reshape(v,{batch,kv,1,dim}),value);
        auto repeat=[&](T* input){
            return reshape(mul(reshape(input,{batch,kv,1,capacity,dim}),
                constant(std::vector<float>(heads/kv,1),{1,1,heads/kv,1,1})),{batch,heads,capacity,dim});
        };
        // The verified27B profile has six query heads per KV head. Group
        // queries inside each KV head instead of expanding both caches sixfold.
        // Keep other profiles on their previously validated graph geometry.
        const bool grouped27B=heads==24&&kv==4&&dim==256;
        T* query=reshape(q,grouped27B?std::vector<int>{batch,kv,heads/kv,dim}:std::vector<int>{batch,heads,1,dim});
        T* scores=mul(matmul(query,grouped27B?nextK:repeat(nextK),true),scalar(1/std::sqrt(float(dim)),4));
        auto* beyond=ew(range2,slots2,nvinfer1::ElementWiseOperation::kGREATER);
        auto* valid=unary(beyond,nvinfer1::UnaryOperation::kNOT);
        scores=select(reshape(valid,{batch,1,1,capacity}),scores,scalar(-1e30f,4));
        T* context=reshape(matmul(softmax(scores,-1),grouped27B?nextV:repeat(nextV)),{batch,heads*dim});
        T* output=maskRows(mul(context,sigmoid(gate)),active);
        return cat({output,reshape(nextK,{batch,kv*capacity*dim}),reshape(nextV,{batch,kv*capacity*dim})},1);
    }
    T* attentionStep(T* x,T* key,T* value,T* positions,T* slots,T* active,
        const std::string& prefix,int heads,int kv,int dim,int rotary,
        const std::array<int,3>& sections,float theta,float eps) {
        auto xs=shape(x),ks=shape(key);
        if(xs.size()!=2||ks.size()!=4)throw std::invalid_argument("Qwen3.5 full attention input rank invalid");
        const int batch=xs[0],width=kv*ks[2]*dim,contextWidth=heads*dim;
        auto* packed=attentionProjected(
            reshape(project(x,weight(prefix+".q_proj.weight")),{batch,heads,2*dim}),
            reshape(project(x,weight(prefix+".k_proj.weight")),{batch,kv,dim}),
            reshape(project(x,weight(prefix+".v_proj.weight")),{batch,kv,dim}),
            key,value,positions,slots,active,prefix,heads,kv,dim,rotary,sections,theta,eps);
        auto* output=maskRows(project(slice(packed,1,0,contextWidth),weight(prefix+".o_proj.weight")),active);
        return cat({output,slice(packed,1,contextWidth,width),slice(packed,1,contextWidth+width,width)},1);
    }
    std::array<T*,3> linearCore(T* projected,T* z,T* betaLogits,T* aLogits,T* history,T* state,T* active,const std::string& prefix,
        int kh,int vh,int kd,int vd,float epsilon) {
        const auto xs=shape(projected),hs=shape(history);
        if(xs.size()!=2||hs.size()!=3||kh<=0||vh<=0||vh%kh||kd<=0||vd<=0||!(epsilon>0))
            throw std::invalid_argument("Qwen3.5 linear attention dimensions invalid");
        const int batch=xs[0],channels=2*kh*kd+vh*vd,kernel=hs[2];
        require(history,{batch,channels,kernel});require(state,{batch,vh,kd,vd});require(active,{batch});
        auto w=[&](const char* suffix){return weight(prefix+suffix);};
        require(projected,{batch,channels});
        auto* convWeight=w(".conv1d.weight");require(convWeight,{channels,1,kernel});
        auto* nextHistory=kernel==1?reshape(projected,{batch,channels,1}):
            cat({slice(history,2,1,kernel-1),reshape(projected,{batch,channels,1})},2);
        auto* convolution=silu(reduce(mul(nextHistory,reshape(cast(convWeight),{1,channels,kernel})),2,false));
        nextHistory=select(reshape(active,{batch,1,1}),nextHistory,history);
        auto part=[&](int start,int heads,int dim)->T* {
            T* value=reshape(slice(convolution,1,start,heads*dim),{batch,heads,dim});
            if(heads==vh)return value;
            auto* expanded=reshape(value,{batch,heads,1,dim});
            return reshape(mul(expanded,constant(std::vector<float>(vh/heads,1),{1,1,vh/heads,1})),{batch,vh,dim});
        };
        T* q=part(0,kh,kd);T* k=part(kh*kd,kh,kd);T* v=part(2*kh*kd,vh,vd);
        auto normalize=[&](T* value){return div(value,unary(add(reduce(mul(value,value),-1,true),scalar(1e-6f,3)),nvinfer1::UnaryOperation::kSQRT));};
        q=mul(normalize(q),scalar(1/std::sqrt(float(kd)),3));k=normalize(k);
        T* beta=sigmoid(betaLogits);
        T* a=add(aLogits,cast(w(".dt_bias")));
        // log(1+exp(-abs(a))) + max(a,0), bounded for extreme gates.
        T* absA=unary(a,nvinfer1::UnaryOperation::kABS);
        T* softplus=add(ew(a,scalar(0,2),nvinfer1::ElementWiseOperation::kMAX),
            unary(add(scalar(1,2),unary(unary(absA,nvinfer1::UnaryOperation::kNEG),nvinfer1::UnaryOperation::kEXP)),nvinfer1::UnaryOperation::kLOG));
        T* decay=unary(mul(unary(unary(cast(w(".A_log")),nvinfer1::UnaryOperation::kEXP),nvinfer1::UnaryOperation::kNEG),softplus),nvinfer1::UnaryOperation::kEXP);
        T* decayed=mul(state,reshape(decay,{batch,vh,1,1}));
        T* memory=reduce(mul(decayed,reshape(k,{batch,vh,kd,1})),2,false);
        T* delta=mul(sub(v,memory),reshape(beta,{batch,vh,1}));
        T* next=add(decayed,mul(reshape(k,{batch,vh,kd,1}),reshape(delta,{batch,vh,1,vd})));
        T* output=reduce(mul(next,reshape(q,{batch,vh,kd,1})),2,false);
        next=select(reshape(active,{batch,1,1,1}),next,state);
        output=select(reshape(active,{batch,1,1}),output,scalar(0,3));
        z=reshape(z,{batch,vh,vd});
        output=mul(rms(output,w(".norm.weight"),epsilon,false),silu(z));
        return {reshape(output,{batch,vh*vd}),nextHistory,next};
    }
    T* linearStep(T* x,T* history,T* state,T* active,const std::string& prefix,
        int kh,int vh,int kd,int vd,float epsilon) {
        auto w=[&](const char* suffix){return weight(prefix+suffix);};
        auto core=linearCore(project(x,w(".in_proj_qkv.weight")),project(x,w(".in_proj_z.weight")),
            project(x,w(".in_proj_b.weight")),project(x,w(".in_proj_a.weight")),history,state,active,prefix,kh,vh,kd,vd,epsilon);
        int batch=shape(x)[0],channels=2*kh*kd+vh*vd,kernel=shape(history)[2];
        T* output=maskRows(project(core[0],w(".out_proj.weight")),active);
        return cat({output,reshape(core[1],{batch,channels*kernel}),reshape(core[2],{batch,vh*kd*vd})},1);
    }
    T* linearDecoderChunk(T* x,T* history,T* state,T* active,const std::string& layer,
        int kh,int vh,int kd,int vd,float eps,bool fusedDelta=false,bool retainInputs=false,bool register128=false,bool joinedInputs=false) {
        if(retainInputs&&!fusedDelta)throw std::invalid_argument("Qwen3.5 retained projections require fused delta");
        if(register128&&(!fusedDelta||kd!=128||vd!=128||kh>64||vh>64))throw std::invalid_argument("Register delta requires fused128 dimensions and bounded heads");
        auto xs=shape(x),hs=shape(history);
        if(xs.size()!=3||xs[1]>128||hs.size()!=3||kh<=0||vh<=0||vh%kh||kd<=0||vd<=0)
            throw std::invalid_argument("Qwen3.5 linear chunk dimensions invalid");
        const int batch=xs[0],tokens=xs[1],hidden=xs[2],channels=2*kh*kd+vh*vd,kernel=hs[2];
        if(joinedInputs&&(tokens!=4||(batch!=1&&batch!=2&&batch!=4&&batch!=8)))throw std::invalid_argument("Joined input projections require decode width4 and batch1/2/4/8");
        require(active,{batch,tokens});
        x=select(reshape(active,{batch,tokens,1}),x,scalar(0,3));
        T* normalized=norm(x,layer+".input_layernorm.weight",eps);
        const auto prefix=layer+".linear_attn";
        auto w=[&](const char* suffix){return weight(prefix+suffix);};
        T *qkv=nullptr,*z=nullptr,*beta=nullptr,*a=nullptr;
        if(joinedInputs){
            auto* matrix=w(".in_proj_joined.weight");require(matrix,{channels+vh*vd+2*vh,hidden});
            auto* values=project(normalized,matrix);
            qkv=slice(values,2,0,channels);z=slice(values,2,channels,vh*vd);
            beta=slice(values,2,channels+vh*vd,vh);a=slice(values,2,channels+vh*vd+vh,vh);
        }else{
            qkv=project(normalized,w(".in_proj_qkv.weight"));z=project(normalized,w(".in_proj_z.weight"));
            beta=project(normalized,w(".in_proj_b.weight"));a=project(normalized,w(".in_proj_a.weight"));
        }
        auto token=[&](T* input,int index){auto s=shape(input);return reshape(slice(input,1,index,1),{batch,s[2]});};
        std::vector<T*> outputs;T* coreOutput=nullptr;
        if(fusedDelta){
            if(kd>128||vd>128||kernel>4||kernel<1||!(eps>0))throw std::invalid_argument("Qwen3.5 fused delta profile unsupported");
            require(history,{batch,channels,kernel});require(state,{batch,vh,kd,vd});
            auto* conv=w(".conv1d.weight");require(conv,{channels,1,kernel});
            auto* dt=w(".dt_bias");require(dt,{vh});auto* alog=w(".A_log");require(alog,{vh});
            auto* scale=w(".norm.weight");require(scale,{vd});
            if(!EnsureQwen35DeltaPluginRegistered())throw std::runtime_error("Qwen3.5 delta plugin registration failed");
            auto* plugin=new Qwen35DeltaPlugin({kh,vh,kd,vd,kernel,eps},register128);plugins.push_back(plugin);
            T* inputs[]={cast(qkv),cast(z),cast(beta),cast(a),cast(history),cast(state),cast(active),cast(conv),cast(dt),cast(alog),cast(scale)};
            auto* result=n.addPluginV2(inputs,11,*plugin);if(!result)throw std::runtime_error("Qwen3.5 fused delta construction failed");
            coreOutput=result->getOutput(0);history=result->getOutput(1);state=result->getOutput(2);
        }else for(int t=0;t<tokens;++t) {
            auto core=linearCore(token(qkv,t),token(z,t),token(beta,t),token(a,t),history,state,
                reshape(slice(active,1,t,1),{batch}),prefix,kh,vh,kd,vd,eps);
            outputs.push_back(reshape(core[0],{batch,1,vh*vd}));history=core[1];state=core[2];
        }
        if(!coreOutput)coreOutput=cat(outputs,1);
        T* attention=project(coreOutput,w(".out_proj.weight"));
        attention=select(reshape(active,{batch,tokens,1}),attention,scalar(0,3));
        T* residual=add(x,attention);
        T* output=add(residual,mlp(norm(residual,layer+".post_attention_layernorm.weight",eps),layer+".mlp"));
        std::vector<T*> packed={reshape(output,{batch,tokens*hidden}),reshape(history,{batch,channels*kernel}),reshape(state,{batch,vh*kd*vd})};
        if(retainInputs){
            packed.push_back(reshape(cast(qkv),{batch,tokens*channels}));packed.push_back(reshape(cast(z),{batch,tokens*vh*vd}));
            packed.push_back(reshape(cast(beta),{batch,tokens*vh}));packed.push_back(reshape(cast(a),{batch,tokens*vh}));
        }
        return cat(packed,1);
    }
    T* linearReplayChunk(T* packed,T* history,T* state,T* active,const std::string& layer,
        int hidden,int tokens,int kh,int vh,int kd,int vd,float eps,bool register128=false) {
        if(register128&&(kd!=128||vd!=128||kh>64||vh>64))throw std::invalid_argument("Register delta replay requires128 dimensions and bounded heads");
        auto ps=shape(packed),hs=shape(history);
        if(ps.size()!=2||hs.size()!=3||hidden<1||tokens<1||tokens>16||kh<1||vh<1||vh%kh||kd<1||kd>128||vd<1||vd>128||!(eps>0))
            throw std::invalid_argument("Qwen3.5 delta replay dimensions invalid");
        int batch=ps[0],channels=2*kh*kd+vh*vd,kernel=hs[2];
        if(kernel<1||kernel>4)throw std::invalid_argument("Qwen3.5 delta replay kernel invalid");
        require(history,{batch,channels,kernel});require(state,{batch,vh,kd,vd});require(active,{batch,tokens});
        int offset=tokens*hidden+channels*kernel+vh*kd*vd;
        require(packed,{batch,offset+tokens*(channels+vh*vd+2*vh)});
        auto take=[&](int width){auto* part=reshape(slice(packed,1,offset,tokens*width),{batch,tokens,width});offset+=tokens*width;return part;};
        auto* qkv=take(channels);auto* z=take(vh*vd);auto* beta=take(vh);auto* a=take(vh);
        const auto prefix=layer+".linear_attn";
        auto* conv=weight(prefix+".conv1d.weight");require(conv,{channels,1,kernel});
        auto* dt=weight(prefix+".dt_bias");require(dt,{vh});auto* alog=weight(prefix+".A_log");require(alog,{vh});
        auto* scale=weight(prefix+".norm.weight");require(scale,{vd});
        if(!EnsureQwen35DeltaPluginRegistered())throw std::runtime_error("Qwen3.5 replay plugin registration failed");
        auto* plugin=new Qwen35DeltaPlugin({kh,vh,kd,vd,kernel,eps},register128);plugins.push_back(plugin);
        T* inputs[]={cast(qkv),cast(z),cast(beta),cast(a),cast(history),cast(state),cast(active),cast(conv),cast(dt),cast(alog),cast(scale)};
        auto* result=n.addPluginV2(inputs,11,*plugin);if(!result)throw std::runtime_error("Qwen3.5 delta replay construction failed");
        return cat({reshape(result->getOutput(1),{batch,channels*kernel}),reshape(result->getOutput(2),{batch,vh*kd*vd})},1);
    }
    T* fullDecoderChunk(T* x,T* key,T* value,T* positions,T* slots,T* active,const std::string& layer,
        int heads,int kv,int dim,int rotary,std::array<int,3> sections,float theta,float eps,bool fusedAttention=false) {
        auto xs=shape(x),ks=shape(key);
        if(xs.size()!=3||xs[1]>128||ks.size()!=4||kv<=0||dim<=0)throw std::invalid_argument("Qwen3.5 full chunk dimensions invalid");
        const int batch=xs[0],tokens=xs[1],hidden=xs[2],capacity=ks[2],width=kv*capacity*dim;
        require(active,{batch,tokens});require(positions,{batch,tokens,3});require(slots,{batch,tokens});
        x=select(reshape(active,{batch,tokens,1}),x,scalar(0,3));
        T* normalized=norm(x,layer+".input_layernorm.weight",eps);std::vector<T*> outputs;
        const int contextWidth=heads*dim;
        // Project the complete static chunk once. Repeating these large matrix
        // reads for every token also repeats work on decode padding slots.
        auto prefix=layer+".self_attn";
        auto* qg=project(normalized,weight(prefix+".q_proj.weight"));
        auto* projectedK=project(normalized,weight(prefix+".k_proj.weight"));
        auto* projectedV=project(normalized,weight(prefix+".v_proj.weight"));
        T* coreOutput=nullptr;
        if(fusedAttention){
            if(heads<=0||heads%kv||rotary<=0||rotary>dim||rotary%2||
                sections[0]<0||sections[1]<0||sections[2]<0||sections[0]+sections[1]+sections[2]!=rotary/2||!(theta>0)||!(eps>0))
                throw std::invalid_argument("Qwen3.5 fused attention dimensions invalid");
            require(key,{batch,kv,capacity,dim});require(value,shape(key));
            auto* projectedQGate=reshape(qg,{batch,tokens,heads,2*dim});
            T* q=rms(slice(projectedQGate,-1,0,dim),weight(prefix+".q_norm.weight"),eps,true);
            T* gate=slice(projectedQGate,-1,dim,dim);
            T* k=rms(reshape(projectedK,{batch,tokens,kv,dim}),weight(prefix+".k_norm.weight"),eps,true);
            std::vector<float> axes,frequencies;
            for(int i=0;i<rotary/2;++i){
                int axis=0;if(i%3==1&&i<sections[1]*3)axis=1;else if(i%3==2&&i<sections[2]*3)axis=2;
                axes.push_back(float(axis));frequencies.push_back(float(std::pow(double(theta),-2.0*i/rotary)));
            }
            auto* indices=cast(constant(axes,{rotary/2}),nvinfer1::DataType::kINT32);
            auto* pos=checked(n.addGather(*positions,*indices,2));
            auto* angle=mul(pos,constant(frequencies,{1,1,rotary/2}));
            angle=reshape(cat({angle,angle},2),{batch,tokens,1,rotary});
            auto* cosine=unary(angle,nvinfer1::UnaryOperation::kCOS);
            auto* sine=unary(angle,nvinfer1::UnaryOperation::kSIN);
            auto rotate=[&](T* input)->T*{
                auto* part=slice(input,-1,0,rotary);
                auto* half=cat({unary(slice(part,-1,rotary/2,rotary/2),nvinfer1::UnaryOperation::kNEG),slice(part,-1,0,rotary/2)},3);
                auto* out=add(mul(part,cosine),mul(half,sine));
                return rotary==dim?out:cat({out,slice(input,-1,rotary,dim-rotary)},3);
            };
            if(!EnsureQwen35AttentionCorePluginRegistered())throw std::runtime_error("Qwen3.5 attention plugin registration failed");
            auto* plugin=new Qwen35AttentionCorePlugin({batch,tokens,heads,kv,capacity,dim});plugins.push_back(plugin);
            T* inputs[]={cast(rotate(q)),cast(rotate(k)),cast(reshape(projectedV,{batch,tokens,kv,dim})),
                cast(gate),cast(key),cast(value),cast(slots),cast(active)};
            auto* result=n.addPluginV2(inputs,8,*plugin);if(!result)throw std::runtime_error("Qwen3.5 fused attention construction failed");
            coreOutput=reshape(result->getOutput(0),{batch,tokens,contextWidth});
            key=result->getOutput(1);value=result->getOutput(2);
        }else for(int t=0;t<tokens;++t) {
            auto* packed=attentionProjected(
                reshape(slice(qg,1,t,1),{batch,heads,2*dim}),
                reshape(slice(projectedK,1,t,1),{batch,kv,dim}),
                reshape(slice(projectedV,1,t,1),{batch,kv,dim}),key,value,
                reshape(slice(positions,1,t,1),{batch,3}),reshape(slice(slots,1,t,1),{batch}),
                reshape(slice(active,1,t,1),{batch}),layer+".self_attn",heads,kv,dim,rotary,sections,theta,eps);
            outputs.push_back(reshape(slice(packed,1,0,contextWidth),{batch,1,contextWidth}));
            key=reshape(slice(packed,1,contextWidth,width),{batch,kv,capacity,dim});
            value=reshape(slice(packed,1,contextWidth+width,width),{batch,kv,capacity,dim});
        }
        if(!coreOutput)coreOutput=cat(outputs,1);
        auto* attention=project(coreOutput,weight(prefix+".o_proj.weight"));
        attention=select(reshape(active,{batch,tokens,1}),attention,scalar(0,3));
        T* residual=add(x,attention);
        T* output=add(residual,mlp(norm(residual,layer+".post_attention_layernorm.weight",eps),layer+".mlp"));
        return cat({reshape(output,{batch,tokens*hidden}),reshape(key,{batch,width}),reshape(value,{batch,width})},1);
    }
    T* mtpChunk(T* normalizedTarget,T* embedding,T* key,T* value,T* positions,T* slots,T* active,
        int heads,int kv,int dim,int rotary,std::array<int,3> sections,float theta,float eps) {
        auto s=shape(normalizedTarget);
        if(s.size()!=3)throw std::invalid_argument("Qwen3.5 MTP target requires rank three");
        require(embedding,s);
        // Target is already normalized by the target model's final norm.
        // Iterative draft inputs may instead be the previous MTP normalized output.
        auto* input=cat({norm(embedding,"mtp.pre_fc_norm_embedding.weight",eps),
            norm(normalizedTarget,"mtp.pre_fc_norm_hidden.weight",eps)},2);
        auto* projected=project(input,weight("mtp.fc.weight"));
        auto* packed=fullDecoderChunk(projected,key,value,positions,slots,active,"mtp.layers.0",
            heads,kv,dim,rotary,sections,theta,eps,true);
        const int hiddenWidth=s[1]*s[2],cacheWidth=kv*shape(key)[2]*dim;
        auto* hidden=reshape(slice(packed,1,0,hiddenWidth),s);
        auto* normalized=norm(hidden,"mtp.norm.weight",eps);
        return cat({reshape(normalized,{s[0],hiddenWidth}),slice(packed,1,hiddenWidth,2*cacheWidth)},1);
    }
    T* vision(T* patches,int gh,int gw,int hidden,int heads,int depth,int patch,
        int temporal,int channels,int merge,int positions,int outHidden,float theta,int diagnosticFP32=0) {
        if(gh<=0||gw<=0||merge<=0||gh%merge||gw%merge||hidden<=0||heads<=0||hidden%heads||
            (hidden/heads)%4||depth<=0||patch<=0||temporal<=0||channels<=0||positions<=0||outHidden<=0||!(theta>0))
            throw std::invalid_argument("Qwen3.5 vision profile invalid");
        const int count=gh*gw,feature=channels*temporal*patch*patch,dim=hidden/heads,unit=merge*merge;
        auto ps=shape(patches);if(ps.size()!=3)throw std::invalid_argument("Qwen3.5 vision patches require rank three");
        int batch=ps[0];require(patches,{batch,count,feature});
        if(diagnosticFP32<0||diagnosticFP32>1||
            (diagnosticFP32&&(batch!=1||count>256||hidden!=1152||outHidden!=5120||depth>27)))
            throw std::invalid_argument("FP32 vision diagnostic requires the bounded Qwen3.8 study profile");
        int side=int(std::sqrt(positions));if(side*side!=positions)throw std::invalid_argument("Qwen3.5 vision position table must be square");
        auto w=[&](const std::string& name){
            auto* value=weight("model.visual."+name);
            return diagnosticFP32?cast(value,nvinfer1::DataType::kFLOAT):value;
        };
        auto linear=[&](T* input,const std::string& prefix){return add(project(input,w(prefix+".weight")),cast(w(prefix+".bias")));};
        auto ln=[&](T* input,const std::string& prefix){
            auto s=shape(input);auto* centered=sub(input,reduce(input,-1,true,true));
            auto* den=unary(add(reduce(mul(centered,centered),-1,true,true),scalar(1e-6f,s.size())),nvinfer1::UnaryOperation::kSQRT);
            return add(mul(div(centered,den),cast(w(prefix+".weight"))),cast(w(prefix+".bias")));
        };
        auto* pw=w("patch_embed.proj.weight");require(pw,{hidden,channels,temporal,patch,patch});
        T* x=add(project(patches,reshape(pw,{hidden,feature})),cast(w("patch_embed.proj.bias")));
        auto* table=w("pos_embed.weight");require(table,{positions,hidden});
        std::vector<float> indices,coefficients,cosines,sines;
        for(int bh=0;bh<gh/merge;++bh)for(int bw=0;bw<gw/merge;++bw)
            for(int ih=0;ih<merge;++ih)for(int iw=0;iw<merge;++iw) {
                int h=bh*merge+ih,col=bw*merge+iw;
                float ph=gh>1?float(h)*float(side-1)/float(gh-1):0,pw=gw>1?float(col)*float(side-1)/float(gw-1):0;
                int h0=int(std::floor(ph)),w0=int(std::floor(pw)),h1=std::min(h0+1,side-1),w1=std::min(w0+1,side-1);
                float dh=ph-h0,dw=pw-w0;
                for(int index:{h0*side+w0,h0*side+w1,h1*side+w0,h1*side+w1})indices.push_back(float(index));
                coefficients.insert(coefficients.end(),{(1-dh)*(1-dw),(1-dh)*dw,dh*(1-dw),dh*dw});
                for(int repeat=0;repeat<2;++repeat)for(int axis=0;axis<2;++axis)for(int d=0;d<dim/4;++d) {
                    float angle=float(axis?col:h)/std::pow(theta,float(2*d)/float(dim/2));
                    cosines.push_back(std::cos(angle));sines.push_back(std::sin(angle));
                }
            }
        auto* index=cast(constant(indices,{count,4}),nvinfer1::DataType::kINT32);
        auto* gathered=cast(checked(n.addGather(*table,*index,0)));
        x=add(x,reduce(mul(gathered,constant(coefficients,{count,4,1})),1,false));
        auto* cosine=constant(cosines,{1,count,1,dim});auto* sine=constant(sines,{1,count,1,dim});
        auto rotate=[&](T* input){
            auto* half=cat({unary(slice(input,-1,dim/2,dim/2),nvinfer1::UnaryOperation::kNEG),slice(input,-1,0,dim/2)},3);
            return add(mul(input,cosine),mul(half,sine));
        };
        for(int layer=0;layer<depth;++layer) {
            auto p="blocks."+std::to_string(layer);
            auto* qkv=linear(ln(x,p+".norm1"),p+".attn.qkv");
            auto* q=transpose(rotate(reshape(slice(qkv,-1,0,hidden),{batch,count,heads,dim})),{0,2,1,3});
            auto* k=transpose(rotate(reshape(slice(qkv,-1,hidden,hidden),{batch,count,heads,dim})),{0,2,1,3});
            auto* v=transpose(reshape(slice(qkv,-1,2*hidden,hidden),{batch,count,heads,dim}),{0,2,1,3});
            auto* scores=mul(matmul(q,k,true),scalar(1/std::sqrt(float(dim)),4));
            auto* attended=reshape(transpose(matmul(softmax(scores,-1),v),{0,2,1,3}),{batch,count,hidden});
            x=add(x,linear(attended,p+".attn.proj"));
            auto* intermediate=linear(ln(x,p+".norm2"),p+".mlp.linear_fc1");
            auto* activated=checked(n.addActivation(*intermediate,nvinfer1::ActivationType::kGELU_TANH));
            x=add(x,linear(activated,p+".mlp.linear_fc2"));
        }
        x=reshape(ln(x,"merger.norm"),{batch,count/unit,hidden*unit});
        auto* intermediate=linear(x,"merger.linear_fc1");
        x=linear(checked(n.addActivation(*intermediate,nvinfer1::ActivationType::kGELU_ERF)),"merger.linear_fc2");
        require(x,{batch,count/unit,outHidden});return x;
    }
};
}
