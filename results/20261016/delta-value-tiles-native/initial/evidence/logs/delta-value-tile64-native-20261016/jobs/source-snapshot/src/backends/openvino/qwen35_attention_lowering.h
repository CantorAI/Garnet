// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <array>
#include <cmath>
#include <stdexcept>
namespace Garnet {
struct Qwen35AttentionWeights { ov::Output<ov::Node> q,k,v,out,qNorm,kNorm; };
struct Qwen35AttentionResult { ov::Output<ov::Node> output,keyCache,valueCache; };
// One token with explicit fixed-capacity KV buffers. The caller supplies a
// valid contiguous-prefix slot for every active row and THW RoPE positions.
inline Qwen35AttentionResult LowerQwen35AttentionStep(
    ov::Output<ov::Node> hidden,ov::Output<ov::Node> keyCache,ov::Output<ov::Node> valueCache,
    ov::Output<ov::Node> positions,ov::Output<ov::Node> slots,ov::Output<ov::Node> active,
    const Qwen35AttentionWeights& w,int heads,int kvHeads,int headDim,int rotaryDim,
    const std::array<int,3>& sections,float theta,float epsilon) {
    namespace op=ov::opset13;
    const auto hs=hidden.get_shape(),cs=keyCache.get_shape();
    if(hs.size()!=2||!hs[0]||!hs[1]||heads<=0||kvHeads<=0||heads%kvHeads||headDim<=0||
       rotaryDim<=0||rotaryDim>headDim||rotaryDim%2||!(theta>0)||!(epsilon>0)||
       sections[0]<0||sections[1]<0||sections[2]<0||sections[0]+sections[1]+sections[2]!=rotaryDim/2||
       cs.size()!=4||cs[0]!=hs[0]||cs[1]!=size_t(kvHeads)||!cs[2]||cs[3]!=size_t(headDim)||
       valueCache.get_shape()!=cs||positions.get_shape()!=ov::Shape{hs[0],3}||
       slots.get_shape()!=ov::Shape{hs[0]}||active.get_shape()!=ov::Shape{hs[0]})
        throw std::invalid_argument("Qwen3.5 full attention shape/configuration is invalid");
    auto require=[](ov::Output<ov::Node> value,const ov::Shape& shape){
        if(value.get_shape()!=shape)throw std::invalid_argument("Qwen3.5 full attention weight dimensions mismatch");
    };
    require(w.q,{size_t(2*heads*headDim),hs[1]});require(w.k,{size_t(kvHeads*headDim),hs[1]});
    require(w.v,{size_t(kvHeads*headDim),hs[1]});require(w.out,{hs[1],size_t(heads*headDim)});
    require(w.qNorm,{size_t(headDim)});require(w.kNorm,{size_t(headDim)});
    auto i64=[](std::vector<int64_t> x){return op::Constant::create(ov::element::i64,ov::Shape{x.size()},x);};
    auto f32=[](float x){return op::Constant::create(ov::element::f32,ov::Shape{},std::vector<float>{x});};
    auto fp=[](ov::Output<ov::Node> x)->ov::Output<ov::Node>{return std::make_shared<op::Convert>(x,ov::element::f32);};
    auto reshape=[&](ov::Output<ov::Node> x,std::vector<int64_t> shape)->ov::Output<ov::Node>{return std::make_shared<op::Reshape>(x,i64(shape),false);};
    auto slice=[&](ov::Output<ov::Node> x,int64_t start,int64_t end)->ov::Output<ov::Node>{return std::make_shared<op::Slice>(x,i64({start}),i64({end}),i64({1}),i64({-1}));};
    auto unsqueeze=[&](ov::Output<ov::Node> x,int axis)->ov::Output<ov::Node>{return std::make_shared<op::Unsqueeze>(x,i64({axis}));};
    const int64_t batch=hs[0],capacity=cs[2];
    hidden=fp(hidden);keyCache=fp(keyCache);valueCache=fp(valueCache);
    auto project=[&](ov::Output<ov::Node> weight)->ov::Output<ov::Node>{return std::make_shared<op::MatMul>(hidden,fp(weight),false,true);};
    auto queryGate=reshape(project(w.q),{batch,heads,2LL*headDim});
    auto q=slice(queryGate,0,headDim);auto gate=reshape(slice(queryGate,headDim,2LL*headDim),{batch,int64_t(heads)*headDim});
    auto k=reshape(project(w.k),{batch,kvHeads,headDim}),v=reshape(project(w.v),{batch,kvHeads,headDim});
    auto norm=[&](ov::Output<ov::Node> x,ov::Output<ov::Node> weight)->ov::Output<ov::Node>{
        auto mean=std::make_shared<op::ReduceMean>(std::make_shared<op::Multiply>(x,x),i64({-1}),true);
        auto denominator=std::make_shared<op::Sqrt>(std::make_shared<op::Add>(mean,f32(epsilon)));
        return std::make_shared<op::Multiply>(std::make_shared<op::Divide>(x,denominator),std::make_shared<op::Add>(fp(weight),f32(1)));
    };
    q=norm(q,w.qNorm);k=norm(k,w.kNorm);
    std::vector<int64_t> axes;std::vector<float> frequencies;
    for(int i=0;i<rotaryDim/2;++i){
        int axis=0;
        if(i%3==1&&i<sections[1]*3)axis=1;
        else if(i%3==2&&i<sections[2]*3)axis=2;
        axes.push_back(axis);frequencies.push_back(float(std::pow(double(theta),-2.0*i/rotaryDim)));
    }
    auto pos=std::make_shared<op::Gather>(fp(positions),i64(axes),i64({1}));
    auto freq=op::Constant::create(ov::element::f32,ov::Shape{frequencies.size()},frequencies);
    auto angles=std::make_shared<op::Multiply>(pos,freq);
    auto fullAngles=std::make_shared<op::Concat>(ov::OutputVector{angles,angles},-1);
    auto cosine=unsqueeze(std::make_shared<op::Cos>(fullAngles),1),sine=unsqueeze(std::make_shared<op::Sin>(fullAngles),1);
    auto rotate=[&](ov::Output<ov::Node> x)->ov::Output<ov::Node>{
        auto part=slice(x,0,rotaryDim);
        auto half=std::make_shared<op::Concat>(ov::OutputVector{std::make_shared<op::Negative>(slice(part,rotaryDim/2,rotaryDim)),slice(part,0,rotaryDim/2)},-1);
        auto rotated=std::make_shared<op::Add>(std::make_shared<op::Multiply>(part,cosine),std::make_shared<op::Multiply>(half,sine));
        if(rotaryDim==headDim)return rotated;
        return std::make_shared<op::Concat>(ov::OutputVector{rotated,slice(x,rotaryDim,headDim)},-1);
    };
    q=rotate(q);k=rotate(k);
    std::vector<int64_t> indices;for(int64_t i=0;i<capacity;++i)indices.push_back(i);
    auto slot=std::make_shared<op::Convert>(slots,ov::element::i64);
    auto mask=std::make_shared<op::Convert>(active,ov::element::boolean);
    auto equals=std::make_shared<op::Equal>(unsqueeze(slot,1),i64(indices));
    auto writeMask=std::make_shared<op::LogicalAnd>(equals,unsqueeze(mask,1));
    auto nextK=std::make_shared<op::Select>(reshape(writeMask,{batch,1,capacity,1}),unsqueeze(k,2),keyCache);
    auto nextV=std::make_shared<op::Select>(reshape(writeMask,{batch,1,capacity,1}),unsqueeze(v,2),valueCache);
    auto repeat=[&](ov::Output<ov::Node> cache)->ov::Output<ov::Node>{
        auto expanded=reshape(cache,{batch,kvHeads,1,capacity,headDim});
        auto repeated=std::make_shared<op::Broadcast>(expanded,i64({batch,kvHeads,heads/kvHeads,capacity,headDim}),ov::op::BroadcastType::BIDIRECTIONAL);
        return reshape(repeated,{batch,heads,capacity,headDim});
    };
    auto scores=std::make_shared<op::Multiply>(std::make_shared<op::MatMul>(unsqueeze(q,2),repeat(nextK),false,true),f32(1.0f/std::sqrt(float(headDim))));
    auto prefix=std::make_shared<op::LessEqual>(i64(indices),unsqueeze(slot,1));
    auto masked=std::make_shared<op::Select>(reshape(prefix,{batch,1,1,capacity}),scores,f32(-1e30f));
    auto probabilities=std::make_shared<op::Softmax>(masked,-1);
    auto context=reshape(std::make_shared<op::MatMul>(probabilities,repeat(nextV)),{batch,int64_t(heads)*headDim});
    auto gated=std::make_shared<op::Multiply>(context,std::make_shared<op::Sigmoid>(gate));
    auto output=std::make_shared<op::MatMul>(gated,fp(w.out),false,true);
    return {std::make_shared<op::Select>(unsqueeze(mask,1),output,f32(0)),nextK,nextV};
}
}
