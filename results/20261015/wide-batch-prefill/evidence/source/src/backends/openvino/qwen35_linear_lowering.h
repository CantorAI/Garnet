// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "causal_conv_lowering.h"
#include "gated_delta_lowering.h"
namespace Garnet {
struct Qwen35LinearWeights {
    ov::Output<ov::Node> qkv,z,b,a,conv,aLog,dtBias,norm,out;
};
struct Qwen35LinearResult { ov::Output<ov::Node> output,convState,recurrentState; };
// OpenVINO 2026.2.1 ARM SoftPlus/Log showed up to 1.8% relative error
// in a Pi FP32 probe. Evaluate log1p(exp(-abs(x))) through its bounded
// atanh series instead: u=t/(2+t) <= 1/3. Seven terms have an analytic
// truncation error <= 1.05e-8, independent of the input magnitude.
inline ov::Output<ov::Node> Qwen35StableSoftplus(ov::Output<ov::Node> x) {
    namespace op=ov::opset13;
    auto scalar=[](float v){return op::Constant::create(ov::element::f32,ov::Shape{},std::vector<float>{v});};
    x=std::make_shared<op::Convert>(x,ov::element::f32);
    auto t=std::make_shared<op::Exp>(std::make_shared<op::Negative>(std::make_shared<op::Abs>(x)));
    auto u=std::make_shared<op::Divide>(t,std::make_shared<op::Add>(scalar(2),t));
    auto square=std::make_shared<op::Multiply>(u,u);
    ov::Output<ov::Node> polynomial=scalar(1.0f/13);
    for(int denominator=11;denominator>=1;denominator-=2)
        polynomial=std::make_shared<op::Add>(scalar(1.0f/denominator),std::make_shared<op::Multiply>(square,polynomial));
    auto log1p=std::make_shared<op::Multiply>(scalar(2),std::make_shared<op::Multiply>(u,polynomial));
    return std::make_shared<op::Add>(std::make_shared<op::Maximum>(x,scalar(0)),log1p);
}
// Single-token GatedDeltaNet, with independent explicit request states.
inline Qwen35LinearResult LowerQwen35LinearStep(
    ov::Output<ov::Node> x, ov::Output<ov::Node> convState,
    ov::Output<ov::Node> recurrentState, ov::Output<ov::Node> active,
    const Qwen35LinearWeights& w, int keyHeads,int valueHeads,int keyDim,int valueDim,float eps) {
    namespace op=ov::opset13;
    const auto xs=x.get_shape();
    if(xs.size()!=2 || !xs[0] || !xs[1] || keyHeads<=0 || valueHeads<=0 ||
       valueHeads%keyHeads || keyDim<=0 || valueDim<=0 || !(eps>0))
        throw std::invalid_argument("Qwen3.5 linear attention dimensions are invalid");
    const int64_t batch=xs[0], channels=2LL*keyHeads*keyDim+int64_t(valueHeads)*valueDim;
    auto require=[](ov::Output<ov::Node> t,const ov::Shape& shape){
        if(t.get_shape()!=shape)throw std::invalid_argument("Qwen3.5 linear attention weight/state shape mismatch");
    };
    require(w.qkv,{size_t(channels),xs[1]});require(w.z,{size_t(valueHeads*valueDim),xs[1]});
    require(w.b,{size_t(valueHeads),xs[1]});require(w.a,{size_t(valueHeads),xs[1]});
    require(w.aLog,{size_t(valueHeads)});require(w.dtBias,{size_t(valueHeads)});
    require(w.norm,{size_t(valueDim)});require(w.out,{xs[1],size_t(valueHeads*valueDim)});
    require(recurrentState,{xs[0],size_t(valueHeads),size_t(keyDim),size_t(valueDim)});
    auto i64=[](std::vector<int64_t> s){return op::Constant::create(ov::element::i64,ov::Shape{s.size()},s);};
    auto f32=[](float v){return op::Constant::create(ov::element::f32,ov::Shape{},std::vector<float>{v});};
    auto fp=[](ov::Output<ov::Node> t)->ov::Output<ov::Node>{return std::make_shared<op::Convert>(t,ov::element::f32);};
    auto reshape=[&](ov::Output<ov::Node> t,std::vector<int64_t> s)->ov::Output<ov::Node>{return std::make_shared<op::Reshape>(t,i64(s),false);};
    x=fp(x);
    auto project=[&](ov::Output<ov::Node> weight)->ov::Output<ov::Node>{return std::make_shared<op::MatMul>(x,fp(weight),false,true);};
    auto convWeight=w.conv;
    const auto cwShape=convWeight.get_shape();
    if(cwShape.size()==3&&cwShape[1]==1)
        convWeight=reshape(convWeight,{int64_t(cwShape[0]),int64_t(cwShape[2])});
    const auto conv=LowerCausalConvStep(project(w.qkv),convWeight,convState,active);
    auto part=[&](int64_t start,int64_t end,int heads,int dim)->ov::Output<ov::Node>{
        auto slice=std::make_shared<op::Slice>(conv.output,i64({start}),i64({end}),i64({1}),i64({1}));
        auto split=reshape(slice,{batch,heads,dim});
        if(heads==valueHeads)return split;
        auto expanded=reshape(split,{batch,heads,1,dim});
        auto repeated=std::make_shared<op::Broadcast>(expanded,i64({batch,heads,valueHeads/heads,dim}),ov::op::BroadcastType::BIDIRECTIONAL);
        return reshape(repeated,{batch,valueHeads,dim});
    };
    const int64_t keyWidth=int64_t(keyHeads)*keyDim;
    auto q=part(0,keyWidth,keyHeads,keyDim),k=part(keyWidth,2*keyWidth,keyHeads,keyDim);
    auto v=part(2*keyWidth,channels,valueHeads,valueDim);
    auto beta=std::make_shared<op::Sigmoid>(project(w.b));
    auto dt=Qwen35StableSoftplus(std::make_shared<op::Add>(project(w.a),fp(w.dtBias)));
    auto g=std::make_shared<op::Multiply>(std::make_shared<op::Negative>(std::make_shared<op::Exp>(fp(w.aLog))),dt);
    const auto delta=LowerGatedDelta(q,k,v,g,beta,recurrentState,active);
    auto mean=std::make_shared<op::ReduceMean>(std::make_shared<op::Multiply>(delta.output,delta.output),i64({-1}),true);
    auto norm=std::make_shared<op::Divide>(delta.output,std::make_shared<op::Sqrt>(std::make_shared<op::Add>(mean,f32(eps))));
    auto z=reshape(project(w.z),{batch,valueHeads,valueDim});
    auto gated=std::make_shared<op::Multiply>(std::make_shared<op::Multiply>(norm,fp(w.norm)),std::make_shared<op::Multiply>(z,std::make_shared<op::Sigmoid>(z)));
    auto output=std::make_shared<op::MatMul>(reshape(gated,{batch,int64_t(valueHeads)*valueDim}),fp(w.out),false,true);
    return {output,conv.state,delta.state};
}
}
