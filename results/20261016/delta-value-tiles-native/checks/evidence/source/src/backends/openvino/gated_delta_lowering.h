// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <cmath>
#include <stdexcept>
namespace Garnet {
struct OpenVINODeltaResult { ov::Output<ov::Node> state, output; };
inline OpenVINODeltaResult LowerGatedDelta(
    ov::Output<ov::Node> q, ov::Output<ov::Node> k, ov::Output<ov::Node> v,
    ov::Output<ov::Node> g, ov::Output<ov::Node> beta,
    ov::Output<ov::Node> state, ov::Output<ov::Node> active) {
    namespace op=ov::opset13;
    const auto qs=q.get_shape(), vs=v.get_shape();
    if(qs.size()!=3 || vs.size()!=3 || qs[0]!=vs[0] || qs[1]!=vs[1] ||
       k.get_shape()!=qs || !qs[0] || !qs[1] || !qs[2] || !vs[2] ||
       state.get_shape()!=ov::Shape{qs[0],qs[1],qs[2],vs[2]} ||
       g.get_shape()!=ov::Shape{qs[0],qs[1]} || beta.get_shape()!=g.get_shape() ||
       active.get_shape()!=ov::Shape{qs[0]})
        throw std::invalid_argument("gated delta graph shape contract is invalid");
    auto i64=[](std::vector<int64_t> x){return op::Constant::create(ov::element::i64,ov::Shape{x.size()},x);};
    auto f32=[](float x){return op::Constant::create(ov::element::f32,ov::Shape{},std::vector<float>{x});};
    auto convert=[](ov::Output<ov::Node> x)->ov::Output<ov::Node>{return std::make_shared<op::Convert>(x,ov::element::f32);};
    auto unsqueeze=[&](ov::Output<ov::Node> x,int64_t axis)->ov::Output<ov::Node>{return std::make_shared<op::Unsqueeze>(x,i64({axis}));};
    auto normalize=[&](ov::Output<ov::Node> x)->ov::Output<ov::Node>{
        auto square=std::make_shared<op::Multiply>(x,x);
        auto sum=std::make_shared<op::ReduceSum>(square,i64({-1}),true);
        auto denominator=std::make_shared<op::Sqrt>(std::make_shared<op::Add>(sum,f32(1e-6f)));
        return std::make_shared<op::Divide>(x,denominator);
    };
    q=std::make_shared<op::Multiply>(normalize(convert(q)),f32(1.0f/std::sqrt(float(qs[2]))));
    k=normalize(convert(k)); v=convert(v); state=convert(state);
    auto decay=unsqueeze(unsqueeze(std::make_shared<op::Exp>(convert(g)),-1),-1);
    auto decayed=std::make_shared<op::Multiply>(state,decay);
    auto memory=std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(decayed,unsqueeze(k,-1)),i64({-2}),false);
    auto delta=std::make_shared<op::Multiply>(std::make_shared<op::Subtract>(v,memory),unsqueeze(convert(beta),-1));
    auto next=std::make_shared<op::Add>(decayed,std::make_shared<op::Multiply>(unsqueeze(k,-1),unsqueeze(delta,-2)));
    auto out=std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(next,unsqueeze(q,-1)),i64({-2}),false);
    auto mask=std::make_shared<op::Convert>(active,ov::element::boolean);
    auto stateMask=std::make_shared<op::Reshape>(mask,i64({int64_t(qs[0]),1,1,1}),false);
    auto outputMask=std::make_shared<op::Reshape>(mask,i64({int64_t(qs[0]),1,1}),false);
    return {std::make_shared<op::Select>(stateMask,next,state),std::make_shared<op::Select>(outputMask,out,f32(0))};
}
}
