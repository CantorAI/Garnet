// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <stdexcept>
namespace Garnet {
struct OpenVINOConvResult { ov::Output<ov::Node> state, output; };
// One token, depthwise causal convolution. State holds K previous projected
// tokens, oldest first. Inactive request rows neither advance nor emit output.
inline OpenVINOConvResult LowerCausalConvStep(
    ov::Output<ov::Node> token, ov::Output<ov::Node> weight,
    ov::Output<ov::Node> state, ov::Output<ov::Node> active) {
    namespace op = ov::opset13;
    const auto ts = token.get_shape(), ss = state.get_shape();
    if (ts.size()!=2 || !ts[0] || !ts[1] || ss.size()!=3 ||
        ss[0]!=ts[0] || ss[1]!=ts[1] || !ss[2] ||
        weight.get_shape()!=ov::Shape{ts[1],ss[2]} || active.get_shape()!=ov::Shape{ts[0]})
        throw std::invalid_argument("causal convolution graph shape contract is invalid");
    auto i64=[](std::vector<int64_t> x){return op::Constant::create(ov::element::i64,ov::Shape{x.size()},x);};
    auto f32=[](float x){return op::Constant::create(ov::element::f32,ov::Shape{},std::vector<float>{x});};
    token=std::make_shared<op::Convert>(token,ov::element::f32);
    weight=std::make_shared<op::Convert>(weight,ov::element::f32);
    state=std::make_shared<op::Convert>(state,ov::element::f32);
    auto previous=std::make_shared<op::Slice>(state,i64({1}),i64({int64_t(ss[2])}),i64({1}),i64({2}));
    auto newest=std::make_shared<op::Unsqueeze>(token,i64({2}));
    auto next=std::make_shared<op::Concat>(ov::OutputVector{previous,newest},2);
    auto convolution=std::make_shared<op::ReduceSum>(
        std::make_shared<op::Multiply>(next,weight),i64({2}),false);
    auto activated=std::make_shared<op::Multiply>(convolution,std::make_shared<op::Sigmoid>(convolution));
    auto mask=std::make_shared<op::Convert>(active,ov::element::boolean);
    auto stateMask=std::make_shared<op::Reshape>(mask,i64({int64_t(ts[0]),1,1}),false);
    auto outputMask=std::make_shared<op::Reshape>(mask,i64({int64_t(ts[0]),1}),false);
    return {std::make_shared<op::Select>(stateMask,next,state),
            std::make_shared<op::Select>(outputMask,activated,f32(0))};
}
}
