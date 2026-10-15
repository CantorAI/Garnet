// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
namespace Garnet {
// One still image per batch row, fixed equal H/W grids. Patches arrive in
// spatial-merge block order, with channel/temporal/patch-height/width flattened.
inline ov::Output<ov::Node> LowerQwen35Vision(
    ov::Output<ov::Node> patches,const std::function<ov::Output<ov::Node>(const std::string&)>& weight,
    int gridH,int gridW,int hidden,int heads,int depth,int patchSize,int temporal,int channels,
    int merge,int positions,int outHidden,float theta=10000) {
    namespace op=ov::opset13;
    const auto shape=patches.get_shape();
    if(gridH<=0||gridW<=0||merge<=0||gridH%merge||gridW%merge||hidden<=0||heads<=0||hidden%heads||
       (hidden/heads)%4||depth<=0||patchSize<=0||temporal<=0||channels<=0||positions<=0||outHidden<=0||!(theta>0))
        throw std::invalid_argument("invalid Qwen3.5 still-image vision profile");
    const int64_t count=int64_t(gridH)*gridW,feature=int64_t(channels)*temporal*patchSize*patchSize;
    if(shape.size()!=3||!shape[0]||shape[1]!=size_t(count)||shape[2]!=size_t(feature))
        throw std::invalid_argument("vision patch input dimensions mismatch");
    const int64_t batch=shape[0],dim=hidden/heads,unit=merge*merge;
    const int side=int(std::sqrt(positions));
    if(side*side!=positions)throw std::invalid_argument("vision position table must be square");
    auto i64=[](std::vector<int64_t> v){return op::Constant::create(ov::element::i64,ov::Shape{v.size()},v);};
    auto f32=[](float v){return op::Constant::create(ov::element::f32,ov::Shape{},std::vector<float>{v});};
    auto fp=[](ov::Output<ov::Node> v)->ov::Output<ov::Node>{return std::make_shared<op::Convert>(v,ov::element::f32);};
    auto reshape=[&](ov::Output<ov::Node> v,std::vector<int64_t> s)->ov::Output<ov::Node>{return std::make_shared<op::Reshape>(v,i64(s),false);};
    auto transpose=[&](ov::Output<ov::Node> v,std::vector<int64_t> order)->ov::Output<ov::Node>{return std::make_shared<op::Transpose>(v,i64(order));};
    auto slice=[&](ov::Output<ov::Node> v,int64_t start,int64_t end)->ov::Output<ov::Node>{return std::make_shared<op::Slice>(v,i64({start}),i64({end}),i64({1}),i64({-1}));};
    auto require=[](ov::Output<ov::Node> v,ov::Shape s){if(v.get_shape()!=s)throw std::invalid_argument("vision checkpoint dimensions mismatch");};
    auto linear=[&](ov::Output<ov::Node> x,const std::string& prefix)->ov::Output<ov::Node>{
        auto w=weight(prefix+".weight"),bias=weight(prefix+".bias");auto ws=w.get_shape();
        if(ws.size()!=2||ws[1]!=x.get_shape().back())throw std::invalid_argument("vision projection shape mismatch");
        require(bias,{ws[0]});return std::make_shared<op::Add>(std::make_shared<op::MatMul>(x,fp(w),false,true),fp(bias));
    };
    auto norm=[&](ov::Output<ov::Node> x,const std::string& prefix)->ov::Output<ov::Node>{
        auto scale=weight(prefix+".weight"),bias=weight(prefix+".bias");require(scale,{x.get_shape().back()});require(bias,scale.get_shape());
        auto mean=std::make_shared<op::ReduceMean>(x,i64({-1}),true);
        auto centered=std::make_shared<op::Subtract>(x,mean);
        auto variance=std::make_shared<op::ReduceMean>(std::make_shared<op::Multiply>(centered,centered),i64({-1}),true);
        auto normalized=std::make_shared<op::Divide>(centered,std::make_shared<op::Sqrt>(std::make_shared<op::Add>(variance,f32(1e-6f))));
        return std::make_shared<op::Add>(std::make_shared<op::Multiply>(normalized,fp(scale)),fp(bias));
    };
    auto patchWeight=weight("model.visual.patch_embed.proj.weight"),patchBias=weight("model.visual.patch_embed.proj.bias");
    require(patchWeight,{size_t(hidden),size_t(channels),size_t(temporal),size_t(patchSize),size_t(patchSize)});require(patchBias,{size_t(hidden)});
    ov::Output<ov::Node> x=std::make_shared<op::Add>(std::make_shared<op::MatMul>(fp(patches),reshape(fp(patchWeight),{hidden,feature}),false,true),fp(patchBias));
    auto table=weight("model.visual.pos_embed.weight");require(table,{size_t(positions),size_t(hidden)});
    std::vector<int64_t> indices;std::vector<float> coefficients,cosines,sines;
    for(int bh=0;bh<gridH/merge;++bh)for(int bw=0;bw<gridW/merge;++bw)
      for(int ih=0;ih<merge;++ih)for(int iw=0;iw<merge;++iw){
        int h=bh*merge+ih,w=bw*merge+iw;
        float ph=gridH>1?float(h)*float(side-1)/float(gridH-1):0,pw=gridW>1?float(w)*float(side-1)/float(gridW-1):0;
        int h0=int(std::floor(ph)),w0=int(std::floor(pw)),h1=std::min(h0+1,side-1),w1=std::min(w0+1,side-1);
        float dh=ph-h0,dw=pw-w0;
        indices.insert(indices.end(),{h0*side+w0,h0*side+w1,h1*side+w0,h1*side+w1});
        coefficients.insert(coefficients.end(),{(1-dh)*(1-dw),(1-dh)*dw,dh*(1-dw),dh*dw});
        for(int repeat=0;repeat<2;++repeat)for(int axis=0;axis<2;++axis)for(int64_t d=0;d<dim/4;++d){
            float angle=float(axis?w:h)/std::pow(theta,float(2*d)/float(dim/2));cosines.push_back(std::cos(angle));sines.push_back(std::sin(angle));
        }
      }
    auto index=op::Constant::create(ov::element::i64,ov::Shape{size_t(count),4},indices);
    auto coefficient=op::Constant::create(ov::element::f32,ov::Shape{size_t(count),4,1},coefficients);
    auto gathered=std::make_shared<op::Gather>(fp(table),index,op::Constant::create(ov::element::i64,ov::Shape{},std::vector<int64_t>{0}));
    auto positional=std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(gathered,coefficient),i64({1}),false);
    x=std::make_shared<op::Add>(x,positional);
    auto cos=op::Constant::create(ov::element::f32,ov::Shape{1,size_t(count),1,size_t(dim)},cosines);
    auto sin=op::Constant::create(ov::element::f32,ov::Shape{1,size_t(count),1,size_t(dim)},sines);
    auto rotate=[&](ov::Output<ov::Node> q)->ov::Output<ov::Node>{
        auto swapped=std::make_shared<op::Concat>(ov::OutputVector{std::make_shared<op::Negative>(slice(q,dim/2,dim)),slice(q,0,dim/2)},-1);
        return std::make_shared<op::Add>(std::make_shared<op::Multiply>(q,cos),std::make_shared<op::Multiply>(swapped,sin));
    };
    for(int layer=0;layer<depth;++layer){
        auto prefix="model.visual.blocks."+std::to_string(layer);
        auto qkv=linear(norm(x,prefix+".norm1"),prefix+".attn.qkv");require(qkv,{size_t(batch),size_t(count),size_t(3*hidden)});
        auto q=transpose(rotate(reshape(slice(qkv,0,hidden),{batch,count,heads,dim})),{0,2,1,3});
        auto k=transpose(rotate(reshape(slice(qkv,hidden,2*hidden),{batch,count,heads,dim})),{0,2,1,3});
        auto v=transpose(reshape(slice(qkv,2*hidden,3*hidden),{batch,count,heads,dim}),{0,2,1,3});
        auto scores=std::make_shared<op::Multiply>(std::make_shared<op::MatMul>(q,k,false,true),f32(1/std::sqrt(float(dim))));
        auto probabilities=std::make_shared<op::Softmax>(scores,-1);
        auto attended=reshape(transpose(std::make_shared<op::MatMul>(probabilities,v,false,false),{0,2,1,3}),{batch,count,hidden});
        x=std::make_shared<op::Add>(x,linear(attended,prefix+".attn.proj"));
        auto intermediate=linear(norm(x,prefix+".norm2"),prefix+".mlp.linear_fc1");
        auto activated=std::make_shared<op::Gelu>(intermediate,ov::op::GeluApproximationMode::TANH);
        x=std::make_shared<op::Add>(x,linear(activated,prefix+".mlp.linear_fc2"));
    }
    x=reshape(norm(x,"model.visual.merger.norm"),{batch,count/unit,hidden*unit});
    auto intermediate=linear(x,"model.visual.merger.linear_fc1");
    x=linear(std::make_shared<op::Gelu>(intermediate,ov::op::GeluApproximationMode::ERF),"model.visual.merger.linear_fc2");
    require(x,{size_t(batch),size_t(count/unit),size_t(outHidden)});return x;
}
}
