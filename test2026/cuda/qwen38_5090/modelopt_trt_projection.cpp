// SPDX-License-Identifier: Apache-2.0
#include "modelopt_trt_projection.h"
#include <cmath>
#include <climits>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {
ITensor* Output(ILayer* layer) {
    if(!layer||!layer->getOutput(0))throw std::runtime_error("ModelOpt TensorRT layer creation failed");
    return layer->getOutput(0);
}
void Scalar(Weights value) {
    if(value.type!=DataType::kFLOAT||value.count!=1||!value.values||
        !std::isfinite(*static_cast<const float*>(value.values))||*static_cast<const float*>(value.values)<=0)
        throw std::invalid_argument("ModelOpt scale must be a positive finite F32 scalar");
}
ITensor* Constant(INetworkDefinition& n,Dims shape,Weights w){return Output(n.addConstant(shape,w));}
ITensor* DQ(INetworkDefinition& n,ITensor& input,ITensor& scale,DataType type,Dims block={}) {
    auto* layer=n.addDequantize(input,scale,type);
    if(!layer||(block.nbDims&&!layer->setBlockShape(block)))
        throw std::runtime_error("ModelOpt dequantization block creation failed");
    return Output(layer);
}
}
ModelOptTRTProjection AddModelOptTRTWeight(INetworkDefinition& n,Dims shape,
    Weights packed,Weights scales,Weights scale2,Weights inputScale,bool nvfp4) {
    Scalar(inputScale);
    if(shape.nbDims!=2||shape.d[0]<=0||shape.d[1]<=0||shape.d[0]>INT_MAX||shape.d[1]>INT_MAX||
        packed.count!=int64_t(shape.d[0])*shape.d[1]||!packed.values||!scales.values)
        throw std::invalid_argument("ModelOpt projection requires a valid logical matrix");
    auto* inputScaleTensor=Constant(n,Dims{},inputScale);
    auto* codes=Constant(n,shape,packed);
    ITensor* weight=nullptr;
    if(nvfp4){
        Scalar(scale2);
        if(packed.type!=DataType::kFP4||scales.type!=DataType::kFP8||shape.d[1]%16||
            scales.count!=int64_t(shape.d[0])*(shape.d[1]/16))
            throw std::invalid_argument("Invalid NVFP4 block16 matrix storage");
        const auto* bytes=static_cast<const unsigned char*>(scales.values);
        for(int64_t i=0;i<scales.count;++i)if(bytes[i]>126)
            throw std::invalid_argument("NVFP4 block scales must be finite nonnegative E4M3");
        auto scaleShape=shape;scaleShape.d[1]/=16;
        auto* quantizedScales=Constant(n,scaleShape,scales);
        auto* globalScale=Constant(n,Dims{},scale2);
        auto* blockScales=DQ(n,*quantizedScales,*globalScale,DataType::kFLOAT);
        auto* decoded=DQ(n,*codes,*blockScales,DataType::kFLOAT,Dims2{1,16});
        weight=Output(n.addCast(*decoded,DataType::kBF16));
    }else{
        Scalar(scales);
        if(packed.type!=DataType::kFP8)throw std::invalid_argument("Invalid per-tensor FP8 matrix storage");
        weight=DQ(n,*codes,*Constant(n,Dims{},scales),DataType::kBF16);
    }
    return {weight,inputScaleTensor,nvfp4};
}
ITensor* ProjectModelOptTRT(INetworkDefinition& n,ITensor& input,const ModelOptTRTProjection& p) {
    if(!p.weight||!p.inputScale)throw std::invalid_argument("Missing ModelOpt projection metadata");
    auto shape=input.getDimensions(),weights=p.weight->getDimensions();
    if(shape.nbDims<2||shape.d[shape.nbDims-1]!=weights.d[1])
        throw std::invalid_argument("ModelOpt projection inner dimensions differ");
    int64_t rows=1;
    for(int i=0;i<shape.nbDims-1;++i){
        if(shape.d[i]<=0||rows>INT_MAX/shape.d[i])throw std::invalid_argument("ModelOpt projection requires bounded rows");
        rows*=shape.d[i];
    }
    auto* cast=input.getType()==DataType::kBF16?&input:Output(n.addCast(input,DataType::kBF16));
    auto* flatten=n.addShuffle(*cast);
    if(!flatten)throw std::runtime_error("ModelOpt reshape failed");
    flatten->setReshapeDimensions(Dims2{int(rows),int(weights.d[1])});
    ITensor* activation=nullptr;
    if(p.nvfp4){
        // TensorRT10.16's double-scale translator requires a scalar block size.
        // V2 with blockShape={1,16} is rejected during tactic compilation.
        auto* q=n.addDynamicQuantize(*flatten->getOutput(0),1,16,DataType::kFP4,DataType::kFP8);
        if(!q)throw std::runtime_error("NVFP4 dynamic activation quantization failed");
        q->setInput(1,*p.inputScale);
        auto* scales=DQ(n,*q->getOutput(1),*p.inputScale,DataType::kFLOAT);
        auto* decoded=DQ(n,*q->getOutput(0),*scales,DataType::kFLOAT,Dims2{1,16});
        activation=Output(n.addCast(*decoded,DataType::kBF16));
    }else{
        auto* q=n.addQuantize(*flatten->getOutput(0),*p.inputScale,DataType::kFP8);
        activation=DQ(n,*Output(q),*p.inputScale,DataType::kBF16);
    }
    auto* result=Output(n.addMatrixMultiply(*activation,MatrixOperation::kNONE,*p.weight,MatrixOperation::kTRANSPOSE));
    shape.d[shape.nbDims-1]=weights.d[0];
    auto* restore=n.addShuffle(*result);
    if(!restore)throw std::runtime_error("ModelOpt output reshape failed");
    restore->setReshapeDimensions(shape);
    return Output(restore);
}
}
