// SPDX-License-Identifier: Apache-2.0
#include "trt_builder.h"
#include "modelopt_nvfp4_plugin.h"
#include "../../cuda/modelopt_nvfp4.cuh"
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
ITensor* TRTBuilder::GetModelOptNVFP4Weight(const std::string& name,const SafeTensorMetadata& entry){
    try{
        if(name.size()<7||name.substr(name.size()-7)!=".weight"||entry.dataType!="U8"||entry.shape.size()!=2||
            entry.shape[0]<=0||entry.shape[0]>1048576||entry.shape[0]%16||entry.shape[1]<=0||entry.shape[1]>32768||
            entry.shape[1]%32||entry.dataSize!=uint64_t(entry.shape[0])*entry.shape[1])
            throw std::invalid_argument("Invalid bounded native NVFP4 weight: "+name);
        const int rows=int(entry.shape[0]),columns=int(entry.shape[1])*2;
        const auto prefix=name.substr(0,name.size()-7);
        auto data=[&](const SafeTensorMetadata& e)->const void*{
            auto& mapping=capturedWeightFiles[e.filePath.string()];
            if(!mapping){mapping=std::make_unique<SafeTensorsMappedFile>();
                if(!mapping->Open(e.filePath,loweringError))throw std::runtime_error(loweringError);}
            auto* bytes=mapping->DataAt(e.dataOffset,e.dataSize);
            if(!bytes)throw std::invalid_argument("NVFP4 tensor span exceeds checkpoint");return bytes;
        };
        auto scalar=[&](const std::string& suffix){
            auto* e=capturedWeightIndex->Find(prefix+suffix);
            if(!e||e->dataType!="F32"||!e->shape.empty()||e->dataSize!=4)
                throw std::invalid_argument("Invalid NVFP4 scalar: "+prefix+suffix);
            float value;std::memcpy(&value,data(*e),4);
            if(!std::isfinite(value)||value<=0)throw std::invalid_argument("NVFP4 scalar must be finite and positive");
            return value;
        };
        const float weightScale=scalar(".weight_scale_2"),inputScale=scalar(".input_scale");
        auto* scale=capturedWeightIndex->Find(prefix+".weight_scale");
        if(!scale||(scale->dataType!="F8_E4M3"&&scale->dataType!="F8_E4M3FN")||
            scale->shape!=std::vector<long long>{rows,columns/16}||scale->dataSize!=uint64_t(rows)*(columns/16))
            throw std::invalid_argument("Invalid NVFP4 block16 scale layout: "+name);
        const auto tiledBytes=ModelOptNVFP4ScaleBytes(rows,columns);
        if(!tiledBytes||tiledBytes>(256ULL<<20))throw std::invalid_argument("NVFP4 tiled scale allocation exceeds256MiB");
        const auto* raw=static_cast<const unsigned char*>(data(*scale));
        nativeNVFP4ScaleStorage.emplace_back(tiledBytes,0);auto& tiled=nativeNVFP4ScaleStorage.back();
        const int blocks=columns/16,paddedColumns=((blocks+3)/4)*4;
        for(int row=0;row<rows;++row)for(int col=0;col<blocks;++col){
            const auto value=raw[size_t(row)*blocks+col];
            if(value>126)throw std::invalid_argument("NVFP4 block scale is negative or nonfinite");
            const size_t offset=size_t(row/128)*128*paddedColumns+size_t(col/4)*512+
                (row%32)*16+((row%128)/32)*4+col%4;
            tiled[offset]=value;
        }
        Weights packed{DataType::kINT8,data(entry),int64_t(entry.dataSize)};
        auto* weight=network->addConstant(Dims2{rows,columns/2},packed);
        Dims scaleShape{};scaleShape.nbDims=1;scaleShape.d[0]=int64_t(tiledBytes);
        auto* scales=network->addConstant(scaleShape,{DataType::kINT8,tiled.data(),int64_t(tiledBytes)});
        // TensorRT10.16's stripped-weight builder asserts for these opaque INT8
        // plugin constants. Keep compressed bytes in the plan until external
        // packed weight bindings can share them across prefill/decode engines.
        if(!weight||!scales||!network->setWeightsName(packed,name.c_str()))
            throw std::runtime_error("Native NVFP4 constant registration failed");
        weight->setName(name.c_str());scales->setName((prefix+".weight_scale.tiled").c_str());
        auto* result=weight->getOutput(0);result->setName(name.c_str());
        nativeNVFP4Weights[result]={scales->getOutput(0),rows,columns,weightScale,inputScale};
        weightTensorMap[name]=result;return result;
    }catch(const std::exception& error){loweringError=error.what();return nullptr;}
}
ITensor* TRTBuilder::ProjectModelOptNVFP4(ITensor* input,ITensor* weight){
    auto found=nativeNVFP4Weights.find(weight);if(found==nativeNVFP4Weights.end())return ProjectModelOptFP8(input,weight);
    const auto& p=found->second;auto shape=input->getDimensions();
    if(shape.nbDims<2||shape.d[shape.nbDims-1]!=p.columns)throw std::invalid_argument("Native NVFP4 projection shape mismatch");
    int64_t rows=1;
    for(int i=0;i<shape.nbDims-1;++i){if(shape.d[i]<=0||rows>1024/shape.d[i])throw std::invalid_argument("Native NVFP4 projection requires1..1024 bounded input rows");rows*=shape.d[i];}
    if(!EnsureModelOptNVFP4PluginRegistered())throw std::runtime_error("Native NVFP4 plugin registration failed");
    auto* cast=input->getType()==DataType::kBF16?nullptr:network->addCast(*input,DataType::kBF16);
    auto* value=input->getType()==DataType::kBF16?input:(cast?cast->getOutput(0):nullptr);
    auto* flatten=value?network->addShuffle(*value):nullptr;
    if(!flatten)throw std::runtime_error("Native NVFP4 input reshape failed");
    flatten->setReshapeDimensions(Dims2{int(rows),p.columns});
    auto* plugin=new ModelOptNVFP4Plugin(p.rows,p.columns,p.weightScale,p.inputScale);ownedPlugins.push_back(plugin);
    ITensor* inputs[]={flatten->getOutput(0),weight,p.scales};
    auto* layer=network->addPluginV2(inputs,3,*plugin);
    if(!layer)throw std::runtime_error("Native NVFP4 projection layer failed");
    auto* restore=network->addShuffle(*layer->getOutput(0));
    if(!restore)throw std::runtime_error("Native NVFP4 output reshape failed");
    shape.d[shape.nbDims-1]=p.rows;restore->setReshapeDimensions(shape);return restore->getOutput(0);
}
}
