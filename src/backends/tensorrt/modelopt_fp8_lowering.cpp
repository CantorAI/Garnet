// SPDX-License-Identifier: Apache-2.0
#include "trt_builder.h"
#include "modelopt_fp8_plugin.h"
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
ITensor* TRTBuilder::GetModelOptFP8Weight(const std::string& name,const SafeTensorMetadata& entry){
    try{
        if(name.size()<7||name.substr(name.size()-7)!=".weight"||(entry.dataType!="F8_E4M3"&&entry.dataType!="F8_E4M3FN")||entry.shape.size()!=2||
            entry.shape[0]<=0||entry.shape[0]>1048576||entry.shape[0]%16||entry.shape[1]<=0||entry.shape[1]>65536||
            entry.shape[1]%16||entry.dataSize!=uint64_t(entry.shape[0])*entry.shape[1])
            throw std::invalid_argument("Invalid bounded native FP8 weight: "+name);
        const int rows=int(entry.shape[0]),columns=int(entry.shape[1]);
        const auto prefix=name.substr(0,name.size()-7);
        auto data=[&](const SafeTensorMetadata& e)->const void*{
            auto& mapping=capturedWeightFiles[e.filePath.string()];
            if(!mapping){mapping=std::make_unique<SafeTensorsMappedFile>();
                if(!mapping->Open(e.filePath,loweringError))throw std::runtime_error(loweringError);}
            auto* bytes=mapping->DataAt(e.dataOffset,e.dataSize);
            if(!bytes)throw std::invalid_argument("FP8 tensor span exceeds checkpoint");return bytes;
        };
        auto scalar=[&](const std::string& suffix){
            auto* e=capturedWeightIndex->Find(prefix+suffix);
            if(!e||e->dataType!="F32"||!e->shape.empty()||e->dataSize!=4)
                throw std::invalid_argument("Invalid FP8 scalar: "+prefix+suffix);
            float value;std::memcpy(&value,data(*e),4);
            if(!std::isfinite(value)||value<=0)throw std::invalid_argument("FP8 scalar must be finite and positive");
            return value;
        };
        const float weightScale=scalar(".weight_scale"),inputScale=scalar(".input_scale");
        const auto* raw=static_cast<const unsigned char*>(data(entry));
        for(uint64_t i=0;i<entry.dataSize;++i)
            if((raw[i]&0x7f)==0x7f)throw std::invalid_argument("FP8 checkpoint contains NaN");
        Weights packed{DataType::kINT8,data(entry),int64_t(entry.dataSize)};
        auto* weight=network->addConstant(Dims2{rows,columns},packed);
        if(!weight||!network->setWeightsName(packed,name.c_str()))
            throw std::runtime_error("Native FP8 constant registration failed");
        weight->setName(name.c_str());auto* result=weight->getOutput(0);result->setName(name.c_str());
        nativeFP8Weights[result]={rows,columns,weightScale,inputScale};
        weightTensorMap[name]=result;return result;
    }catch(const std::exception& error){loweringError=error.what();return nullptr;}
}
ITensor* TRTBuilder::ProjectModelOptFP8(ITensor* input,ITensor* weight){
    auto found=nativeFP8Weights.find(weight);if(found==nativeFP8Weights.end())return ProjectCTInt4(input,weight);
    const auto& p=found->second;auto shape=input->getDimensions();
    if(shape.nbDims<2||shape.d[shape.nbDims-1]!=p.columns)throw std::invalid_argument("Native FP8 projection shape mismatch");
    int64_t rows=1;
    for(int i=0;i<shape.nbDims-1;++i){if(shape.d[i]<=0||rows>4096/shape.d[i])throw std::invalid_argument("Native FP8 projection requires1..4096 bounded input rows");rows*=shape.d[i];}
    if(!EnsureModelOptFP8PluginRegistered())throw std::runtime_error("Native FP8 plugin registration failed");
    auto* cast=input->getType()==DataType::kBF16?nullptr:network->addCast(*input,DataType::kBF16);
    auto* value=input->getType()==DataType::kBF16?input:(cast?cast->getOutput(0):nullptr);
    auto* flatten=value?network->addShuffle(*value):nullptr;
    if(!flatten)throw std::runtime_error("Native FP8 input reshape failed");
    flatten->setReshapeDimensions(Dims2{int(rows),p.columns});
    auto* plugin=new ModelOptFP8Plugin(p.rows,p.columns,p.weightScale,p.inputScale);ownedPlugins.push_back(plugin);
    ITensor* inputs[]={flatten->getOutput(0),weight};
    auto* layer=network->addPluginV2(inputs,2,*plugin);
    if(!layer)throw std::runtime_error("Native FP8 projection layer failed");
    auto* restore=network->addShuffle(*layer->getOutput(0));
    if(!restore)throw std::runtime_error("Native FP8 output reshape failed");
    shape.d[shape.nbDims-1]=p.rows;restore->setReshapeDimensions(shape);return restore->getOutput(0);
}
}
