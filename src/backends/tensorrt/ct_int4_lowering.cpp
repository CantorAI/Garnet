// SPDX-License-Identifier: Apache-2.0
#include "trt_builder.h"
#include "ct_int4_plugin.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
ITensor* TRTBuilder::GetCTInt4Weight(const std::string& name){try{
 if(name.size()<7||name.substr(name.size()-7)!=".weight")throw std::invalid_argument("INT4 requires a logical weight name");
 const auto prefix=name.substr(0,name.size()-7);
 auto get=[&](const char* suffix){auto* e=capturedWeightIndex->Find(prefix+suffix);if(!e)throw std::invalid_argument("Missing INT4 metadata: "+prefix+suffix);return e;};
 auto* packed=get(".weight_packed");auto* scales=get(".weight_scale");auto* shape=get(".weight_shape");
 auto data=[&](const SafeTensorMetadata& e){auto& map=capturedWeightFiles[e.filePath.string()];
  if(!map){map=std::make_unique<SafeTensorsMappedFile>();if(!map->Open(e.filePath,loweringError))throw std::runtime_error(loweringError);}
  const void* p=map->DataAt(e.dataOffset,e.dataSize);if(!p)throw std::invalid_argument("INT4 tensor span unavailable");return p;
 };
 if(shape->dataType!="I64"||shape->shape!=std::vector<long long>{2}||shape->dataSize!=16)throw std::invalid_argument("INT4 logical shape metadata invalid");
 int64_t geometry[2];std::memcpy(geometry,data(*shape),16);const auto n=geometry[0],k=geometry[1];
 if(n<1||n>1048576||k<128||k>65536||k%128||packed->dataType!="I32"||packed->shape!=std::vector<long long>{n,k/8}||packed->dataSize!=uint64_t(n)*k/2||
  scales->dataType!="BF16"||scales->shape!=std::vector<long long>{n,k/128}||scales->dataSize!=uint64_t(n)*(k/128)*2)
  throw std::invalid_argument("INT4 G128 packed/scales geometry invalid");
 for(const auto* suffix:{".weight_zero_point",".weight_g_idx",".g_idx"})if(capturedWeightIndex->Find(prefix+suffix))throw std::invalid_argument("Unsupported INT4 zero-point/reordered groups");
 const auto* bytes=static_cast<const unsigned char*>(data(*scales));
 for(uint64_t i=0;i<scales->dataSize;i+=2){unsigned short v;std::memcpy(&v,bytes+i,2);if(!v||(v&0x8000)||(v&0x7f80)==0x7f80)throw std::invalid_argument("INT4 scale must be positive finite BF16");}
 Weights pw{DataType::kINT8,data(*packed),n*k/2},sw{DataType::kBF16,data(*scales),n*k/128};
 auto* p=network->addConstant(Dims2{int(n),int(k/2)},pw);auto* s=network->addConstant(Dims2{int(n),int(k/128)},sw);
 if(!p||!s||!network->setWeightsName(pw,(prefix+".weight_packed").c_str())||!network->setWeightsName(sw,(prefix+".weight_scale").c_str()))throw std::runtime_error("INT4 constants registration failed");
 p->setName((prefix+".weight_packed").c_str());s->setName((prefix+".weight_scale").c_str());auto* value=p->getOutput(0);value->setName(name.c_str());
 nativeCTInt4Weights[value]={s->getOutput(0),int(n),int(k)};weightTensorMap[name]=value;return value;
}catch(const std::exception& e){loweringError=e.what();return nullptr;}}
ITensor* TRTBuilder::ProjectCTInt4(ITensor* input,ITensor* weight){
 auto found=nativeCTInt4Weights.find(weight);if(found==nativeCTInt4Weights.end())return nullptr;const auto& p=found->second;auto shape=input->getDimensions();
 if(shape.nbDims<2||shape.d[shape.nbDims-1]!=p.columns)throw std::invalid_argument("INT4 projection inner dimensions differ");
 int64_t rows=1;for(int i=0;i<shape.nbDims-1;++i){if(shape.d[i]<1||rows>128/shape.d[i])throw std::invalid_argument("INT4 projection requires1..128 input rows");rows*=shape.d[i];}
 if(!EnsureCTInt4G128PluginRegistered())throw std::runtime_error("INT4 plugin registration failed");
 auto* cast=input->getType()==DataType::kBF16?nullptr:network->addCast(*input,DataType::kBF16);
 auto* value=input->getType()==DataType::kBF16?input:(cast?cast->getOutput(0):nullptr);auto* flatten=value?network->addShuffle(*value):nullptr;
 if(!flatten)throw std::runtime_error("INT4 input reshape failed");flatten->setReshapeDimensions(Dims2{int(rows),p.columns});
 auto* plugin=new CTInt4G128Plugin(p.rows,p.columns);ownedPlugins.push_back(plugin);ITensor* inputs[]={flatten->getOutput(0),weight,p.scales};
 auto* layer=network->addPluginV2(inputs,3,*plugin);auto* restore=layer?network->addShuffle(*layer->getOutput(0)):nullptr;
 if(!restore)throw std::runtime_error("INT4 projection/restore failed");shape.d[shape.nbDims-1]=p.rows;restore->setReshapeDimensions(shape);return restore->getOutput(0);
}
}
