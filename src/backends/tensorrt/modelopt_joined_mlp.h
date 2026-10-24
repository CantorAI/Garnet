// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "safetensors_index.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
namespace Garnet {
// A synthetic, explicit opt-in name; never an alias for arbitrary matrices.
inline bool JoinedNVFP4MLPNames(const std::string& name,std::string& gate,std::string& up){
    const std::string prefix="model.language_model.layers.",suffix=".mlp.gate_up_proj.weight";
    if(name.size()<=prefix.size()+suffix.size()||name.rfind(prefix,0)!=0||
        name.compare(name.size()-suffix.size(),suffix.size(),suffix)!=0)return false;
    const auto index=name.substr(prefix.size(),name.size()-prefix.size()-suffix.size());
    if(index.size()>4||(index.size()>1&&index.front()=='0'))return false;
    unsigned int layer=0;
    for(char c:index){if(c<'0'||c>'9')return false;layer=layer*10+unsigned(c-'0');}
    if(layer>=1024)return false;
    const auto base=prefix+index+".mlp.";gate=base+"gate_proj.weight";up=base+"up_proj.weight";return true;
}
struct JoinedNVFP4MLPMetadata {
    const SafeTensorMetadata *gate=nullptr,*up=nullptr,*gateScales=nullptr,*upScales=nullptr;
    std::string gateName,upName;
    int halfRows=0,columns=0;
    float weightScale=0,inputScale=0;
};
inline JoinedNVFP4MLPMetadata ValidateJoinedNVFP4MLP(const SafeTensorsIndex& index,const std::string& name){
    JoinedNVFP4MLPMetadata p;
    if(!JoinedNVFP4MLPNames(name,p.gateName,p.upName))throw std::invalid_argument("Invalid joined NVFP4 MLP name");
    p.gate=index.Find(p.gateName);p.up=index.Find(p.upName);
    if(!p.gate||!p.up||p.gate->dataType!="U8"||p.up->dataType!="U8"||p.gate->shape!=p.up->shape||
        p.gate->shape.size()!=2||p.gate->shape[0]<128||p.gate->shape[0]>524288||p.gate->shape[0]%128||
        p.gate->shape[1]<32||p.gate->shape[1]>32768||p.gate->shape[1]%32||
        p.gate->dataSize!=uint64_t(p.gate->shape[0])*p.gate->shape[1]||p.up->dataSize!=p.gate->dataSize||
        p.gate->dataSize>(256ULL<<20))throw std::invalid_argument("Joined NVFP4 MLP geometry/storage invalid");
    p.halfRows=int(p.gate->shape[0]);p.columns=int(p.gate->shape[1])*2;
    auto scalar=[&](const std::string& matrix,const char* suffix){
        const auto key=matrix.substr(0,matrix.size()-7)+suffix;const auto* e=index.Find(key);
        if(!e||e->dataType!="F32"||!e->shape.empty()||e->dataSize!=4)throw std::invalid_argument("Joined NVFP4 scalar metadata invalid");
        SafeTensorsMappedFile mapping;std::string error;if(!mapping.Open(e->filePath,error))throw std::runtime_error(error);
        const auto* raw=mapping.DataAt(e->dataOffset,4);if(!raw)throw std::invalid_argument("Joined NVFP4 scalar span invalid");
        float value;std::memcpy(&value,raw,4);if(!std::isfinite(value)||value<=0)throw std::invalid_argument("Joined NVFP4 scalar nonpositive/nonfinite");return value;
    };
    p.weightScale=scalar(p.gateName,".weight_scale_2");p.inputScale=scalar(p.gateName,".input_scale");
    if(p.weightScale!=scalar(p.upName,".weight_scale_2")||p.inputScale!=scalar(p.upName,".input_scale")||
        !std::isfinite(p.weightScale*p.inputScale)||p.weightScale*p.inputScale<=0)
        throw std::invalid_argument("Joined NVFP4 gate/up requires identical positive FP32 calibration");
    auto scales=[&](const std::string& matrix){
        const auto* e=index.Find(matrix.substr(0,matrix.size()-7)+".weight_scale");
        if(!e||(e->dataType!="F8_E4M3"&&e->dataType!="F8_E4M3FN")||
            e->shape!=std::vector<long long>{p.halfRows,p.columns/16}||e->dataSize!=uint64_t(p.halfRows)*(p.columns/16))
            throw std::invalid_argument("Joined NVFP4 block16 scales invalid");return e;
    };
    p.gateScales=scales(p.gateName);p.upScales=scales(p.upName);return p;
}
}
