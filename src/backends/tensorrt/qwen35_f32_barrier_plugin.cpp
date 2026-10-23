// SPDX-License-Identifier: Apache-2.0
#include "qwen35_f32_barrier_plugin.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <mutex>
#include <cstdint>
namespace Garnet {
using namespace nvinfer1;
IPluginV2DynamicExt* Qwen35F32BarrierPlugin::clone() const noexcept {
    try{auto* p=new Qwen35F32BarrierPlugin;p->ns_=ns_;return p;}catch(...){return nullptr;}
}
DimsExprs Qwen35F32BarrierPlugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder&) noexcept {
    return in&&index==0&&count==1?in[0]:DimsExprs{};
}
bool Qwen35F32BarrierPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* d,int in,int out) noexcept {
    return d&&in==1&&out==1&&pos>=0&&pos<2&&d[pos].type==DataType::kFLOAT&&d[pos].format==TensorFormat::kLINEAR;
}
int Qwen35F32BarrierPlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* inputs,void* const* outputs,void*,cudaStream_t stream) noexcept {
    if(!in||!out||!inputs||!outputs||!inputs[0]||!outputs[0]||in[0].type!=DataType::kFLOAT||out[0].type!=DataType::kFLOAT||
        in[0].format!=TensorFormat::kLINEAR||out[0].format!=TensorFormat::kLINEAR||in[0].dims.nbDims!=3||out[0].dims.nbDims!=3)return 1;
    const int limits[]={8,128,16384};size_t elements=1;
    for(int i=0;i<3;++i){int d=in[0].dims.d[i];if(d<1||d>limits[i]||out[0].dims.d[i]!=d)return 1;elements*=size_t(d);}
    if(inputs[0]==outputs[0])return 0;
    const auto a=reinterpret_cast<uintptr_t>(inputs[0]),b=reinterpret_cast<uintptr_t>(outputs[0]);
    if((a>b?a-b:b-a)<elements*sizeof(float))return 1;
    return cudaMemcpyAsync(outputs[0],inputs[0],elements*sizeof(float),cudaMemcpyDeviceToDevice,stream)==cudaSuccess?0:1;
}
class Qwen35F32BarrierCreator final:public IPluginCreator {
    std::string ns_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen35F32Barrier";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection* fields) noexcept override {
        if(fields&&fields->nbFields)return nullptr;
        try{auto* p=new Qwen35F32BarrierPlugin;p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}
    }
    IPluginV2* deserializePlugin(const char*,const void*,size_t bytes) noexcept override{return bytes?nullptr:createPlugin(nullptr,nullptr);}
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35F32BarrierPluginRegistered(){
    static Qwen35F32BarrierCreator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
