// SPDX-License-Identifier: Apache-2.0
#include "qwen35_attention_core_plugin.h"
#include <NvInfer.h>
#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
Qwen35AttentionCorePlugin::Qwen35AttentionCorePlugin(Qwen35AttentionCoreConfig c):c_(c){
    if(!Qwen35AttentionCoreWorkspace(c))throw std::invalid_argument("Invalid Qwen3.5 attention-core profile");
}
bool Qwen35AttentionCorePlugin::matches(const PluginTensorDesc* in,int count) const noexcept {
    if(!in||count!=8)return false;
    const int expected[8][4]={{c_.batch,c_.tokens,c_.heads,c_.dim},{c_.batch,c_.tokens,c_.kv,c_.dim},
        {c_.batch,c_.tokens,c_.kv,c_.dim},{c_.batch,c_.tokens,c_.heads,c_.dim},
        {c_.batch,c_.kv,c_.capacity,c_.dim},{c_.batch,c_.kv,c_.capacity,c_.dim},
        {c_.batch,c_.tokens,0,0},{c_.batch,c_.tokens,0,0}};
    for(int i=0;i<8;++i){
        int rank=i<6?4:2;
        if(in[i].dims.nbDims!=rank||in[i].type!=DataType::kFLOAT||in[i].format!=TensorFormat::kLINEAR)return false;
        for(int d=0;d<rank;++d)if(in[i].dims.d[d]!=expected[i][d])return false;
    }return true;
}
IPluginV2DynamicExt* Qwen35AttentionCorePlugin::clone() const noexcept {
    try{auto* p=new Qwen35AttentionCorePlugin(c_);p->ns_=ns_;return p;}catch(...){return nullptr;}
}
DimsExprs Qwen35AttentionCorePlugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder&) noexcept {
    if(!in||count!=8||index<0||index>=3)return {};
    return in[index==0?0:index==1?4:5];
}
bool Qwen35AttentionCorePlugin::supportsFormatCombination(int pos,const PluginTensorDesc* d,int in,int out) noexcept {
    return d&&in==8&&out==3&&pos>=0&&pos<11&&d[pos].type==DataType::kFLOAT&&d[pos].format==TensorFormat::kLINEAR;
}
size_t Qwen35AttentionCorePlugin::getWorkspaceSize(const PluginTensorDesc* in,int count,const PluginTensorDesc*,int out) const noexcept {
    return out==3&&matches(in,count)?Qwen35AttentionCoreWorkspace(c_):0;
}
int Qwen35AttentionCorePlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,
    const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    if(!matches(in,8)||!out||!inputs||!outputs)return 1;
    for(int i=0;i<3;++i){
        int source=i==0?0:i==1?4:5;
        if(out[i].type!=DataType::kFLOAT||out[i].format!=TensorFormat::kLINEAR||out[i].dims.nbDims!=4)return 1;
        for(int d=0;d<4;++d)if(out[i].dims.d[d]!=in[source].dims.d[d])return 1;
        for(int j=0;j<8;++j)if(outputs[i]==inputs[j])return 1;
        for(int j=0;j<i;++j)if(outputs[i]==outputs[j])return 1;
    }
    return RunQwen35AttentionCore(c_,inputs,outputs,workspace,stream);
}
void Qwen35AttentionCorePlugin::serialize(void* data) const noexcept {if(data)std::memcpy(data,&c_,sizeof(c_));}
class Qwen35AttentionCoreCreator final:public IPluginCreator {
    std::string ns_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen35AttentionCore";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override {
        if(!data||size!=sizeof(Qwen35AttentionCoreConfig))return nullptr;
        Qwen35AttentionCoreConfig c;std::memcpy(&c,data,size);
        if(!Qwen35AttentionCoreWorkspace(c))return nullptr;
        try{auto* p=new Qwen35AttentionCorePlugin(c);p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35AttentionCorePluginRegistered(){
    static Qwen35AttentionCoreCreator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
