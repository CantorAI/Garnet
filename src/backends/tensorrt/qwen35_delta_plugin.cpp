// SPDX-License-Identifier: Apache-2.0
#include "qwen35_delta_plugin.h"
#include <NvInfer.h>
#include <cstring>
#include <mutex>
namespace Garnet {
using namespace nvinfer1;
IPluginV2DynamicExt* Qwen35DeltaPlugin::clone() const noexcept {
    try{auto* p=new Qwen35DeltaPlugin(c_,register128_);p->ns_=ns_;return p;}catch(...){return nullptr;}
}
DimsExprs Qwen35DeltaPlugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder&) noexcept {
    if(count!=11||index<0||index>2)return {};
    return in[index==0?1:index==1?4:5];
}
bool Qwen35DeltaPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* d,int in,int out) noexcept {
    return in==11&&out==3&&pos>=0&&pos<14&&d[pos].type==DataType::kFLOAT&&d[pos].format==TensorFormat::kLINEAR;
}
size_t Qwen35DeltaPlugin::getWorkspaceSize(const PluginTensorDesc* in,int count,const PluginTensorDesc*,int) const noexcept {
    if(count!=11||in[0].dims.nbDims!=3)return 0;
    return size_t(in[0].dims.d[0])*in[0].dims.d[1]*in[0].dims.d[2]*sizeof(float);
}
int Qwen35DeltaPlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc*,const void* const* inputs,
    void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    if(!in||in[0].dims.nbDims!=3)return 1;
    return register128_?RunQwen35DeltaRegister128(c_,int(in[0].dims.d[0]),int(in[0].dims.d[1]),inputs,outputs,workspace,stream):
        RunQwen35Delta(c_,int(in[0].dims.d[0]),int(in[0].dims.d[1]),inputs,outputs,workspace,stream);
}
void Qwen35DeltaPlugin::serialize(void* data) const noexcept{std::memcpy(data,&c_,sizeof(c_));}
class Qwen35DeltaCreator final:public IPluginCreator {
    std::string ns_;PluginFieldCollection fields_{};bool register128_=false;
public:
    explicit Qwen35DeltaCreator(bool register128=false):register128_(register128){}
    const char* getPluginName() const noexcept override{return register128_?"GarnetQwen35DeltaRegister128":"GarnetQwen35Delta";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override {
        if(!data||size!=sizeof(Qwen35DeltaConfig))return nullptr;
        Qwen35DeltaConfig c;std::memcpy(&c,data,size);
        if(c.kh<=0||c.vh<=0||c.vh%c.kh||c.kd<=0||c.kd>128||c.vd<=0||c.vd>128||c.kernel<1||c.kernel>4||!(c.epsilon>0))return nullptr;
        if(register128_&&(c.kd!=128||c.vd!=128||c.kh>64||c.vh>64))return nullptr;
        try{auto* p=new Qwen35DeltaPlugin(c,register128_);p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35DeltaPluginRegistered(){
    static Qwen35DeltaCreator creator,registerCreator(true);static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"")&&registry->registerCreator(registerCreator,"");});return okay;
}
}
