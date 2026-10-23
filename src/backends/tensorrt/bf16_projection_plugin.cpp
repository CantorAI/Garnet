// SPDX-License-Identifier: Apache-2.0
#include "bf16_projection_plugin.h"
#include <NvInfer.h>
#include <climits>
#include <mutex>
namespace Garnet {
using namespace nvinfer1;
IPluginV2DynamicExt* BF16ProjectionPlugin::clone() const noexcept {
    try{auto* p=new BF16ProjectionPlugin(workspace_);p->namespace_=namespace_;p->context_=context_;return p;}catch(...){return nullptr;}
}
DimsExprs BF16ProjectionPlugin::getOutputDimensions(int index,const DimsExprs* inputs,int count,IExprBuilder&) noexcept {
    if(index||count!=2||inputs[0].nbDims!=2||inputs[1].nbDims!=2)return DimsExprs{};
    auto output=inputs[0];output.d[1]=inputs[1].d[0];return output;
}
bool BF16ProjectionPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* values,int inputs,int outputs) noexcept {
    return inputs==2&&outputs==1&&pos>=0&&pos<3&&values[pos].type==DataType::kBF16&&values[pos].format==TensorFormat::kLINEAR;
}
int BF16ProjectionPlugin::initialize() noexcept {
    try{if(!context_)context_=CreateBF16ProjectionContext();return context_?0:1;}catch(...){return 1;}
}
int BF16ProjectionPlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,
    const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    if(!in||!out||!context_||in[0].dims.nbDims!=2||in[1].dims.nbDims!=2||
        in[0].dims.d[1]!=in[1].dims.d[1])return 1;
    const auto m=in[0].dims.d[0],k=in[0].dims.d[1],n=in[1].dims.d[0];
    if(m<=0||k<=0||n<=0||m>INT_MAX||k>INT_MAX||n>INT_MAX)return 1;
    if(workspace_&&!workspace)return 1;
    return RunBF16Projection(context_,inputs[0],inputs[1],outputs[0],int(m),int(k),int(n),stream,workspace_?workspace:nullptr,workspace_?16384:0);
}
class BF16ProjectionCreator final:public IPluginCreator {
    std::string ns_;PluginFieldCollection fields_{};bool workspace_;
public:
    explicit BF16ProjectionCreator(bool workspace):workspace_(workspace){}
    const char* getPluginName() const noexcept override{return "GarnetBF16Projection";}
    const char* getPluginVersion() const noexcept override{return workspace_?"2":"1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override {
        try{auto* p=new BF16ProjectionPlugin(workspace_);p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}
    }
    IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override{
        if(!workspace_)return size?nullptr:createPlugin(nullptr,nullptr);
        if(!data||size!=sizeof(uint32_t))return nullptr;
        uint32_t bytes;std::memcpy(&bytes,data,sizeof(bytes));return bytes==16384?createPlugin(nullptr,nullptr):nullptr;
    }
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureBF16ProjectionPluginRegistered(){
    static BF16ProjectionCreator creator(false),workspaceCreator(true);static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"")&&registry->registerCreator(workspaceCreator,"");});return okay;
}
}
