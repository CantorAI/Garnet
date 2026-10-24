// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_plugin.h"
#include "modelopt_cublaslt.h"
#include "../../cuda/modelopt_fp8.cuh"
#include <NvInfer.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {
constexpr size_t BlasWorkspace=4ULL<<20;
size_t Aligned(size_t bytes){return (bytes+255)&~size_t(255);}
}
ModelOptFP8Plugin::ModelOptFP8Plugin(int rows,int columns,float weightScale,float inputScale)
    :state_{rows,columns,weightScale,inputScale}{
    static_assert(sizeof(State)==16,"Stable FP8 plugin wire format");
    if(rows<=0||rows>1048576||rows%16||columns<=0||columns>65536||columns%16||
        !std::isfinite(weightScale)||weightScale<=0||
        !std::isfinite(inputScale)||inputScale<=0||!std::isfinite(weightScale*inputScale)||weightScale*inputScale<=0)
        throw std::invalid_argument("Invalid ModelOpt FP8 plugin geometry/scales");
}
IPluginV2DynamicExt* ModelOptFP8Plugin::clone() const noexcept{
    try{auto* p=new ModelOptFP8Plugin(state_.rows,state_.columns,state_.weightScale,state_.inputScale);
        p->namespace_=namespace_;p->context_=context_;return p;}catch(...){return nullptr;}
}
ModelOptFP8Plugin* ModelOptFP8Plugin::Deserialize(const void* data,size_t length) noexcept{
    if(!data||length!=sizeof(State))return nullptr;
    State s;std::memcpy(&s,data,sizeof(s));
    try{return new ModelOptFP8Plugin(s.rows,s.columns,s.weightScale,s.inputScale);}catch(...){return nullptr;}
}
DimsExprs ModelOptFP8Plugin::getOutputDimensions(int index,const DimsExprs* input,int count,IExprBuilder& builder) noexcept{
    if(index||count!=2||input[0].nbDims!=2||input[1].nbDims!=2)return DimsExprs{};
    auto shape=input[0];shape.d[1]=builder.constant(state_.rows);return shape;
}
bool ModelOptFP8Plugin::supportsFormatCombination(int pos,const PluginTensorDesc* values,int inputs,int outputs) noexcept{
    if(!values||inputs!=2||outputs!=1||pos<0||pos>=3||values[pos].format!=TensorFormat::kLINEAR)return false;
    if(pos==0||pos==2)return values[pos].type==DataType::kBF16;
    // INT32 is opaque four-byte storage for caller-owned compressed buffers.
    // The kernel reads the same packed bytes; no numerical conversion occurs.
    return values[pos].type==DataType::kINT8||values[pos].type==DataType::kINT32;
}
size_t ModelOptFP8Plugin::getWorkspaceSize(const PluginTensorDesc* in,int count,const PluginTensorDesc*,int) const noexcept{
    if(!in||count!=2||in[0].dims.nbDims!=2||in[0].dims.d[0]<1||in[0].dims.d[0]>4096||in[0].dims.d[1]!=state_.columns)return 0;
    return Aligned(size_t(in[0].dims.d[0])*state_.columns)+BlasWorkspace;
}
int ModelOptFP8Plugin::initialize() noexcept{
    try{if(!context_)context_=CreateModelOptNVFP4Context();return context_?0:1;}catch(...){return 1;}
}
int ModelOptFP8Plugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* input,
    void* const* output,void* workspace,cudaStream_t stream) noexcept{
    if(!in||!out||!input||!output||!workspace||!context_||!getWorkspaceSize(in,2,out,1)||
        in[1].dims.nbDims!=2||in[1].dims.d[0]!=state_.rows||in[1].dims.d[1]!=state_.columns/(in[1].type==DataType::kINT32?4:1)||
        out[0].dims.nbDims!=2||out[0].dims.d[0]!=in[0].dims.d[0]||out[0].dims.d[1]!=state_.rows)return 1;
    PluginTensorDesc descriptors[3];
    for(int i=0;i<2;++i){if(!input[i])return 1;descriptors[i]=in[i];}
    if(!output[0])return 1;descriptors[2]=out[0];
    for(int i=0;i<3;++i)if(!supportsFormatCombination(i,descriptors,2,1))return 1;
    const int batch=int(in[0].dims.d[0]);
    auto* packed=static_cast<unsigned char*>(workspace);
    auto* blasWorkspace=packed+Aligned(size_t(batch)*state_.columns);
    if(QuantizeModelOptFP8Activation(input[0],packed,state_.inputScale,batch,state_.columns,stream)!=cudaSuccess)return 1;
    return RunModelOptFP8MatmulContext(context_,input[1],packed,output[0],batch,state_.rows,state_.columns,
        state_.weightScale*state_.inputScale,blasWorkspace,BlasWorkspace,stream)==CUBLAS_STATUS_SUCCESS?0:1;
}
class ModelOptFP8Creator final:public IPluginCreator {
    std::string namespace_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetModelOptFP8";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    // Native lowering supplies validated metadata directly. Wire deserialization
    // is the supported creator path; no ambiguous default projection is admitted.
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t length) noexcept override{
        auto* p=ModelOptFP8Plugin::Deserialize(data,length);if(p)p->setPluginNamespace(namespace_.c_str());return p;
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureModelOptFP8PluginRegistered(){
    static ModelOptFP8Creator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
