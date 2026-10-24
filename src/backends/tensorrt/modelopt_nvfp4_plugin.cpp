// SPDX-License-Identifier: Apache-2.0
#include "modelopt_nvfp4_plugin.h"
#include "modelopt_cublaslt.h"
#include "../../cuda/modelopt_nvfp4.cuh"
#include <NvInfer.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {
size_t Aligned(size_t bytes){return (bytes+255)&~size_t(255);}
}
ModelOptNVFP4Plugin::ModelOptNVFP4Plugin(int rows,int columns,float weightScale,float inputScale,bool downCandidate,bool decodeOnly)
    :state_{rows,columns,weightScale,inputScale},candidatePolicy_(downCandidate?(decodeOnly?2:1):0){
    static_assert(sizeof(State)==16,"Stable NVFP4 plugin wire format");
    if(rows<=0||rows>1048576||rows%16||columns<=0||columns>65536||columns%64||
        !ModelOptNVFP4ScaleBytes(rows,columns)||ModelOptNVFP4ScaleBytes(rows,columns)>(256ULL<<20)||
        !std::isfinite(weightScale)||weightScale<=0||
        !std::isfinite(inputScale)||inputScale<=0||!std::isfinite(weightScale*inputScale)||weightScale*inputScale<=0)
        throw std::invalid_argument("Invalid ModelOpt NVFP4 plugin geometry/scales");
    if(candidatePolicy_&&(rows!=5120||columns!=17408))throw std::invalid_argument("Unqualified NVFP4 down-candidate geometry");
}
IPluginV2DynamicExt* ModelOptNVFP4Plugin::clone() const noexcept{
    try{auto* p=new ModelOptNVFP4Plugin(state_.rows,state_.columns,state_.weightScale,state_.inputScale,candidatePolicy_!=0,candidatePolicy_==2);
        p->namespace_=namespace_;p->context_=context_;return p;}catch(...){return nullptr;}
}
ModelOptNVFP4Plugin* ModelOptNVFP4Plugin::Deserialize(const void* data,size_t length) noexcept{
    if(!data||(length!=sizeof(State)&&length!=sizeof(State)+sizeof(uint32_t)))return nullptr;
    State s;std::memcpy(&s,data,sizeof(s));
    uint32_t policy=0;
    if(length>sizeof(State)){std::memcpy(&policy,static_cast<const unsigned char*>(data)+sizeof(State),sizeof(policy));if(policy!=1&&policy!=2)return nullptr;}
    try{return new ModelOptNVFP4Plugin(s.rows,s.columns,s.weightScale,s.inputScale,policy!=0,policy==2);}catch(...){return nullptr;}
}
DimsExprs ModelOptNVFP4Plugin::getOutputDimensions(int index,const DimsExprs* input,int count,IExprBuilder& builder) noexcept{
    if(index||count!=3||input[0].nbDims!=2||input[1].nbDims!=2||input[2].nbDims!=1)return DimsExprs{};
    auto shape=input[0];shape.d[1]=builder.constant(state_.rows);return shape;
}
bool ModelOptNVFP4Plugin::supportsFormatCombination(int pos,const PluginTensorDesc* values,int inputs,int outputs) noexcept{
    if(!values||inputs!=3||outputs!=1||pos<0||pos>=4||values[pos].format!=TensorFormat::kLINEAR)return false;
    if(pos==0||pos==3)return values[pos].type==DataType::kBF16;
    // INT32 is opaque four-byte storage for caller-owned compressed buffers.
    // The kernel reads the same packed bytes; no numerical conversion occurs.
    return values[pos].type==DataType::kINT8||values[pos].type==DataType::kINT32;
}
size_t ModelOptNVFP4Plugin::getWorkspaceSize(const PluginTensorDesc* in,int count,const PluginTensorDesc*,int) const noexcept{
    if(!in||count!=3||in[0].dims.nbDims!=2||in[0].dims.d[0]<1||in[0].dims.d[0]>1024||in[0].dims.d[1]!=state_.columns)return 0;
    return Aligned(size_t(in[0].dims.d[0])*state_.columns/2)+
        Aligned(ModelOptNVFP4ScaleBytes(int(in[0].dims.d[0]),state_.columns))+blasWorkspace(int(in[0].dims.d[0]));
}
int ModelOptNVFP4Plugin::initialize() noexcept{
    try{if(!context_)context_=CreateModelOptNVFP4Context();return context_?0:1;}catch(...){return 1;}
}
int ModelOptNVFP4Plugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* input,
    void* const* output,void* workspace,cudaStream_t stream) noexcept{
    if(!in||!out||!input||!output||!workspace||!context_||!getWorkspaceSize(in,3,out,1)||
        in[1].dims.nbDims!=2||in[1].dims.d[0]!=state_.rows||in[1].dims.d[1]!=state_.columns/(in[1].type==DataType::kINT32?8:2)||
        in[2].dims.nbDims!=1||in[2].dims.d[0]!=int64_t(ModelOptNVFP4ScaleBytes(state_.rows,state_.columns)/(in[2].type==DataType::kINT32?4:1))||
        out[0].dims.nbDims!=2||out[0].dims.d[0]!=in[0].dims.d[0]||out[0].dims.d[1]!=state_.rows)return 1;
    PluginTensorDesc descriptors[4];
    for(int i=0;i<3;++i){if(!input[i])return 1;descriptors[i]=in[i];}
    if(!output[0])return 1;descriptors[3]=out[0];
    for(int i=0;i<4;++i)if(!supportsFormatCombination(i,descriptors,3,1))return 1;
    const int batch=int(in[0].dims.d[0]);
    auto* packed=static_cast<unsigned char*>(workspace);
    auto* scales=packed+Aligned(size_t(batch)*state_.columns/2);
    auto* blasWorkspace=scales+Aligned(ModelOptNVFP4ScaleBytes(batch,state_.columns));
    if(QuantizeModelOptNVFP4Activation(input[0],packed,scales,state_.inputScale,batch,state_.columns,stream)!=cudaSuccess)return 1;
    return RunModelOptNVFP4MatmulContext(context_,input[1],input[2],packed,scales,output[0],batch,state_.rows,state_.columns,
        state_.weightScale*state_.inputScale,blasWorkspace,this->blasWorkspace(batch),stream,useDownCandidate(batch))==CUBLAS_STATUS_SUCCESS?0:1;
}
class ModelOptNVFP4Creator final:public IPluginCreator {
    std::string namespace_;PluginFieldCollection fields_{};
    uint32_t policy_;
public:
    explicit ModelOptNVFP4Creator(uint32_t policy):policy_(policy){}
    const char* getPluginName() const noexcept override{return "GarnetModelOptNVFP4";}
    const char* getPluginVersion() const noexcept override{return policy_==2?"3":policy_==1?"2":"1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    // Native lowering supplies validated metadata directly. Wire deserialization
    // is the supported creator path; no ambiguous default projection is admitted.
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t length) noexcept override{
        if(!data||length!=(policy_?20:16))return nullptr;
        if(policy_){uint32_t tag=0;std::memcpy(&tag,static_cast<const unsigned char*>(data)+16,sizeof(tag));if(tag!=policy_)return nullptr;}
        auto* p=ModelOptNVFP4Plugin::Deserialize(data,length);if(p)p->setPluginNamespace(namespace_.c_str());return p;
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureModelOptNVFP4PluginRegistered(){
    static ModelOptNVFP4Creator legacy(0),historical(1),decode(2);static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(legacy,"")&&registry->registerCreator(historical,"")&&registry->registerCreator(decode,"");});return okay;
}
}
