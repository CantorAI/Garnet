// SPDX-License-Identifier: Apache-2.0
#include "ct_int4_plugin.h"
#include "../../cuda/ct_int4.cuh"
#include <NvInfer.h>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
CTInt4G128Plugin::CTInt4G128Plugin(int rows,int columns):state_{rows,columns}{
 static_assert(sizeof(State)==8,"Stable INT4 plugin format");
 if(rows<1||rows>1048576||columns<128||columns>65536||columns%128)throw std::invalid_argument("Invalid INT4 G128 geometry");
}
IPluginV2DynamicExt* CTInt4G128Plugin::clone() const noexcept{
 try{auto* p=new CTInt4G128Plugin(state_.rows,state_.columns);p->namespace_=namespace_;return p;}catch(...){return nullptr;}
}
CTInt4G128Plugin* CTInt4G128Plugin::Deserialize(const void* data,size_t length) noexcept{
 if(!data||length!=sizeof(State))return nullptr;State s;std::memcpy(&s,data,sizeof(s));
 try{return new CTInt4G128Plugin(s.rows,s.columns);}catch(...){return nullptr;}
}
DimsExprs CTInt4G128Plugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder& b) noexcept{
 if(index||count!=3||in[0].nbDims!=2||in[1].nbDims!=2||in[2].nbDims!=2)return DimsExprs{};
 auto shape=in[0];shape.d[1]=b.constant(state_.rows);return shape;
}
bool CTInt4G128Plugin::supportsFormatCombination(int pos,const PluginTensorDesc* v,int inputs,int outputs) noexcept{
 if(inputs!=3||outputs!=1||pos<0||pos>=4||v[pos].format!=TensorFormat::kLINEAR)return false;
 return v[pos].type==(pos==1?DataType::kINT8:DataType::kBF16);
}
size_t CTInt4G128Plugin::getWorkspaceSize(const PluginTensorDesc*,int,const PluginTensorDesc*,int) const noexcept{return 0;}
int CTInt4G128Plugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* input,
 void* const* output,void*,cudaStream_t stream) noexcept{
 if(!in||!out||!input||!output||in[0].dims.nbDims!=2||in[0].dims.d[0]<1||in[0].dims.d[0]>128||in[0].dims.d[1]!=state_.columns||
  in[1].dims.nbDims!=2||in[1].dims.d[0]!=state_.rows||in[1].dims.d[1]!=state_.columns/2||
  in[2].dims.nbDims!=2||in[2].dims.d[0]!=state_.rows||in[2].dims.d[1]!=state_.columns/128||
  out[0].dims.nbDims!=2||out[0].dims.d[0]!=in[0].dims.d[0]||out[0].dims.d[1]!=state_.rows)return 1;
 return ProjectCTInt4G128(input[0],static_cast<const unsigned char*>(input[1]),input[2],output[0],int(in[0].dims.d[0]),state_.rows,state_.columns,stream)==cudaSuccess?0:1;
}
class CTInt4G128Creator final:public IPluginCreator {
 std::string namespace_;PluginFieldCollection fields_{};
public:
 const char* getPluginName() const noexcept override{return "GarnetCTInt4G128";}
 const char* getPluginVersion() const noexcept override{return "1";}
 const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
 IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
 IPluginV2* deserializePlugin(const char*,const void* data,size_t length) noexcept override{
  auto* p=CTInt4G128Plugin::Deserialize(data,length);if(p)p->setPluginNamespace(namespace_.c_str());return p;
 }
 void setPluginNamespace(const char* v) noexcept override{namespace_=v?v:"";}
 const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureCTInt4G128PluginRegistered(){
 static CTInt4G128Creator creator;static std::once_flag once;static bool okay=false;
 std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
