#include "garnet_tensor.h"
#include "openvino_builder.h"
#include "gated_delta.h"
#include <cmath>
#include <filesystem>
#include <iostream>
class DeltaGraphProbe {
public:

 X::Value RunConv(const X::Value& graph,const X::Value& symbols,int batch) {
  if(batch!=1&&batch!=2)throw X::Error("unsupported test batch");
  const int channels=9,kernel=4;
  std::vector<std::vector<int64_t>> shapes={{batch,channels},{channels,kernel},{batch,channels,kernel},{batch}};
  std::vector<std::vector<float>> data;
  for(size_t i=0;i<shapes.size();++i){
   size_t count=1;for(auto d:shapes[i])count*=d;
   data.emplace_back(count);
   for(size_t n=0;n<count;++n)data.back()[n]=float(int((n*13+i*7)%23)-11)/13;
  }
  data[3].assign(batch,1);
  X::ARGS symbolic,arguments;
  for(uint64_t i=0;i<symbols.Size();++i)symbolic.push_back(symbols.Get(i));
  arguments=symbolic;
  const auto path=(std::filesystem::temp_directory_path()/("garnet_pi5b_conv_"+std::to_string(batch)+".blob")).string();
  Garnet::OpenVINOBuilder builder;std::string error;
  if(!builder.BuildCapturedGraph(graph,X::Value(),arguments,symbolic,nullptr,path,"bf16",error))throw X::Error(error);
  auto expectedState=data[2];
  for(int step=0;step<7;++step){
   for(size_t n=0;n<data[0].size();++n)data[0][n]=float(int((n*5+step*11)%19)-9)/7;
   // Pause and then resume row 1, without advancing its convolution history.
   if(batch==2)data[3][1]=(step>=2&&step<=4)?0:1;
   auto values=X::Value::List(Host());
   for(size_t i=0;i<data.size();++i)
    values.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shapes[i],data[i].data(),data[i].size()*sizeof(float)));
   std::vector<float> expected(batch*channels);
   for(int b=0;b<batch;++b)for(int c=0;c<channels;++c){
    const auto offset=(b*channels+c)*kernel;
    if(data[3][b]==0)continue;
    for(int j=0;j<kernel-1;++j)expectedState[offset+j]=expectedState[offset+j+1];
    expectedState[offset+kernel-1]=data[0][b*channels+c];
    double dot=0;for(int j=0;j<kernel;++j)dot+=double(expectedState[offset+j])*data[1][c*kernel+j];
    expected[b*channels+c]=float(dot/(1+std::exp(-dot)));
   }
   auto result=builder.RunCapturedEngine(path,values,nullptr,X::Value(),false,200+batch,error);
   if(!result.IsValid())throw X::Error(error);
   X::Tensor tensor(result);const auto info=tensor.Info();
   if(info.dtype!=X3_TENSOR_FLOAT32||info.byte_size!=size_t(batch*channels*(kernel+1))*sizeof(float))throw X::Error("invalid convolution output");
   auto actual=static_cast<const float*>(info.data);
   for(int bc=0;bc<batch*channels;++bc){
    const auto value=actual[bc*(kernel+1)];
    if(!std::isfinite(value)||std::abs(value-expected[bc])>3e-5)throw X::Error("convolution output mismatch");
    for(int j=0;j<kernel;++j){
     const auto stateValue=actual[bc*(kernel+1)+1+j];
     if(!std::isfinite(stateValue)||std::abs(stateValue-expectedState[bc*kernel+j])>3e-5)throw X::Error("convolution history mismatch");
     data[2][bc*kernel+j]=stateValue;
    }
   }
  }
  Garnet::OpenVINOBuilder::ReleaseCachedSession(path,200+batch);
  return X::Value(true);
 }
 X::Value Run(const X::Value& graph,const X::Value& symbols,int batch) {
  if(batch!=1&&batch!=2)throw X::Error("unsupported test batch");
  const int heads=2,kd=7,vd=5;
  std::vector<std::vector<int64_t>> shapes={{batch,heads,kd},{batch,heads,kd},{batch,heads,vd},{batch,heads},{batch,heads},{batch,heads,kd,vd},{batch}};
  std::vector<std::vector<float>> data;
  for(size_t i=0;i<shapes.size();++i){
   size_t size=1;for(auto d:shapes[i])size*=d;
   data.emplace_back(size);
   for(size_t n=0;n<size;++n)data.back()[n]=float(int((n*13+i*7)%23)-11)/13;
  }
  for(auto& v:data[3])v=-std::abs(v);
  for(auto& v:data[4])v=(v+1)/2;
  data[6].assign(batch,1);
  auto values=X::Value::List(Host());
  for(size_t i=0;i<data.size();++i)
   values.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shapes[i],data[i].data(),data[i].size()*sizeof(float)));
  X::ARGS symbolic,arguments;
  for(uint64_t i=0;i<symbols.Size();++i)symbolic.push_back(symbols.Get(i));
  arguments=symbolic;
  const auto path=(std::filesystem::temp_directory_path()/ ("garnet_pi5b_delta_"+std::to_string(batch)+".blob")).string();
  Garnet::OpenVINOBuilder builder;std::string error;
  if(!builder.BuildCapturedGraph(graph,X::Value(),arguments,symbolic,nullptr,path,"bf16",error))throw X::Error(error);
  auto expectedState=data[5];std::vector<float> expected(batch*heads*vd);
  std::vector<unsigned char> active(batch,1);
  for(int step=0;step<4;++step){
   if(batch==2&&step>=2){active[1]=0;data[6][1]=0;}
   values=X::Value::List(Host());
   for(size_t i=0;i<data.size();++i)
    values.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shapes[i],data[i].data(),data[i].size()*sizeof(float)));
   if(!Garnet::GatedDeltaStep(data[0].data(),data[1].data(),data[2].data(),data[3].data(),data[4].data(),active.data(),batch,heads,kd,vd,expectedState.data(),expected.data(),error))throw X::Error(error);
   auto result=builder.RunCapturedEngine(path,values,nullptr,X::Value(),false,100+batch,error);
   if(!result.IsValid())throw X::Error(error);
   X::Tensor tensor(result);auto info=tensor.Info();
   if(info.dtype!=X3_TENSOR_FLOAT32||info.byte_size!=size_t(batch*heads*(vd+kd*vd))*sizeof(float))throw X::Error("invalid graph output");
   const auto actual=static_cast<const float*>(info.data);
   for(int bh=0;bh<batch*heads;++bh){
    for(int v=0;v<vd;++v)if(!std::isfinite(actual[bh*(vd+kd*vd)+v])||std::abs(actual[bh*(vd+kd*vd)+v]-expected[bh*vd+v])>3e-5)throw X::Error("graph output mismatch");
    for(int n=0;n<kd*vd;++n){
     float s=actual[bh*(vd+kd*vd)+vd+n];
     if(!std::isfinite(s)||std::abs(s-expectedState[bh*kd*vd+n])>3e-5)throw X::Error("graph state mismatch");
     data[5][bh*kd*vd+n]=s;
    }
   }
  }
  Garnet::OpenVINOBuilder::ReleaseCachedSession(path,100+batch);
  return X::Value(true);
 }
 BEGIN_PACKAGE(DeltaGraphProbe)
  APISET().AddClass<0,Garnet::GarnetTensor>("tensor");
  APISET().AddFunc<3>("run_graph",&DeltaGraphProbe::Run);
  APISET().AddFunc<3>("run_conv",&DeltaGraphProbe::RunConv);
 END_PACKAGE
};
int main(int argc,char**argv){
 try {
  if(argc!=3)throw std::runtime_error("expected source root and script");
  DeltaGraphProbe probe;X::Runtime runtime;auto package=runtime.RegisterPackage("garnet",probe);
  runtime.AddImportRoot(argv[1]);runtime.EvalFile(argv[2]);
  std::cout<<"Garnet captured graph/OpenVINO recurrence PASS\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
