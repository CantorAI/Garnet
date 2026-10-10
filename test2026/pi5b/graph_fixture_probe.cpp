#include "garnet_tensor.h"
#include "openvino_builder.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
class GraphFixtureProbe {
public:
 nlohmann::json fixture;
 std::filesystem::path fixturePath;
 X::Value ValueFromJSON(const nlohmann::json& item) {
  if(item.is_object()) {
   auto value=X::Value::Dict(Host());
   for(auto it=item.begin();it!=item.end();++it)value.Set(it.key().c_str(),ValueFromJSON(*it));
   return value;
  }
  if(item.is_array()) {
   auto value=X::Value::List(Host());for(const auto& child:item)value.Append(ValueFromJSON(child));return value;
  }
  if(item.is_string())return X::Value::String(Host(),item.get<std::string>());
  if(item.is_boolean())return X::Value(item.get<bool>());
  if(item.is_number_float())return X::Value(item.get<double>());
  if(item.is_number_integer())return X::Value(item.get<int64_t>());
  throw X::Error("unsupported fixture configuration value");
 }
 X::Value Config() {
  return ValueFromJSON(fixture["config"]);
 }
 X::Value Run(const X::Value& graph,const X::Value& symbols) {
  Garnet::SafeTensorsIndex weights;std::string error;
  if(!weights.Open(fixture["checkpoint"].get<std::string>(),error))throw X::Error(error);
  X::ARGS symbolic,arguments;
  std::vector<std::vector<int64_t>> shapes;
  for(uint64_t i=0;i<symbols.Size();++i){
   auto symbol=symbols.Get(i);symbolic.push_back(symbol);
   const auto info=X::Tensor(symbol).Info();shapes.emplace_back(info.shape,info.shape+info.rank);
  }
  arguments=symbolic;
  auto path=(fixturePath.parent_path()/"fixture.blob").string();
  Garnet::OpenVINOBuilder builder;
  const auto start=std::chrono::steady_clock::now();
  const bool memoryOnly=fixture.value("memory_only",false);
  if(!builder.BuildCapturedGraph(graph,X::Value(),arguments,symbolic,&weights,path,"bf16",error,memoryOnly))throw X::Error(error);
  if(memoryOnly&&std::filesystem::exists(path))throw X::Error("memory-only graph unexpectedly wrote an engine file");
  const auto compileEnd=std::chrono::steady_clock::now();
  if(!builder.PrepareCapturedEngine(path,&weights,error))throw X::Error(error);
  int steps=0;
  std::vector<double> inferenceTimes;
  for(const auto& step:fixture["steps"]){
   if(step["inputs"].size()!=shapes.size())throw X::Error("fixture input count mismatch");
   auto inputs=X::Value::List(Host());
   for(size_t i=0;i<shapes.size();++i){
    auto data=step["inputs"][i].get<std::vector<float>>();
    size_t count=1;for(auto d:shapes[i])count*=size_t(d);
    if(data.size()!=count)throw X::Error("fixture input dimensions mismatch");
    inputs.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shapes[i],data.data(),data.size()*sizeof(float)));
   }
   const auto inferStart=std::chrono::steady_clock::now();
   auto actual=builder.RunCapturedEngine(path,inputs,&weights,X::Value(),false,501,error);
   inferenceTimes.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-inferStart).count());
   if(!actual.IsValid())throw X::Error(error);
   X::Tensor output(actual);const auto info=output.Info();
   auto expected=step["expected"].get<std::vector<float>>();
   if(info.dtype!=X3_TENSOR_FLOAT32||info.byte_size!=expected.size()*sizeof(float))throw X::Error("fixture output dimensions mismatch");
   const float* data=static_cast<const float*>(info.data);
   const double atol=fixture.value("atol",5e-6),rtol=fixture.value("rtol",5e-5);
   for(size_t i=0;i<expected.size();++i) {
    if(!std::isfinite(data[i])||std::abs(double(data[i])-expected[i])>atol+rtol*std::abs(expected[i]))
     throw X::Error("captured fixture numerical mismatch at step "+std::to_string(steps)+" element "+std::to_string(i));
   }
   ++steps;
  }
  Garnet::OpenVINOBuilder::ReleaseCachedSession(path,501);
  Garnet::OpenVINOBuilder::ReleaseCachedExecutions(fixturePath.parent_path().string());
  const auto end=std::chrono::steady_clock::now();
  std::cout<<"captured checkpoint graph PASS steps="<<steps
   <<" compile_ms="<<std::chrono::duration<double,std::milli>(compileEnd-start).count()
   <<" validation_ms="<<std::chrono::duration<double,std::milli>(end-compileEnd).count()
   <<" inference_ms="<<nlohmann::json(inferenceTimes).dump()<<'\n';
  return X::Value(true);
 }
 BEGIN_PACKAGE(GraphFixtureProbe)
  APISET().AddClass<0,Garnet::GarnetTensor>("tensor");
  APISET().AddFunc<0>("config",&GraphFixtureProbe::Config);
  APISET().AddFunc<2>("run_fixture",&GraphFixtureProbe::Run);
 END_PACKAGE
};
int main(int argc,char**argv){
 try {
  if(argc!=4)throw std::runtime_error("expected source root, script, and fixture");
  GraphFixtureProbe probe;probe.fixturePath=argv[3];std::ifstream stream(probe.fixturePath);stream>>probe.fixture;
  X::Runtime runtime;auto package=runtime.RegisterPackage("garnet",probe);
  runtime.AddImportRoot(argv[1]);runtime.EvalFile(argv[2]);return 0;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
