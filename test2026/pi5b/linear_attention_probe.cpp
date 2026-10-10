#include "qwen35_linear_lowering.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <iostream>
int main(){
 try {
  using json=nlohmann::json;json j;std::cin>>j;
  if(j.value("kind",std::string())=="softplus") {
   auto data=j["x"].get<std::vector<float>>();
   auto p=std::make_shared<ov::opset13::Parameter>(ov::element::f32,ov::Shape{data.size()});
   auto model=std::make_shared<ov::Model>(ov::OutputVector{Garnet::Qwen35StableSoftplus(p)},ov::ParameterVector{p});
   ov::Core core;auto compiled=core.compile_model(model,"CPU",{{"INFERENCE_PRECISION_HINT",ov::element::f32},{"INFERENCE_NUM_THREADS",2}});
   auto request=compiled.create_infer_request();ov::Tensor input(ov::element::f32,{data.size()});
   std::memcpy(input.data(),data.data(),data.size()*sizeof(float));request.set_input_tensor(input);request.infer();
   auto output=request.get_output_tensor();auto ptr=output.data<const float>();
   std::cout<<json{{"out",std::vector<float>(ptr,ptr+output.get_size())}};return 0;
  }
  size_t b=j["batch"],kh=j["key_heads"],vh=j["value_heads"],kd=j["key_dim"],vd=j["value_dim"],d=j["hidden_dim"],ks=j["kernel_size"];
  size_t c=2*kh*kd+vh*vd;
  const char* names[]={"x","conv_state","state","active","qkv","z","b_proj","a_proj","conv","a_log","dt_bias","norm","out_proj"};
  std::vector<ov::Shape> shapes={{b,d},{b,c,ks},{b,vh,kd,vd},{b},{c,d},{vh*vd,d},{vh,d},{vh,d},{c,1,ks},{vh},{vh},{vd},{d,vh*vd}};
  ov::ParameterVector p;
  for(auto& s:shapes)p.push_back(std::make_shared<ov::opset13::Parameter>(ov::element::f32,s));
  Garnet::Qwen35LinearWeights weights{p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12]};
  auto result=Garnet::LowerQwen35LinearStep(p[0],p[1],p[2],p[3],weights,kh,vh,kd,vd,1e-6f);
  auto model=std::make_shared<ov::Model>(ov::OutputVector{result.output,result.convState,result.recurrentState},p);
  ov::Core core;auto compiled=core.compile_model(model,"CPU",{{"INFERENCE_PRECISION_HINT",ov::element::f32},{"INFERENCE_NUM_THREADS",2}});
  auto request=compiled.create_infer_request();
  for(size_t i=0;i<p.size();++i){
   auto data=j[names[i]].get<std::vector<float>>();ov::Tensor tensor(ov::element::f32,shapes[i]);
   if(data.size()!=tensor.get_size())return 2;
   std::memcpy(tensor.data(),data.data(),data.size()*sizeof(float));request.set_input_tensor(i,tensor);
  }
  request.infer();json output;const char* outputs[]={"out","conv_state","state"};
  for(size_t i=0;i<3;++i){auto t=request.get_output_tensor(i);auto ptr=t.data<const float>();output[outputs[i]]=std::vector<float>(ptr,ptr+t.get_size());}
  std::cout<<output;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
