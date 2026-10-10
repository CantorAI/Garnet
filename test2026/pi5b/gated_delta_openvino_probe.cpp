#include "gated_delta_lowering.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <iostream>
int main(){
 using json=nlohmann::json;json j;std::cin>>j;
 size_t b=j["batch"],h=j["heads"],kd=j["key_dim"],vd=j["value_dim"];
 ov::ParameterVector p;
 for(auto shape : std::vector<ov::Shape>{{b,h,kd},{b,h,kd},{b,h,vd},{b,h},{b,h},{b,h,kd,vd},{b}})
  p.push_back(std::make_shared<ov::opset13::Parameter>(ov::element::f32,shape));
 auto result=Garnet::LowerGatedDelta(p[0],p[1],p[2],p[3],p[4],p[5],p[6]);
 auto model=std::make_shared<ov::Model>(ov::OutputVector{result.state,result.output},p);
 ov::Core core;
 auto compiled=core.compile_model(model,"CPU",{{"INFERENCE_PRECISION_HINT",ov::element::f32},{"INFERENCE_NUM_THREADS",2}});
 auto request=compiled.create_infer_request();
 const char* names[]={"q","k","v","g","beta","state","active"};
 for(size_t i=0;i<7;++i){
  auto data=j[names[i]].get<std::vector<float>>();
  ov::Tensor tensor(ov::element::f32,p[i]->get_shape());
  if(data.size()!=tensor.get_size())return 2;
  std::memcpy(tensor.data(),data.data(),data.size()*sizeof(float));
  request.set_input_tensor(i,tensor);
 }
 request.infer();
 json output;
 for(size_t i=0;i<2;++i){
  auto tensor=request.get_output_tensor(i);const auto data=tensor.data<const float>();
  output[i==0?"state":"out"]=std::vector<float>(data,data+tensor.get_size());
 }
 std::cout<<output;
}
