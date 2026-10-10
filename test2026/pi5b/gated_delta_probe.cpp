#include "gated_delta.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <vector>
int main(){
 using json=nlohmann::json; json j;std::cin>>j;
 auto q=j["q"].get<std::vector<float>>(),k=j["k"].get<std::vector<float>>(),v=j["v"].get<std::vector<float>>();
 auto g=j["g"].get<std::vector<float>>(),beta=j["beta"].get<std::vector<float>>(),state=j["state"].get<std::vector<float>>();
 auto active=j["active"].get<std::vector<unsigned char>>();
 size_t b=j["batch"],h=j["heads"],kd=j["key_dim"],vd=j["value_dim"];
 if(q.size()!=b*h*kd||k.size()!=q.size()||v.size()!=b*h*vd||g.size()!=b*h||beta.size()!=b*h||state.size()!=b*h*kd*vd||active.size()!=b)return 2;
 std::vector<float> out(b*h*vd);std::string error;
 if(!Garnet::GatedDeltaStep(q.data(),k.data(),v.data(),g.data(),beta.data(),active.data(),b,h,kd,vd,state.data(),out.data(),error)){std::cerr<<error;return 3;}
 std::cout<<json{{"state",state},{"out",out}};
}
