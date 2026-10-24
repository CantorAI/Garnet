// SPDX-License-Identifier: Apache-2.0
// Actual checkpoint parser + shared admission helper, entirely CPU-only.
#include "modelopt_joined_mlp.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>
using Json=nlohmann::json;
using Bytes=std::vector<unsigned char>;
const std::string base="model.language_model.layers.0.mlp.";
const std::string joined=base+"gate_up_proj.weight";
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void FloatAt(Json& header,Bytes& data,const std::string& name,float value){
    const size_t offset=header.at(name).at("data_offsets").at(0).get<size_t>();
    std::memcpy(data.data()+offset,&value,4);
}
void Fixture(Json& header,Bytes& data){
    auto add=[&](std::string name,std::string dtype,Json shape,size_t bytes,unsigned char value){
        size_t start=data.size();data.resize(start+bytes,value);
        header[name]={{"dtype",dtype},{"shape",shape},{"data_offsets",{start,data.size()}}};
    };
    for(const auto& projection:{"gate_proj","up_proj"}){
        const auto prefix=base+projection;
        add(prefix+".weight","U8",{128,32},128*32,0x12);
        add(prefix+".weight_scale","F8_E4M3",{128,4},128*4,56);
        add(prefix+".weight_scale_2","F32",Json::array(),4,0);
        add(prefix+".input_scale","F32",Json::array(),4,0);
        FloatAt(header,data,prefix+".weight_scale_2",.125f);
        FloatAt(header,data,prefix+".input_scale",.25f);
    }
}
bool Admit(const std::filesystem::path& folder,int number,const std::function<void(Json&,Bytes&)>& mutate={}){
    Json header;Bytes data;Fixture(header,data);if(mutate)mutate(header,data);
    const auto file=folder/(std::to_string(number)+".safetensors");
    auto encoded=header.dump();while(encoded.size()%8)encoded.push_back(' ');
    const uint64_t size=encoded.size();std::ofstream output(file,std::ios::binary);
    output.write(reinterpret_cast<const char*>(&size),8);output.write(encoded.data(),encoded.size());
    output.write(reinterpret_cast<const char*>(data.data()),data.size());output.close();
    Garnet::SafeTensorsIndex index;std::string error;
    if(!index.Open(file,error))return false;
    try{
        const auto p=Garnet::ValidateJoinedNVFP4MLP(index,joined);
        Require(p.halfRows==128&&p.columns==64&&p.weightScale==.125f&&p.inputScale==.25f,"Valid geometry/calibration differs");
        Require(p.gateName==base+"gate_proj.weight"&&p.upName==base+"up_proj.weight","Gate/up order changed");
        return true;
    }catch(const std::invalid_argument&){return false;}
}
int main(int argc,char** argv){try{
    Require(argc==2,"Pass a fixed test fixture directory");std::filesystem::path folder(argv[1]);std::filesystem::create_directories(folder);
    std::string gate,up;Require(Garnet::JoinedNVFP4MLPNames(joined,gate,up),"Valid name rejected");
    for(const auto& name:{"model.language_model.layers.00.mlp.gate_up_proj.weight","model.language_model.layers.-1.mlp.gate_up_proj.weight","model.language_model.layers.1024.mlp.gate_up_proj.weight","mtp.0.mlp.gate_up_proj.weight","model.language_model.layers.0.mlp.gate_proj.weight"})
        Require(!Garnet::JoinedNVFP4MLPNames(name,gate,up),"Invalid joined name admitted");
    Require(Admit(folder,0),"Valid fixture rejected");int number=0;
    auto reject=[&](const std::function<void(Json&,Bytes&)>& mutate){Require(!Admit(folder,++number,mutate),"Invalid calibration/geometry admitted");};
    reject([](Json& h,Bytes& d){FloatAt(h,d,base+"up_proj.input_scale",.5f);});
    reject([](Json& h,Bytes& d){FloatAt(h,d,base+"up_proj.weight_scale_2",.5f);});
    reject([](Json& h,Bytes& d){FloatAt(h,d,base+"gate_proj.input_scale",0.f);});
    reject([](Json& h,Bytes& d){FloatAt(h,d,base+"gate_proj.weight_scale_2",-1.f);});
    reject([](Json& h,Bytes& d){FloatAt(h,d,base+"gate_proj.input_scale",std::numeric_limits<float>::quiet_NaN());});
    reject([](Json& h,Bytes& d){FloatAt(h,d,base+"gate_proj.weight_scale_2",std::numeric_limits<float>::infinity());});
    reject([](Json& h,Bytes&){h[base+"gate_proj.input_scale"]["shape"]={1};});
    reject([](Json& h,Bytes&){h.erase(base+"up_proj.input_scale");});
    reject([](Json& h,Bytes&){h[base+"up_proj.weight"]["dtype"]="F8_E4M3";});
    reject([](Json& h,Bytes&){h[base+"up_proj.weight"]["shape"]={256,32};});
    reject([](Json& h,Bytes&){h[base+"gate_proj.weight"]["shape"]={64,64};h[base+"up_proj.weight"]["shape"]={64,64};});
    reject([](Json& h,Bytes&){h[base+"gate_proj.weight_scale"]["dtype"]="U8";});
    reject([](Json& h,Bytes&){h[base+"up_proj.weight_scale"]["shape"]={64,8};});
    reject([](Json& h,Bytes&){h.erase(base+"up_proj.weight_scale");});
    std::cout<<"JOINED_MLP_METADATA_CPU_PASS: valid fixture, "<<number<<" rejection fixtures, five invalid names; no GPU or performance claim\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
