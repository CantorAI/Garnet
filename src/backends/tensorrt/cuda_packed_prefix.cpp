// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_packed_prefix.h"
#include "safetensors_index.h"
#include "tensor_helper.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
namespace Garnet { namespace {
void require(bool okay,const char* message){if(!okay)throw X::Error(message);}
std::vector<float> weight(const SafeTensorsIndex& index,const std::string& name,const std::vector<long long>& shape){
    const auto* e=index.Find(name);require(e&&e->shape==shape,"Packed prefix weight shape mismatch");
    size_t n=1;for(auto d:shape)n*=size_t(d);
    require(n<=65536&&(e->dataType=="F32"||e->dataType=="BF16"),"Packed prefix weight dtype/size unsupported");
    size_t item=e->dataType=="F32"?4:2;require(e->dataSize==n*item,"Packed prefix weight byte size mismatch");
    std::vector<unsigned char> bytes(n*item);std::ifstream file(e->filePath,std::ios::binary);
    require(bool(file),"Packed prefix weight file open failed");file.seekg(e->dataOffset);file.read((char*)bytes.data(),bytes.size());
    require(bool(file),"Packed prefix weight read failed");std::vector<float> result(n);
    for(size_t i=0;i<n;++i){
        if(item==4)std::memcpy(&result[i],bytes.data()+4*i,4);
        else{uint16_t small;std::memcpy(&small,bytes.data()+2*i,2);uint32_t bits=uint32_t(small)<<16;std::memcpy(&result[i],&bits,4);}
        require(std::isfinite(result[i]),"Packed prefix weights must be finite");
    }return result;
}
void tensor(const X3TensorInfo& v,int b,int64_t width){
    require(!v.symbolic&&!v.readonly&&v.device_type==TensorHelper::CudaDevice&&v.dtype==X3_TENSOR_FLOAT32&&
        v.data&&v.rank==2&&v.shape[0]==b&&v.shape[1]==width&&v.byte_size>=uint64_t(b)*width*4,
        "Packed prefix requires matching writable FP32 CUDA matrices");
    require(!v.strides||(v.strides[0]==width*4&&v.strides[1]==4),"Packed prefix matrices must be contiguous");
}
}
void CUDAPackedPrefix::CloseUnlocked(){ready_=false;linear_=full_=parameters_=workspace_=X::Tensor();config_={};}
CUDAPackedPrefix::~CUDAPackedPrefix(){Close();}
void CUDAPackedPrefix::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDAPackedPrefix::Build(const std::string& checkpoint,const std::string& specification){
    std::lock_guard<std::mutex> lock(mutex_);require(!ready_,"Close packed prefix before rebuilding");
    auto j=nlohmann::json::parse(specification);auto m=j.at("config");Qwen35PackedCommitConfig c{};
    c.batch=j.at("batch");c.tokens=j.at("tokens");c.capacity=j.at("capacity");
    c.key_heads=m.at("linear_num_key_heads");c.value_heads=m.at("linear_num_value_heads");c.kernel=m.at("linear_conv_kernel_dim");
    c.kv_heads=m.at("num_key_value_heads");c.head_dim=m.at("head_dim");
    c.state_width=j.at("state_width");c.packed_width=j.at("packed_width");c.hidden_width=j.at("hidden_width");
    require(m.at("linear_key_head_dim")==128&&m.at("linear_value_head_dim")==128,"Packed prefix requires128-wide K/V state");
    int hidden=m.at("hidden_size");require(hidden>=1&&hidden<=16384&&c.tokens>=1&&c.tokens<=16&&c.hidden_width==int64_t(hidden)*c.tokens,
        "Packed prefix hidden layout invalid");
    // Validate every interval before creating device storage or launching work.
    const auto& layers=j.at("layout");const auto& kinds=m.at("layer_types");
    require(layers.is_array()&&layers.size()==kinds.size()&&layers.size()<=128,"Packed prefix layer list invalid");
    std::vector<Qwen35PackedLinearLayout> linear;std::vector<int64_t> full;std::vector<float> parameters;
    SafeTensorsIndex index;std::string error;require(index.Open(checkpoint,error),"Packed prefix checkpoint open failed");
    int64_t cursor=0,payload=c.hidden_width+c.state_width;
    int channels=2*c.key_heads*128+c.value_heads*128;
    require(c.key_heads>=1&&c.key_heads<=64&&c.value_heads>=1&&c.value_heads<=64&&c.value_heads%c.key_heads==0&&c.kernel>=1&&c.kernel<=4,
        "Packed prefix linear dimensions invalid");
    require(c.batch>=1&&c.batch<=8&&c.capacity>=32&&c.capacity<=8192&&c.kv_heads>=1&&c.kv_heads<=32&&c.head_dim>=1&&c.head_dim<=256,
        "Packed prefix batch/cache dimensions invalid");
    for(size_t i=0;i<layers.size();++i){
        const auto& l=layers[i];std::string kind=kinds[i];require(l.at("index")==i&&l.at("kind")==kind&&l.at("offset")==cursor,"Packed prefix layer order invalid");
        int64_t w1=0,w2=0;
        if(kind=="linear_attention"){
            w1=int64_t(channels)*c.kernel;w2=int64_t(c.value_heads)*128*128;
            int64_t n=int64_t(c.tokens)*(channels+c.value_heads*128+2*c.value_heads);
            require(l.at("payload_offset")==payload&&l.at("payload_width")==n,"Packed prefix payload invalid");
            linear.push_back({cursor,payload});payload+=n;
            auto prefix="model.language_model.layers."+std::to_string(i)+".linear_attn.";
            for(auto pair: {std::make_pair("conv1d.weight",std::vector<long long>{channels,1,c.kernel}),
                            std::make_pair("dt_bias",std::vector<long long>{c.value_heads}),
                            std::make_pair("A_log",std::vector<long long>{c.value_heads})}){
                auto data=weight(index,prefix+pair.first,pair.second);parameters.insert(parameters.end(),data.begin(),data.end());
            }
        }else{
            require(kind=="full_attention","Packed prefix layer kind unsupported");
            w1=w2=int64_t(c.kv_heads)*c.capacity*c.head_dim;full.push_back(cursor);
        }
        require(l.at("state1_width")==w1&&l.at("state2_width")==w2,"Packed prefix state widths invalid");cursor+=w1+w2;
    }
    c.linear_layers=int(linear.size());c.full_layers=int(full.size());
    require(cursor==c.state_width&&payload==c.packed_width,"Packed prefix complete layout mismatch");
    auto workspaceBytes=Qwen35PackedCommitWorkspace(c);require(workspaceBytes>0,"Packed prefix config outside admitted range");
    // Metadata uses integer storage and the exact POD byte representation.
    auto a=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(linear.size()),2},linear.data());
    X::Tensor b;if(!full.empty())b=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(full.size())},full.data());
    auto p=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(parameters.size())},parameters.data());
    auto w=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(workspaceBytes/4)});
    linear_=a;full_=b;parameters_=p;workspace_=w;config_=c;ready_=true;return true;
}
void CUDAPackedPrefix::Commit(X::Value packed,X::Value canonical,X::Value spans){
    std::lock_guard<std::mutex> lock(mutex_);require(ready_,"Packed prefix is closed");
    X::Tensor source(packed),state(canonical);tensor(source.Info(),config_.batch,config_.packed_width);tensor(state.Info(),config_.batch,config_.state_width);
    require(spans.IsList()&&spans.Size()==uint64_t(2*config_.batch),"Packed prefix span count mismatch");
    Qwen35PackedCommitSpans s{};
    for(int b=0;b<config_.batch;++b){
        double start=spans.Get(uint64_t(2*b)).ToDouble(),count=spans.Get(uint64_t(2*b+1)).ToDouble();
        require(std::isfinite(start)&&std::isfinite(count)&&start==std::floor(start)&&count==std::floor(count)&&
            start>=0&&start<=config_.capacity&&count>=0&&count<=config_.tokens&&count<=config_.capacity-start,"Packed prefix spans out of range");
        s.start[b]=int(start);s.count[b]=int(count);
    }
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors{{source,X3_TENSOR_READ},{state,X3_TENSOR_WRITE},
        {linear_,X3_TENSOR_READ},{parameters_,X3_TENSOR_READ},{workspace_,X3_TENSOR_WRITE}};
    if(full_.IsValid())tensors.emplace_back(full_,X3_TENSOR_READ);
    auto use=TensorHelper::AcquireGPU(tensors);
    int status=RunQwen35PackedCommit(config_,s,(const float*)source.Info().data,(float*)state.Info().data,
        (const Qwen35PackedLinearLayout*)linear_.Info().data,full_.IsValid()?(const int64_t*)full_.Info().data:nullptr,
        (const float*)parameters_.Info().data,(float*)workspace_.Info().data,CurrentExecutionStream());
    require(status==0,"Packed prefix kernel launch failed; canonical completion is unknown");use.Finish();
}
std::string CUDAPackedPrefix::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);auto bytes=[](const X::Tensor& t)->uint64_t{return t.IsValid()?t.Info().byte_size:0;};
    return nlohmann::json{{"ready",ready_},{"workspace_bytes",bytes(workspace_)},{"parameter_bytes",bytes(parameters_)},
        {"layout_bytes",bytes(linear_)+bytes(full_)},{"owned_bytes",bytes(workspace_)+bytes(parameters_)+bytes(linear_)+bytes(full_)},
        {"method","native state-only in-place prefix commit"},{"scope","Owned helper buffers only; borrowed packed/canonical state excluded. Not process VRAM or peak."}}.dump();
}
}
#endif
