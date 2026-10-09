// SPDX-License-Identifier: Apache-2.0
// JSON belongs to the host compiler, not the CUDA translation unit.
#include "gpt_oss_peer_group_options.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdint>
namespace Garnet {
bool ParseGptOssPeerGroupOptions(const char* options,size_t bytes,size_t maximum,
        std::array<size_t,2>& counts,size_t& capacity,int& ctas){
    try{
        const auto v=nlohmann::json::parse(options,options+bytes);
        if(!v.is_object() || v.size()!=2 || !v.contains("phase_elements") || !v.contains("ctas") ||
           !v.at("ctas").is_number_integer() ||
           !v.at("phase_elements").is_array() || v.at("phase_elements").size()!=2)return false;
        const auto requestedCtas=v.at("ctas").get<int64_t>();
        if(requestedCtas!=64 && requestedCtas!=128 && requestedCtas!=188)return false;
        ctas=static_cast<int>(requestedCtas);
        for(int phase=0;phase<2;++phase){
            const auto& n=v.at("phase_elements").at(phase);
            if(!n.is_number_integer() || n.get<int64_t>()<=0 ||
               uint64_t(n.get<int64_t>())>maximum || n.get<int64_t>()%8)return false;
            counts[phase]=size_t(n.get<int64_t>());capacity=std::max(capacity,counts[phase]);
        }
        return true;
    }catch(...){return false;}
}
std::string GptOssPeerGroupDescription(const std::array<size_t,2>& counts,
        size_t capacity,int ctas,size_t ownedBytes,size_t mappedBytes){
    return nlohmann::json({{"schema",2},{"id","gpt_oss"},{"backend","tensorrt"},
        {"protocol","owned-mapped-bf16-peer-v2"},{"ctas",ctas},{"phase_elements",counts},
        {"capacity_elements",capacity},{"owned_bytes_per_rank",ownedBytes},
        {"mapped_host_bytes",mappedBytes},{"concurrency","serial paired phases, one invocation per rank"}}).dump();
}
}
