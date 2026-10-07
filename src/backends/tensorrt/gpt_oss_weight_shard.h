// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>

namespace Garnet {
// Intermediate TP2 preserves all experts and selects contiguous gate/up row
// pairs and down-projection groups. Mode3 keeps bias only on rank0 so the
// following all-reduce adds it once. Packed MXFP4 bytes remain opaque.
inline std::string GptOssIntermediateShardName(const std::string& source,int rank,int mode) {
    return "__garnet_gpt_oss_tp2_inner_v1_rank"+std::to_string(rank)+"_mode"+std::to_string(mode)+"__:"+source;
}
inline bool ParseGptOssIntermediateShardName(const std::string& name,std::string& source,int& rank,int& mode) {
    for(int r=0;r<2;++r)for(int m=1;m<=3;++m) {
        const auto prefix=GptOssIntermediateShardName("",r,m);
        if(name.compare(0,prefix.size(),prefix)==0&&name.size()>prefix.size()) {
            source=name.substr(prefix.size());rank=r;mode=m;return true;
        }
    }
    return false;
}
template<class Integer> inline bool GatherGptOssIntermediateShard(const void* source,size_t bytes,
    const std::vector<Integer>& shape,size_t elementBytes,int rank,int mode,
    std::vector<unsigned char>& output,std::string& error) {
    if(!source||(rank!=0&&rank!=1)||mode<1||mode>3||shape.size()<2||shape.size()>4||
        (elementBytes!=1&&elementBytes!=2&&elementBytes!=4)||
        (mode==2&&shape.size()<3)||(mode==3&&shape.size()!=2)) {
        error="invalid GPT-OSS intermediate shard descriptor";return false;
    }
    size_t elements=1;
    for(auto extent:shape) {
        if(extent<=0||uint64_t(extent)>std::numeric_limits<size_t>::max()/elements) {
            error="GPT-OSS intermediate shard shape overflow";return false;
        }
        elements*=size_t(extent);
    }
    if(elements>std::numeric_limits<size_t>::max()/elementBytes||elements*elementBytes!=bytes) {
        error="GPT-OSS intermediate shard byte count mismatch";return false;
    }
    if(mode==3) {
        output.resize(bytes);
        if(rank==0)std::memcpy(output.data(),source,bytes);
        else std::memset(output.data(),0,bytes);
        return true;
    }
    const size_t axis=mode==1?1:2;
    if(shape[axis]%2) {error="GPT-OSS intermediate shard axis must divide equally";return false;}
    size_t outer=1,inner=elementBytes;
    for(size_t i=0;i<axis;++i)outer*=size_t(shape[i]);
    for(size_t i=axis+1;i<shape.size();++i)inner*=size_t(shape[i]);
    const size_t stride=size_t(shape[axis])*inner,half=stride/2;
    output.resize(bytes/2);
    const auto* data=static_cast<const unsigned char*>(source);
    for(size_t i=0;i<outer;++i)
        std::memcpy(output.data()+i*half,data+i*stride+rank*half,half);
    return true;
}
// Derived names preserve enough information to regenerate the exact constant
// from the original checkpoint when loading a stripped/refittable engine.
inline std::string GptOssExpertShardName(const std::string& source, int rank) {
    return "__garnet_gpt_oss_ep2_v1_rank" + std::to_string(rank) + "__:" + source;
}
inline bool ParseGptOssExpertShardName(const std::string& name,
    std::string& source, int& rank) {
    for (int r = 0; r < 2; ++r) {
        const auto prefix = GptOssExpertShardName("", r);
        if (name.compare(0, prefix.size(), prefix) == 0 && name.size() > prefix.size()) {
            source = name.substr(prefix.size()); rank = r; return true;
        }
    }
    return false;
}
inline bool GatherGptOssExpertShard(const void* source, size_t bytes,
    int64_t experts, int rank, std::vector<unsigned char>& output,
    std::string& error) {
    if (!source || experts < 2 || (rank != 0 && rank != 1) ||
        !bytes || uint64_t(experts) > bytes || bytes % size_t(experts)) {
        error = "invalid GPT-OSS expert-axis checkpoint shard";
        return false;
    }
    const size_t stride = bytes / size_t(experts);
    const size_t localExperts = (size_t(experts) + 1 - rank) / 2;
    output.resize(localExperts * stride);
    const auto* original = static_cast<const unsigned char*>(source);
    for (size_t local = 0; local < localExperts; ++local)
        std::memcpy(output.data() + local * stride,
            original + (2 * local + rank) * stride, stride);
    return true;
}
}
