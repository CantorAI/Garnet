// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_weight_shard.h"
#include <iostream>
#include <stdexcept>
#include <limits>

static void require(bool value) {
    if (!value) throw std::runtime_error("expert weight shard test failed");
}
int main() {
    int intermediateCases=0;
    for(int experts:{2,5,128})for(size_t elementBytes:{1,2,4})for(int mode:{1,2,3}) {
        std::vector<std::vector<long long>> shapes;
        if(mode==1)shapes={{experts,128},{experts,128,3},{experts,128,3,16}};
        if(mode==2)shapes={{experts,96,4},{experts,96,4,16}};
        if(mode==3)shapes={{experts,96}};
        for(const auto& shape:shapes) {
            size_t elements=1;for(auto extent:shape)elements*=size_t(extent);
            std::vector<unsigned char> original(elements*elementBytes);
            for(size_t i=0;i<original.size();++i)original[i]=(i*73+i/31)%251;
            for(int rank:{0,1}) {
                std::vector<unsigned char> shard;std::string error,source;int decodedRank=-1,decodedMode=0;
                require(Garnet::GatherGptOssIntermediateShard(original.data(),original.size(),shape,
                    elementBytes,rank,mode,shard,error));
                auto localShape=shape;
                if(mode!=3)localShape[mode==1?1:2]/=2;
                const size_t count=mode==3?elements:elements/2;
                require(shard.size()==count*elementBytes);
                // Independent coordinate oracle for every element/component.
                for(size_t i=0;i<count;++i) {
                    size_t rest=i;std::vector<size_t> coordinates(shape.size());
                    for(size_t reverse=shape.size();reverse-->0;) {
                        coordinates[reverse]=rest%size_t(localShape[reverse]);rest/=size_t(localShape[reverse]);
                    }
                    if(mode!=3)coordinates[mode==1?1:2]+=rank*size_t(shape[mode==1?1:2])/2;
                    size_t position=0;for(size_t axis=0;axis<shape.size();++axis)
                        position=position*size_t(shape[axis])+coordinates[axis];
                    for(size_t byte=0;byte<elementBytes;++byte)
                        require(shard[i*elementBytes+byte]==(mode==3&&rank==1?0:original[position*elementBytes+byte]));
                }
                require(Garnet::ParseGptOssIntermediateShardName(
                    Garnet::GptOssIntermediateShardName("block.17.weight",rank,mode),source,decodedRank,decodedMode));
                require(source=="block.17.weight"&&decodedRank==rank&&decodedMode==mode);
                std::vector<unsigned char> regenerated;
                require(Garnet::GatherGptOssIntermediateShard(original.data(),original.size(),shape,
                    elementBytes,decodedRank,decodedMode,regenerated,error)&&regenerated==shard);
                ++intermediateCases;
            }
        }
    }
    {
        std::vector<unsigned char> source(64),out;std::string error,name;int rank=-1,mode=0;
        require(!Garnet::GatherGptOssIntermediateShard(source.data(),63,std::vector<long long>{2,32},1,0,1,out,error));
        require(!Garnet::GatherGptOssIntermediateShard(source.data(),64,std::vector<long long>{2,32},1,2,1,out,error));
        require(!Garnet::GatherGptOssIntermediateShard(source.data(),64,std::vector<long long>{2,32},1,0,2,out,error));
        require(!Garnet::GatherGptOssIntermediateShard(source.data(),6,std::vector<long long>{2,3},1,0,1,out,error));
        require(!Garnet::GatherGptOssIntermediateShard(source.data(),64,
            std::vector<long long>{std::numeric_limits<long long>::max(),32},4,0,1,out,error));
        require(!Garnet::ParseGptOssIntermediateShardName(Garnet::GptOssIntermediateShardName("x",2,1),name,rank,mode));
        require(!Garnet::ParseGptOssIntermediateShardName(Garnet::GptOssIntermediateShardName("",0,1),name,rank,mode));
    }
    std::cout<<"intermediate TP2 coordinate/byte/bias/refit cases passed: "<<intermediateCases<<"\n";
    for (int experts : {2, 5, 128}) for (int rowBytes : {1, 2, 12, 64}) {
        std::vector<unsigned char> full(size_t(experts) * rowBytes);
        for (size_t i = 0; i < full.size(); ++i) full[i] = (i * 73 + i / rowBytes) % 251;
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<unsigned char> shard;
            std::string error;
            require(Garnet::GatherGptOssExpertShard(full.data(), full.size(), experts, rank, shard, error));
            require(shard.size() == size_t((experts + 1 - rank) / 2) * rowBytes);
            for (size_t i = 0; i < shard.size(); ++i)
                require(shard[i] == full[(2 * (i / rowBytes) + rank) * rowBytes + i % rowBytes]);
            std::string source;
            int decoded = -1;
            require(Garnet::ParseGptOssExpertShardName(
                Garnet::GptOssExpertShardName("block.17.mlp.mlp1_weight.blocks", rank), source, decoded));
            require(source == "block.17.mlp.mlp1_weight.blocks" && decoded == rank);
            // Refit regeneration must reproduce the initial constant byte for byte.
            std::vector<unsigned char> regenerated;
            require(Garnet::GatherGptOssExpertShard(full.data(), full.size(), experts,
                decoded, regenerated, error) && regenerated == shard);
        }
    }
    std::vector<unsigned char> source(15), output;
    std::string error, decoded;
    int rank = -1;
    require(!Garnet::GatherGptOssExpertShard(source.data(), 14, 5, 0, output, error));
    require(!Garnet::GatherGptOssExpertShard(source.data(), 15, 5, 2, output, error));
    require(!Garnet::GatherGptOssExpertShard(nullptr, 15, 5, 0, output, error));
    require(!Garnet::ParseGptOssExpertShardName("block.0.mlp.mlp1_weight.blocks", decoded, rank));
    require(!Garnet::ParseGptOssExpertShardName(Garnet::GptOssExpertShardName("", 0), decoded, rank));
    require(!Garnet::ParseGptOssExpertShardName(Garnet::GptOssExpertShardName("x", 2), decoded, rank));
    std::cout << "expert weight shard byte mapping and refit regeneration passed\n";
}
