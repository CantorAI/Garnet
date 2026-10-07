// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_weight_shard.h"
#include <iostream>
#include <stdexcept>

static void require(bool value) {
    if (!value) throw std::runtime_error("expert weight shard test failed");
}
int main() {
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
