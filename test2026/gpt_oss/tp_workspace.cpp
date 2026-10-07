// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_tp_workspace.h"
#ifndef GARNET_GPT_OSS_WORKSPACE_CPU_ONLY
#include "gpt_oss_plugin.h"
#include <cstring>
#include <vector>
#endif
#include <cstdio>
#include <limits>

int main() {
    struct Case { int batch, sequence, phase, eligible; size_t expected; };
    const Case cases[] = {
        {512,3,1,1,512*3*32*4}, {128,32,1,1,128*32*32*4},
        {1,128,1,1,128*32*4}, {64,2,1,1,128*32*4},
        {1,127,1,1,0}, {512,1,0,1,512*32*4}, {512,3,1,0,0},
        {127,1,0,1,0}, {128,1,0,1,128*32*4}, {144,1,0,1,144*32*4},
        {448,1,0,1,448*32*4}, {513,1,0,1,0}, {128,2,0,1,0},
        {448,1,0,0,0}, {128,1,2,1,0}, {128,1,0,2,0},
        {0,3,1,1,0}, {512,0,1,1,0}, {-1,3,1,1,0},
        {std::numeric_limits<int>::max(),2,1,1,0}
    };
    for (const auto& test : cases) {
        const auto bytes=Garnet::GptOssTpBf16Workspace(3,test.eligible,test.phase,
            0,32,3,test.batch,test.sequence,32);
        if (bytes!=test.expected) return 1;
#ifndef GARNET_GPT_OSS_WORKSPACE_CPU_ONLY
        Garnet::GptOssOptions options{};
        options.kind=3; options.hidden=32; options.tpRank=0;
        options.prefill=test.phase; options.bf16Communication=test.eligible;
        Garnet::GptOssPlugin plugin(options);
        nvinfer1::PluginTensorDesc input{};
        input.dims.nbDims=3;
        input.dims.d[0]=test.batch; input.dims.d[1]=test.sequence; input.dims.d[2]=32;
        const auto actual=plugin.getWorkspaceSize(&input,1,nullptr,1);
        if (actual!=test.expected) {
            std::fprintf(stderr,"TP workspace batch%d sequence%d phase%d eligible%d: %zu != %zu\n",
                test.batch,test.sequence,test.phase,test.eligible,actual,test.expected);
            return 1;
        }
        if (std::strcmp(plugin.getPluginVersion(),"10")) return 1;
        std::vector<unsigned char> serialized(plugin.getSerializationSize());
        plugin.serialize(serialized.data());
        Garnet::GptOssPlugin restored(serialized.data(),serialized.size());
        auto* cloned=plugin.clone();
        if (!cloned || cloned->getWorkspaceSize(&input,1,nullptr,1)!=actual ||
            restored.getWorkspaceSize(&input,1,nullptr,1)!=actual) return 1;
        cloned->destroy();
#endif
    }
    // Both ranks, supported decode boundary/interior, and actual model H2880.
    for (int rank=0;rank<2;++rank) for (int batch=1;batch<=600;++batch) {
        const size_t expected=batch>=128&&batch<=512?size_t(batch)*2880*4:0;
        if (Garnet::GptOssTpBf16Workspace(3,1,0,rank,2880,3,batch,1,2880)!=expected)
            return 1;
    }
    struct Hazard { int kind,eligible,phase,rank,hidden,dimensions;
                    std::int64_t batch,sequence,width; };
    const Hazard hazards[]={
        {4,1,0,0,32,3,128,1,32}, {3,1,0,-1,32,3,128,1,32},
        {3,1,0,2,32,3,128,1,32}, {3,1,0,0,32,2,128,1,32},
        {3,1,0,0,32,3,128,1,64}, {3,1,0,0,0,3,128,1,0},
        {3,1,1,0,32,3,128,std::numeric_limits<std::int64_t>::max(),32},
        {3,1,1,0,std::numeric_limits<int>::max(),3,
            std::numeric_limits<int>::max(),1,std::numeric_limits<int>::max()}
    };
    for (const auto& h:hazards)
        if (Garnet::GptOssTpBf16Workspace(h.kind,h.eligible,h.phase,h.rank,
            h.hidden,h.dimensions,h.batch,h.sequence,h.width)) return 1;
    std::puts("TP V10 BF16 pair workspace phase/rank/eligibility/shape/overflow checks passed.");
    return 0;
}
