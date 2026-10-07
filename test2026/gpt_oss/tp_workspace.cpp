// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_plugin.h"
#include <cstdio>
#include <limits>

int main() {
    struct Case { int batch, sequence, phase, eligible; size_t expected; };
    const Case cases[] = {
        {512,3,1,1,512*3*32*4}, {128,32,1,1,128*32*32*4},
        {1,128,1,1,128*32*4}, {64,2,1,1,128*32*4},
        {1,127,1,1,0}, {512,1,0,1,0}, {512,3,1,0,0},
        {0,3,1,1,0}, {512,0,1,1,0}, {-1,3,1,1,0},
        {std::numeric_limits<int>::max(),2,1,1,0}
    };
    for (const auto& test : cases) {
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
    }
    std::puts("TP prefill workspace batched threshold, phase, eligibility and invalid shape checks passed.");
    return 0;
}
