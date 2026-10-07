// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_marlin_chunks.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>
int main() {
    using namespace Garnet;
    for(int rows:{4097,4608,7168,8020,8192}) {
        if(GptOssMarlinBoundedPrefillRows(rows,1,4096,true)!=4096)return 1;
        std::vector<float> activation(std::size_t(rows)*96);
        std::vector<unsigned char> constant(8);
        const void* inputs[9];inputs[0]=activation.data();
        for(int i=1;i<9;++i)inputs[i]=constant.data()+i-1;
        std::vector<int> written(rows);
        for(int begin=0;begin<rows;begin+=4096) {
            int count=std::min(4096,rows-begin);
            auto segment=GptOssMarlinChunkInputs(inputs,std::size_t(begin),96);
            if(segment[0]!=activation.data()+std::size_t(begin)*96)return 1;
            for(int i=1;i<9;++i)if(segment[i]!=inputs[i])return 1;
            auto* data=static_cast<const float*>(segment[0]);
            if(data+std::size_t(count)*96>activation.data()+activation.size())return 1;
            for(int row=begin;row<begin+count;++row)++written[row];
        }
        for(int n:written)if(n!=1)return 1;
        if(inputs[0]!=activation.data())return 1;
    }
    for(int rows:{-1,0,1,512,4096,8193,std::numeric_limits<int>::max()})
        if(GptOssMarlinBoundedPrefillRows(rows,1,4096,true))return 1;
    for(int phase:{0,2,-1})
        if(GptOssMarlinBoundedPrefillRows(7168,phase,4096,true))return 1;
    for(int limit:{8,512,8192})
        if(GptOssMarlinBoundedPrefillRows(7168,1,limit,true))return 1;
    if(GptOssMarlinBoundedPrefillRows(7168,1,4096,false))return 1;
    std::puts("Bounded Marlin prefill extent/phase/flag/offset/constant/tail checks PASS");
}
