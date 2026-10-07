// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace Garnet {
// Marlin tile/scale mapping follows the vLLM v0.31.0 Apache-2.0 sources
// already attributed in repository NOTICE and third_party/marlin/PROVENANCE.txt.
// Compose TP selection with packing; never materialize an original GPU copy.
struct GptOssMarlinPackSpec {
    int projection=0; // 0: down, 1: interleaved gate/up
    int partition=0; // 0: all, 1: expert axis, 2: up rows, 3: down K groups
    int rank=-1;
    bool scales=false;
};
inline bool ValidGptOssMarlinPackSpec(const GptOssMarlinPackSpec& s) {
    return (s.projection==0||s.projection==1) &&
        ((s.partition==0&&s.rank==-1) ||
         (s.rank>=0&&s.rank<2&&(s.partition==1||
          (s.partition==2&&s.projection==1)||(s.partition==3&&s.projection==0))));
}
inline std::string GptOssMarlinPackName(const std::string& source,const GptOssMarlinPackSpec& s) {
    return "__garnet_gpt_oss_marlin_v1_u"+std::to_string(s.projection)+"_p"+
        std::to_string(s.partition)+"_r"+std::to_string(s.rank)+"_s"+
        std::to_string(int(s.scales))+"__:"+source;
}
inline bool ParseGptOssMarlinPackName(const std::string& name,std::string& source,GptOssMarlinPackSpec& s) {
    for(int up=0;up<2;++up)for(int p=0;p<4;++p)for(int rank=-1;rank<2;++rank)for(int scale=0;scale<2;++scale) {
        GptOssMarlinPackSpec candidate{up,p,rank,scale!=0};
        if(!ValidGptOssMarlinPackSpec(candidate))continue;
        const auto prefix=GptOssMarlinPackName("",candidate);
        if(name.size()>prefix.size()&&name.compare(0,prefix.size(),prefix)==0) {
            source=name.substr(prefix.size());s=candidate;return true;
        }
    }
    return false;
}
template<class F> inline void GptOssPackRanges(size_t count,F function) {
    const unsigned workers=count>=size_t(1<<20)
        ? std::min(16u,std::max(1u,std::thread::hardware_concurrency())) : 1u;
    if(workers==1){function(0,count);return;}
    std::vector<std::thread> threads;
    threads.reserve(workers-1);
    // Join already-created workers if thread creation fails; never terminate
    // while unwinding through a joinable std::thread.
    try {
        for(unsigned i=1;i<workers;++i)
            threads.emplace_back([&,i]{function(count*i/workers,count*(i+1)/workers);});
        function(0,count/workers);
    } catch(...) {
        for(auto& thread:threads)thread.join();
        throw;
    }
    for(auto& thread:threads)thread.join();
}
template<class Integer> inline bool PackGptOssMarlinWeight(const void* source,size_t bytes,
    const std::vector<Integer>& shape,const GptOssMarlinPackSpec& s,
    std::vector<unsigned char>& output,std::vector<int64_t>& packedShape,std::string& error) {
    if(!source||!ValidGptOssMarlinPackSpec(s)||shape.size()!=size_t(s.scales?3:4)) {
        error="invalid GPT-OSS Marlin packing descriptor";return false;
    }
    size_t elements=1;
    for(auto extent:shape) {
        if(extent<=0||uint64_t(extent)>std::numeric_limits<size_t>::max()/elements) {
            error="GPT-OSS Marlin source shape overflow";return false;
        }
        elements*=size_t(extent);
    }
    if(elements!=bytes||shape[0]>256||shape[1]>131072||shape[2]>2048||(!s.scales&&shape[3]!=16)) {
        error="invalid GPT-OSS Marlin source byte count or dimensions";return false;
    }
    const int experts=int(shape[0]),originalN=int(shape[1]),originalK=int(shape[2])*32;
    if((s.projection==1&&(originalK>16384||originalN%64))||
       (s.projection==0&&(originalN>16384||originalN%32))||
       (s.partition==1&&experts<2)||
       (s.partition==2&&originalN%128)||
       (s.partition==3&&originalK%64)) {
        error="unsupported GPT-OSS Marlin source/partition alignment";return false;
    }
    const int localE=s.partition==1?(experts+1-s.rank)/2:experts;
    const int localN=s.partition==2?originalN/2:originalN;
    const int localK=s.partition==3?originalK/2:originalK;
    const int paddedN=(localN+(s.projection?127:63))/(s.projection?128:64)*(s.projection?128:64);
    const int paddedK=(localK+(s.projection?63:127))/(s.projection?64:128)*(s.projection?64:128);
    const size_t perExpert=size_t(paddedK)*paddedN/(s.scales?32:2);
    if(size_t(localE)>std::numeric_limits<size_t>::max()/perExpert) {
        error="GPT-OSS Marlin packed size overflow";return false;
    }
    packedShape=s.scales?std::vector<int64_t>{localE,paddedK/32,paddedN}
                       :std::vector<int64_t>{localE,paddedK/32,paddedN,16};
    output.resize(size_t(localE)*perExpert);
    const auto* raw=static_cast<const unsigned char*>(source);
    const int nOffset=s.partition==2?s.rank*localN:0;
    const int kOffset=s.partition==3?s.rank*localK:0;
    if(s.scales) {
        std::atomic<bool> invalid{false};
        GptOssPackRanges(output.size(),[&](size_t begin,size_t end) {
            for(size_t index=begin;index<end;++index) {
                const int e=int(index/perExpert),expert=s.partition==1?2*e+s.rank:e;
                const size_t local=index%perExpert;
                const int group=int(local/paddedN);
                int n=int(local%paddedN);
                n=(n&~3)|((n&1)<<1)|((n&2)>>1);
                n=(n/64)*64+(n%64)/8+8*(n%8);
                unsigned char value=127;
                if(n<localN&&group<localK/32) {
                    value=raw[(size_t(expert)*originalN+n+nOffset)*(originalK/32)+group+kOffset/32];
                    if(value<2||value>249)invalid.store(true,std::memory_order_relaxed);
                }
                output[index]=value;
            }
        });
        if(invalid.load()){error="unsupported GPT-OSS Marlin UE8M0 scale";return false;}
    } else {
        const size_t wordsPerExpert=perExpert/4;
        GptOssPackRanges(output.size()/4,[&](size_t begin,size_t end) {
            constexpr int offsets[4]={0,1,8,9},permutation[8]={0,2,4,6,1,3,5,7};
            for(size_t index=begin;index<end;++index) {
                const int e=int(index/wordsPerExpert),expert=s.partition==1?2*e+s.rank:e;
                const size_t local=index%wordsPerExpert;
                const int lane=int(local%128)/4,warp=int(local%4),tile=int(local/128);
                const int firstN=(tile%(paddedN/64))*64+warp*16+lane/4;
                const int firstK=(tile/(paddedN/64))*16+(lane%4)*2;
                uint32_t word=0;
                for(int p=0;p<8;++p) {
                    const int v=permutation[p],n=firstN+(v>=4?8:0),k=firstK+offsets[v%4];
                    unsigned code=0;
                    if(n<localN&&k<localK) {
                        const unsigned byte=raw[(size_t(expert)*originalN+n+nOffset)*(originalK/2)+(k+kOffset)/2];
                        code=(byte>>(((k+kOffset)%2)*4))&15;
                    }
                    word|=code<<(p*4);
                }
                // Explicit little-endian bytes match CUDA uint32 words on all
                // supported hosts, without alignment or aliasing assumptions.
                for(int b=0;b<4;++b)output[index*4+b]=static_cast<unsigned char>(word>>(b*8));
            }
        });
    }
    return true;
}
}
