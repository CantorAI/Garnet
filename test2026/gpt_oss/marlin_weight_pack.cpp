// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_marlin_pack.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok){if(!ok)throw std::runtime_error("Marlin weight packing oracle failed");}
static uint32_t word(const std::vector<unsigned char>& packed,size_t index) {
    uint32_t value=0;for(int b=0;b<4;++b)value|=uint32_t(packed[index*4+b])<<(8*b);return value;
}
int main() {
    int cases=0;
    for(int hidden:{96,2880})for(int intermediate:{64,2880})for(int up:{0,1})for(int partition:{0,1,2,3}) {
        if((partition==2&&!up)||(partition==3&&up))continue;
        for(int rank:{-1,0,1})for(int scales:{0,1}) {
            Garnet::GptOssMarlinPackSpec spec{up,partition,rank,scales!=0};
            if(!Garnet::ValidGptOssMarlinPackSpec(spec))continue;
            const int E=3,N=up?2*intermediate:hidden,K=up?hidden:intermediate;
            std::vector<int64_t> shape{E,N,K/32};if(!scales)shape.push_back(16);
            std::vector<unsigned char> raw(size_t(E)*N*K/(scales?32:2));
            for(size_t i=0;i<raw.size();++i)raw[i]=scales?(i*29+i/7)%248+2:(i*73+i/17+i/(K/2))%256;
            std::vector<unsigned char> packed;std::vector<int64_t> dims;std::string error;
            require(Garnet::PackGptOssMarlinWeight(raw.data(),raw.size(),shape,spec,packed,dims,error));
            const int localE=partition==1?(E+1-rank)/2:E;
            const int localN=partition==2?N/2:N,localK=partition==3?K/2:K;
            const int padN=(localN+(up?127:63))/(up?128:64)*(up?128:64);
            const int padK=(localK+(up?63:127))/(up?64:128)*(up?64:128);
            require(dims[0]==localE&&dims[1]==padK/32&&dims[2]==padN&&
                packed.size()==size_t(localE)*padK*padN/(scales?32:2));
            const int offsetN=partition==2?rank*localN:0,offsetK=partition==3?rank*localK:0;
            // Inverse coordinate oracle, unlike the production word-forward
            // permutation: locate each logical code/scale in its target tile.
            for(int e=0;e<localE;++e)for(int n=0;n<padN;++n) {
                const int originalE=partition==1?2*e+rank:e;
                if(scales) {
                    const int shuffled=(n/64)*64+(n%8)*8+(n%64)/8;
                    const int targetN=(shuffled&~3)|((shuffled&1)<<1)|((shuffled&2)>>1);
                    for(int group=0;group<padK/32;++group) {
                        const unsigned wanted=n<localN&&group<localK/32
                            ?raw[(size_t(originalE)*N+n+offsetN)*(K/32)+group+offsetK/32]:127;
                        require(packed[(size_t(e)*(padK/32)+group)*padN+targetN]==wanted);
                    }
                } else for(int k=0;k<padK;++k) {
                    const int lane=(n%8)*4+(k%8)/2,warp=(n%64)/16;
                    const size_t target=(size_t(e)*(padK/16)*(padN/64)+size_t(k/16)*(padN/64)+n/64)*128+lane*4+warp;
                    constexpr int inverse[8]={0,4,1,5,2,6,3,7};
                    const int codePosition=inverse[(n%16>=8?4:0)+(k%16>=8?2:0)+k%2];
                    unsigned wanted=0;
                    if(n<localN&&k<localK) {
                        const unsigned byte=raw[(size_t(originalE)*N+n+offsetN)*(K/2)+(k+offsetK)/2];
                        wanted=(byte>>(((k+offsetK)%2)*4))&15;
                    }
                    require(((word(packed,target)>>(codePosition*4))&15)==wanted);
                }
            }
            std::string source;Garnet::GptOssMarlinPackSpec decoded;
            require(Garnet::ParseGptOssMarlinPackName(Garnet::GptOssMarlinPackName("block.17.weight",spec),source,decoded));
            require(source=="block.17.weight");
            std::vector<unsigned char> refit;std::vector<int64_t> refitShape;
            require(Garnet::PackGptOssMarlinWeight(raw.data(),raw.size(),shape,decoded,refit,refitShape,error));
            require(refit==packed&&refitShape==dims);
            ++cases;
        }
    }
    std::vector<unsigned char> source(3*128*3,127),output;
    std::vector<int64_t> shape{3,128,3},dims;std::string error,name;
    Garnet::GptOssMarlinPackSpec spec{1,0,-1,true},decoded;
    for(unsigned char bad:{0,1,250,255}) {
        source[0]=bad;
        require(!Garnet::PackGptOssMarlinWeight(source.data(),source.size(),shape,spec,output,dims,error));
    }
    source[0]=127;
    require(!Garnet::PackGptOssMarlinWeight(source.data(),source.size()-1,shape,spec,output,dims,error));
    require(!Garnet::PackGptOssMarlinWeight(nullptr,source.size(),shape,spec,output,dims,error));
    require(!Garnet::ParseGptOssMarlinPackName(Garnet::GptOssMarlinPackName("",spec),name,decoded));
    spec.rank=0;require(!Garnet::PackGptOssMarlinWeight(source.data(),source.size(),shape,spec,output,dims,error));
    std::cout<<cases<<" independent code/scale/padding/shard/refit packing cases passed; invalid inputs fail closed\n";
}
