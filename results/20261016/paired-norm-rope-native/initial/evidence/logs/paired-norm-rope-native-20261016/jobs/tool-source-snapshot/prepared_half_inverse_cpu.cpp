// Exhaustive original-codec and independent floating reconstruction controls; CPU only.
#include "exact_half_codec.h"
#include "prepared_half_inverse_bits.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
static void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static uint32_t reference(uint16_t half,int32_t scale){
    uint32_t exponent=(half>>10)&31,fraction=half&1023;
    need(exponent!=31,"Floating reference only accepts finite half");
    // Independent exact binary arithmetic, followed by the old float ldexp step.
    double value=exponent?std::ldexp(1.+double(fraction)/1024.,int(exponent)-15):std::ldexp(double(fraction),-24);
    if(half&0x8000)value=-value;
    float restored=std::ldexp(static_cast<float>(value),-scale);uint32_t bits;std::memcpy(&bits,&restored,4);return bits;
}
int main(){try{
    uint64_t bf16Cases=0,accepted=0,rejected=0,halfCases=0,finite=0,nonfinite=0,boundary=0;
    for(int32_t scale=-64;scale<=64;++scale)for(uint32_t value=0;value<65536;++value){
        uint16_t encoded=0x55aa;bool good=Garnet::EncodeExactScaledHalf(uint16_t(value),scale,encoded);++bf16Cases;
        if(!good){++rejected;continue;}++accepted;uint32_t bits=0x7fcafeed;
        need(Garnet::RestorePreparedHalfBits(encoded,scale,bits),"Original admitted encoding rejected by integer inverse");
        need(bits==(value<<16),"Integer inverse changed original BF16-expanded F32 bits");
        need(bits==reference(encoded,scale),"Integer inverse differs from independent floating restore");
    }
    for(int32_t scale=-64;scale<=64;++scale)for(uint32_t value=0;value<65536;++value){
        uint32_t bits=0x7fcafeed;bool good=Garnet::RestorePreparedHalfBits(uint16_t(value),scale,bits);++halfCases;
        if(((value>>10)&31)==31){++nonfinite;need(!good&&bits==0x7fcafeed,"Nonfinite half admitted or rejected output changed");}
        else{++finite;need(good&&bits==reference(uint16_t(value),scale),"Finite half integer/floating reconstruction differs");}
    }
    for(int32_t scale:{std::numeric_limits<int32_t>::min(),-1000,-65,65,1000,std::numeric_limits<int32_t>::max()})for(uint32_t value=0;value<65536;++value){
        uint32_t bits=0x7fcafeed;uint16_t encoded=0x55aa;
        need(!Garnet::RestorePreparedHalfBits(uint16_t(value),scale,bits)&&bits==0x7fcafeed,"Invalid scale admitted or inverse output changed");
        need(!Garnet::EncodeExactScaledHalf(uint16_t(value),scale,encoded)&&encoded==0x55aa,"Original codec invalid-scale guard differs");++boundary;
    }
    need(bf16Cases==8454144&&halfCases==8454144&&boundary==393216,"Exhaustive coverage count differs");
    printf("{\"kind\":\"complete\",\"bf16_scale_cases\":%llu,\"original_codec_admitted\":%llu,\"original_codec_rejected\":%llu,\"half_scale_cases\":%llu,\"finite_half_exact\":%llu,\"nonfinite_half_rejected\":%llu,\"invalid_scale_cases\":%llu,\"original_codec_and_float_reference_bits_exact\":true,\"rejected_outputs_unchanged\":true,\"gpu_or_model_test\":false}\n",(unsigned long long)bf16Cases,(unsigned long long)accepted,(unsigned long long)rejected,(unsigned long long)halfCases,(unsigned long long)finite,(unsigned long long)nonfinite,(unsigned long long)boundary);return 0;
}catch(const std::exception& error){fprintf(stderr,"prepared_half_inverse_failed: %s\n",error.what());return 1;}}
