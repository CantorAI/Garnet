// Private CPU setup experiment. Exact inverse of a finite half scaled by 2^-scale.
// This does not replace the serving codec or change any GPU arithmetic.
#pragma once
#include <cstdint>
namespace Garnet {
inline bool RestorePreparedHalfBits(uint16_t half,int32_t scale,uint32_t& restored){
    if(scale<-64||scale>64)return false;
    const uint32_t sign=uint32_t(half&0x8000)<<16,exponent=(half>>10)&31,fraction=half&1023;
    if(exponent==31)return false;
    if(!exponent&&!fraction){restored=sign;return true;}
    int32_t resultExponent;uint32_t resultFraction;
    if(exponent){resultExponent=int32_t(exponent)+112-scale;resultFraction=fraction<<13;}
    else{
        unsigned highest;
#if defined(__clang__) || defined(__GNUC__)
        highest=31u-unsigned(__builtin_clz(fraction));
#else
        highest=0;for(uint32_t value=fraction;value>>=1;)++highest;
#endif
        resultExponent=int32_t(highest)+103-scale;
        resultFraction=(fraction-(uint32_t(1)<<highest))<<(23-highest);
    }
    // All finite half values within the supported scale range are normal F32.
    // Preserve explicit admission guards instead of introducing rounding/clipping.
    if(resultExponent<1||resultExponent>254)return false;
    restored=sign|(uint32_t(resultExponent)<<23)|resultFraction;return true;
}
}
