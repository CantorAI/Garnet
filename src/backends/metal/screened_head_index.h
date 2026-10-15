// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
namespace Garnet {
#ifdef GARNET_METAL_INTERLEAVED_SCREEN_INDEX
inline constexpr bool kScreenedHeadInterleaved=true;
#else
inline constexpr bool kScreenedHeadInterleaved=false;
#endif
#ifdef GARNET_METAL_SMALL_SCREEN_SEED
inline constexpr size_t kScreenedHeadSeedRows=1024;
#else
inline constexpr size_t kScreenedHeadSeedRows=8192;
#endif
// Static admission is deliberately bounded. Original BF16 bytes remain authoritative.
inline bool EncodeScreenedHeadRow(const uint16_t* source,size_t count,int8_t* encoded,float* metadata) {
    if(!source||!encoded||!metadata||(count!=896&&count!=1024)||std::fegetround()!=FE_TONEAREST)return false;
    auto value=[](uint16_t raw){uint32_t bits=uint32_t(raw)<<16;float result;std::memcpy(&result,&bits,4);return double(result);};
    double largest=0;
    for(size_t i=0;i<count;++i){double x=value(source[i]);if(!std::isfinite(x))return false;largest=std::max(largest,std::abs(x));}
    double scale=1;
    if(largest){int exponent=0;double fraction=std::frexp(largest/127.,&exponent);scale=std::ldexp(1.,fraction==.5?exponent-1:exponent);}
    if(!std::isfinite(scale)||!(scale>0)||scale>1e6||double(float(scale))!=scale)return false;
    double error=0,original=0,reconstructed=0;
    for(size_t i=0;i<count;++i){double x=value(source[i]),q=std::nearbyint(x/scale),restored=q*scale;
        if(q<-127||q>127||double(float(restored))!=restored)return false;
        encoded[i]=int8_t(q);double delta=x-restored;error+=delta*delta;original+=x*x;reconstructed+=restored*restored;
    }
    const double squared[]={error,original,reconstructed};
    for(size_t i=0;i<3;++i){double norm=std::sqrt(squared[i]);float upper=std::nextafter(float(norm*1.0001),std::numeric_limits<float>::infinity());
        if(!std::isfinite(upper)||upper>1e6||double(upper)<norm)return false;metadata[i]=upper;
    }
    metadata[3]=float(scale);return true;
}
}
