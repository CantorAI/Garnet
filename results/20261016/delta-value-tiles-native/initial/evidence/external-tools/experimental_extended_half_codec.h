// Experimental only; serving codec is unchanged.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace Garnet {
// Magnitude bits are monotonically ordered for finite positive BF16 values.
inline bool ExactHalfRowScale(uint16_t largest,int32_t& scale) {
    const int exponent=(largest>>7)&255;
    if(!largest){scale=0;return true;}
    if(exponent==0||exponent==255)return false;
    scale=142-exponent;
    return scale>=-112&&scale<=141;
}
// Encode only representable values. No rounding, saturation or underflow to zero.
inline bool EncodeExactScaledHalf(uint16_t value,int32_t scale,uint16_t& encoded) {
    if(scale<-112||scale>141)return false;
    const uint16_t sign=value&0x8000,magnitude=value&0x7fff;
    if(!magnitude){encoded=sign;return true;}
    const int sourceExponent=(magnitude>>7)&255;
    if(sourceExponent==0||sourceExponent==255)return false;
    const int exponent=sourceExponent-127+scale;
    if(exponent>15||exponent<-24)return false;
    if(exponent>=-14) {
        encoded=sign|uint16_t((exponent+15)<<10)|uint16_t((magnitude&127)<<3);
        return true;
    }
    const uint32_t significand=128+(magnitude&127);
    const int shift=exponent+17;
    uint32_t fraction;
    if(shift>=0)fraction=significand<<shift;
    else {
        const int discarded=-shift;
        if(significand&((1u<<discarded)-1))return false;
        fraction=significand>>discarded;
    }
    encoded=sign|uint16_t(fraction);return true;
}
}
