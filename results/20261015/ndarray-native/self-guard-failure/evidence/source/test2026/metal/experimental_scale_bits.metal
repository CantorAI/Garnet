// Experimental exact power-of-two scaling, preserving subnormal result bits.
// Assumes a finite FP32 input; does not change learned weights or arithmetic.
inline uint scale_bits(uint bits,int scale) {
    uint sign=bits&0x80000000u,exponent=(bits>>23)&255u,fraction=bits&0x7fffffu;
    if(exponent==255u)return bits;
    if(exponent==0u&&fraction==0u)return bits;
    int actualExponent=int(exponent);
    uint significand=fraction|0x800000u;
    if(exponent==0u) {
        int shift=int(clz(fraction))-8;
        significand=fraction<<shift;actualExponent=1-shift;
    }
    int target=actualExponent+scale;
    if(target>=255)return sign|0x7f800000u;
    if(target>0)return sign|(uint(target)<<23)|(significand&0x7fffffu);
    int shift=1-target;
    if(shift>=25)return sign;
    uint truncated=significand>>shift,mask=(1u<<shift)-1u,tail=significand&mask,halfway=1u<<(shift-1);
    if(tail>halfway||(tail==halfway&&(truncated&1u)))++truncated;
    return sign|truncated;
}
