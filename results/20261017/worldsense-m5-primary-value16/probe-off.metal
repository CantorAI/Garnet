#include <metal_stdlib>
using namespace metal;
inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    uint4 signMant=uint4(as_type<uchar4>(sm));
    uint4 exponents=uint4(tags,tags>>4,tags>>8,tags>>12)&15u;
    return ushort4((signMant&127u)|((signMant&128u)<<8)|((exponents+base)<<7));
}
kernel void value16_probe(device const uint* input [[buffer(0)]],device ushort4* output [[buffer(1)]],uint id [[thread_position_in_grid]]) {
 output[id]=primary_quad_kword(input[id*3],ushort(input[id*3+1]),input[id*3+2]);
}
