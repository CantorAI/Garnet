#include <metal_stdlib>
using namespace metal;
inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    ushort4 signMant=ushort4(as_type<uchar4>(sm));
    ushort4 exponents=ushort4(tags,tags>>4,tags>>8,tags>>12)&ushort(15);
    ushort4 placed=(signMant*ushort(257))&ushort(0x807f);
    return placed|((exponents+ushort(base))<<ushort(7));
}
kernel void value16_probe(device const uint* input [[buffer(0)]],device ushort4* output [[buffer(1)]],uint id [[thread_position_in_grid]]) {
 output[id]=primary_quad_kword(input[id*3],ushort(input[id*3+1]),input[id*3+2]);
}
