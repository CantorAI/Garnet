#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    uint4 signMant=uint4(as_type<uchar4>(sm));
    uint4 exponents=uint4(tags,tags>>4,tags>>8,tags>>12)&15u;
    return ushort4((signMant&127u)|((signMant&128u)<<8)|((exponents+base)<<7));
}
