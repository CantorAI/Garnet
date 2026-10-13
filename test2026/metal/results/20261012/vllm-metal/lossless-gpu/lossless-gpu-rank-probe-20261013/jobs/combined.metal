#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
kernel void bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
            if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
        if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
// Experimental exact codec, appended to the current native BF16 decode shader.
// One SIMD group reconstructs 128 values. No floating-point conversion is
// performed until the original BF16 bits have been reconstructed.
inline uint below(uint limit) {return limit>=32?0xffffffffu:((1u<<limit)-1u);}
inline ushort4 unpack4(device const uchar* payload,device const uint* tags,
    device const ulong* metadata,device const uchar* exceptions,device const uchar* ranks,ulong block,uint lane) {
    uint bit=(lane%8)*12,word=bit/32,shift=bit%32;
    ulong start=block*12+(lane/8)*3;
    uint packed=tags[start+word]>>shift;
    if(shift>20)packed|=tags[start+word+1]<<(32-shift);
    uint4 tag=uint4(packed,packed>>3,packed>>6,packed>>9)&7;
    uint4 escaped=uint4(tag==7);
    uint w0=tags[start],w1=tags[start+1],w2=tags[start+2];
    uint n=(lane%8)*4,limit=n*3;
    uint flags0=(w0&(w0>>1)&(w0>>2))&0x09249249u;
    uint flags1=(w1&(w1>>1)&(w1>>2))&0x12492492u;
    uint flags2=(w2&(w2>>1)&(w2>>2))&0x24924924u;
    uint prefix=uint(ranks[block*4+lane/8])+popcount(flags0&below(limit));
    if(limit>32)prefix+=popcount(flags1&below(limit-32));
    if(limit>64)prefix+=popcount(flags2&below(limit-64));
    if(n>10)prefix+=uint((((w0>>30)|(w1<<2))&7)==7);
    if(n>21)prefix+=uint((((w1>>31)|(w2<<1))&7)==7);
    ulong meta=metadata[block],offset=(meta>>8)+prefix;
    uint base=uint(meta&255);
    uchar4 low=*(device const uchar4*)(payload+block*128+lane*4);
    ushort4 bits;
    for(uint j=0;j<4;++j) {
        uint exponent=escaped[j]?uint(exceptions[offset++]):base-tag[j];
        bits[j]=ushort((uint(low[j])&127)|((uint(low[j])&128)<<8)|(exponent<<7));
    }
    return bits;
}
kernel void unpack_bits(device const uchar* payload [[buffer(0)]],device const uint* tags [[buffer(1)]],
    device const ulong* metadata [[buffer(2)]],device const uchar* exceptions [[buffer(3)]],
    device ushort* output [[buffer(4)]],constant ulong& blocks [[buffer(5)]],device const uchar* ranks [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong block=ulong(group)*4+simd;
    if(block>=blocks)return;
    *(device ushort4*)(output+block*128+lane*4)=unpack4(payload,tags,metadata,exceptions,ranks,block,lane);
}
kernel void packed_decode(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    device const uint* tags [[buffer(4)]],device const ulong* metadata [[buffer(5)]],
    device const uchar* exceptions [[buffer(6)]],device const uchar* ranks [[buffer(7)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    for(ulong base=0;base<p.inner;base+=128) {
        ulong k=base+lane*4;
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        ushort4 b0=unpack4(payload,tags,metadata,exceptions,ranks,(column*p.inner+base)/128,lane);
        ushort4 b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
        if(column+1<p.columns)b1=unpack4(payload,tags,metadata,exceptions,ranks,((column+1)*p.inner+base)/128,lane);
        if(column+2<p.columns)b2=unpack4(payload,tags,metadata,exceptions,ranks,((column+2)*p.inner+base)/128,lane);
        if(column+3<p.columns)b3=unpack4(payload,tags,metadata,exceptions,ranks,((column+3)*p.inner+base)/128,lane);
        for(uint j=0;j<4;++j) {
            float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
            sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
        }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
