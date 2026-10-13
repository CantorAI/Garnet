// Experimental exact codec, appended to the current native BF16 decode shader.
// One SIMD group reconstructs 128 values. No floating-point conversion is
// performed until the original BF16 bits have been reconstructed.
inline ushort4 unpack4(device const uchar* payload,device const ushort* tags,
    device const ulong* metadata,device const uchar* exceptions,ulong block,uint lane) {
    uint packed=uint(tags[block*32+lane]);
    ushort4 tag=ushort4(packed,packed>>4,packed>>8,packed>>12)&15;
    ulong meta=metadata[block];ushort base=ushort(meta&255);
    ushort4 low=ushort4(*(device const uchar4*)(payload+block*128+lane*4));
    ushort4 exponents=base-tag;
    // Uniform per SIMD group; most real checkpoint blocks take this path.
    if((meta>>63)==0) {
        uint4 escaped=uint4(tag==15);
        uint count=escaped.x+escaped.y+escaped.z+escaped.w;
        uint prefix=simd_prefix_exclusive_sum(count);
        ulong offset=((meta&0x7ffffffffffffffful)>>8)+prefix;
        for(uint j=0;j<4;++j)if(escaped[j])exponents[j]=ushort(exceptions[offset++]);
    }
    return (low&127)|((low&128)<<8)|(exponents<<7);
}
kernel void unpack_bits(device const uchar* payload [[buffer(0)]],device const ushort* tags [[buffer(1)]],
    device const ulong* metadata [[buffer(2)]],device const uchar* exceptions [[buffer(3)]],
    device ushort* output [[buffer(4)]],constant ulong& blocks [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong block=ulong(group)*4+simd;
    if(block>=blocks)return;
    *(device ushort4*)(output+block*128+lane*4)=unpack4(payload,tags,metadata,exceptions,block,lane);
}
kernel void packed_decode(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    device const ushort* tags [[buffer(4)]],device const ulong* metadata [[buffer(5)]],
    device const uchar* exceptions [[buffer(6)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    for(ulong base=0;base<p.inner;base+=128) {
        ulong k=base+lane*4;
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        ushort4 b0=unpack4(payload,tags,metadata,exceptions,(column*p.inner+base)/128,lane);
        ushort4 b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
        if(column+1<p.columns)b1=unpack4(payload,tags,metadata,exceptions,((column+1)*p.inner+base)/128,lane);
        if(column+2<p.columns)b2=unpack4(payload,tags,metadata,exceptions,((column+2)*p.inner+base)/128,lane);
        if(column+3<p.columns)b3=unpack4(payload,tags,metadata,exceptions,((column+3)*p.inner+base)/128,lane);
        for(uint j=0;j<4;++j) {
            float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
            sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
        }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
