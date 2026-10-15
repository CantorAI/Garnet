// Private greedy component only: derived INT8 screening index, original BF16 refinement.
// Screening is not a public complete-logit operation or a quantized model replacement.
kernel void prepare_head_bounds(device const float* input [[buffer(0)]],device const float* seed [[buffer(1)]],
    device float* stats [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong stride=p.inner/128+2,seedRows=min(ulong(8192),p.columns);
    threadgroup float partial[4];float best=-INFINITY;
    for(ulong j=tid;j<seedRows;j+=128)best=max(best,seed[row*seedRows+j]);
    best=simd_max(best);if(lane==0)partial[simd]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)stats[row*stride]=max(max(partial[0],partial[1]),max(partial[2],partial[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float square=0;for(ulong j=tid;j<p.inner;j+=128){float x=input[row*p.aStride+j];square+=x*x;}
    square=simd_sum(square);if(lane==0)partial[simd]=square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)stats[row*stride+1]=sqrt(partial[0]+partial[1]+partial[2]+partial[3])*1.01f+1e-20f;
}
kernel void screened_head(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const float* stats [[buffer(4)]],
    device const float4* metadata [[buffer(5)]],device uint* visits [[buffer(6)]],device const float* seed [[buffer(7)]],
    device const char* approximate [[buffer(8)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;if(column>=p.columns)return;
    ulong blocks=p.inner/128,stride=blocks+2,seedRows=min(ulong(8192),p.columns);
    if(column<seedRows){if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){
        out[column+j]=seed[column+j];if(p.rows==2)out[p.oStride+column+j]=seed[seedRows+column+j];
        visits[column+j]=0;visits[p.columns+column+j]=p.rows==2?3:1;
    }return;}
    bool4 valid=bool4(true,column+1<p.columns,column+2<p.columns,column+3<p.columns);
    float4 sum0=0,sum1=0;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128){
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        char4 b0=*(device const char4*)(approximate+column*p.inner+k),b1=char4(0),b2=char4(0),b3=char4(0);
        if(valid.y)b1=*(device const char4*)(approximate+(column+1)*p.inner+k);
        if(valid.z)b2=*(device const char4*)(approximate+(column+2)*p.inner+k);
        if(valid.w)b3=*(device const char4*)(approximate+(column+3)*p.inner+k);
        for(uint j=0;j<4;++j){float4 w=float4(b0[j],b1[j],b2[j],b3[j]);sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;}
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    if(p.rows==2)sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    bool4 keep0=valid,keep1=valid&bool4(p.rows==2);
    for(uint j=0;j<4;++j)if(valid[j]){
        float4 m=metadata[column+j];
        // Error norm plus deliberately inflated FP32-dot/norm allowances.
        // This experiment validates every actual rejected row against full GPU logits;
        // it does not certify these bounds for all future input magnitudes.
        float error=1.01f*m.x+0.001f*(m.y+m.z);
        keep0[j]=!(sum0[j]*m.w+stats[1]*error+1e-6f<stats[0]);
        if(p.rows==2)keep1[j]=!(sum1[j]*m.w+stats[stride+1]*error+1e-6f<stats[stride]);
    }
    bool4 needed=keep0|keep1;sum0=0;sum1=0;
    if(any(needed))for(ulong k=ulong(lane)*4;k<p.inner;k+=128){
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        ushort4 b0=ushort4(0),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
        if(needed.x)b0=*(device const ushort4*)(weight+column*p.inner+k);
        if(needed.y)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
        if(needed.z)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
        if(needed.w)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
        for(uint j=0;j<4;++j){float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;}
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    if(p.rows==2)sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){
        out[column+j]=keep0[j]?sum0[j]:-3.402823466e+38f;
        if(p.rows==2)out[p.oStride+column+j]=keep1[j]?sum1[j]:-3.402823466e+38f;
        visits[column+j]=needed[j]?uint(blocks):0;
        visits[p.columns+column+j]=uint(keep0[j])|(uint(keep1[j])<<1);
    }
}
