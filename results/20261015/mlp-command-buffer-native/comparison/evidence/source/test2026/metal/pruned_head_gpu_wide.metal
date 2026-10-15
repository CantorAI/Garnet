// Four independent output columns amortize loads and bound scheduling.
// No public full-logit operation is replaced by this experiment.
#ifndef HEAD_BOUND_INTERVAL
#define HEAD_BOUND_INTERVAL 1
#endif
kernel void pruned_head_wide(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const float* stats [[buffer(4)]],
    device const float* remaining [[buffer(5)]],device uint* visits [[buffer(6)]],device const float* seed [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;if(column>=p.columns)return;
    ulong blocks=p.inner/128,stride=blocks+2,seedRows=min(ulong(8192),p.columns);
    if(column<seedRows) {
        if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=seed[column+j];if(p.rows==2)out[p.oStride+column+j]=seed[seedRows+column+j];visits[column+j]=0;visits[p.columns+column+j]=p.rows==2?3:1;}
        return;
    }
    bool4 valid=bool4(true,column+1<p.columns,column+2<p.columns,column+3<p.columns);
    bool4 alive0=valid,alive1=valid&bool4(p.rows==2);float4 sum0=0,sum1=0;uint4 evaluated=0;
    float threshold0=stats[0],threshold1=p.rows==2?stats[stride]:0;
    float4 fullWeight=0;
    for(uint j=0;j<4;++j)if(valid[j])fullWeight[j]=remaining[(column+j)*(blocks+1)];
    float scaled=float(2*p.inner+64)*5.960464477539063e-8f,allowance=2*scaled/(1-scaled)+0.00001f;
    float4 error0=allowance*sqrt(fullWeight*stats[1]);
    float4 error1=p.rows==2?allowance*sqrt(fullWeight*stats[stride+1]):float4(0);
    for(ulong block=0;block<blocks&&any(alive0|alive1);++block) {
        bool4 needed=alive0|alive1;ulong k=block*128+lane*4;
        ushort4 b0=ushort4(0),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
        if(needed.x)b0=*(device const ushort4*)(weight+column*p.inner+k);
        if(needed.y)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
        if(needed.z)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
        if(needed.w)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        for(uint j=0;j<4;++j) {
            float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
            sum0+=x[j]*select(float4(0),w,alive0);
            if(p.rows==2)sum1+=y[j]*select(float4(0),w,alive1);
        }
        evaluated+=uint4(needed);
        // A completed row needs no remaining-work bound. Check cadence is
        // experimental: fewer reductions trade against later pruning.
        if(block+1==blocks||(block+1)%HEAD_BOUND_INTERVAL!=0)continue;
        float4 partial0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
        float4 partial1=p.rows==2?float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w)):float4(0);
        float4 residual=0;for(uint j=0;j<4;++j)if(valid[j])residual[j]=remaining[(column+j)*(blocks+1)+block+1];
        alive0=alive0&!(partial0+sqrt(residual*stats[block+2])*1.001f+error0<threshold0);
        if(p.rows==2)alive1=alive1&!(partial1+sqrt(residual*stats[stride+block+2])*1.001f+error1<threshold1);
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    if(p.rows==2)sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j) {
        out[column+j]=alive0[j]?sum0[j]:-3.402823466e+38f;
        if(p.rows==2)out[p.oStride+column+j]=alive1[j]?sum1[j]:-3.402823466e+38f;
        visits[column+j]=evaluated[j];visits[p.columns+column+j]=uint(alive0[j])|(uint(alive1[j])<<1);
    }
}
