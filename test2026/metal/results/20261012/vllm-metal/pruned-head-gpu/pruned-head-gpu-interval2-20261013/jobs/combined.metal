#define HEAD_BOUND_INTERVAL 2
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
struct ArgmaxParams {ulong rows,columns,iStride,oStride;};
kernel void argmax_rows(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float maximum[4];threadgroup uint indices[4];threadgroup uint invalid[4];
    float best=-INFINITY;uint index=0xffffffff,bad=0;
    for(ulong j=tid;j<p.columns;j+=128) {
        float value=input[row*p.iStride+j];bad|=!isfinite(value);
        if(value>best||(value==best&&j<index)){best=value;index=uint(j);}
    }
    float groupBest=simd_max(best);uint groupIndex=simd_min(best==groupBest?index:0xffffffff);
    uint groupBad=simd_max(bad);
    if(lane==0){maximum[simd]=groupBest;indices[simd]=groupIndex;invalid[simd]=groupBad;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best=lane<4?maximum[lane]:-INFINITY;index=lane<4?indices[lane]:0xffffffff;bad=lane<4?invalid[lane]:0;
    groupBest=simd_max(best);groupIndex=simd_min(best==groupBest?index:0xffffffff);groupBad=simd_max(bad);
    if(tid==0){output[row*p.oStride]=float(groupIndex);output[row*p.oStride+1]=float(groupBad!=0);}
}
// Isolated greedy-head experiment. The complete public logit API is unchanged.
// Current native bf16_decode and argmax_rows kernels precede this source.
kernel void prepare_head_bounds(device const float* input [[buffer(0)]],device const float* seed [[buffer(1)]],
    device float* stats [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong blocks=p.inner/128,seedRows=min(ulong(8192),p.columns),stride=blocks+2;
    threadgroup float partial[4],norm[64];
    float best=-INFINITY;
    for(ulong i=tid;i<seedRows;i+=128)best=max(best,seed[row*seedRows+i]);
    best=simd_max(best);if(lane==0)partial[simd]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)stats[row*stride]=max(max(partial[0],partial[1]),max(partial[2],partial[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong block=0;block<blocks;++block) {
        float x=input[row*p.aStride+block*128+tid];float sum=simd_sum(x*x);
        if(lane==0)partial[simd]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0)norm[block]=partial[0]+partial[1]+partial[2]+partial[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) {
        float tail=0;stats[row*stride+blocks+1]=0;
        for(long block=long(blocks)-1;block>=0;--block){tail+=norm[block];stats[row*stride+1+block]=tail*1.001f;}
    }
}
kernel void pruned_head(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const float* stats [[buffer(4)]],
    device const float* remaining [[buffer(5)]],device uint* visits [[buffer(6)]],device const float* seed [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*4+simd;if(column>=p.columns)return;
    ulong blocks=p.inner/128,stride=blocks+2,seedRows=min(ulong(8192),p.columns);
    if(column<seedRows) {
        if(lane==0){out[column]=seed[column];if(p.rows==2)out[p.oStride+column]=seed[seedRows+column];visits[column]=0;visits[p.columns+column]=p.rows==2?3:1;}
        return;
    }
    float sum0=0,sum1=0;bool alive0=true,alive1=p.rows==2;uint evaluated=0;
    float threshold0=stats[0],threshold1=p.rows==2?stats[stride]:0;
    float fullWeight=remaining[column*(blocks+1)];
    float scaled=float(2*p.inner+64)*5.960464477539063e-8f;
    float allowance=2*scaled/(1-scaled)+0.00001f;
    float error0=allowance*sqrt(fullWeight*stats[1]);
    float error1=p.rows==2?allowance*sqrt(fullWeight*stats[stride+1]):0;
    for(ulong block=0;block<blocks&&(alive0||alive1);++block) {
        ulong k=block*128+lane*4;
        ushort4 bits=*(device const ushort4*)(weight+column*p.inner+k);
        float4 w=as_type<float4>(uint4(bits)<<16),x=*(device const float4*)(a+k);
        float4 y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        for(uint j=0;j<4;++j){if(alive0)sum0+=x[j]*w[j];if(alive1)sum1+=y[j]*w[j];}
        ++evaluated;float partial0=simd_sum(sum0),partial1=p.rows==2?simd_sum(sum1):0;
        float residual=remaining[column*(blocks+1)+block+1];
        if(alive0&&partial0+sqrt(residual*stats[block+2])*1.001f+error0<threshold0)alive0=false;
        if(alive1&&partial1+sqrt(residual*stats[stride+block+2])*1.001f+error1<threshold1)alive1=false;
    }
    float result0=simd_sum(sum0),result1=p.rows==2?simd_sum(sum1):0;
    if(lane==0) {
        // A finite sentinel lets the unchanged native argmax reject actual
        // nonfinite values. These are not public complete vocabulary logits.
        out[column]=alive0?result0:-3.402823466e+38f;
        if(p.rows==2)out[p.oStride+column]=alive1?result1:-3.402823466e+38f;
        visits[column]=evaluated;visits[p.columns+column]=uint(alive0)|(uint(alive1)<<1);
    }
}
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
