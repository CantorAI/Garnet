#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
struct HalfRescaleParams {ulong rows,columns,iStride,oStride;};
kernel void half_rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],
    device float* output [[buffer(2)]],constant HalfRescaleParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,column=id%p.columns;
    output[row*p.oStride+column]=ldexp(input[row*p.iStride+column],-scales[column]);
}
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
kernel void bf16_decode_small(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*4+simd*2;
    if(column>=p.columns)return;
    float2 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float2 w=as_type<float2>(uint2(b0[j],b1[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float2 w=0.0f;
        w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float2(simd_sum(sum0.x),simd_sum(sum0.y));
    sum1=float2(simd_sum(sum1.x),simd_sum(sum1.y));
    if(lane==0)for(uint j=0;j<2&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
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
// Private greedy component only: derived INT8 screening index, original BF16 refinement.
// Screening is not a public complete-logit operation or a quantized model replacement.
kernel void prepare_head_bounds(device const float* input [[buffer(0)]],device const float* seed [[buffer(1)]],
    device float* stats [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong stride=p.inner/128+2,seedRows=min(ulong(1024),p.columns);
    threadgroup float partial[4];threadgroup uint invalid[4];float best=-INFINITY;
    for(ulong j=tid;j<seedRows;j+=128)best=max(best,seed[row*seedRows+j]);
    best=simd_max(best);if(lane==0)partial[simd]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)stats[row*stride]=max(max(partial[0],partial[1]),max(partial[2],partial[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float square=0;uint bad=0;
    for(ulong j=tid;j<p.inner;j+=128){float x=input[row*p.aStride+j];bad|=uint(!isfinite(x)||abs(x)>1e6f);square=fma(x,x,square);}
    bad=simd_max(bad);if(lane==0)invalid[simd]=bad;
    square=simd_sum(square);if(lane==0)partial[simd]=square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){
        float norm=sqrt(partial[0]+partial[1]+partial[2]+partial[3])*1.01f+1e-20f;
        stats[row*stride+1]=norm;
        stats[row*stride+2]=float(!(invalid[0]|invalid[1]|invalid[2]|invalid[3])&&isfinite(norm)&&norm>=1e-10f&&norm<=1e10f);
    }
}
kernel void screened_head(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const float* stats [[buffer(4)]],
    device const float4* metadata [[buffer(5)]],device uint* visits [[buffer(6)]],device const float* seed [[buffer(7)]],
    device const float* approximate [[buffer(8)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;if(column>=p.columns)return;
    ulong blocks=p.inner/128,stride=blocks+2,seedRows=min(ulong(1024),p.columns);
    if(column<seedRows){if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){
        out[column+j]=seed[column+j];if(p.rows==2)out[p.oStride+column+j]=seed[seedRows+column+j];
        visits[column+j]=0;visits[p.columns+column+j]=p.rows==2?3:1;
    }return;}
    bool4 valid=bool4(true,column+1<p.columns,column+2<p.columns,column+3<p.columns);
    float4 sum0=float4(approximate[column],valid.y?approximate[column+1]:0,valid.z?approximate[column+2]:0,valid.w?approximate[column+3]:0);
    float4 sum1=p.rows==2?float4(approximate[p.columns+column],valid.y?approximate[p.columns+column+1]:0,valid.z?approximate[p.columns+column+2]:0,valid.w?approximate[p.columns+column+3]:0):float4(0);
    bool4 keep0=valid,keep1=valid&bool4(p.rows==2);
    for(uint j=0;j<4;++j)if(valid[j]){
        float4 m=metadata[column+j];
        // Error norm plus deliberately inflated FP32-dot/norm allowances.
        // This experiment validates every actual rejected row against full GPU logits;
        // it does not certify these bounds for all future input magnitudes.
        float error=1.01f*m.x+0.001f*(m.y+m.z);
        keep0[j]=stats[2]==0||!(sum0[j]*m.w+stats[1]*error+1e-6f<stats[0]);
        if(p.rows==2)keep1[j]=stats[stride+2]==0||!(sum1[j]*m.w+stats[stride+1]*error+1e-6f<stats[stride]);
    }
    bool4 needed=keep0|keep1;sum0=0;sum1=0;
    if(any(needed))for(ulong k=ulong(lane)*4;k<p.inner;k+=128){
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        ushort4 b0=ushort4(0),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
        if(needed.x)b0=*(device const ushort4*)(weight+column*p.inner+k);
        if(needed.y)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
        if(needed.z)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
        if(needed.w)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
        for(uint j=0;j<4;++j){float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}
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

// Private greedy component only: derived INT8 screening index, original BF16 refinement.
// Screening is not a public complete-logit operation or a quantized model replacement.
kernel void control_prepare_head_bounds(device const float* input [[buffer(0)]],device const float* seed [[buffer(1)]],
    device float* stats [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong stride=p.inner/128+2,seedRows=min(ulong(1024),p.columns);
    threadgroup float partial[4];threadgroup uint invalid[4];float best=-INFINITY;
    for(ulong j=tid;j<seedRows;j+=128)best=max(best,seed[row*seedRows+j]);
    best=simd_max(best);if(lane==0)partial[simd]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)stats[row*stride]=max(max(partial[0],partial[1]),max(partial[2],partial[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float square=0;uint bad=0;
    for(ulong j=tid;j<p.inner;j+=128){float x=input[row*p.aStride+j];bad|=uint(!isfinite(x)||abs(x)>1e6f);square=fma(x,x,square);}
    bad=simd_max(bad);if(lane==0)invalid[simd]=bad;
    square=simd_sum(square);if(lane==0)partial[simd]=square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){
        float norm=sqrt(partial[0]+partial[1]+partial[2]+partial[3])*1.01f+1e-20f;
        stats[row*stride+1]=norm;
        stats[row*stride+2]=float(!(invalid[0]|invalid[1]|invalid[2]|invalid[3])&&isfinite(norm)&&norm>=1e-10f&&norm<=1e10f);
    }
}
kernel void control_screened_head(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const float* stats [[buffer(4)]],
    device const float4* metadata [[buffer(5)]],device uint* visits [[buffer(6)]],device const float* seed [[buffer(7)]],
    device const char* approximate [[buffer(8)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;if(column>=p.columns)return;
    ulong blocks=p.inner/128,stride=blocks+2,seedRows=min(ulong(1024),p.columns);
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
        for(uint j=0;j<4;++j){float4 w=float4(b0[j],b1[j],b2[j],b3[j]);sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}
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
        keep0[j]=stats[2]==0||!(sum0[j]*m.w+stats[1]*error+1e-6f<stats[0]);
        if(p.rows==2)keep1[j]=stats[stride+2]==0||!(sum1[j]*m.w+stats[stride+1]*error+1e-6f<stats[stride]);
    }
    bool4 needed=keep0|keep1;sum0=0;sum1=0;
    if(any(needed))for(ulong k=ulong(lane)*4;k<p.inner;k+=128){
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        ushort4 b0=ushort4(0),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
        if(needed.x)b0=*(device const ushort4*)(weight+column*p.inner+k);
        if(needed.y)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
        if(needed.z)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
        if(needed.w)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
        for(uint j=0;j<4;++j){float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}
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
