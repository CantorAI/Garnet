#include <metal_stdlib>
using namespace metal;
struct VisionParams {ulong rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
kernel void vision_softmax(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    ulong packedRow=((row/p.tokens)*p.heads+head)*p.tokens+row%p.tokens;
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]=b[packedRow*p.bStride+key]*rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)out[row*p.oStride+head*p.tokens+key]=scratch[key]/total;
}
kernel void vision_op(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],device float* out [[buffer(3)]],
    constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.outRows*p.outColumns)return;
    ulong row=id/p.outColumns,col=id%p.outColumns;
    if(p.mode==0){out[row*p.oStride+col]=a[row*p.aStride+col]+b[col];return;}
    if(p.mode==1){
        // Compute row statistics once rather than repeating both sums per column.
        if(col)return;
        float mean=0;for(ulong j=0;j<p.columns;++j)mean+=a[row*p.aStride+j];mean/=float(p.columns);
        float variance=0;for(ulong j=0;j<p.columns;++j){float x=a[row*p.aStride+j]-mean;variance+=x*x;}
        float inverse=rsqrt(variance/float(p.columns)+1e-6f);
        for(ulong j=0;j<p.columns;++j)out[row*p.oStride+j]=(a[row*p.aStride+j]-mean)*inverse*b[j]+c[j];return;
    }
    if(p.mode==2||p.mode==3){
        float x=a[row*p.aStride+col],activation;
        // Avoid native tanh overflow; at |argument| >= 10 its FP32 value saturates.
        if(p.mode==2)activation=tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f));
        else {
            // Abramowitz-Stegun erf approximation (absolute error < 1.5e-7).
            float z=abs(x)*0.7071067811865475f,t=1.0f/(1.0f+0.3275911f*z);
            float v=1.0f-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-z*z);
            activation=x<0?-v:v;
        }
        out[row*p.oStride+col]=0.5f*x*(1.0f+activation);return;
    }
    if(p.mode==4){out[row*p.oStride+col]=a[(row*p.unit+col/p.columns)*p.aStride+col%p.columns];return;}
    if(p.mode==9){out[row*p.oStride+col]=a[(row/p.tokens)*p.aStride+(row%p.tokens)*p.outColumns+col];return;}
    if(p.mode==5){
        ulong token=row%p.tokens;float position=0;
        for(ulong i=0;i<4;++i)position+=b[ulong(c[token*p.cStride+i])*p.bStride+col]*c[token*p.cStride+4+i];
        out[row*p.oStride+col]=a[row*p.aStride+col]+position;return;
    }
    ulong dim=p.hidden/p.heads;
    if(p.mode==6){
        if(col>=2*p.hidden){out[row*p.oStride+col]=a[row*p.aStride+col];return;}
        ulong d=col%dim,base=col-d,other=(d+dim/2)%dim;
        float angle=b[(row%p.tokens)*p.bStride+d];
        out[row*p.oStride+col]=a[row*p.aStride+col]*cos(angle)+(d<dim/2?-1.0f:1.0f)*a[row*p.aStride+base+other]*sin(angle);return;
    }
    if(p.mode==7){
        // One thread handles a whole softmax row; dispatch grid includes
        // columns, but only its first column performs the reduction.
        if(col%p.tokens)return;
        ulong head=col/p.tokens,batchBase=(row/p.tokens)*p.tokens;float maximum=-INFINITY;
        for(ulong key=0;key<p.tokens;++key){
            float score=0;for(ulong d=0;d<dim;++d)score+=a[row*p.aStride+head*dim+d]*a[(batchBase+key)*p.aStride+p.hidden+head*dim+d];
            score*=rsqrt(float(dim));out[row*p.oStride+head*p.tokens+key]=score;maximum=max(maximum,score);
        }
        float total=0;for(ulong key=0;key<p.tokens;++key){float v=exp(out[row*p.oStride+head*p.tokens+key]-maximum);out[row*p.oStride+head*p.tokens+key]=v;total+=v;}
        for(ulong key=0;key<p.tokens;++key)out[row*p.oStride+head*p.tokens+key]/=total;return;
    }
    if(p.mode==8){
        ulong head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=a[row*p.aStride+head*p.tokens+key]*b[(batchBase+key)*p.bStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+col]=result;
    }
}

kernel void vision_prefetch8(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong row=id/p.outColumns,col=id%p.outColumns;if(row>=p.outRows)return;
    ulong dim=p.hidden/p.heads,head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;
    float result=0;ulong key=0;
    for(;key+8<=p.tokens;key+=8) {
        float p0=a[row*p.aStride+head*p.tokens+key+0],v0=b[(batchBase+key+0)*p.bStride+2*p.hidden+head*dim+d];
        float p1=a[row*p.aStride+head*p.tokens+key+1],v1=b[(batchBase+key+1)*p.bStride+2*p.hidden+head*dim+d];
        float p2=a[row*p.aStride+head*p.tokens+key+2],v2=b[(batchBase+key+2)*p.bStride+2*p.hidden+head*dim+d];
        float p3=a[row*p.aStride+head*p.tokens+key+3],v3=b[(batchBase+key+3)*p.bStride+2*p.hidden+head*dim+d];
        float p4=a[row*p.aStride+head*p.tokens+key+4],v4=b[(batchBase+key+4)*p.bStride+2*p.hidden+head*dim+d];
        float p5=a[row*p.aStride+head*p.tokens+key+5],v5=b[(batchBase+key+5)*p.bStride+2*p.hidden+head*dim+d];
        float p6=a[row*p.aStride+head*p.tokens+key+6],v6=b[(batchBase+key+6)*p.bStride+2*p.hidden+head*dim+d];
        float p7=a[row*p.aStride+head*p.tokens+key+7],v7=b[(batchBase+key+7)*p.bStride+2*p.hidden+head*dim+d];
        result=fma(p0,v0,result);
        result=fma(p1,v1,result);
        result=fma(p2,v2,result);
        result=fma(p3,v3,result);
        result=fma(p4,v4,result);
        result=fma(p5,v5,result);
        result=fma(p6,v6,result);
        result=fma(p7,v7,result);
    }
    for(;key<p.tokens;++key)result=fma(a[row*p.aStride+head*p.tokens+key],b[(batchBase+key)*p.bStride+2*p.hidden+head*dim+d],result);
    out[row*p.oStride+col]=result;
}

kernel void vision_vector2(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong columns=p.outColumns/2,row=id/columns,col=(id%columns)*2;if(row>=p.outRows)return;
    ulong dim=p.hidden/p.heads,head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;
    float2 result=0;ulong key=0;
    for(;key+8<=p.tokens;key+=8) {
        float p0=a[row*p.aStride+head*p.tokens+key+0];float2 v0=*(device const float2*)(b+(batchBase+key+0)*p.bStride+2*p.hidden+head*dim+d);
        float p1=a[row*p.aStride+head*p.tokens+key+1];float2 v1=*(device const float2*)(b+(batchBase+key+1)*p.bStride+2*p.hidden+head*dim+d);
        float p2=a[row*p.aStride+head*p.tokens+key+2];float2 v2=*(device const float2*)(b+(batchBase+key+2)*p.bStride+2*p.hidden+head*dim+d);
        float p3=a[row*p.aStride+head*p.tokens+key+3];float2 v3=*(device const float2*)(b+(batchBase+key+3)*p.bStride+2*p.hidden+head*dim+d);
        float p4=a[row*p.aStride+head*p.tokens+key+4];float2 v4=*(device const float2*)(b+(batchBase+key+4)*p.bStride+2*p.hidden+head*dim+d);
        float p5=a[row*p.aStride+head*p.tokens+key+5];float2 v5=*(device const float2*)(b+(batchBase+key+5)*p.bStride+2*p.hidden+head*dim+d);
        float p6=a[row*p.aStride+head*p.tokens+key+6];float2 v6=*(device const float2*)(b+(batchBase+key+6)*p.bStride+2*p.hidden+head*dim+d);
        float p7=a[row*p.aStride+head*p.tokens+key+7];float2 v7=*(device const float2*)(b+(batchBase+key+7)*p.bStride+2*p.hidden+head*dim+d);
        result=fma(float2(p0),v0,result);
        result=fma(float2(p1),v1,result);
        result=fma(float2(p2),v2,result);
        result=fma(float2(p3),v3,result);
        result=fma(float2(p4),v4,result);
        result=fma(float2(p5),v5,result);
        result=fma(float2(p6),v6,result);
        result=fma(float2(p7),v7,result);
    }
    for(;key<p.tokens;++key)result=fma(float2(a[row*p.aStride+head*p.tokens+key]),*(device const float2*)(b+(batchBase+key)*p.bStride+2*p.hidden+head*dim+d),result);
    *(device float2*)(out+row*p.oStride+col)=result;
}

kernel void vision_vector4(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong columns=p.outColumns/4,row=id/columns,col=(id%columns)*4;if(row>=p.outRows)return;
    ulong dim=p.hidden/p.heads,head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;
    float4 result=0;ulong key=0;
    for(;key+8<=p.tokens;key+=8) {
        float p0=a[row*p.aStride+head*p.tokens+key+0];float4 v0=*(device const float4*)(b+(batchBase+key+0)*p.bStride+2*p.hidden+head*dim+d);
        float p1=a[row*p.aStride+head*p.tokens+key+1];float4 v1=*(device const float4*)(b+(batchBase+key+1)*p.bStride+2*p.hidden+head*dim+d);
        float p2=a[row*p.aStride+head*p.tokens+key+2];float4 v2=*(device const float4*)(b+(batchBase+key+2)*p.bStride+2*p.hidden+head*dim+d);
        float p3=a[row*p.aStride+head*p.tokens+key+3];float4 v3=*(device const float4*)(b+(batchBase+key+3)*p.bStride+2*p.hidden+head*dim+d);
        float p4=a[row*p.aStride+head*p.tokens+key+4];float4 v4=*(device const float4*)(b+(batchBase+key+4)*p.bStride+2*p.hidden+head*dim+d);
        float p5=a[row*p.aStride+head*p.tokens+key+5];float4 v5=*(device const float4*)(b+(batchBase+key+5)*p.bStride+2*p.hidden+head*dim+d);
        float p6=a[row*p.aStride+head*p.tokens+key+6];float4 v6=*(device const float4*)(b+(batchBase+key+6)*p.bStride+2*p.hidden+head*dim+d);
        float p7=a[row*p.aStride+head*p.tokens+key+7];float4 v7=*(device const float4*)(b+(batchBase+key+7)*p.bStride+2*p.hidden+head*dim+d);
        result=fma(float4(p0),v0,result);
        result=fma(float4(p1),v1,result);
        result=fma(float4(p2),v2,result);
        result=fma(float4(p3),v3,result);
        result=fma(float4(p4),v4,result);
        result=fma(float4(p5),v5,result);
        result=fma(float4(p6),v6,result);
        result=fma(float4(p7),v7,result);
    }
    for(;key<p.tokens;++key)result=fma(float4(a[row*p.aStride+head*p.tokens+key]),*(device const float4*)(b+(batchBase+key)*p.bStride+2*p.hidden+head*dim+d),result);
    *(device float4*)(out+row*p.oStride+col)=result;
}
