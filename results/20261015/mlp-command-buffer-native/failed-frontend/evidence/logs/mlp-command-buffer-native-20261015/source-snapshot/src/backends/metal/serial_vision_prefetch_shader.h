// Generated from the audited original-order component.
#pragma once
inline constexpr const char* kSerialVisionPrefetchShader=R"GARNET(
#include <metal_stdlib>
using namespace metal;
struct VisionParams {ulong rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
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
)GARNET";
