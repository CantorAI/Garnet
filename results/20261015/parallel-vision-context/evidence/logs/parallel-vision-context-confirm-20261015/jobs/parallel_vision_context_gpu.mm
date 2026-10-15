// Selected softmax + value contraction versus one FP32 threadgroup kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct Params {uint64_t rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
struct Matrix {id<MTLBuffer> data;uint64_t stride,rows,columns;};
static Matrix matrix(id<MTLDevice> d,uint64_t rows,uint64_t columns) {
    auto bytes=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
    auto data=[d newBufferWithLength:rows*bytes options:MTLResourceStorageModeShared];
    if(!data)throw std::runtime_error("Allocation failed");std::fill_n((float*)data.contents,rows*bytes/4,99.5f);
    return {data,bytes/4,rows,columns};
}
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected extracted shader");
    auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];NSError* error=nil;
    if(!d||!queue)throw std::runtime_error("Metal unavailable");
    auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    auto library=[d newLibraryWithSource:source options:nil error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLComputePipelineState> kernels[2];
    NSArray* names=@[@"vision_softmax_context",@"vision_parallel_context"];
    for(int i=0;i<2;++i){kernels[i]=[d newComputePipelineStateWithFunction:[library newFunctionWithName:names[i]] error:&error];if(!kernels[i])throw std::runtime_error("Pipeline failed");}
    for(uint64_t batch:{1,2})for(uint64_t tokens:{4,24,216})for(uint64_t heads:{2,12})for(uint64_t dim:{8,64}) {@autoreleasepool {
        auto rows=batch*tokens,hidden=heads*dim;
        auto qkv=matrix(d,rows,hidden*3),scores=matrix(d,rows*heads,tokens),prob=matrix(d,rows,heads*tokens),out=matrix(d,rows,hidden);
        auto encode=[&](id<MTLCommandBuffer> cb,int mode) {
            auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:kernels[mode]];
            Params p{rows,hidden*3,rows,hidden,qkv.stride,scores.stride,1,out.stride,tokens,hidden,heads,4,10};
            [enc setBuffer:qkv.data offset:0 atIndex:0];[enc setBuffer:scores.data offset:0 atIndex:1];
            [enc setBuffer:qkv.data offset:0 atIndex:2];[enc setBuffer:out.data offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            [enc setThreadgroupMemoryLength:(tokens*4+15)&~uint64_t(15) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(rows*heads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [enc endEncoding];
        };
        auto run=[&](bool fused) {
            auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];
            encode(cb,fused?1:0);
            [cb commit];[cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("GPU dispatch failed");
            if(!(cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0))throw std::runtime_error("GPU timestamp unavailable");
            return std::vector<double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
        };
        bool exact=true;double peak=0,worst=0,controlWorst=0;
        for(int generation=0;generation<4;++generation) {
            auto a=(float*)qkv.data.contents,b=(float*)scores.data.contents;
            for(uint64_t r=0;r<rows;++r)for(uint64_t c=0;c<hidden*3;++c)a[r*qkv.stride+c]=std::sin(float(r*hidden*3+c)*.013f+generation)*.3f;
            for(uint64_t r=0;r<rows*heads;++r)for(uint64_t k=0;k<tokens;++k)b[r*scores.stride+k]=std::cos(float(r*tokens+k)*.017f+generation)*float(generation==3?80:4);
            run(false);std::vector<float> baseline(rows*out.stride);std::memcpy(baseline.data(),out.data.contents,baseline.size()*4);
            std::fill_n((float*)out.data.contents,baseline.size(),99.5f);run(true);
            exact&=std::memcmp(baseline.data(),out.data.contents,baseline.size()*4)==0;
            auto actual=(float*)out.data.contents;
            for(uint64_t r=0;r<rows;++r)for(uint64_t h=0;h<heads;++h) {
                auto packed=((r/tokens)*heads+h)*tokens+r%tokens;double maximum=-INFINITY,total=0;
                for(uint64_t k=0;k<tokens;++k)maximum=std::max(maximum,double(b[packed*scores.stride+k])/std::sqrt(double(dim)));
                std::vector<double> weights(tokens);
                for(uint64_t k=0;k<tokens;++k){weights[k]=std::exp(double(b[packed*scores.stride+k])/std::sqrt(double(dim))-maximum);total+=weights[k];}
                for(uint64_t c=0;c<dim;++c) {
                    double expected=0;for(uint64_t k=0;k<tokens;++k)expected+=weights[k]/total*a[((r/tokens)*tokens+k)*qkv.stride+2*hidden+h*dim+c];
                    auto index=r*out.stride+h*dim+c;double value=actual[index];
                    if(!std::isfinite(value))throw std::runtime_error("Nonfinite output");
                    peak=std::max(peak,std::abs(value-double(baseline[index])));
                    if(std::abs(value-double(baseline[index]))>5e-6+5e-5*std::abs(double(baseline[index])))throw std::runtime_error("Original candidate/control envelope exceeded");
                    controlWorst=std::max(controlWorst,std::abs(double(baseline[index])-expected)/(5e-6+5e-5*std::abs(expected)));
                    worst=std::max(worst,std::abs(value-expected)/(5e-6+5e-5*std::abs(expected)));
                }
                for(uint64_t c=hidden;c<out.stride;++c)if(actual[r*out.stride+c]!=99.5f)throw std::runtime_error("Padding modified");
            }
        }
        if(worst>1||controlWorst>1)throw std::runtime_error("Correctness failed");
        printf("{\"kind\":\"check\",\"batch\":%llu,\"tokens\":%llu,\"heads\":%llu,\"dim\":%llu,\"generations\":4,\"bits_exact\":%s,\"control_fp64_worst_ratio\":%.9g,\"fp64_worst_ratio\":%.9g,\"max_abs_error\":%.9g}\n",batch,tokens,heads,dim,exact?"true":"false",controlWorst,worst,peak);fflush(stdout);
        auto commonWarm=std::chrono::steady_clock::now();size_t warmEpoch=0;
        do{run((warmEpoch++%2)!=0);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-commonWarm).count()<.25);
        for(int round=0;round<2;++round)for(bool fused:{false,true,true,false}) {
            for(int warm=0;warm<3;++warm)run(fused);std::vector<double> gpu,wall;
            for(int trial=0;trial<20;++trial){auto time=run(fused);gpu.push_back(time[0]);wall.push_back(time[1]);}
            auto rawGpu=gpu,rawWall=wall;
            std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
            printf("{\"kind\":\"samples\",\"batch\":%llu,\"tokens\":%llu,\"heads\":%llu,\"dim\":%llu,\"round\":%d,\"candidate\":%s,\"gpu_ms\":[",batch,tokens,heads,dim,round,fused?"true":"false");
            for(size_t i=0;i<rawGpu.size();++i)printf("%s%.12g",i?",":"",rawGpu[i]);printf("],\"wall_ms\":[");
            for(size_t i=0;i<rawWall.size();++i)printf("%s%.12g",i?",":"",rawWall[i]);puts("]}");
            printf("{\"kind\":\"timing\",\"batch\":%llu,\"tokens\":%llu,\"heads\":%llu,\"dim\":%llu,\"round\":%d,\"policy\":\"%s\",\"samples\":20,\"gpu_ms_median\":%.9g,\"wall_ms_median\":%.9g}\n",batch,tokens,heads,dim,round,fused?"fused":"control",(gpu[9]+gpu[10])/2,(wall[9]+wall[10])/2);fflush(stdout);
        }
    }}} catch(const std::exception& e){fprintf(stderr,"ERROR: %s\n",e.what());return 1;}return 0;}}
