// GPU self-verified prefix lookup versus extracted general cache-slot scan.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct Params {uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
struct Matrix {id<MTLBuffer> data;uint64_t stride,rows,columns;};
static Matrix matrix(id<MTLDevice> device,uint64_t rows,uint64_t columns) {
    uint64_t stride=((columns+3)/4)*4+4;auto data=[device newBufferWithLength:rows*stride*4 options:MTLResourceStorageModeShared];
    if(!data)throw std::runtime_error("Allocation failed");std::fill_n((float*)data.contents,rows*stride,99.5f);return {data,stride,rows,columns};
}
int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc!=2)throw std::runtime_error("Expected shader");auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
    if(!device||!queue)throw std::runtime_error("Metal unavailable");
    auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    auto library=[device newLibraryWithSource:source options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLComputePipelineState> kernels[2];NSArray* names=@[@"attention_chunk",@"attention_chunk_prefix"];
    for(int i=0;i<2;++i){kernels[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:names[i]] error:&error];if(!kernels[i])throw std::runtime_error("Pipeline creation failed");}
    struct Layout {uint64_t heads,kv,dim;};
    for(uint64_t batch:{1,2})for(uint64_t chunk:{16,32,80})for(auto shape:std::vector<Layout>{{16,8,128},{14,2,64},{8,2,256}}) {
        @autoreleasepool {
        uint64_t heads=shape.heads,kv=shape.kv,dim=shape.dim,capacity=256,rows=batch*chunk;
        Matrix inputs[]={matrix(device,rows,heads*dim),matrix(device,rows,kv*dim),matrix(device,rows,kv*dim),matrix(device,batch,kv*capacity*dim),matrix(device,batch,kv*capacity*dim),matrix(device,batch,chunk),matrix(device,batch,chunk),matrix(device,rows,heads*dim)};
        auto out=matrix(device,rows,heads*dim);Params p{batch,chunk,heads,kv,dim,capacity,inputs[0].stride,inputs[1].stride,inputs[2].stride,inputs[3].stride,inputs[4].stride,inputs[5].stride,inputs[6].stride,inputs[7].stride,out.stride,0};
        auto run=[&](bool prefix) {
            auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:kernels[prefix]];
            for(int i=0;i<8;++i)[encoder setBuffer:inputs[i].data offset:0 atIndex:i];[encoder setBuffer:out.data offset:0 atIndex:8];[encoder setBytes:&p length:sizeof(p) atIndex:9];
            [encoder setThreadgroupMemoryLength:capacity*4 atIndex:0];[encoder setThreadgroupMemoryLength:capacity*4 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(rows*heads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[cb commit];[cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("GPU dispatch failed");if(!(cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0))throw std::runtime_error("Timestamp unavailable");
            return std::vector<double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
        };
        auto controls=[&](int pattern) {
            auto slots=(float*)inputs[5].data.contents,active=(float*)inputs[6].data.contents;
            for(uint64_t b=0;b<batch;++b)for(uint64_t t=0;t<chunk;++t) {
                float first=pattern==0?0:17+float(b)*3;float position=first+float(t),flag=1;
                if(pattern==2&&t>=chunk-3)flag=0;
                if(pattern==3)position=first+float(t/2);
                if(pattern==4)position=first+float(t*2);
                if(pattern==5&&(t==3||t==7))flag=0;
                if(pattern==6&&t<5)flag=0;
                slots[b*inputs[5].stride+t]=position;active[b*inputs[6].stride+t]=flag;
            }
        };
        bool exact=true;double worst=0,peak=0;int checks=0;
        for(int generation=0;generation<2;++generation) {
            for(int i:{0,1,2,3,4,7}) {
                auto data=(float*)inputs[i].data.contents;
                for(uint64_t r=0;r<inputs[i].rows;++r)for(uint64_t c=0;c<inputs[i].columns;++c)data[r*inputs[i].stride+c]=std::sin(float(r*inputs[i].columns+c)*(.011f+i*.003f)+generation+i)*.3f;
            }
            for(int pattern=0;pattern<7;++pattern)for(uint64_t gated:{0,1}) {
                p.gated=gated;controls(pattern);run(false);std::vector<float> baseline(rows*out.stride);std::memcpy(baseline.data(),out.data.contents,baseline.size()*4);
                std::fill_n((float*)out.data.contents,baseline.size(),99.5f);run(true);auto actual=(float*)out.data.contents;
                exact&=std::memcmp(actual,baseline.data(),baseline.size()*4)==0;++checks;
                auto q=(float*)inputs[0].data.contents,k=(float*)inputs[1].data.contents,v=(float*)inputs[2].data.contents,kc=(float*)inputs[3].data.contents,vc=(float*)inputs[4].data.contents,slots=(float*)inputs[5].data.contents,active=(float*)inputs[6].data.contents,gate=(float*)inputs[7].data.contents;
                for(uint64_t row=0;row<batch;++row)for(uint64_t token=0;token<chunk;++token) {
                    uint64_t flat=row*chunk+token,visible=std::min(capacity,uint64_t(slots[row*p.sStride+token])+1);
                    if(!active[row*p.aStride+token]){for(uint64_t d=0;d<heads*dim;++d)if(actual[flat*out.stride+d]!=0)throw std::runtime_error("Inactive query modified");}
                    else for(uint64_t h=0;h<heads;++h) {
                        uint64_t kh=h/(heads/kv);std::vector<int> selected(visible,-1);std::vector<double> weights(visible);double maximum=-INFINITY,total=0;
                        for(uint64_t s=0;s<visible;++s) {
                            for(uint64_t t=0;t<=token;++t)if(active[row*p.aStride+t]&&slots[row*p.sStride+t]==float(s))selected[s]=int(t);
                            double score=0;for(uint64_t d=0;d<dim;++d) {double key=selected[s]<0?kc[row*p.kcStride+(kh*capacity+s)*dim+d]:k[(row*chunk+selected[s])*p.kStride+kh*dim+d];score+=q[flat*p.qStride+h*dim+d]*key;}
                            weights[s]=score/std::sqrt(double(dim));maximum=std::max(maximum,weights[s]);
                        }
                        for(auto& w:weights){w=std::exp(w-maximum);total+=w;}
                        for(uint64_t d=0;d<dim;++d) {
                            double expected=0;for(uint64_t s=0;s<visible;++s){double value=selected[s]<0?vc[row*p.vcStride+(kh*capacity+s)*dim+d]:v[(row*chunk+selected[s])*p.vStride+kh*dim+d];expected+=weights[s]/total*value;}
                            if(gated)expected/=1+std::exp(-double(gate[flat*p.gStride+h*dim+d]));
                            double value=actual[flat*out.stride+h*dim+d];if(!std::isfinite(value))throw std::runtime_error("Nonfinite result");
                            worst=std::max(worst,std::abs(value-expected)/(5e-6+5e-5*std::abs(expected)));peak=std::max(peak,std::abs(value-double(baseline[flat*out.stride+h*dim+d])));
                        }
                    }
                    for(uint64_t d=heads*dim;d<out.stride;++d)if(actual[flat*out.stride+d]!=99.5f)throw std::runtime_error("Padding modified");
                }
            }
        }
        if(!exact||worst>1)throw std::runtime_error("Reference failed");
        printf("{\"kind\":\"check\",\"batch\":%llu,\"chunk\":%llu,\"heads\":%llu,\"kv_heads\":%llu,\"dim\":%llu,\"cases\":%d,\"bits_exact\":true,\"fp64_worst_ratio\":%.9g,\"max_abs_error\":%.9g}\n",batch,chunk,heads,kv,dim,checks,worst,peak);fflush(stdout);
        controls(0);p.gated=dim==256;
        for(int round=0;round<2;++round)for(bool prefix:{false,true,true,false}) {
            for(int warm=0;warm<3;++warm)run(prefix);std::vector<double> gpu,wall;
            for(int trial=0;trial<12;++trial){auto t=run(prefix);gpu.push_back(t[0]);wall.push_back(t[1]);}
            std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
            printf("{\"kind\":\"timing\",\"batch\":%llu,\"chunk\":%llu,\"heads\":%llu,\"kv_heads\":%llu,\"dim\":%llu,\"round\":%d,\"policy\":\"%s\",\"samples\":12,\"gpu_ms_median\":%.9g,\"wall_ms_median\":%.9g}\n",batch,chunk,heads,kv,dim,round,prefix?"prefix":"control",(gpu[5]+gpu[6])/2,(wall[5]+wall[6])/2);fflush(stdout);
        }
        }
    }
    } catch(const std::exception& e){fprintf(stderr,"ERROR: %s\n",e.what());return 1;}return 0;}}
