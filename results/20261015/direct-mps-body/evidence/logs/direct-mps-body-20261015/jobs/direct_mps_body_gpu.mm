// Component experiment through the existing Garnet context; no serving route.
#import <Foundation/Foundation.h>
#include "metal_context.h"
namespace Garnet {extern bool gDirectMPS;}
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"JSON serialization failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t v){uint32_t b=uint32_t(v)<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"Expected immutable asset folder");NSString* folder=[NSString stringWithUTF8String:argv[1]];
    auto json=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(json!=nil,"Manifest missing");
    NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:json options:0 error:nil];need(manifest!=nil,"Bad manifest");
    Garnet::MetalContext gpu;
    for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
        NSString* name=item[@"name"];size_t n=[item[@"rows"] unsignedLongLongValue],k=[item[@"inner"] unsignedLongLongValue];
        need(n>0&&k>0&&n*k==[item[@"count"] unsignedLongLongValue],"Bad matrix dimensions");
        auto data=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:name] stringByAppendingPathComponent:@"raw.bin"]];
        need(data.length==n*k*2,"Raw learned matrix size mismatch");std::vector<uint16_t> bits(n*k);std::memcpy(bits.data(),data.bytes,data.length);
        auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        if(!weight.GlobalHalfReady()){emit(@{@"kind":@"skipped",@"matrix":name,@"reason":@"No admitted lossless common-scale half representation"});continue;}
        for(size_t batch:{size_t(1),size_t(2)}){
            const size_t physical[]={batch,batch};std::vector<Garnet::MetalMatrix> input(2),output(2);
            for(size_t p=0;p<2;++p)output[p]=gpu.Allocate(physical[p],n);
            auto upload=[&](int pattern){for(size_t p=0;p<2;++p){std::vector<float> x(physical[p]*k,0);
                for(size_t r=0;r<batch;++r)for(size_t j=0;j<k;++j){float v=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;
                    if(pattern==1)v=0;else if(pattern==2)v=(j%2?-.125f:.125f);else if(pattern==3)v*=1e-4f;x[r*k+j]=v;}
                input[p]=gpu.Upload(x,physical[p],k);}};
            auto run=[&](size_t policy){Garnet::gDirectMPS=policy!=0;auto begin=std::chrono::steady_clock::now();gpu.Begin();gpu.EncodeBF16MatVec(input[policy],weight,output[policy]);gpu.SubmitAndWait();
                double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                need(gpu.LastGlobalHalfAlphaDispatches()==(policy?1:0),"Requested actual native projection route did not execute");
                return std::pair<double,double>{gpu.LastGPUMs(),wall};};
            bool admitted[]={true,true};
            for(int pattern=0;pattern<4;++pattern){upload(pattern);run(0);auto reference=gpu.Download(output[0]),x=gpu.Download(input[0]);
                std::vector<double> exact(batch*n),bound(batch*n);const double eps=std::ldexp(1.,-23),terms=std::ceil(double(k)/32)+32,gamma=terms*eps/(1-terms*eps);double controlWorst=0;
                for(size_t r=0;r<batch;++r)for(size_t c=0;c<n;++c){double sum=0,absolute=0;for(size_t j=0;j<k;++j){double product=double(x[r*k+j])*double(expand(bits[c*k+j]));sum+=product;absolute+=std::abs(product);}
                    size_t i=r*n+c;exact[i]=sum;bound[i]=gamma*absolute+1e-7;need(std::isfinite(sum)&&std::isfinite(reference[i]),"Nonfinite original control");controlWorst=std::max(controlWorst,std::abs(double(reference[i])-sum)/bound[i]);}
                need(controlWorst<=1,"Original control exceeds unchanged FP64 envelope");
                for(size_t p=1;p<2;++p){run(p);auto actual=gpu.Download(output[p]);double worst=0,comparison=0,peak=0;bool zero=true,finite=true,bitsExact=true;
                    for(size_t r=0;r<physical[p];++r)for(size_t c=0;c<n;++c){size_t i=r*n+c;finite=finite&&std::isfinite(actual[i]);
                        if(r>=batch){zero=zero&&actual[i]==0;continue;}double error=std::abs(double(actual[i])-reference[i]);
                        if(!std::isfinite(actual[i])){worst=comparison=INFINITY;continue;}
                        worst=std::max(worst,std::abs(double(actual[i])-exact[i])/bound[i]);comparison=std::max(comparison,error/(5e-6+5e-5*std::abs(double(reference[i]))));peak=std::max(peak,error);bitsExact=bitsExact&&std::memcmp(&actual[i],&reference[i],4)==0;}
                    bool passed=finite&&zero&&worst<=1&&comparison<=1;admitted[p]=admitted[p]&&passed;
                    emit(@{@"kind":@"check",@"matrix":name,@"batch":@(batch),@"physical_rows":@(physical[p]),@"pattern":@(pattern),@"global_half_dispatches":@1,@"control_fp64_worst":@(controlWorst),@"fp64_worst":@(std::isfinite(worst)?worst:-1),@"comparison_worst":@(std::isfinite(comparison)?comparison:-1),@"max_abs":@(peak),@"bits_exact":@(bitsExact),@"finite":@(finite),@"no_extra_rows":@(zero),@"passed":@(passed)});
                }
            }
            upload(0);auto warm=std::chrono::steady_clock::now();do{run(0);for(size_t p=1;p<2;++p)if(admitted[p])run(p);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25);
            for(size_t p=1;p<2;++p)if(admitted[p])for(int round=0;round<2;++round){int position=0;for(bool candidate:{false,true,true,false}){size_t policy=candidate?p:0;run(policy);run(policy);NSMutableArray* g=[NSMutableArray array];NSMutableArray* w=[NSMutableArray array];
                for(int sample=0;sample<8;++sample){auto result=run(policy);need(result.first>0&&result.second>0,"Invalid measured interval");[g addObject:@(result.first)];[w addObject:@(result.second)];}
                emit(@{@"kind":@"samples",@"matrix":name,@"batch":@(batch),@"physical_rows":@(physical[p]),@"round":@(round),@"position":@(position++),@"candidate":@(candidate),@"gpu_ms":g,@"wall_ms":w});}}
        }
    }}return 0;
}catch(const std::exception& e){fprintf(stderr,"padded_mps_body_failed: %s\n",e.what());return 1;}}}
