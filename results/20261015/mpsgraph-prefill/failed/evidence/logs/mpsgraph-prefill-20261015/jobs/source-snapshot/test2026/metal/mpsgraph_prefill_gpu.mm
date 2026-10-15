// Isolated MPSGraph FP32 matmul versus Garnet's retained native prefill.
#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "metal_context.h"
#include "exact_half_codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"JSON failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t bits){uint32_t word=uint32_t(bits)<<16;float value;std::memcpy(&value,&word,4);return value;}
struct GraphProduct {
    MPSGraph* graph;MPSGraphTensor* a;MPSGraphTensor* w;MPSGraphTensor* output;
    NSDictionary* feeds;NSDictionary* results;id<MTLBuffer> values;
};
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"Expected assets and model");NSString* folder=[NSString stringWithUTF8String:argv[1]];std::string model=argv[2];
    auto manifestData=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(manifestData!=nil,"Missing manifest");
    NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:manifestData options:0 error:nil];need(manifest!=nil,"Bad manifest");
    Garnet::MetalContext native;auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];need(queue!=nil,"MPSGraph queue unavailable");
    for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
        size_t n=[item[@"rows"] unsignedLongLongValue],k=[item[@"inner"] unsignedLongLongValue];NSString* name=item[@"name"];std::string tensor=[item[@"tensor"] UTF8String],suffix=".self_attn.k_proj.weight";
        bool floatKey=tensor.rfind("model.layers.",0)==0&&tensor.size()>=suffix.size()&&tensor.compare(tensor.size()-suffix.size(),suffix.size(),suffix)==0;
        auto raw=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:name] stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==n*k*2,"Raw weight size mismatch");std::vector<uint16_t> bits(n*k);std::memcpy(bits.data(),raw.bytes,raw.length);
        auto weight=native.UploadBF16(bits.data(),bits.size()*2,n,k,floatKey);
        if(!weight.GlobalHalfReady()){emit(@{@"kind":@"skipped",@"matrix":name,@"reason":@"Original native policy has no common-alpha half weight"});continue;}
        std::vector<float> w32(n*k);std::vector<uint16_t> w16(n*k);uint16_t largest=0;for(uint16_t b:bits)largest=std::max(largest,uint16_t(b&32767));int32_t scale=0;need(Garnet::ExactHalfRowScale(largest,scale),"Common half scale mismatch");
        for(size_t i=0;i<bits.size();++i){w32[i]=expand(bits[i]);need(Garnet::EncodeExactScaledHalf(bits[i],scale,w16[i]),"Common half admission mismatch");_Float16 half;std::memcpy(&half,&w16[i],2);float recovered=std::ldexp(float(half),-scale);need(std::memcmp(&recovered,&w32[i],4)==0,"Common half did not recover original BF16 bits");}
        auto original=[device newBufferWithBytes:w32.data() length:w32.size()*4 options:MTLResourceStorageModeShared];auto encoded=[device newBufferWithBytes:w16.data() length:w16.size()*2 options:MTLResourceStorageModeShared];need(original&&encoded,"Graph weight allocation failed");
        for(size_t batch:{size_t(1),size_t(2)}){@autoreleasepool{
            size_t rows=model=="Qwen3.5-0.8B"?80*batch:model=="Qwen3-0.6B"?(batch==1?24:44):26*batch;
            auto graphInput=[device newBufferWithLength:rows*k*4 options:MTLResourceStorageModeShared];need(graphInput!=nil,"Graph input allocation failed");
            auto output=native.Allocate(rows,n);Garnet::MetalMatrix input;GraphProduct products[2];
            for(int policy=1;policy<=2;++policy){auto& p=products[policy-1];p.graph=[MPSGraph new];auto descriptor=[MPSGraphCompilationDescriptor new];descriptor.optimizationLevel=MPSGraphOptimizationLevel1;descriptor.waitForCompilationCompletion=YES;p.graph.compilationDescriptor=descriptor;
                p.a=[p.graph placeholderWithShape:@[@(rows),@(k)] dataType:MPSDataTypeFloat32 name:@"original_fp32_activation"];
                p.w=[p.graph placeholderWithShape:@[@(n),@(k)] dataType:policy==1?MPSDataTypeFloat32:MPSDataTypeFloat16 name:@"lossless_weight_storage"];
                auto expanded=policy==1?p.w:[p.graph castTensor:p.w toType:MPSDataTypeFloat32 name:@"exact_half_to_fp32"];
                auto transposed=[p.graph transposeTensor:expanded dimension:0 withDimension:1 name:nil];auto product=[p.graph matrixMultiplicationWithPrimaryTensor:p.a secondaryTensor:transposed name:@"fp32_matmul"];
                if(policy==2){auto alpha=[p.graph constantWithScalar:std::ldexp(1.0,-scale) dataType:MPSDataTypeFloat32];product=[p.graph multiplicationWithPrimaryTensor:product secondaryTensor:alpha name:@"exact_common_alpha"];}p.output=product;
                need(p.a.dataType==MPSDataTypeFloat32&&expanded.dataType==MPSDataTypeFloat32&&product.dataType==MPSDataTypeFloat32,"Graph arithmetic dtype mismatch");
                auto aData=[[MPSGraphTensorData alloc] initWithMTLBuffer:graphInput shape:@[@(rows),@(k)] dataType:MPSDataTypeFloat32];auto wData=[[MPSGraphTensorData alloc] initWithMTLBuffer:policy==1?original:encoded shape:@[@(n),@(k)] dataType:policy==1?MPSDataTypeFloat32:MPSDataTypeFloat16];
                p.values=[device newBufferWithLength:rows*n*4 options:MTLResourceStorageModeShared];need(p.values!=nil,"Graph result allocation failed");auto result=[[MPSGraphTensorData alloc] initWithMTLBuffer:p.values shape:@[@(rows),@(n)] dataType:MPSDataTypeFloat32];p.feeds=@{p.a:aData,p.w:wData};p.results=@{p.output:result};
            }
            auto run=[&](int policy){auto start=std::chrono::steady_clock::now();if(policy){auto& p=products[policy-1];[p.graph runWithMTLCommandQueue:queue feeds:p.feeds targetOperations:nil resultsDictionary:p.results];}else{native.Begin();native.EncodeBF16MatVec(input,weight,output);native.SubmitAndWait();need(native.LastGlobalHalfAlphaDispatches()==1,"Original control route mismatch");}return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
            bool admitted[]={true,true,true};
            for(int pattern=0;pattern<4;++pattern){std::vector<float> x(rows*k);for(size_t r=0;r<rows;++r)for(size_t j=0;j<k;++j){float v=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;if(pattern==1)v*=float(j%2?-16:16);if(pattern==2)v=j==((r*53+17)%k)?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(j%9)-4);x[r*k+j]=v;}
                input=native.Upload(x,rows,k);std::memcpy(graphInput.contents,x.data(),x.size()*4);run(0);auto control=native.Download(output);std::vector<double> exact(rows*n),bound(rows*n);double eps=std::ldexp(1.,-23),terms=std::ceil(double(k)/32)+32,gamma=terms*eps/(1-terms*eps),controlWorst=0;
                for(size_t r=0;r<rows;++r)for(size_t c=0;c<n;++c){double sum=0,absolute=0;for(size_t j=0;j<k;++j){double term=double(x[r*k+j])*w32[c*k+j];sum+=term;absolute+=std::abs(term);}size_t i=r*n+c;exact[i]=sum;bound[i]=gamma*absolute+1e-7;need(std::isfinite(control[i]),"Nonfinite native control");controlWorst=std::max(controlWorst,std::abs(double(control[i])-sum)/bound[i]);}need(controlWorst<=1,"Native control exceeded FP64 bound");
                for(int policy=1;policy<=2;++policy){double elapsed=run(policy);if(pattern==0)emit(@{@"kind":@"first_graph_run",@"matrix":name,@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"wall_ms":@(elapsed)});auto actual=static_cast<const float*>(products[policy-1].values.contents);bool finite=true;double worst=0,comparison=0;
                    for(size_t i=0;i<rows*n;++i){if(!std::isfinite(actual[i])){finite=false;continue;}worst=std::max(worst,std::abs(double(actual[i])-exact[i])/bound[i]);comparison=std::max(comparison,std::abs(double(actual[i])-control[i])/(5e-6+5e-5*std::abs(double(control[i]))));}
                    bool passed=finite&&worst<=1&&comparison<=1;admitted[policy]&=passed;emit(@{@"kind":@"check",@"matrix":name,@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),@"values":@(rows*n),@"fp32_activation_product_output":@YES,@"original_weight_bits_recovered":@YES,@"control_fp64_worst":@(controlWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"finite":@(finite),@"passed":@(passed)});
                }
            }
            auto warm=std::chrono::steady_clock::now();do{run(0);for(int policy=1;policy<=2;++policy)if(admitted[policy])run(policy);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25);
            for(int policy=1;policy<=2;++policy)if(admitted[policy])for(int round=0;round<2;++round){int position=0;for(bool candidate:{false,true,true,false}){run(candidate?policy:0);run(candidate?policy:0);NSMutableArray* times=[NSMutableArray array];for(int sample=0;sample<8;++sample){double elapsed=run(candidate?policy:0);need(std::isfinite(elapsed)&&elapsed>0,"Bad wall interval");[times addObject:@(elapsed)];}emit(@{@"kind":@"samples",@"matrix":name,@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"round":@(round),@"position":@(position++),@"candidate":@(candidate),@"wall_ms":times});}}
        }}
    }}return 0;
}catch(const std::exception& error){fprintf(stderr,"mpsgraph_prefill_failed: %s\n",error.what());return 1;}}}
