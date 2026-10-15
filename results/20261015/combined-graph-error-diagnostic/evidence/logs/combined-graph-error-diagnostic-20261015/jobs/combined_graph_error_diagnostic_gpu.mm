// Private complete independent projection sets: one native submission or one graph execution.
#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static void emit(NSDictionary* value){auto d=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(d!=nil,"JSON failed");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t bits){uint32_t b=uint32_t(bits)<<16;float value;std::memcpy(&value,&b,4);return value;}
struct Projection {
    std::string name;size_t n,k;Garnet::MetalBF16Weight nativeWeight;std::vector<float> weights,control;std::vector<double> exact,bound;
    Garnet::MetalMatrix input,output;id<MTLBuffer> graphInput,graphWeight,graphOutput[2];
};
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"Expected immutable assets and model");NSString* folder=[NSString stringWithUTF8String:argv[1]];std::string model=argv[2];
    auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(data!=nil,"Manifest missing");NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(manifest!=nil,"Bad manifest");
    Garnet::MetalContext native;auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];need(queue!=nil,"Graph queue unavailable");std::vector<Projection> projections;
    for(NSDictionary* c in manifest[@"cases"]){Projection p;p.n=[c[@"rows"] unsignedLongLongValue];p.k=[c[@"inner"] unsignedLongLongValue];auto raw=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:c[@"name"]] stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==p.n*p.k*2,"Raw size mismatch");
        std::vector<uint16_t> bits(p.n*p.k);std::memcpy(bits.data(),raw.bytes,raw.length);p.weights.resize(bits.size());for(size_t i=0;i<bits.size();++i)p.weights[i]=expand(bits[i]);
        std::string name=[c[@"tensor"] UTF8String],suffix=".self_attn.k_proj.weight";p.name=name;bool key=name.rfind("model.layers.",0)==0&&name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
        p.nativeWeight=native.UploadBF16(bits.data(),bits.size()*2,p.n,p.k,key);p.graphWeight=[device newBufferWithBytes:p.weights.data() length:p.weights.size()*4 options:MTLResourceStorageModeShared];need(p.graphWeight!=nil,"Graph weight allocation failed");projections.push_back(std::move(p));
    }
    for(size_t batch:{size_t(1),size_t(2)}){@autoreleasepool{
        size_t rows=model=="Qwen3.5-0.8B"?80*batch:model=="Qwen3-0.6B"?(batch==1?24:44):26*batch;
        MPSGraph* graphs[2]={[MPSGraph new],[MPSGraph new]};NSMutableDictionary* feeds[2]={[NSMutableDictionary dictionary],[NSMutableDictionary dictionary]};NSMutableDictionary* results[2]={[NSMutableDictionary dictionary],[NSMutableDictionary dictionary]};size_t total=0;
        for(auto& p:projections){p.output=native.Allocate(rows,p.n);p.graphInput=[device newBufferWithLength:rows*p.k*4 options:MTLResourceStorageModeShared];need(p.graphInput!=nil,"Graph input allocation failed");total+=rows*p.n;
            for(int index=0;index<2;++index){auto graph=graphs[index];auto a=[graph placeholderWithShape:@[@(rows),@(p.k)] dataType:MPSDataTypeFloat32 name:nil];MPSGraphTensor* w;
                if(index==0){w=[graph placeholderWithShape:@[@(p.n),@(p.k)] dataType:MPSDataTypeFloat32 name:nil];feeds[index][w]=[[MPSGraphTensorData alloc] initWithMTLBuffer:p.graphWeight shape:@[@(p.n),@(p.k)] dataType:MPSDataTypeFloat32];}
                else{auto bytes=[NSData dataWithBytes:p.weights.data() length:p.weights.size()*4];w=[graph constantWithData:bytes shape:@[@(p.n),@(p.k)] dataType:MPSDataTypeFloat32];}
                auto transpose=[graph transposeTensor:w dimension:0 withDimension:1 name:nil];auto product=[graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:transpose name:nil];need(a.dataType==MPSDataTypeFloat32&&w.dataType==MPSDataTypeFloat32&&product.dataType==MPSDataTypeFloat32,"Graph tensor dtype changed");
                feeds[index][a]=[[MPSGraphTensorData alloc] initWithMTLBuffer:p.graphInput shape:@[@(rows),@(p.k)] dataType:MPSDataTypeFloat32];p.graphOutput[index]=[device newBufferWithLength:rows*p.n*4 options:MTLResourceStorageModeShared];need(p.graphOutput[index]!=nil,"Graph output allocation failed");results[index][product]=[[MPSGraphTensorData alloc] initWithMTLBuffer:p.graphOutput[index] shape:@[@(rows),@(p.n)] dataType:MPSDataTypeFloat32];
            }
        }
        auto run=[&](int policy){auto start=std::chrono::steady_clock::now();if(policy){int index=policy-1;[graphs[index] runWithMTLCommandQueue:queue feeds:feeds[index] targetOperations:nil resultsDictionary:results[index]];}else{native.Begin();for(auto& p:projections)native.EncodeBF16MatVec(p.input,p.nativeWeight,p.output);native.SubmitAndWait();}return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
        bool admitted[]={true,true,true};
        auto inputs=[&](int pattern,bool reference){for(auto& p:projections){std::vector<float> x(rows*p.k);for(size_t r=0;r<rows;++r)for(size_t j=0;j<p.k;++j){float v=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;if(pattern==1)v*=float(j%2?-16:16);if(pattern==2)v=j==((r*53+17)%p.k)?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(j%9)-4);x[r*p.k+j]=v;}
            p.input=native.Upload(x,rows,p.k);std::memcpy(p.graphInput.contents,x.data(),x.size()*4);if(!reference)continue;p.exact.resize(rows*p.n);p.bound.resize(rows*p.n);double eps=std::ldexp(1.,-23),terms=std::ceil(double(p.k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<rows;++r)for(size_t c=0;c<p.n;++c){double sum=0,absolute=0;for(size_t k=0;k<p.k;++k){double product=double(x[r*p.k+k])*p.weights[c*p.k+k];sum+=product;absolute+=std::abs(product);}p.exact[r*p.n+c]=sum;p.bound[r*p.n+c]=gamma*absolute+1e-7;}}};
        for(int pattern=0;pattern<4;++pattern){inputs(pattern,true);run(0);double controlWorst=0;for(auto& p:projections){p.control=native.Download(p.output);for(size_t i=0;i<p.control.size();++i){need(std::isfinite(p.control[i]),"Nonfinite native control");controlWorst=std::max(controlWorst,std::abs(double(p.control[i])-p.exact[i])/p.bound[i]);}}need(controlWorst<=1,"Native control exceeded original FP64 bound");
            for(int policy=1;policy<=2;++policy){double first=run(policy);if(pattern==0)emit(@{@"kind":@"first_graph_run",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"wall_ms":@(first)});double worst=0,comparison=0;bool finite=true;
                size_t matrix=0;
                for(auto& p:projections){
                    auto actual=static_cast<const float*>(p.graphOutput[policy-1].contents);
                    double pw=0,pc=0,cw=0;size_t at=0;bool pf=true;
                    for(size_t i=0;i<rows*p.n;++i){
                        cw=std::max(cw,std::abs(double(p.control[i])-p.exact[i])/p.bound[i]);
                        if(!std::isfinite(actual[i])){pf=false;continue;}
                        pw=std::max(pw,std::abs(double(actual[i])-p.exact[i])/p.bound[i]);
                        double ratio=std::abs(double(actual[i])-p.control[i])/(5e-6+5e-5*std::abs(double(p.control[i])));
                        if(ratio>pc){pc=ratio;at=i;}
                    }
                    finite&=pf;worst=std::max(worst,pw);comparison=std::max(comparison,pc);
                    emit(@{@"kind":@"projection_check",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),
                        @"matrix":@(matrix++),@"tensor":[NSString stringWithUTF8String:p.name.c_str()],@"columns":@(p.n),@"inner":@(p.k),@"values":@(rows*p.n),
                        @"control_fp64_worst":@(cw),@"fp64_worst":@(pw),@"comparison_worst":@(pc),@"finite":@(pf),@"passed":@(pf&&cw<=1&&pw<=1&&pc<=1),
                        @"worst_index":@(at),@"control_at_worst":@(p.control[at]),@"candidate_at_worst":@(actual[at]),@"fp64_at_worst":@(p.exact[at]),@"fp64_bound_at_worst":@(p.bound[at]),
                        @"comparison_bound_at_worst":@(5e-6+5e-5*std::abs(double(p.control[at])))});
                }
                bool passed=finite&&worst<=1&&comparison<=1;admitted[policy]&=passed;emit(@{@"kind":@"check",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),@"projections":@(projections.size()),@"values":@(total),@"fp32_activation_weight_product_output":@YES,@"control_fp64_worst":@(controlWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"finite":@(finite),@"passed":@(passed)});
            }
        }
    }}return 0;
}catch(const std::exception& e){fprintf(stderr,"combined_graph_prefill_failed: %s\n",e.what());return 1;}}}
