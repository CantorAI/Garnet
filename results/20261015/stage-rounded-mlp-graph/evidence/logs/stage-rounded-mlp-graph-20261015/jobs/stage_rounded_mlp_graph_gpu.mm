// A complete dependent MLP: gate/up -> SiLU multiplication -> down projection.
#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"JSON failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t bits){uint32_t value=uint32_t(bits)<<16;float result;std::memcpy(&result,&value,4);return result;}
struct Weight {
    size_t n=0,k=0;std::vector<float> values;Garnet::MetalBF16Weight native;
    id<MTLBuffer> graph=nil;
};
struct Mlp {
    Weight weights[3];size_t width=0,intermediate=0;
    Garnet::MetalMatrix input,gate,up,activated,output;
    id<MTLBuffer> graphInput=nil,graphOutput[2]={nil,nil};
    MPSGraph* graphs[2]={nil,nil};
    NSMutableDictionary* feeds[2]={nil,nil};
    NSMutableDictionary* results[2]={nil,nil};
    std::vector<float> control;std::vector<double> reference,bounds;
};

int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"Expected immutable assets and model");
    NSString* folder=[NSString stringWithUTF8String:argv[1]];std::string model=argv[2];
    auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(bytes!=nil,"Manifest missing");
    NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];need(manifest!=nil,"Bad manifest");
    Garnet::MetalContext native;auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];need(queue!=nil,"Graph queue missing");
    std::map<std::string,Mlp> groups;
    for(NSDictionary* item in manifest[@"cases"]){
        std::string tensor=[item[@"tensor"] UTF8String];int index=-1;std::string prefix;
        const char* suffixes[]={".mlp.gate_proj.weight",".mlp.up_proj.weight",".mlp.down_proj.weight"};
        for(int i=0;i<3;++i){std::string suffix=suffixes[i];if(tensor.size()>=suffix.size()&&tensor.compare(tensor.size()-suffix.size(),suffix.size(),suffix)==0){index=i;prefix=tensor.substr(0,tensor.size()-suffix.size());break;}}
        if(index<0)continue;
        auto& weight=groups[prefix].weights[index];need(weight.n==0,"Duplicate MLP weight");
        weight.n=[item[@"rows"] unsignedLongLongValue];weight.k=[item[@"inner"] unsignedLongLongValue];
        auto raw=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:item[@"name"]] stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==weight.n*weight.k*2,"Raw weight size mismatch");
        std::vector<uint16_t> bits(weight.n*weight.k);std::memcpy(bits.data(),raw.bytes,raw.length);weight.values.resize(bits.size());
        for(size_t i=0;i<bits.size();++i)weight.values[i]=expand(bits[i]);
        weight.native=native.UploadBF16(bits.data(),raw.length,weight.n,weight.k);
        weight.graph=[device newBufferWithBytes:weight.values.data() length:weight.values.size()*4 options:MTLResourceStorageModeShared];need(weight.graph!=nil,"Weight buffer missing");
    }
    need(groups.size()==(model=="Qwen3.5-0.8B"?2:1),"Incomplete MLP fixture groups");
    for(auto& entry:groups){auto& p=entry.second;p.width=p.weights[0].k;p.intermediate=p.weights[0].n;
        need(p.width&&p.intermediate&&p.weights[1].n==p.intermediate&&p.weights[1].k==p.width&&p.weights[2].n==p.width&&p.weights[2].k==p.intermediate,"MLP dependency shapes mismatch");}
    for(size_t batch:{size_t(1),size_t(2)}){@autoreleasepool{
        size_t rows=model=="Qwen3.5-0.8B"?80*batch:model=="Qwen3-0.6B"?(batch==1?24:44):26*batch;
        size_t total=0;
        for(auto& entry:groups){auto& p=entry.second;
            p.gate=native.Allocate(rows,p.intermediate);p.up=native.Allocate(rows,p.intermediate);p.activated=native.Allocate(rows,p.intermediate);p.output=native.Allocate(rows,p.width);
            p.graphInput=[device newBufferWithLength:rows*p.width*4 options:MTLResourceStorageModeShared];need(p.graphInput!=nil,"Input allocation failed");total+=rows*p.width;
            for(int policy=0;policy<2;++policy){p.graphs[policy]=[MPSGraph new];p.feeds[policy]=[NSMutableDictionary dictionary];p.results[policy]=[NSMutableDictionary dictionary];auto graph=p.graphs[policy];
                auto input=[graph placeholderWithShape:@[@(rows),@(p.width)] dataType:MPSDataTypeFloat32 name:nil];
                p.feeds[policy][input]=[[MPSGraphTensorData alloc] initWithMTLBuffer:p.graphInput shape:@[@(rows),@(p.width)] dataType:MPSDataTypeFloat32];
                MPSGraphTensor* weights[3];
                for(int i=0;i<3;++i){auto& w=p.weights[i];NSArray* shape=@[@(w.n),@(w.k)];
                    if(policy==0){weights[i]=[graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];p.feeds[policy][weights[i]]=[[MPSGraphTensorData alloc] initWithMTLBuffer:w.graph shape:shape dataType:MPSDataTypeFloat32];}
                    else weights[i]=[graph constantWithData:[NSData dataWithBytes:w.values.data() length:w.values.size()*4] shape:shape dataType:MPSDataTypeFloat32];
                    need(weights[i].dataType==MPSDataTypeFloat32,"Weight dtype changed");}
                auto product=[&](MPSGraphTensor* a,MPSGraphTensor* w){return [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:[graph transposeTensor:w dimension:0 withDimension:1 name:nil] name:nil];};
                auto gate=product(input,weights[0]),up=product(input,weights[1]);
                float oneValue=1.f;auto one=[graph constantWithData:[NSData dataWithBytes:&oneValue length:4] shape:@[@1] dataType:MPSDataTypeFloat32];
                auto neg=[graph negativeWithTensor:gate name:nil],exponential=[graph exponentWithTensor:neg name:nil];
                auto denominator=[graph additionWithPrimaryTensor:one secondaryTensor:exponential name:nil];
                auto silu=[graph divisionWithPrimaryTensor:gate secondaryTensor:denominator name:nil];
                auto activated=[graph multiplicationWithPrimaryTensor:silu secondaryTensor:up name:nil];
                auto result=product(activated,weights[2]);
                for(MPSGraphTensor* tensor in @[input,gate,up,one,neg,exponential,denominator,silu,activated,result])need(tensor.dataType==MPSDataTypeFloat32,"MLP tensor dtype changed");
                p.graphOutput[policy]=[device newBufferWithLength:rows*p.width*4 options:MTLResourceStorageModeShared];need(p.graphOutput[policy]!=nil,"Output allocation failed");
                p.results[policy][result]=[[MPSGraphTensorData alloc] initWithMTLBuffer:p.graphOutput[policy] shape:@[@(rows),@(p.width)] dataType:MPSDataTypeFloat32];
            }
        }
        auto run=[&](int policy){auto start=std::chrono::steady_clock::now();
            for(auto& entry:groups){auto& p=entry.second;
                if(policy){int index=policy-1;[p.graphs[index] runWithMTLCommandQueue:queue feeds:p.feeds[index] targetOperations:nil resultsDictionary:p.results[index]];}
                else {native.Begin();native.EncodeBF16MatVec(p.input,p.weights[0].native,p.gate);native.EncodeBF16MatVec(p.input,p.weights[1].native,p.up);native.EncodeSiluMul(p.gate,p.up,p.activated);native.EncodeBF16MatVec(p.activated,p.weights[2].native,p.output);native.SubmitAndWait();need(native.LastCoalescedBlockDispatches()==0,"Decode layout used during MLP prefill");}
            }
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
        auto inputs=[&](int pattern,bool reference){for(auto& entry:groups){auto& p=entry.second;std::vector<float> x(rows*p.width);
            for(size_t r=0;r<rows;++r)for(size_t k=0;k<p.width;++k){float v=std::sin(float((k*17+r*53)%1301)*.013f)*.2f;if(pattern==1)v*=float(k%2?-16:16);if(pattern==2)v=k==((r*53+17)%p.width)?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(k%9)-4);x[r*p.width+k]=v;}
            p.input=native.Upload(x,rows,p.width);std::memcpy(p.graphInput.contents,x.data(),x.size()*4);if(!reference)continue;
            p.reference.resize(rows*p.width);p.bounds.resize(rows*p.width);std::vector<float> activated(p.intermediate);
            for(size_t r=0;r<rows;++r){
                for(size_t j=0;j<p.intermediate;++j){double g=0,u=0;for(size_t k=0;k<p.width;++k){g+=double(x[r*p.width+k])*p.weights[0].values[j*p.width+k];u+=double(x[r*p.width+k])*p.weights[1].values[j*p.width+k];}float gate=static_cast<float>(g),up=static_cast<float>(u);activated[j]=(gate/(1.f+std::exp(-gate)))*up;}
                for(size_t c=0;c<p.width;++c){double sum=0;for(size_t j=0;j<p.intermediate;++j)sum+=activated[j]*p.weights[2].values[c*p.intermediate+j];need(std::isfinite(sum),"Nonfinite full FP64 reference");float rounded=static_cast<float>(sum);p.reference[r*p.width+c]=rounded;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(double(rounded));}
            }
        }};
        bool admitted[]={true,true,true};
        for(int pattern=0;pattern<4;++pattern){inputs(pattern,true);run(0);double controlWorst=0;bool controlFinite=true;
            for(auto& entry:groups){auto& p=entry.second;p.control=native.Download(p.output);for(size_t i=0;i<p.control.size();++i){if(!std::isfinite(p.control[i])){controlFinite=false;continue;}controlWorst=std::max(controlWorst,std::abs(double(p.control[i])-p.reference[i])/p.bounds[i]);}}
            for(int policy=1;policy<=2;++policy){double elapsed=run(policy);if(pattern==0)emit(@{@"kind":@"first_graph_run",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"wall_ms":@(elapsed)});
                double worst=0,comparison=0;bool finite=controlFinite;
                for(auto& entry:groups){auto& p=entry.second;auto actual=static_cast<const float*>(p.graphOutput[policy-1].contents);for(size_t i=0;i<p.control.size();++i){if(!std::isfinite(actual[i])){finite=false;continue;}worst=std::max(worst,std::abs(double(actual[i])-p.reference[i])/p.bounds[i]);comparison=std::max(comparison,std::abs(double(actual[i])-p.control[i])/(5e-6+5e-5*std::abs(double(p.control[i]))));}}
                bool passed=finite&&controlWorst<=1&&worst<=1&&comparison<=1;admitted[policy]&=passed;
                emit(@{@"kind":@"check",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),@"projections":@(groups.size()*3),@"mlp_blocks":@(groups.size()),@"dependent_matmuls":@(groups.size()*3),@"values":@(total),@"fp32_activation_weight_product_output":@YES,@"control_fp64_worst":@(controlWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"finite":@(finite),@"passed":@(passed)});
            }
        }
        inputs(0,false);auto warm=std::chrono::steady_clock::now();do{run(0);for(int policy=1;policy<=2;++policy)if(admitted[policy])run(policy);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25);
        for(int policy=1;policy<=2;++policy)if(admitted[policy])for(int round=0;round<2;++round){int position=0;for(bool candidate:{false,true,true,false}){run(candidate?policy:0);run(candidate?policy:0);NSMutableArray* samples=[NSMutableArray array];for(int i=0;i<8;++i){double elapsed=run(candidate?policy:0);need(std::isfinite(elapsed)&&elapsed>0,"Invalid MLP wall interval");[samples addObject:@(elapsed)];}emit(@{@"kind":@"samples",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"round":@(round),@"position":@(position++),@"candidate":@(candidate),@"wall_ms":samples});}}
    }}return 0;
}catch(const std::exception& error){fprintf(stderr,"dependent_mlp_graph_failed: %s\n",error.what());return 1;}}}
