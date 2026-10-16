// Isolated native component: original FP32 prefill versus packed BF16 loads.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace Garnet {extern int gBlockPackedPrefillPolicy;}
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static void emit(NSDictionary* v){auto d=[NSJSONSerialization dataWithJSONObject:v options:0 error:nil];need(d!=nil,"JSON failed");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t v){uint32_t b=uint32_t(v)<<16;float x;std::memcpy(&x,&b,4);return x;}
struct Projection{size_t n,k;Garnet::MetalBF16Weight weight;bool selected,floatKey;std::vector<uint16_t> bits;Garnet::MetalMatrix input,output;std::vector<float> control;std::vector<double> exact,bound;};
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==5,"Expected assets, model, mode and timing policy");bool timing=std::strcmp(argv[3],"timing")==0;need(timing||std::strcmp(argv[3],"numeric")==0,"Unknown mode");int timingPolicy=std::atoi(argv[4]);need(!timing||(timingPolicy>=1&&timingPolicy<=2),"Bad timing policy");NSString* folder=[NSString stringWithUTF8String:argv[1]];std::string model=argv[2];
    auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(data!=nil,"Manifest missing");
    NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(manifest!=nil,"Bad manifest");Garnet::MetalContext gpu;std::vector<Projection> projections;
    for(NSDictionary* c in manifest[@"cases"]){Projection p;p.n=[c[@"rows"] unsignedLongLongValue];p.k=[c[@"inner"] unsignedLongLongValue];
        auto raw=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:c[@"name"]] stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==p.n*p.k*2,"Bad raw size");p.bits.resize(p.n*p.k);std::memcpy(p.bits.data(),raw.bytes,raw.length);
        std::string name=[c[@"tensor"] UTF8String],suffix=".self_attn.k_proj.weight";
        p.floatKey=name.rfind("model.layers.",0)==0&&name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
        auto packStart=std::chrono::steady_clock::now();p.weight=gpu.UploadBF16(p.bits.data(),p.bits.size()*2,p.n,p.k,p.floatKey);p.selected=p.weight.BlockPackedPrefillBytes()>0;need(p.selected==(p.weight.InterleavedBodyReady()&&p.n>1024),"Block packed selection changed");emit(@{@"kind":@"packing",@"matrix":c[@"name"],@"selected":@(p.selected),@"additional_bytes":@(p.weight.BlockPackedPrefillBytes()),@"upload_ms":@(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-packStart).count())});projections.push_back(std::move(p));}
    for(size_t batch:{size_t(1),size_t(2)}){
        size_t rows=model=="Qwen3.5-0.8B"?80*batch:model=="Qwen3-0.6B"?(batch==1?24:44):26*batch;
        auto inputs=[&](int pattern,bool reference){for(auto& p:projections){std::vector<float> x(rows*p.k);for(size_t r=0;r<rows;++r)for(size_t j=0;j<p.k;++j){float v=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;if(pattern==1)v*=float(j%2?-16:16);if(pattern==2)v=j==((r*53+17)%p.k)?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(j%9)-4);x[r*p.k+j]=v;}
            p.input=gpu.Upload(x,rows,p.k);p.output=gpu.Allocate(rows,p.n);if(!reference)continue;p.exact.resize(rows*p.n);p.bound.resize(rows*p.n);
            double eps=std::ldexp(1.,-23),terms=std::ceil(double(p.k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<rows;++r)for(size_t c=0;c<p.n;++c){double sum=0,absolute=0;for(size_t j=0;j<p.k;++j){double product=double(x[r*p.k+j])*expand(p.bits[c*p.k+j]);sum+=product;absolute+=std::abs(product);}p.exact[r*p.n+c]=sum;p.bound[r*p.n+c]=gamma*absolute+1e-7;}}};
        auto run=[&](int policy){Garnet::gBlockPackedPrefillPolicy=policy;auto start=std::chrono::steady_clock::now();gpu.Begin();size_t expected=0;for(auto& p:projections){expected+=policy&&p.selected;gpu.EncodeBF16MatVec(p.input,p.weight,p.output);}gpu.SubmitAndWait();
            need(gpu.LastBlockPackedPrefillDispatches()==expected,"Packed prefill actual route mismatch");return std::pair<double,double>{gpu.LastGPUMs(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};};
        bool admitted[]={true,true,true,true};
        if(!timing)for(int pattern=0;pattern<4;++pattern){inputs(pattern,true);run(0);double controlWorst=0;for(auto& p:projections){p.control=gpu.Download(p.output);for(size_t i=0;i<p.control.size();++i){need(std::isfinite(p.control[i]),"Nonfinite control");controlWorst=std::max(controlWorst,std::abs(double(p.control[i])-p.exact[i])/p.bound[i]);}}need(controlWorst<=1,"MPS control exceeded unchanged FP64 envelope");
            for(int policy=1;policy<=2;++policy){run(policy);double worst=0,comparison=0;size_t selected=0,values=0;bool finite=true;for(auto& p:projections){selected+=p.selected;auto actual=gpu.Download(p.output);values+=actual.size();for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i])){finite=false;continue;}worst=std::max(worst,std::abs(double(actual[i])-p.exact[i])/p.bound[i]);comparison=std::max(comparison,std::abs(double(actual[i])-p.control[i])/(5e-6+5e-5*std::abs(double(p.control[i]))));}}
                bool passed=finite&&worst<=1&&comparison<=1;admitted[policy]=admitted[policy]&&passed;need(selected>0,"No packed projection selected");emit(@{@"kind":@"check",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),@"projections":@(projections.size()),@"selected_projections":@(selected),@"values":@(values),@"control_fp64_worst":@(controlWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"finite":@(finite),@"passed":@(passed)});}
        }
        if(!timing)continue;inputs(0,false);auto warm=std::chrono::steady_clock::now();
        do{run(0);run(timingPolicy);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.);
        for(bool null:{true,false})for(int round=0;round<10;++round)for(int position=0;position<4;++position){bool candidate=position==1||position==2;int actual=candidate&&!null?timingPolicy:0;
            for(int w=0;w<3;++w)run(actual);
            for(int sample=0;sample<5;++sample){auto t=run(actual);need(std::isfinite(t.first)&&std::isfinite(t.second)&&t.first>0&&t.second>0,"Bad interval");emit(@{@"kind":@"timing",@"policy":@(timingPolicy),@"batch":@(batch),@"rows":@(rows),@"null":@(null),@"round":@(round),@"position":@(position),@"sample":@(sample),@"candidate":@(candidate),@"actual_policy":@(actual),@"gpu_ms":@(t.first),@"wall_ms":@(t.second)});}
        }
    }emit(@{@"kind":@"complete",@"mode":timing?@"timing":@"numeric",@"numeric_checks":@(timing?0:16),@"timing_samples":@(timing?800:0),@"full_goal_met":@NO});return 0;
}catch(const std::exception& e){fprintf(stderr,"packed_body_prefill_failed: %s\n",e.what());return 1;}}}
