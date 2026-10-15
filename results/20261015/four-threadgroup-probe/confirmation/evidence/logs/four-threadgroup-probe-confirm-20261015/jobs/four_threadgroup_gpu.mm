// Isolated native component: coalesced exact12-bit blocks versus current raw BF16 decode.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
namespace Garnet {extern bool gCoalescedBlockEnabled;extern size_t gCoalescedBlockThreads;}
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static NSString* currentMatrix=nil;
static void emit(NSDictionary* v){if(currentMatrix){auto tagged=[v mutableCopy];tagged[@"matrix"]=currentMatrix;v=tagged;}auto d=[NSJSONSerialization dataWithJSONObject:v options:0 error:nil];need(d!=nil,"JSON failed");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t v){uint32_t b=uint32_t(v)<<16;float x;std::memcpy(&x,&b,4);return x;}
struct Projection{std::string name;size_t n,k;Garnet::MetalBF16Weight weight;bool selected,floatKey;std::vector<uint16_t> bits;Garnet::MetalMatrix input,output;std::vector<float> control;std::vector<double> exact,bound;};

static void controls(){
    Garnet::MetalContext gpu,other;size_t checked=0;
    for(auto shape:std::vector<std::pair<size_t,size_t>>{{257,4096},{1024,1024},{8192,128},{8193,128},{256,4096},{1024,1025},{1024,512}}){
        size_t n=shape.first,k=shape.second;std::vector<uint16_t> bits(n*k,0x3e80);
        Garnet::gCoalescedBlockEnabled=false;auto original=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        Garnet::gCoalescedBlockEnabled=true;auto wide=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);Garnet::gCoalescedBlockEnabled=false;
        bool selected=n>256&&n<=8192&&k%128==0&&n*k>=1048576;
        need(wide.CoalescedBlockReady()==selected&&original.CoalescedBlockReady()==selected,"Wide admission mismatch");
        for(size_t rows:{size_t(1),size_t(2),size_t(3)}){
            auto input=gpu.Upload(std::vector<float>(rows*k,.125f),rows,k),out=gpu.Allocate(rows,n);
            Garnet::gCoalescedBlockEnabled=false;gpu.Begin();gpu.EncodeBF16MatVec(input,original,out);gpu.SubmitAndWait();auto control=gpu.Download(out);
            for(size_t threads:{size_t(64),size_t(128),size_t(256)}){
                Garnet::gCoalescedBlockEnabled=true;Garnet::gCoalescedBlockThreads=threads;gpu.Begin();gpu.EncodeBF16MatVec(input,wide,out);bool rejected=false;
                try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending wide output readable");
                other.Begin();rejected=false;try{other.EncodeBF16MatVec(input,wide,out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Foreign wide writer accepted");other.Abort();
                gpu.SubmitAndWait();auto actual=gpu.Download(out);need(actual.size()==control.size()&&!std::memcmp(actual.data(),control.data(),actual.size()*4),"Wide tail/fallback/prefill changed original bits");
                need(gpu.LastCoalescedBlockDispatches()==size_t(selected&&rows<=2),"Wide route count mismatch");need(gpu.LastTunedBodyDispatches()==size_t(selected&&rows<=2),"Two safety route mismatch");
                gpu.Begin();gpu.EncodeBF16MatVec(input,wide,out);gpu.Abort();need(gpu.LastCoalescedBlockDispatches()==0,"Wide abort counter survived");need(gpu.LastTunedBodyDispatches()==0,"Two abort counter survived");
                rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted wide output readable");
                gpu.Begin();gpu.EncodeBF16MatVec(input,wide,out);gpu.SubmitAndWait();need(gpu.Download(out)==control,"Wide abort recovery changed values");
                auto uninitialized=gpu.Allocate(rows,k);gpu.Begin();rejected=false;try{gpu.EncodeBF16MatVec(uninitialized,wide,out);}catch(const std::logic_error&){rejected=true;}
                need(rejected&&gpu.LastCoalescedBlockDispatches()==0,"Uninitialized wide input accepted");gpu.Abort();++checked;
            }
        }
    }
    emit(@{@"kind":@"controls",@"profiles":@(checked),@"passed":@YES});
}

int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"Expected assets and model");NSString* folder=[NSString stringWithUTF8String:argv[1]];std::string model=argv[2];
    auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(data!=nil,"Manifest missing");
    NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(manifest!=nil,"Bad manifest");controls();Garnet::MetalContext gpu;std::vector<Projection> projections;
    for(NSDictionary* c in manifest[@"cases"]){Projection p;p.name=[c[@"name"] UTF8String];p.n=[c[@"rows"] unsignedLongLongValue];p.k=[c[@"inner"] unsignedLongLongValue];
        auto raw=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:c[@"name"]] stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==p.n*p.k*2,"Bad raw size");p.bits.resize(p.n*p.k);std::memcpy(p.bits.data(),raw.bytes,raw.length);
        std::string name=[c[@"tensor"] UTF8String],suffix=".self_attn.k_proj.weight";
        p.floatKey=name.rfind("model.layers.",0)==0&&name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
        p.weight=gpu.UploadBF16(p.bits.data(),p.bits.size()*2,p.n,p.k,p.floatKey);p.selected=p.weight.CoalescedBlockReady();
        emit(@{@"kind":@"packing",@"matrix":c[@"name"],@"blocks":@(p.weight.CoalescedBlockCount()),@"compact_blocks":@(p.weight.CoalescedCompactCount()),@"selected":@(p.selected)});
        projections.push_back(std::move(p));}
    auto allProjections=std::move(projections);
    for(const auto& original:allProjections){
        if(!original.selected)continue;
        currentMatrix=[NSString stringWithUTF8String:original.name.c_str()];
        projections.clear();projections.push_back(original);
    for(size_t batch:{size_t(1),size_t(2)}){
        size_t rows=batch;
        auto inputs=[&](int pattern,bool reference){for(auto& p:projections){std::vector<float> x(rows*p.k);for(size_t r=0;r<rows;++r)for(size_t j=0;j<p.k;++j){float v=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;if(pattern==1)v*=float(j%2?-16:16);if(pattern==2)v=j==((r*53+17)%p.k)?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(j%9)-4);x[r*p.k+j]=v;}
            p.input=gpu.Upload(x,rows,p.k);p.output=gpu.Allocate(rows,p.n);if(!reference)continue;p.exact.resize(rows*p.n);p.bound.resize(rows*p.n);
            double eps=std::ldexp(1.,-23),terms=std::ceil(double(p.k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<rows;++r)for(size_t c=0;c<p.n;++c){double sum=0,absolute=0;for(size_t j=0;j<p.k;++j){double product=double(x[r*p.k+j])*expand(p.bits[c*p.k+j]);sum+=product;absolute+=std::abs(product);}p.exact[r*p.n+c]=sum;p.bound[r*p.n+c]=gamma*absolute+1e-7;}}};
        auto run=[&](int policy){Garnet::gCoalescedBlockEnabled=policy!=0;Garnet::gCoalescedBlockThreads=policy==1?64:policy==2?128:256;auto start=std::chrono::steady_clock::now();gpu.Begin();size_t expected=0;for(auto& p:projections){expected+=p.selected;gpu.EncodeBF16MatVec(p.input,p.weight,p.output);}gpu.SubmitAndWait();
            need(gpu.LastCoalescedBlockDispatches()==expected,"Packed prefill actual route mismatch");need(gpu.LastTunedBodyDispatches()==(policy?expected:0),"Two body actual route mismatch");return std::pair<double,double>{gpu.LastGPUMs(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};};
        bool admitted[]={true,true,true,true};
        for(int pattern=0;pattern<4;++pattern){inputs(pattern,true);run(0);double controlWorst=0;for(auto& p:projections){p.control=gpu.Download(p.output);for(size_t i=0;i<p.control.size();++i){need(std::isfinite(p.control[i]),"Nonfinite control");controlWorst=std::max(controlWorst,std::abs(double(p.control[i])-p.exact[i])/p.bound[i]);}}need(controlWorst<=1,"MPS control exceeded unchanged FP64 envelope");
            for(int policy=1;policy<=3;++policy){run(policy);double worst=0,comparison=0;size_t selected=0,values=0;bool finite=true,bitwise=true;for(auto& p:projections){selected+=p.selected;auto actual=gpu.Download(p.output);values+=actual.size();bitwise=bitwise&&!std::memcmp(actual.data(),p.control.data(),actual.size()*4);for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i])){finite=false;continue;}worst=std::max(worst,std::abs(double(actual[i])-p.exact[i])/p.bound[i]);comparison=std::max(comparison,std::abs(double(actual[i])-p.control[i])/(5e-6+5e-5*std::abs(double(p.control[i]))));}}
                bool passed=finite&&bitwise&&worst<=1&&comparison<=1;admitted[policy]=admitted[policy]&&passed;need(selected>0,"No packed projection selected");emit(@{@"kind":@"check",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),@"projections":@(projections.size()),@"selected_projections":@(selected),@"values":@(values),@"control_fp64_worst":@(controlWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"finite":@(finite),@"bitwise_equal":@(bitwise),@"passed":@(passed)});}
        }
        inputs(0,false);auto warm=std::chrono::steady_clock::now();do{run(0);for(int p=1;p<=3;++p)if(admitted[p])run(p);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25);
        for(int policy=1;policy<=3;++policy)if(admitted[policy])for(int round=0;round<2;++round){int position=0;for(bool candidate:{false,true,true,false}){run(candidate?policy:0);run(candidate?policy:0);NSMutableArray* g=[NSMutableArray array];NSMutableArray* w=[NSMutableArray array];for(int i=0;i<8;++i){auto t=run(candidate?policy:0);need(t.first>0&&t.second>0,"Bad GPU interval");[g addObject:@(t.first)];[w addObject:@(t.second)];}emit(@{@"kind":@"samples",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"round":@(round),@"position":@(position++),@"candidate":@(candidate),@"gpu_ms":g,@"wall_ms":w});}}
    }}return 0;
}catch(const std::exception& e){fprintf(stderr,"coalesced_body_block_failed: %s\n",e.what());return 1;}}}
