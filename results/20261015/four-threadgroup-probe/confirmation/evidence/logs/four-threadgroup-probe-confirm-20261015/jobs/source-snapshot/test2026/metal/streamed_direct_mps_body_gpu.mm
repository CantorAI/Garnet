// Native-context component: independent real projections in one submission.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace Garnet {extern bool gDirectMPS;}
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static void emit(NSDictionary* v){auto d=[NSJSONSerialization dataWithJSONObject:v options:0 error:nil];need(d!=nil,"JSON failed");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout);}
static float expand(uint16_t v){uint32_t b=uint32_t(v)<<16;float x;std::memcpy(&x,&b,4);return x;}
struct Projection{std::string name;size_t n,k;Garnet::MetalBF16Weight weight;bool admitted,mlp;std::vector<uint16_t> bits;Garnet::MetalMatrix input,output;std::vector<float> reference;std::vector<double> exact,bound;};
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"Expected assets");NSString* folder=[NSString stringWithUTF8String:argv[1]];auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(data!=nil,"Manifest missing");
    NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(manifest!=nil,"Bad manifest");Garnet::MetalContext gpu;std::vector<Projection> projections;
    for(NSDictionary* c in manifest[@"cases"]){Projection p;p.name=[c[@"name"] UTF8String];p.n=[c[@"rows"] unsignedLongLongValue];p.k=[c[@"inner"] unsignedLongLongValue];p.mlp=[c[@"tensor"] rangeOfString:@".mlp."].location!=NSNotFound;
        auto raw=[NSData dataWithContentsOfFile:[[folder stringByAppendingPathComponent:c[@"name"]] stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==p.n*p.k*2,"Bad raw size");p.bits.resize(p.n*p.k);std::memcpy(p.bits.data(),raw.bytes,raw.length);
        p.weight=gpu.UploadBF16(p.bits.data(),p.bits.size()*2,p.n,p.k);p.admitted=p.weight.GlobalHalfReady();projections.push_back(std::move(p));}
    for(size_t batch:{size_t(1),size_t(2)}){
        for(auto& p:projections){std::vector<float> x(batch*p.k);for(size_t r=0;r<batch;++r)for(size_t j=0;j<p.k;++j)x[r*p.k+j]=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;
            p.input=gpu.Upload(x,batch,p.k);p.output=gpu.Allocate(batch,p.n);p.exact.resize(batch*p.n);p.bound.resize(batch*p.n);
            double eps=std::ldexp(1.,-23),terms=std::ceil(double(p.k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<batch;++r)for(size_t c=0;c<p.n;++c){double sum=0,absolute=0;for(size_t j=0;j<p.k;++j){double product=double(x[r*p.k+j])*double(expand(p.bits[c*p.k+j]));sum+=product;absolute+=std::abs(product);}p.exact[r*p.n+c]=sum;p.bound[r*p.n+c]=gamma*absolute+1e-7;}}
        auto selected=[](const Projection& p,int policy){return p.admitted&&(policy==1||(policy==2&&p.mlp)||(policy==3&&p.n*p.k>=1024*1024));};
        auto run=[&](int policy){size_t expected=0;auto start=std::chrono::steady_clock::now();gpu.Begin();for(auto& p:projections){Garnet::gDirectMPS=selected(p,policy);expected+=Garnet::gDirectMPS;gpu.EncodeBF16MatVec(p.input,p.weight,p.output);}gpu.SubmitAndWait();
            double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();need(gpu.LastGlobalHalfAlphaDispatches()==expected,"Actual streamed route counts differ");return std::pair<double,double>{gpu.LastGPUMs(),wall};};
        run(0);double rawWorst=0;for(auto& p:projections){p.reference=gpu.Download(p.output);for(size_t i=0;i<p.reference.size();++i){need(std::isfinite(p.reference[i])&&std::isfinite(p.exact[i]),"Nonfinite reference");rawWorst=std::max(rawWorst,std::abs(double(p.reference[i])-p.exact[i])/p.bound[i]);}}need(rawWorst<=1,"Original streamed control exceeds FP64 envelope");
        bool admitted[]={true,true,true,true};
        for(int policy=1;policy<4;++policy){run(policy);double worst=0,comparison=0;size_t values=0,forced=0;bool finite=true;
            for(auto& p:projections){forced+=selected(p,policy);auto actual=gpu.Download(p.output);values+=actual.size();for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i])){finite=false;continue;}worst=std::max(worst,std::abs(double(actual[i])-p.exact[i])/p.bound[i]);comparison=std::max(comparison,std::abs(double(actual[i])-p.reference[i])/(5e-6+5e-5*std::abs(double(p.reference[i]))));}}
            admitted[policy]=finite&&worst<=1&&comparison<=1;emit(@{@"kind":@"check",@"batch":@(batch),@"policy":@(policy),@"projections":@(projections.size()),@"forced_mps_projections":@(forced),@"values":@(values),@"finite":@(finite),@"raw_fp64_worst":@(rawWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"passed":@(admitted[policy])});}
        auto warm=std::chrono::steady_clock::now();do{run(0);for(int p=1;p<4;++p)if(admitted[p])run(p);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25);
        for(int policy=1;policy<4;++policy)if(admitted[policy])for(int round=0;round<2;++round){int position=0;for(bool candidate:{false,true,true,false}){int p=candidate?policy:0;run(p);run(p);NSMutableArray* g=[NSMutableArray array];NSMutableArray* w=[NSMutableArray array];for(int i=0;i<8;++i){auto t=run(p);need(t.first>0&&t.second>0,"Bad interval");[g addObject:@(t.first)];[w addObject:@(t.second)];}emit(@{@"kind":@"samples",@"batch":@(batch),@"policy":@(policy),@"round":@(round),@"position":@(position++),@"candidate":@(candidate),@"gpu_ms":g,@"wall_ms":w});}}
    }return 0;
}catch(const std::exception& e){fprintf(stderr,"streamed_direct_mps_failed: %s\n",e.what());return 1;}}}
