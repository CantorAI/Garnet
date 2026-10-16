// Private native-context learned component: all layers first, predetermined endpoints timed.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal_context.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>
using Clock=std::chrono::steady_clock;
static void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static std::string sha(const void* p,size_t n){unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,CC_LONG(n),d);std::ostringstream s;for(auto v:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v);return s.str();}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"Cannot encode finite JSON");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static constexpr float variants[]={1,-1,.5f,2};
static float rowScale(size_t row){static constexpr float values[]={1,.5f,-1,2};return values[row%4];}
struct Reference {
    size_t k,n;std::vector<float> base;std::vector<double> dots[2],sumabs[2];
    Reference(const uint16_t* gate,const uint16_t* up,size_t width,size_t columns):k(width),n(columns),base(k){
        uint32_t state=0x49f62b17;for(size_t c=0;c<k;++c){state=state*1664525u+1013904223u;base[c]=float(int((state>>16)%513)-256)/1024.f;}
        for(size_t side=0;side<2;++side){dots[side].resize(n);sumabs[side].resize(n);auto raw=side?up:gate;
            for(size_t c=0;c<n;++c){double sum=0,absolute=0;for(size_t j=0;j<k;++j){uint32_t bits=uint32_t(raw[c*k+j])<<16;float weight;std::memcpy(&weight,&bits,4);double product=double(base[j])*double(weight);sum+=product;absolute+=std::abs(product);}dots[side][c]=sum;sumabs[side][c]=absolute;}
        }
    }
    std::vector<float> input(size_t rows,int variant)const{std::vector<float> values(rows*k);for(size_t r=0;r<rows;++r)for(size_t j=0;j<k;++j)values[r*k+j]=base[j]*rowScale(r)*variants[variant];return values;}
    float activation(size_t row,size_t col,int variant)const{double scale=double(rowScale(row))*variants[variant];float g=float(dots[0][col]*scale),u=float(dots[1][col]*scale);return (g/(1.f+std::exp(-g)))*u;}
};
struct Ratios{double fp64=0,comparison=0,activation=0,pair=0;size_t checked=0;
    void projections(const Reference& ref,const std::vector<float>& values,size_t rows,int variant,size_t side,bool joined){
        size_t stride=ref.n*(joined?2:1);need(values.size()==rows*stride,"Projection logical shape differs");double epsilon=std::ldexp(1.,-23),steps=double((ref.k+31)/32+32),gamma=steps*epsilon/(1-steps*epsilon);
        for(size_t row=0;row<rows;++row)for(size_t col=0;col<ref.n;++col){float actual=values[row*stride+col+(joined?side*ref.n:0)];need(std::isfinite(actual),"Nonfinite projection");double scale=double(rowScale(row))*variants[variant],expected=ref.dots[side][col]*scale,error=std::abs(double(actual)-expected);
            fp64=std::max(fp64,error/(gamma*ref.sumabs[side][col]*std::abs(scale)+1e-7));comparison=std::max(comparison,error/(5e-6+5e-5*std::abs(expected)));++checked;}
    }
    void activated(const Reference& ref,const std::vector<float>& values,size_t rows,int variant){need(values.size()==rows*ref.n,"Activation logical shape differs");for(size_t row=0;row<rows;++row)for(size_t col=0;col<ref.n;++col){float actual=values[row*ref.n+col],expected=ref.activation(row,col,variant);need(std::isfinite(actual)&&std::isfinite(expected),"Nonfinite activation");activation=std::max(activation,std::abs(double(actual)-expected)/(5e-6+5e-5*std::abs(double(expected))));++checked;}}
    void compare(const std::vector<float>& control,const std::vector<float>& candidate){need(control.size()==candidate.size(),"Comparison shape differs");for(size_t i=0;i<control.size();++i)pair=std::max(pair,std::abs(double(candidate[i])-control[i])/(5e-6+5e-5*std::abs(double(control[i]))));}
    bool passed()const{return fp64<=1&&comparison<=1&&activation<=1&&pair<=1;}
};
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"Expected CPU report and numeric/timing mode");std::string mode=argv[2];need(mode=="numeric"||mode=="timing","Unknown mode");
    auto data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];need(data!=nil,"CPU report missing");auto report=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(report!=nil,"CPU report invalid");
    Garnet::MetalContext anchor;size_t profiles=0,numericCases=0,fallbackCases=0,timedProfiles=0;Ratios complete;size_t allChecked=0;
    for(NSDictionary* model in report[@"models"]){@autoreleasepool{
        auto file=[NSData dataWithContentsOfFile:model[@"checkpoint"] options:NSDataReadingMappedIfSafe error:nil];need(file!=nil,"Checkpoint missing");NSString* name=model[@"model"];size_t layers=[model[@"layers"] unsignedLongLongValue];
        size_t firstRows=[name isEqualToString:@"Qwen3-0.6B"]?24:([name isEqualToString:@"Qwen2.5-Coder-0.5B-Instruct"]?26:80);
        size_t secondRows=[name isEqualToString:@"Qwen3-0.6B"]?44:firstRows*2;
        for(NSDictionary* pair in model[@"pairs"]){@autoreleasepool{
            size_t layer=[pair[@"layer"] unsignedLongLongValue];bool endpoint=layer==0||layer+1==layers;if(mode=="timing"&&!endpoint)continue;
            size_t k=[pair[@"inner"] unsignedLongLongValue],n=[pair[@"rows"] unsignedLongLongValue];const uint16_t* raw[2];
            for(size_t side=0;side<2;++side){NSDictionary* item=pair[@"sides"][side];size_t offset=[item[@"data_offset"] unsignedLongLongValue],bytes=[item[@"original_bytes"] unsignedLongLongValue];need(bytes==n*k*2&&offset<=file.length&&bytes<=file.length-offset,"Learned slice bounds differ");auto source=static_cast<const unsigned char*>(file.bytes)+offset;need(sha(source,bytes)==std::string([item[@"raw_sha256"] UTF8String]),"Learned slice hash differs");raw[side]=reinterpret_cast<const uint16_t*>(source);}
            Garnet::MetalContext gpu;auto uploadStart=Clock::now();auto gate=gpu.UploadBF16(raw[0],n*k*2,n,k),up=gpu.UploadBF16(raw[1],n*k*2,n,k);double uploadMs=ms(uploadStart,Clock::now());Reference ref(raw[0],raw[1],k,n);
            for(size_t rows:{firstRows,secondRows}){@autoreleasepool{
                bool admitted=[pair[@"union_admitted"] boolValue];auto prepareStart=Clock::now();bool prepared=gpu.PrepareUnionMlp(gate,up,rows);double preparationMs=ms(prepareStart,Clock::now());need(prepared==admitted,"Actual prepared admission differs from all-value CPU evidence");
                auto input=gpu.Allocate(rows,k),g=gpu.Allocate(rows,n),u=gpu.Allocate(rows,n),output=gpu.Allocate(rows,n);Ratios ratios;size_t executions=0,epoch=0;
                auto execute=[&](bool candidate,int variant){@autoreleasepool{auto values=ref.input(rows,variant);gpu.Update(input,values);auto begin=Clock::now();gpu.Begin();
                    if(candidate)need(gpu.EncodePreparedUnionMlp(input,gate,up,output),"Prepared candidate unexpectedly fell back");else{gpu.EncodeBF16MatVec(input,gate,g);gpu.EncodeBF16MatVec(input,up,u);gpu.EncodeSiluMul(g,u,output);}auto encoded=Clock::now();gpu.SubmitAndWait();auto done=Clock::now();need(gpu.LastPreparedUnionMlpDispatches()==size_t(candidate),"Actual union dispatch count differs");++executions;return std::array<double,3>{ms(begin,encoded),ms(begin,done),gpu.LastGPUMs()};}};
                auto validate=[&](bool candidate,int variant){auto activation=gpu.Download(output);ratios.activated(ref,activation,rows,variant);if(candidate){auto joined=gpu.DownloadPreparedUnion(gate,up,rows);ratios.projections(ref,joined,rows,variant,0,true);ratios.projections(ref,joined,rows,variant,1,true);}else{ratios.projections(ref,gpu.Download(g),rows,variant,0,false);ratios.projections(ref,gpu.Download(u),rows,variant,1,false);}if(!ratios.passed())emit(@{@"kind":@"numerical_failure",@"model":name,@"layer":@(layer),@"rows":@(rows),@"candidate":@(candidate),@"variant":@(variant),@"fp64_ratio":@(ratios.fp64),@"comparison_ratio":@(ratios.comparison),@"activation_ratio":@(ratios.activation),@"control_candidate_ratio":@(ratios.pair)});need(ratios.passed(),"Original projection/activation numerical envelope failed");return activation;};
                std::array<double,3> firstControl{},firstCandidate{};
                for(int variant=0;variant<4;++variant){auto time=execute(false,variant);if(!variant)firstControl=time;auto control=validate(false,variant);
                    if(prepared){time=execute(true,variant);if(!variant)firstCandidate=time;auto actual=validate(true,variant);ratios.compare(control,actual);auto joined=gpu.DownloadPreparedUnion(gate,up,rows);auto cg=gpu.Download(g),cu=gpu.Download(u);for(size_t row=0;row<rows;++row)for(size_t col=0;col<n;++col){double a=joined[row*2*n+col],b=joined[row*2*n+n+col];ratios.pair=std::max(ratios.pair,std::abs(a-cg[row*n+col])/(5e-6+5e-5*std::abs(double(cg[row*n+col]))));ratios.pair=std::max(ratios.pair,std::abs(b-cu[row*n+col])/(5e-6+5e-5*std::abs(double(cu[row*n+col]))));}}
                    else{gpu.Begin();need(!gpu.EncodePreparedUnionMlp(input,gate,up,output)&&gpu.LastPreparedUnionMlpDispatches()==0,"Rejected learned pair encoded candidate work");gpu.Abort();need(gpu.Download(output)==control,"Rejected learned pair changed original output");++fallbackCases;}
                    ++numericCases;
                }
                emit(@{@"kind":@"numeric",@"mode":[NSString stringWithUTF8String:mode.c_str()],@"model":name,@"layer":@(layer),@"rows":@(rows),@"columns":@(n),@"inner":@(k),@"prepared":@(prepared),@"variants":@4,@"checked_values":@(ratios.checked),@"fp64_ratio":@(ratios.fp64),@"comparison_ratio":@(ratios.comparison),@"activation_ratio":@(ratios.activation),@"control_candidate_ratio":@(ratios.pair),@"passed":@(ratios.passed()),@"upload_ms":@(uploadMs),@"prepare_ms":@(preparationMs),@"extra_union_weight_bytes":@(gpu.PreparedUnionWeightBytes()),@"logical_joined_scratch_bytes":@(rows*2*n*4),@"first_control_wall_ms":@(firstControl[1]),@"first_control_gpu_ms":@(firstControl[2]),@"first_candidate_wall_ms":@(firstCandidate[1]),@"first_candidate_gpu_ms":@(firstCandidate[2])});
                need(ratios.passed(),"Original control/candidate envelope failed");allChecked+=ratios.checked;++profiles;
                if(mode=="timing"){
                    need(endpoint&&prepared,"Timing requires predeclared admitted endpoint");
                    for(bool nullMode:{true,false}){
                        auto warmStart=Clock::now();do{execute(false,int(epoch%4));execute(!nullMode,int(epoch++%4));}while(ms(warmStart,Clock::now())<2000);double warmMs=ms(warmStart,Clock::now());
                        emit(@{@"kind":@"warmup",@"model":name,@"layer":@(layer),@"rows":@(rows),@"null":@(nullMode),@"wall_ms":@(warmMs)});
                        for(size_t round=0;round<10;++round)for(size_t window=0;window<4;++window){bool b=window==1||window==2,candidate=b&&!nullMode;
                            for(size_t i=0;i<3;++i)execute(candidate,int(epoch++%4));auto encodes=[NSMutableArray new],walls=[NSMutableArray new],gpus=[NSMutableArray new];int last=0;
                            for(size_t i=0;i<5;++i){last=int(epoch++%4);auto sample=execute(candidate,last);need(sample[2]>0&&sample[1]>0,"Invalid GPU/wall clock");[encodes addObject:@(sample[0])];[walls addObject:@(sample[1])];[gpus addObject:@(sample[2])];}
                            validate(candidate,last);emit(@{@"kind":@"timing",@"model":name,@"layer":@(layer),@"rows":@(rows),@"round":@(round),@"window":@(window),@"null":@(nullMode),@"label_b":@(b),@"candidate":@(candidate),@"encode_ms":encodes,@"wall_ms":walls,@"gpu_ms":gpus,@"complete_output_checked_after_window":@YES});
                        }
                    }++timedProfiles;
                }
                complete.fp64=std::max(complete.fp64,ratios.fp64);complete.comparison=std::max(complete.comparison,ratios.comparison);complete.activation=std::max(complete.activation,ratios.activation);complete.pair=std::max(complete.pair,ratios.pair);
                emit(@{@"kind":@"profile_complete",@"model":name,@"layer":@(layer),@"rows":@(rows),@"executions":@(executions),@"passed":@(ratios.passed())});
            }}
        }}
    }}
    need(profiles==(mode=="numeric"?152:12)&&fallbackCases==(mode=="numeric"?8:0),"Declared complete profile/fallback coverage differs");struct rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"RSS unavailable");
    emit(@{@"kind":@"complete",@"mode":[NSString stringWithUTF8String:mode.c_str()],@"profiles":@(profiles),@"changed_input_cases":@(numericCases),@"fallback_cases":@(fallbackCases),@"timed_profiles":@(timedProfiles),@"checked_values":@(allChecked),@"worst_fp64_ratio":@(complete.fp64),@"worst_comparison_ratio":@(complete.comparison),@"worst_activation_ratio":@(complete.activation),@"worst_control_candidate_ratio":@(complete.pair),@"process_peak_rss_bytes":@(usage.ru_maxrss),@"scope":@"Learned component with structured synthetic inputs; not full-model inference or captured activations"});return 0;
}catch(const std::exception& error){fprintf(stderr,"prepared_union_learned_failed: %s\n",error.what());return 1;}}}
