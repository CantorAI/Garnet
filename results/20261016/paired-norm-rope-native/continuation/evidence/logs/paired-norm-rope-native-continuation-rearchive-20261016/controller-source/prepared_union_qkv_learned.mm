// All learned separate Q/K/V groups, original FP32 keys, exact scatter, predetermined timings.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <CommonCrypto/CommonDigest.h>
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
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static std::string sha(const void* p,size_t n){unsigned char result[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,CC_LONG(n),result);std::ostringstream s;for(auto v:result)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v);return s.str();}
static void emit(NSDictionary* object){auto data=[NSJSONSerialization dataWithJSONObject:object options:0 error:nil];need(data!=nil,"Finite JSON encoding failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static constexpr float variants[]={1,-1,.5f,2};
static float rowScale(size_t row){static constexpr float values[]={1,.5f,-1,2};return values[row%4];}
struct Reference {
    size_t inner,total=0;std::array<size_t,3> columns;std::vector<float> base;std::array<std::vector<double>,3> dots,sumabs;
    Reference(const std::array<const uint16_t*,3>& raw,size_t k,std::array<size_t,3> n):inner(k),columns(n),base(k){
        uint32_t state=0x49f62b17;for(size_t j=0;j<k;++j){state=state*1664525u+1013904223u;base[j]=float(int((state>>16)%513)-256)/1024.f;}
        for(size_t side=0;side<3;++side){total+=n[side];dots[side].resize(n[side]);sumabs[side].resize(n[side]);for(size_t col=0;col<n[side];++col){double sum=0,absolute=0;for(size_t j=0;j<k;++j){uint32_t bits=uint32_t(raw[side][col*k+j])<<16;float weight;std::memcpy(&weight,&bits,4);double product=double(base[j])*double(weight);sum+=product;absolute+=std::abs(product);}dots[side][col]=sum;sumabs[side][col]=absolute;}}
    }
    std::vector<float> input(size_t rows,int variant)const{std::vector<float> result(rows*inner);for(size_t r=0;r<rows;++r)for(size_t j=0;j<inner;++j)result[r*inner+j]=base[j]*rowScale(r)*variants[variant];return result;}
};
struct Ratios {
    double fp64=0,comparison=0,pair=0;size_t checked=0,scatter=0;
    void values(const Reference& ref,const std::vector<float>& actual,size_t rows,int variant,size_t side,bool joined){
        size_t stride=joined?ref.total:ref.columns[side],offset=0;if(joined)for(size_t s=0;s<side;++s)offset+=ref.columns[s];need(actual.size()==rows*stride,"Projection shape differs");double epsilon=std::ldexp(1.,-23),steps=double((ref.inner+31)/32+32),gamma=steps*epsilon/(1-steps*epsilon);
        for(size_t row=0;row<rows;++row)for(size_t col=0;col<ref.columns[side];++col){float got=actual[row*stride+offset+col];need(std::isfinite(got),"Nonfinite projection");double scale=double(rowScale(row))*variants[variant],expected=ref.dots[side][col]*scale,error=std::abs(double(got)-expected);fp64=std::max(fp64,error/(gamma*ref.sumabs[side][col]*std::abs(scale)+1e-7));comparison=std::max(comparison,error/(5e-6+5e-5*std::abs(expected)));++checked;}
    }
    void compare(const std::vector<float>& control,const std::vector<float>& candidate){need(control.size()==candidate.size(),"Control/candidate shape differs");for(size_t i=0;i<control.size();++i)pair=std::max(pair,std::abs(double(candidate[i])-control[i])/(5e-6+5e-5*std::abs(double(control[i]))));}
    void scatterBits(const Reference& ref,const std::array<std::vector<float>,3>& outputs,const std::vector<float>& joined,size_t rows){size_t offset=0;for(size_t side=0;side<3;++side){for(size_t row=0;row<rows;++row)for(size_t col=0;col<ref.columns[side];++col){need(!std::memcmp(&outputs[side][row*ref.columns[side]+col],&joined[row*ref.total+offset+col],4),"Scatter changed original joined bits");++scatter;}offset+=ref.columns[side];}}
    bool passed()const{return fp64<=1&&comparison<=1&&pair<=1;}
};
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"Expected captured CPU report and numeric/timing mode");std::string mode=argv[2];need(mode=="numeric"||mode=="timing","Unknown mode");
    auto data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];need(data!=nil,"Report unavailable");NSDictionary* report=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(report!=nil,"Invalid report");
    Garnet::MetalContext anchor;size_t profiles=0,cases=0,fallback=0,timed=0,checked=0,scatter=0,fp32Keys=0;Ratios complete;
    for(NSDictionary* model in report[@"models"]){@autoreleasepool{
        NSString* name=model[@"model"];auto file=[NSData dataWithContentsOfFile:model[@"checkpoint"] options:NSDataReadingMappedIfSafe error:nil];need(file!=nil,"Checkpoint unavailable");size_t firstRows=[name isEqualToString:@"Qwen3-0.6B"]?24:([name isEqualToString:@"Qwen2.5-Coder-0.5B-Instruct"]?26:80),secondRows=firstRows==24?44:firstRows*2;
        size_t first=SIZE_MAX,last=0;for(NSDictionary* group in model[@"groups"])if([group[@"union_admitted"] boolValue]){size_t layer=[group[@"layer"] unsignedLongLongValue];first=std::min(first,layer);last=std::max(last,layer);}need(first!=SIZE_MAX,"No admitted endpoints");
        for(NSDictionary* group in model[@"groups"]){@autoreleasepool{
            size_t layer=[group[@"layer"] unsignedLongLongValue];bool endpoint=layer==first||layer==last;if(mode=="timing"&&!endpoint)continue;
            size_t inner=[group[@"inner"] unsignedLongLongValue];std::array<size_t,3> n;std::array<const uint16_t*,3> raw;std::array<bool,3> floatKey;
            for(size_t side=0;side<3;++side){NSDictionary* item=group[@"sides"][side];n[side]=[item[@"rows"] unsignedLongLongValue];size_t offset=[item[@"data_offset"] unsignedLongLongValue],bytes=[item[@"original_bytes"] unsignedLongLongValue];need(bytes==n[side]*inner*2&&offset<=file.length&&bytes<=file.length-offset,"Learned slice bounds differ");auto source=static_cast<const unsigned char*>(file.bytes)+offset;need(sha(source,bytes)==std::string([item[@"raw_sha256"] UTF8String]),"Learned slice hash differs");raw[side]=reinterpret_cast<const uint16_t*>(source);floatKey[side]=[item[@"retained_float_key"] boolValue];fp32Keys+=floatKey[side];}
            Garnet::MetalContext gpu;auto uploadStart=Clock::now();std::array<Garnet::MetalBF16Weight,3> weights;
            for(size_t side=0;side<3;++side)weights[side]=gpu.UploadBF16(raw[side],n[side]*inner*2,n[side],inner,floatKey[side]);double uploadMs=ms(uploadStart,Clock::now());auto& q=weights[0];auto& k=weights[1];auto& v=weights[2];Reference ref(raw,inner,n);
            for(size_t rows:{firstRows,secondRows}){@autoreleasepool{
                bool admitted=[group[@"union_admitted"] boolValue];auto prepareStart=Clock::now();bool prepared=gpu.PrepareUnionQkv(q,k,v,rows);double prepareMs=ms(prepareStart,Clock::now());need(prepared==admitted,"GPU-context preparation differs from CPU admission");auto input=gpu.Allocate(rows,inner);std::array<Garnet::MetalMatrix,3> outputs{gpu.Allocate(rows,n[0]),gpu.Allocate(rows,n[1]),gpu.Allocate(rows,n[2])};Ratios ratios;size_t executions=0,epoch=0;
                auto execute=[&](bool candidate,int variant){@autoreleasepool{auto values=ref.input(rows,variant);gpu.Update(input,values);auto begin=Clock::now();gpu.Begin();if(candidate)need(gpu.EncodePreparedUnionQkv(input,q,k,v,outputs[0],outputs[1],outputs[2]),"Admitted QKV unexpectedly fell back");else for(size_t side=0;side<3;++side)gpu.EncodeBF16MatVec(input,weights[side],outputs[side]);auto encoded=Clock::now();gpu.SubmitAndWait();auto done=Clock::now();need(gpu.LastPreparedUnionQkvDispatches()==size_t(candidate),"Actual QKV dispatch counter differs");++executions;return std::array<double,3>{ms(begin,encoded),ms(begin,done),gpu.LastGPUMs()};}};
                auto validate=[&](bool candidate,int variant){std::array<std::vector<float>,3> actual;for(size_t side=0;side<3;++side){actual[side]=gpu.Download(outputs[side]);ratios.values(ref,actual[side],rows,variant,side,false);}if(candidate){auto joined=gpu.DownloadPreparedUnionQkv(q,k,v,rows);for(size_t side=0;side<3;++side)ratios.values(ref,joined,rows,variant,side,true);ratios.scatterBits(ref,actual,joined,rows);}if(!ratios.passed())emit(@{@"kind":@"numerical_failure",@"model":name,@"layer":@(layer),@"rows":@(rows),@"variant":@(variant),@"candidate":@(candidate),@"fp64_ratio":@(ratios.fp64),@"comparison_ratio":@(ratios.comparison),@"control_candidate_ratio":@(ratios.pair)});need(ratios.passed(),"Original projection/reference envelope failed");return actual;};
                std::array<double,3> firstControl{},firstCandidate{};
                for(int variant=0;variant<4;++variant){auto time=execute(false,variant);if(!variant)firstControl=time;auto control=validate(false,variant);
                    if(prepared){time=execute(true,variant);if(!variant)firstCandidate=time;auto actual=validate(true,variant);for(size_t side=0;side<3;++side)ratios.compare(control[side],actual[side]);}
                    else{gpu.Begin();need(!gpu.EncodePreparedUnionQkv(input,q,k,v,outputs[0],outputs[1],outputs[2])&&gpu.LastPreparedUnionQkvDispatches()==0,"Required fallback encoded candidate work");gpu.Abort();for(size_t side=0;side<3;++side)need(gpu.Download(outputs[side])==control[side],"Rejected group changed original output");++fallback;}++cases;
                }
                emit(@{@"kind":@"numeric",@"mode":[NSString stringWithUTF8String:mode.c_str()],@"model":name,@"layer":@(layer),@"rows":@(rows),@"inner":@(inner),@"q_columns":@(n[0]),@"k_columns":@(n[1]),@"v_columns":@(n[2]),@"prepared":@(prepared),@"fp32_key":@(floatKey[1]),@"variants":@4,@"checked_values":@(ratios.checked),@"scatter_bit_checks":@(ratios.scatter),@"fp64_ratio":@(ratios.fp64),@"comparison_ratio":@(ratios.comparison),@"control_candidate_ratio":@(ratios.pair),@"passed":@(ratios.passed()),@"upload_ms":@(uploadMs),@"prepare_ms":@(prepareMs),@"extra_union_weight_bytes":@(gpu.PreparedUnionQkvWeightBytes()),@"logical_joined_scratch_bytes":@(rows*ref.total*4),@"first_control_wall_ms":@(firstControl[1]),@"first_control_gpu_ms":@(firstControl[2]),@"first_candidate_wall_ms":@(firstCandidate[1]),@"first_candidate_gpu_ms":@(firstCandidate[2])});
                need(ratios.passed(),"Original control/candidate comparison envelope failed");++profiles;checked+=ratios.checked;scatter+=ratios.scatter;
                if(mode=="timing"){
                    need(endpoint&&prepared,"Timing requires predetermined admitted endpoint");
                    for(bool nullMode:{true,false}){auto warm=Clock::now();do{execute(false,int(epoch%4));execute(!nullMode,int(epoch++%4));}while(ms(warm,Clock::now())<2000);emit(@{@"kind":@"warmup",@"model":name,@"layer":@(layer),@"rows":@(rows),@"null":@(nullMode),@"wall_ms":@(ms(warm,Clock::now()))});
                        for(size_t round=0;round<10;++round)for(size_t window=0;window<4;++window){bool b=window==1||window==2,candidate=b&&!nullMode;for(size_t i=0;i<3;++i)execute(candidate,int(epoch++%4));auto encodes=[NSMutableArray new],walls=[NSMutableArray new],gpus=[NSMutableArray new];int lastVariant=0;
                            for(size_t i=0;i<5;++i){lastVariant=int(epoch++%4);auto sample=execute(candidate,lastVariant);need(sample[1]>0&&sample[2]>0,"Invalid wall/GPU timing");[encodes addObject:@(sample[0])];[walls addObject:@(sample[1])];[gpus addObject:@(sample[2])];}validate(candidate,lastVariant);emit(@{@"kind":@"timing",@"model":name,@"layer":@(layer),@"rows":@(rows),@"round":@(round),@"window":@(window),@"null":@(nullMode),@"label_b":@(b),@"candidate":@(candidate),@"encode_ms":encodes,@"wall_ms":walls,@"gpu_ms":gpus,@"complete_output_checked_after_window":@YES});}
                    }++timed;
                }
                complete.fp64=std::max(complete.fp64,ratios.fp64);complete.comparison=std::max(complete.comparison,ratios.comparison);complete.pair=std::max(complete.pair,ratios.pair);emit(@{@"kind":@"profile_complete",@"model":name,@"layer":@(layer),@"rows":@(rows),@"executions":@(executions),@"passed":@(ratios.passed())});
            }}
        }}
    }}
    need(profiles==(mode=="numeric"?116:12)&&fallback==(mode=="numeric"?16:0)&&fp32Keys==(mode=="numeric"?52:4),"Complete learned coverage/fallback/key count differs");struct rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"Peak RSS unavailable");emit(@{@"kind":@"complete",@"mode":[NSString stringWithUTF8String:mode.c_str()],@"profiles":@(profiles),@"changed_input_cases":@(cases),@"fallback_cases":@(fallback),@"timed_profiles":@(timed),@"original_fp32_key_matrices":@(fp32Keys),@"checked_values":@(checked),@"scatter_bit_checks":@(scatter),@"worst_fp64_ratio":@(complete.fp64),@"worst_comparison_ratio":@(complete.comparison),@"worst_control_candidate_ratio":@(complete.pair),@"process_peak_rss_bytes":@(usage.ru_maxrss),@"scope":@"Structured learned component inputs only; no full-model/native performance proof"});return 0;
}catch(const std::exception& error){fprintf(stderr,"prepared_qkv_learned_failed: %s\n",error.what());return 1;}}}
