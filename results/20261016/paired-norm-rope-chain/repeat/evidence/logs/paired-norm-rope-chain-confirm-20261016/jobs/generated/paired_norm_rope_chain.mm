// Actual learned Q/K projections/norm weights. Compare every FP32 output bit to original calls.
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
static NSString* hash(const std::vector<float>& v){return [NSString stringWithUTF8String:sha(v.data(),v.size()*sizeof(float)).c_str()];}
static void emit(NSDictionary* object){auto data=[NSJSONSerialization dataWithJSONObject:object options:0 error:nil];need(data!=nil,"JSON failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static std::vector<float> hidden(size_t n,uint32_t seed){std::vector<float> result(n);for(auto& v:result){seed=seed*1664525u+1013904223u;v=float(int((seed>>8)%513)-256)/1024.f;}return result;}
static const unsigned char* slice(NSData* file,NSDictionary* item,size_t bytes){size_t offset=[item[@"data_offset"] unsignedLongLongValue];need(offset<=file.length&&bytes<=file.length-offset,"Learned slice bounds differ");auto source=static_cast<const unsigned char*>(file.bytes)+offset;need(sha(source,bytes)==std::string([item[@"raw_sha256"] UTF8String]),"Learned slice hash differs");return source;}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==3,"Expected learned report and mode");std::string mode=argv[2];need(mode=="numeric"||mode=="timing","Invalid mode");auto data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];need(data!=nil,"Report unavailable");NSDictionary* report=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(report!=nil,"Invalid report");
 Garnet::MetalContext anchor;size_t profiles=0,cases=0,checked=0,timed=0;
 for(NSDictionary* model in report[@"models"]){@autoreleasepool{
  NSString* name=model[@"model"];NSDictionary* c=model[@"rope"];size_t qHeads=[c[@"query_heads"] unsignedLongLongValue],kHeads=[c[@"key_heads"] unsignedLongLongValue],dim=[c[@"dim"] unsignedLongLongValue],rotary=[c[@"rotary"] unsignedLongLongValue];bool normalize=[c[@"normalize"] boolValue],gated=[c[@"query_gated"] boolValue];float offset=[c[@"weight_offset"] floatValue],epsilon=[c[@"epsilon"] floatValue];double theta=[c[@"theta"] doubleValue];size_t h=[c[@"sections"][1] unsignedLongLongValue],w=[c[@"sections"][2] unsignedLongLongValue];
  auto file=[NSData dataWithContentsOfFile:model[@"checkpoint"] options:NSDataReadingMappedIfSafe error:nil];need(file!=nil,"Checkpoint unavailable");size_t first=SIZE_MAX,last=0;for(NSDictionary* g in model[@"groups"]){size_t layer=[g[@"layer"] unsignedLongLongValue];first=std::min(first,layer);last=std::max(last,layer);}
  for(NSDictionary* group in model[@"groups"]){@autoreleasepool{
   size_t layer=[group[@"layer"] unsignedLongLongValue];bool endpoint=layer==first||layer==last;if(mode=="timing"&&!endpoint)continue;size_t inner=[group[@"inner"] unsignedLongLongValue];Garnet::MetalContext gpu;
   std::array<Garnet::MetalBF16Weight,2> projections;std::array<size_t,2> projectionColumns{};
   for(size_t side=0;side<2;++side){NSDictionary* item=group[@"sides"][side];size_t n=[item[@"rows"] unsignedLongLongValue],bytes=[item[@"original_bytes"] unsignedLongLongValue];need(bytes==n*inner*2,"Projection bytes differ");auto raw=slice(file,item,bytes);projections[side]=gpu.UploadBF16(raw,bytes,n,inner,[item[@"retained_float_key"] boolValue]);projectionColumns[side]=n;}
   need(projectionColumns[0]==qHeads*dim*(gated?2:1)&&projectionColumns[1]==kHeads*dim,"Projection geometry differs");
   std::array<Garnet::MetalMatrix,2> weights;
   for(size_t side=0;side<2;++side){NSDictionary* item=group[@"norms"][side];std::vector<float> values(dim,1);if(![item[@"ones"] boolValue]){size_t bytes=[item[@"bytes"] unsignedLongLongValue];auto raw=slice(file,item,bytes);bool bf16=[item[@"dtype"] isEqualToString:@"BF16"];need(bytes==dim*(bf16?2:4),"Norm bytes differ");for(size_t i=0;i<dim;++i){if(bf16){uint16_t word;std::memcpy(&word,raw+2*i,2);uint32_t bits=uint32_t(word)<<16;std::memcpy(&values[i],&bits,4);}else std::memcpy(&values[i],raw+4*i,4);}}weights[side]=gpu.Upload(values,1,dim);}
   std::array<size_t,4> rowCounts{1,2,[c[@"prefill_rows"][0] unsignedLongLongValue],[c[@"prefill_rows"][1] unsignedLongLongValue]};
   for(size_t rows:rowCounts){@autoreleasepool{
    auto source=gpu.Allocate(rows,inner),qRaw=gpu.Allocate(rows,projectionColumns[0]),kRaw=gpu.Allocate(rows,projectionColumns[1]);std::array<std::array<std::vector<float>,2>,4> inputs;std::array<std::vector<std::array<std::vector<float>,2>>,4> expected;
    auto inputHashes=[NSMutableArray new],outputHashes=[NSMutableArray new];
    for(size_t variant=0;variant<4;++variant){gpu.Update(source,hidden(rows*inner,uint32_t(0x39d1+layer*101+variant*991)));gpu.Begin();gpu.EncodeBF16MatVec(source,projections[0],qRaw);gpu.EncodeBF16MatVec(source,projections[1],kRaw);gpu.SubmitAndWait();auto raw=gpu.Download(qRaw);inputs[variant][1]=gpu.Download(kRaw);if(gated){inputs[variant][0].resize(rows*qHeads*dim);for(size_t row=0;row<rows;++row)for(size_t head=0;head<qHeads;++head)std::memcpy(inputs[variant][0].data()+(row*qHeads+head)*dim,raw.data()+row*projectionColumns[0]+head*2*dim,dim*sizeof(float));}else inputs[variant][0]=std::move(raw);[inputHashes addObject:@[hash(inputs[variant][0]),hash(inputs[variant][1])]];}
    auto query=gpu.Upload(inputs[0][0],rows,qHeads*dim),key=gpu.Upload(inputs[0][1],rows,kHeads*dim);
    constexpr size_t chainStages=32;std::vector<std::array<Garnet::MetalMatrix,2>> chain;for(size_t stage=0;stage<chainStages;++stage)chain.push_back({gpu.Allocate(rows,qHeads*dim),gpu.Allocate(rows,kHeads*dim)});
    std::vector<float> positions(rows*3),frequencies(rotary);for(size_t row=0;row<rows;++row)for(size_t axis=0;axis<3;++axis)positions[row*3+axis]=float((row*17+axis*11)%193);
    for(size_t i=0;i<rotary/2;++i){frequencies[i]=float(std::pow(theta,-2.*double(i)/double(rotary)));frequencies[rotary/2+i]=(i%3==1&&i<h*3)?1.f:((i%3==2&&i<w*3)?2.f:0.f);}
    auto p=gpu.Upload(positions,rows,3),f=gpu.Upload(frequencies,2,rotary/2);size_t epoch=0,executions=0,bitChecks=0;
    auto execute=[&](bool candidate,int variant){gpu.Update(query,inputs[variant][0]);gpu.Update(key,inputs[variant][1]);auto begin=Clock::now();gpu.Begin();for(size_t stage=0;stage<chainStages;++stage){const auto& qInput=stage?chain[stage-1][0]:query;const auto& kInput=stage?chain[stage-1][1]:key;const auto& qo=chain[stage][0];const auto& ko=chain[stage][1];if(candidate)need(gpu.EncodePairedNormRope(qInput,kInput,weights[0],weights[1],p,f,qo,ko,qHeads,kHeads,dim,rotary,epsilon,offset,normalize),"Paired learned route unexpectedly fell back");else{gpu.EncodeNormRope(qInput,weights[0],p,f,qo,qHeads,dim,rotary,epsilon,offset,normalize);gpu.EncodeNormRope(kInput,weights[1],p,f,ko,kHeads,dim,rotary,epsilon,offset,normalize);}}auto encoded=Clock::now();gpu.SubmitAndWait();auto end=Clock::now();need(gpu.LastPairedNormRopeDispatches()==(candidate?chainStages:0),"Actual paired chain route differs");++executions;return std::array<double,3>{ms(begin,encoded),ms(begin,end),gpu.LastGPUMs()};};
    auto validate=[&](int variant){std::array<std::vector<float>,2> final;
     need(expected[variant].size()==chainStages,"Original carried reference missing");
     for(size_t stage=0;stage<chainStages;++stage){std::array<std::vector<float>,2> actual{gpu.Download(chain[stage][0]),gpu.Download(chain[stage][1])};for(size_t side=0;side<2;++side){const auto& control=expected[variant][stage][side];need(actual[side].size()==control.size()&&!std::memcmp(actual[side].data(),control.data(),actual[side].size()*sizeof(float)),"Original carried norm-RoPE stage output bits differ");for(float v:actual[side])need(std::isfinite(v),"Nonfinite carried output");}if(stage+1==chainStages)final=std::move(actual);}return final;};
    for(int variant=0;variant<4;++variant){execute(false,variant);expected[variant].clear();for(size_t stage=0;stage<chainStages;++stage)expected[variant].push_back({gpu.Download(chain[stage][0]),gpu.Download(chain[stage][1])});execute(true,variant);auto actual=validate(variant);bitChecks+=chainStages*(actual[0].size()+actual[1].size());auto hashes=[NSMutableArray new];for(const auto& stage:expected[variant])[hashes addObject:@[hash(stage[0]),hash(stage[1])]];[outputHashes addObject:hashes];++cases;}
    emit(@{@"kind":@"numeric",@"model":name,@"layer":@(layer),@"rows":@(rows),@"query_heads":@(qHeads),@"key_heads":@(kHeads),@"dim":@(dim),@"rotary":@(rotary),@"normalize":@(normalize),@"weight_offset":@(offset),@"variants":@4,@"original_output_bits_checked":@(bitChecks),@"input_sha256":inputHashes,@"stage_output_sha256":outputHashes,@"chain_stages":@(chainStages),@"candidate_dispatches_per_execution":@(chainStages),@"original_bits_equal":@YES});++profiles;checked+=bitChecks;
    if(mode=="timing"){
     for(bool nullMode:{true,false}){auto warm=Clock::now();do{execute(false,int(epoch%4));execute(!nullMode,int(epoch++%4));}while(ms(warm,Clock::now())<2000);emit(@{@"kind":@"warmup",@"model":name,@"layer":@(layer),@"rows":@(rows),@"null":@(nullMode),@"wall_ms":@(ms(warm,Clock::now()))});
      for(size_t round=0;round<10;++round)for(size_t window=0;window<4;++window){bool b=window==1||window==2,candidate=b&&!nullMode;for(size_t i=0;i<3;++i)execute(candidate,int(epoch++%4));auto encodes=[NSMutableArray new],walls=[NSMutableArray new],gpus=[NSMutableArray new];int variant=0;for(size_t i=0;i<5;++i){variant=int(epoch++%4);auto t=execute(candidate,variant);need(t[1]>0&&t[2]>0,"Invalid timing");[encodes addObject:@(t[0])];[walls addObject:@(t[1])];[gpus addObject:@(t[2])];}validate(variant);emit(@{@"kind":@"timing",@"model":name,@"layer":@(layer),@"rows":@(rows),@"round":@(round),@"window":@(window),@"null":@(nullMode),@"label_b":@(b),@"candidate":@(candidate),@"encode_ms":encodes,@"wall_ms":walls,@"gpu_ms":gpus,@"all_stage_output_bits_checked_after_window":@YES,@"chain_stages":@(chainStages),@"paired_dispatches_per_execution":@(candidate?chainStages:0)});}
     }++timed;
    }
    emit(@{@"kind":@"profile_complete",@"model":name,@"layer":@(layer),@"rows":@(rows),@"executions":@(executions),@"original_bits_equal":@YES,@"chain_stages":@(chainStages)});
   }}
  }}
 }}
 need(profiles==(mode=="numeric"?232:24),"Full learned coverage differs");struct rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"RSS unavailable");emit(@{@"kind":@"complete",@"mode":[NSString stringWithUTF8String:mode.c_str()],@"profiles":@(profiles),@"changed_input_cases":@(cases),@"original_output_bits_checked":@(checked),@"timed_profiles":@(timed),@"process_peak_rss_bytes":@(usage.ru_maxrss),@"scope":@"32-stage carried learned component only; every stage output retained and checked, no native/model speed proof"});return 0;
}catch(const std::exception& error){fprintf(stderr,"paired_norm_rope_learned_failed: %s\n",error.what());return 1;}}}
