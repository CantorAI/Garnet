// Private rejection-index experiment. Final scores use original BF16/FP32 math.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace Garnet {extern bool gNibbleScreenEnabled;}
static void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static void emit(NSDictionary* value){auto bytes=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(bytes!=nil,"JSON failed");fwrite(bytes.bytes,1,bytes.length,stdout);putchar('\n');fflush(stdout);}
static std::vector<float> loadFloats(NSString* folder,NSString* name){auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:name]];need(bytes!=nil&&bytes.length%4==0,"Float fixture missing");std::vector<float> result(bytes.length/4);std::memcpy(result.data(),bytes.bytes,bytes.length);return result;}
static bool equal(float a,float b){return std::isfinite(a)&&std::isfinite(b)?std::memcmp(&a,&b,4)==0:std::isnan(a)?std::isnan(b):a==b;}

int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==4,"Expected assets, model and numeric/timing");NSString* folder=[NSString stringWithUTF8String:argv[1]];std::string model=argv[2];bool timing=std::strcmp(argv[3],"timing")==0;need(timing||std::strcmp(argv[3],"numeric")==0,"Unknown mode");
 auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(bytes!=nil,"Manifest missing");NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];need(manifest!=nil,"Bad manifest");
 size_t n=[manifest[@"rows"] unsignedLongLongValue],k=[manifest[@"inner"] unsignedLongLongValue];need(n>8192&&(k==896||k==1024),"Unsupported head");
 auto raw=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"raw.bin"]];need(raw.length==n*k*2,"Bad raw head size");Garnet::MetalContext gpu;Garnet::MetalBF16Weight weights[2];
 for(int policy=0;policy<2;++policy){auto start=std::chrono::steady_clock::now();Garnet::gNibbleScreenEnabled=policy;weights[policy]=gpu.UploadBF16(raw.bytes,raw.length,n,k);need(gpu.PrepareScreenedHead(weights[policy]),"Index preparation failed");need(weights[policy].NibbleScreenReady()==bool(policy),"Index mode mismatch");need(weights[policy].ScreenIndexBytes()==((n+3)/4*4)*k/(policy?2:1),"Index byte mismatch");
  emit(@{@"kind":@"packing",@"policy":@(policy),@"rows":@(n),@"inner":@(k),@"index_bytes":@(weights[policy].ScreenIndexBytes()),@"upload_and_index_ms":@(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count())});}
 Garnet::gNibbleScreenEnabled=false;size_t checked=0,fallback=0,timed=0,ownership=0;
 for(NSDictionary* query in manifest[@"queries"]){@autoreleasepool{
  NSString* name=query[@"name"];size_t batch=[query[@"batch"] unsignedLongLongValue];need(batch==1||batch==2,"Bad batch");auto values=loadFloats(folder,[name stringByAppendingString:@".input.bin"]),reference=loadFloats(folder,[name stringByAppendingString:@".reference.bin"]);need(values.size()==batch*k&&reference.size()==batch*n,"Bad query shape");
  auto input=gpu.Upload(values,batch,k),output=gpu.Allocate(batch,2),fullOutput=gpu.Allocate(batch,n);
  auto full=[&](){gpu.Begin();gpu.EncodeBF16MatVec(input,weights[0],fullOutput);gpu.SubmitAndWait();return gpu.Download(fullOutput);};
  auto run=[&](int policy){auto start=std::chrono::steady_clock::now();gpu.Begin();need(gpu.EncodeScreenedArgmax(input,weights[policy],output),"Screen route unavailable");gpu.SubmitAndWait();need(gpu.LastScreenedHeadDispatches()==1&&gpu.LastNibbleScreenDispatches()==size_t(policy),"Actual index route mismatch");return std::pair<double,double>{gpu.LastGPUMs(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};};
  auto validate=[&](int policy,const std::vector<float>& expected,const char* kind,int pattern){run(policy);auto diagnostics=gpu.ScreenedDiagnostics();auto prediction=gpu.Download(output);size_t kept=0,rejected=0;need(diagnostics.rows==batch&&diagnostics.columns==n,"Diagnostic shape mismatch");
   for(size_t row=0;row<batch;++row){float threshold=-INFINITY,best=-INFINITY;size_t winner=0;bool invalid=false;
    for(size_t c=0;c<n;++c){float v=expected[row*n+c];invalid=invalid||!std::isfinite(v);if(v>best){best=v;winner=c;}if(c<1024)threshold=std::max(threshold,v);}
    for(size_t c=0;c<n;++c){size_t i=row*n+c;if(diagnostics.kept[i]){++kept;need(equal(diagnostics.maskedLogits[i],expected[i]),"Retained original score bits changed");}else{++rejected;need(expected[i]<threshold,"Rejected original logit can beat exact seed");}if(!diagnostics.querySafe[row])need(diagnostics.kept[i],"Unsafe query dropped row");}
    need(prediction[row*2+1]==float(invalid),"Greedy error flag mismatch");if(!invalid)need(prediction[row*2]==float(winner),"Original greedy token mismatch");
   }
   emit(@{@"kind":[NSString stringWithUTF8String:kind],@"query":name,@"batch":@(batch),@"policy":@(policy),@"pattern":@(pattern),@"kept":@(kept),@"rejected":@(rejected),@"all_original_scores_checked":@YES,@"retained_bits_exact":@YES,@"rejected_below_exact_seed":@YES,@"greedy_and_error_flags_match":@YES});
  };
  if(!timing){auto actual=full();need(actual.size()==reference.size()&&!std::memcmp(actual.data(),reference.data(),actual.size()*4),"Original native full logits differ from frozen capture");emit(@{@"kind":@"raw_check",@"query":name,@"batch":@(batch),@"values":@(reference.size()),@"bits_exact":@YES});
   for(int policy=0;policy<2;++policy){validate(policy,reference,"check",-1);++checked;auto abandoned=gpu.Allocate(batch,2);gpu.Begin();need(gpu.EncodeScreenedArgmax(input,weights[policy],abandoned),"Pending route missing");bool rejected=false;try{gpu.Download(abandoned);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending output readable");Garnet::MetalContext other;other.Begin();rejected=false;try{other.EncodeScreenedArgmax(input,weights[policy],abandoned);}catch(const std::logic_error&){rejected=true;}need(rejected,"Foreign pending writer accepted");other.Abort();gpu.Abort();need(gpu.LastNibbleScreenDispatches()==0&&gpu.LastScreenedHeadDispatches()==0,"Abort count survived");rejected=false;try{gpu.Download(abandoned);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted output readable");run(policy);auto empty=gpu.Allocate(batch,k);gpu.Begin();rejected=false;try{gpu.EncodeScreenedArgmax(empty,weights[policy],output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Uninitialized input accepted");gpu.Abort();++ownership;}
   if([name isEqualToString:@"b1-step0"]||[name isEqualToString:@"b2-step0"]){for(int pattern=0;pattern<(batch==2?6:5);++pattern){auto changed=values;for(size_t row=0;row<batch;++row)for(size_t j=0;j<k;++j){float v=pattern==0?0.f:pattern==1?1e-20f:pattern==2?1e20f:pattern==3?NAN:INFINITY;changed[row*k+j]=(pattern==5&&row==1)?values[row*k+j]:v;}gpu.Update(input,changed);auto original=full();for(int policy=0;policy<2;++policy){validate(policy,original,"fallback_check",pattern);++fallback;}}gpu.Update(input,values);}
   continue;
  }
  // Mixed-row diagnostic remains numerical only; all eight ordinary steps at both batches are timed.
  if(![name hasPrefix:@"b1-step"]&&![name hasPrefix:@"b2-step"])continue;
  if([name isEqualToString:@"b1-step0"]||[name isEqualToString:@"b2-step0"]){auto start=std::chrono::steady_clock::now();do{run(0);run(1);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<2.);}
  for(bool null:{true,false})for(int round=0;round<10;++round)for(int position=0;position<4;++position){bool candidate=position==1||position==2;int policy=int(candidate&&!null);for(int warm=0;warm<3;++warm)run(policy);for(int sample=0;sample<5;++sample){auto interval=run(policy);need(std::isfinite(interval.first)&&std::isfinite(interval.second)&&interval.first>0&&interval.second>0,"Invalid interval");emit(@{@"kind":@"timing",@"query":name,@"batch":@(batch),@"null":@(null),@"round":@(round),@"position":@(position),@"sample":@(sample),@"candidate":@(candidate),@"actual_policy":@(policy),@"gpu_ms":@(interval.first),@"wall_ms":@(interval.second)});++timed;}}
 }}
 emit(@{@"kind":@"complete",@"mode":timing?@"timing":@"numeric",@"checks":@(checked),@"fallback_checks":@(fallback),@"ownership_checks":@(ownership),@"timing_samples":@(timed),@"full_goal_met":@NO});return 0;
}catch(const std::exception& e){fprintf(stderr,"nibble_screen_index_failed: %s\n",e.what());return 1;}}}
