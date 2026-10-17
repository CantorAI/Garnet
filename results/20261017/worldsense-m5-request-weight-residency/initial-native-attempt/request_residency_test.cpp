// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "weight_residency_budget.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static uint64_t field(MetalContext& gpu,const char* key){auto text=gpu.RequestWeightResidencyStatsJSON();auto name=std::string("\"")+key+"\":";auto at=text.find(name);need(at!=std::string::npos,"Missing native request counter");return std::stoull(text.substr(at+name.size()));}
int main(){try{
 WeightResidencyBudget budget(65536);need(budget.Acquire(1,32768)&&budget.Acquire(1,32768),"Budget acquire");need(!budget.Acquire(1,1)&&!budget.Acquire(2,UINT64_MAX)&&!budget.Acquire(2,0),"Budget bound/overflow");budget.Release(1,32768);need(budget.Acquire(1,32768),"Budget reuse");budget.Release(1,65536);need(budget.Used(1)==0,"Budget release");
 MetalContext gpu;
#ifndef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 need(!gpu.BeginRequestWeightResidency({}),"Disabled request admitted");need(!gpu.RequestWeightResidencyActive(),"Disabled request active");gpu.EndRequestWeightResidency();
#else
 std::vector<uint16_t> words(128*128,0x3f80);auto make=[&](MetalContext& context){return context.UploadBF16(words.data(),words.size()*2,128,128,false,false,false);};auto w=make(gpu),alias=w;
 need(gpu.BeginRequestWeightResidency({w,w}),"Raw request not admitted");need(field(gpu,"last_count")==1&&field(gpu,"active_bytes")>0&&gpu.RequestWeightsCovered({alias}),"Raw dedup/coverage");
 bool rejected=false;try{gpu.BeginRequestWeightResidency({w});}catch(...){rejected=true;}need(rejected&&gpu.RequestWeightResidencyActive(),"Nested request altered active lease");
 w=MetalBF16Weight();auto input=gpu.Upload(std::vector<float>(128,1),1,128);auto output=gpu.Allocate(1,128);gpu.Begin();gpu.EncodeBF16MatVec(input,alias,output);alias=MetalBF16Weight();rejected=false;try{gpu.EndRequestWeightResidency();}catch(...){rejected=true;}need(rejected&&gpu.RequestWeightResidencyActive(),"Open batch ended request");gpu.SubmitAndWait();need(gpu.RequestWeightResidencyActive(),"Normal submit ended request");for(auto value:gpu.Download(output))need(value==128,"Leased raw GPU projection changed");gpu.EndRequestWeightResidency();gpu.EndRequestWeightResidency();need(field(gpu,"active_bytes")==0&&field(gpu,"global_used_bytes")==0,"End retained owners/budget");
 auto again=make(gpu);need(gpu.BeginRequestWeightResidency({again}),"Readmission failed");gpu.Begin();gpu.EncodeBF16MatVec(input,again,output);gpu.Flush();gpu.EncodeBF16MatVec(input,again,output);gpu.Abort();need(!gpu.RequestWeightResidencyActive()&&field(gpu,"global_used_bytes")==0,"Partial abort retained budget");
 need(gpu.BeginRequestWeightResidency({again}),"After Abort admission");auto foreign=make(gpu);need(!gpu.RequestWeightsCovered({foreign}),"Unprepared weights covered");gpu.CancelRequestWeightResidency(8);need(field(gpu,"reason")==8&&field(gpu,"active_bytes")==0,"Coverage fallback cleanup");
 need(!gpu.BeginRequestWeightResidency({}),"Empty request admitted");need(field(gpu,"reason")==2,"Empty reason");gpu.EndRequestWeightResidency();
 rejected=false;try{gpu.BeginRequestWeightResidency({MetalBF16Weight()});}catch(...){rejected=true;}need(rejected&&field(gpu,"active_bytes")==0&&field(gpu,"global_used_bytes")==0,"Invalid owner cleanup");
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 auto packed=gpu.UploadBF16(words.data(),words.size()*2,128,128);need(packed.PrimaryReady(),"Primary negative test did not pack");need(!gpu.BeginRequestWeightResidency({packed})&&field(gpu,"reason")==7,"Packed request falsely admitted");gpu.EndRequestWeightResidency();
#endif
#ifdef GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS
 need(field(gpu,"limit_bytes")==65536,"Fault budget differs from declared cap");
 for(int stage:{1,2}){MetalContext::RequestResidencyTestFailAfter(stage);rejected=false;try{gpu.BeginRequestWeightResidency({again});}catch(...){rejected=true;}need(rejected&&field(gpu,"global_used_bytes")==0&&field(gpu,"active_bytes")==0,"Injected begin failure leaked");need(gpu.BeginRequestWeightResidency({again}),"Injected failure readmission");gpu.EndRequestWeightResidency();}
 MetalContext a,b,c;auto wa=make(a),wb=make(b),wc=make(c);need(a.BeginRequestWeightResidency({wa})&&b.BeginRequestWeightResidency({wb}),"Two bounded contexts not admitted");need(!c.BeginRequestWeightResidency({wc})&&field(c,"reason")==5,"Global contention not bounded");c.EndRequestWeightResidency();a.EndRequestWeightResidency();need(c.BeginRequestWeightResidency({wc}),"Global budget not reusable");c.EndRequestWeightResidency();b.EndRequestWeightResidency();
#else
 need(field(gpu,"limit_bytes")==uint64_t(4)*1024*1024*1024,"Production 4GiB cap changed");std::vector<MetalBF16Weight> many;uint16_t one=0x3f80;for(int i=0;i<2049;++i)many.push_back(gpu.UploadBF16(&one,2,1,1,false,false,false));need(!gpu.BeginRequestWeightResidency(many)&&field(gpu,"reason")==4&&field(gpu,"attempted_count")==2049,"Unique allocation cap not enforced");need(field(gpu,"attempted_bytes")<uint64_t(4)*1024*1024*1024,"Count test hit byte cap");gpu.EndRequestWeightResidency();
#endif
 {MetalContext temporary;auto weight=make(temporary);need(temporary.BeginRequestWeightResidency({weight}),"Destructor scope admission");}need(field(gpu,"global_used_bytes")==0,"Destructor retained budget");
#endif
 std::cout<<"Bounded request residency GPU projection/dedup/lifetime/submit/partial-Abort/coverage/nesting/caps/budget/failure/cleanup PASS"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
