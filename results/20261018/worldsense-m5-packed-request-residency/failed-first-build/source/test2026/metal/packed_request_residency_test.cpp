// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "weight_residency_budget.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
template<class F>static void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}need(bad,"Invalid packed request operation accepted");}
static json stats(MetalContext& gpu){return json::parse(gpu.PackedRequestWeightResidencyStatsJSON());}
static void clean(MetalContext& gpu){auto s=stats(gpu);need(s["active_bytes"]==0&&s["global_used_bytes"]==0&&!s["scope_open"].get<bool>(),"Packed scope/budget retained");
#ifdef GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY
 need(s["set_allocation_count"]==0&&s["owner_count"]==0&&!s["attached"].get<bool>()&&!s["requested"].get<bool>(),"Actual packed set/owner/attachment retained");
#endif
}
static std::vector<float> project(MetalContext& gpu,const MetalBF16Weight& weight,unsigned pattern){
 std::vector<float> x(128);float sum=0;for(size_t j=0;j<x.size();++j){x[j]=pattern?float(int(j%7)-3)*.25f:1.f;sum+=x[j];}
 auto input=gpu.Upload(x,1,128),output=gpu.Allocate(1,260);gpu.Begin();gpu.EncodeBF16MatVec(input,weight,output);gpu.SubmitAndWait();auto y=gpu.Download(output);need(y.size()==260,"Projection output shape changed");for(float v:y)need(v==sum,"Packed lease changed exact binary-fraction projection");return y;
}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: packed_request_test output.bin");std::ofstream output(argv[1],std::ios::binary),leased(std::string(argv[1])+".leased.bin",std::ios::binary);need(bool(output)&&bool(leased),"Cannot create complete packed projection outputs");
 WeightResidencyBudget budget(uint64_t(8)*1024*1024*1024);need(budget.Acquire(1,uint64_t(8)*1024*1024*1024),"Exact budget bound rejected");need(!budget.Acquire(1,1)&&!budget.Acquire(2,UINT64_MAX)&&!budget.Acquire(2,0),"Budget overflow/zero admitted");budget.Release(1,uint64_t(8)*1024*1024*1024);need(budget.Used(1)==0,"Pure budget retained reservation");
 MetalContext gpu;auto initial=MetalContext::PrimaryStats();std::vector<uint16_t> words(260*128,0x3f80);auto make=[&](MetalContext& context){return context.UploadBF16(words.data(),words.size()*2,260,128,false,false,true);};auto w=make(gpu);
 auto write=[&](std::ofstream& stream,unsigned pattern,const std::vector<float>& values){uint64_t shape[4]={260,128,pattern,values.size()};stream.write(reinterpret_cast<const char*>(shape),sizeof(shape));stream.write(reinterpret_cast<const char*>(values.data()),values.size()*4);};
 for(unsigned pattern:{0,1})write(output,pattern,project(gpu,w,pattern));
#ifndef GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY
 need(!gpu.BeginPackedRequestWeightResidency({}),"Disabled packed API admitted");need(!gpu.PackedRequestWeightResidencyActive(),"Disabled packed API active");gpu.EndPackedRequestWeightResidency();clean(gpu);
#else
 need(w.PrimaryReady(),"Packed fixture not admitted by original primary codec");auto info=MetalContext::WeightBufferInfo(w);need(info.size()==2,"Packed owner lacks payload and metadata");uint64_t actual=0;for(const auto& row:info)actual+=row.allocatedBytes;need(actual>0,"Empty actual allocation tuple");
 uint16_t one=0x3f80;auto raw=gpu.UploadBF16(&one,2,1,1,false,false,false);
 need(!gpu.BeginPackedRequestWeightResidency({raw})&&stats(gpu)["reason"]==2,"Raw-only hint falsely admitted");gpu.EndPackedRequestWeightResidency();clean(gpu);
 need(gpu.BeginPackedRequestWeightResidency({w,w,raw}),"Packed request not admitted");auto state=stats(gpu);need(state["last_count"]==2&&state["last_bytes"]==actual&&state["owner_count"]==1&&state["set_allocation_count"]==2&&state["attached"].get<bool>()&&state["requested"].get<bool>(),"Actual two-allocation/owner tuple incorrect");need(gpu.PackedRequestWeightsCovered({w,raw})&&!gpu.PackedRequestWeightsCovered({MetalBF16Weight()}),"Packed/raw coverage validation");
 rejects([&]{gpu.BeginPackedRequestWeightResidency({w});});need(gpu.PackedRequestWeightResidencyActive()&&stats(gpu)["active_bytes"]==actual,"Nested begin altered lease");
 for(unsigned pattern:{0,1}){write(leased,pattern,project(gpu,w,pattern));need(gpu.PackedRequestWeightResidencyActive()&&stats(gpu)["set_allocation_count"]==2,"Submit released packed request");}
 auto alias=w;w=MetalBF16Weight();auto input=gpu.Upload(std::vector<float>(128,1),1,128),out=gpu.Allocate(1,260);gpu.Begin();gpu.EncodeBF16MatVec(input,alias,out);alias=MetalBF16Weight();need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights+1,"Lease lost strong original owner");rejects([&]{gpu.EndPackedRequestWeightResidency();});need(gpu.PackedRequestWeightResidencyActive(),"Open End released lease");gpu.SubmitAndWait();for(float v:gpu.Download(out))need(v==128,"Released-owner GPU result changed");gpu.EndPackedRequestWeightResidency();gpu.EndPackedRequestWeightResidency();clean(gpu);need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights,"Ended lease retained logical owner");
 auto again=make(gpu);need(gpu.BeginPackedRequestWeightResidency({again}),"Readmission failed");gpu.Begin();gpu.EncodeBF16MatVec(input,again,out);gpu.Flush();gpu.EncodeBF16MatVec(input,again,out);gpu.Abort();clean(gpu);
 need(gpu.BeginPackedRequestWeightResidency({again}),"Post-Abort readmission failed");{auto uncovered=make(gpu);need(!gpu.PackedRequestWeightsCovered({uncovered}),"Uncovered packed buffers accepted");}gpu.CancelPackedRequestWeightResidency(8);need(stats(gpu)["reason"]==8,"Coverage cancel reason changed");clean(gpu);
 need(!gpu.BeginPackedRequestWeightResidency({}),"Empty packed set admitted");need(stats(gpu)["reason"]==2,"Empty reason changed");gpu.EndPackedRequestWeightResidency();rejects([&]{gpu.BeginPackedRequestWeightResidency({MetalBF16Weight()});});clean(gpu);
#ifdef GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS
 need(stats(gpu)["limit_bytes"]==131072,"New packed fault budget differs from pre-source probe");
 for(int stage:{1,2}){MetalContext::PackedRequestResidencyTestFailAfter(stage);rejects([&]{gpu.BeginPackedRequestWeightResidency({again});});clean(gpu);need(gpu.BeginPackedRequestWeightResidency({again}),"Fault recovery readmission failed");gpu.EndPackedRequestWeightResidency();clean(gpu);}
 {auto second=make(gpu);need(!gpu.BeginPackedRequestWeightResidency({again,second})&&stats(gpu)["reason"]==4&&stats(gpu)["attempted_bytes"].get<uint64_t>()>131072,"Actual packed byte cap not enforced");gpu.EndPackedRequestWeightResidency();clean(gpu);}
 {MetalContext a,b;auto wa=make(a),wb=make(b);need(a.BeginPackedRequestWeightResidency({wa}),"First real packed fault-budget request rejected");need(!b.BeginPackedRequestWeightResidency({wb})&&stats(b)["reason"]==5,"Global packed budget contention not enforced");b.EndPackedRequestWeightResidency();a.EndPackedRequestWeightResidency();need(b.BeginPackedRequestWeightResidency({wb}),"Budget release did not allow readmission");b.EndPackedRequestWeightResidency();clean(a);clean(b);}
#else
 need(stats(gpu)["limit_bytes"]==uint64_t(8)*1024*1024*1024,"Production8GiB cap changed");
 {std::vector<MetalBF16Weight> many;for(size_t i=0;i<1025;++i)many.push_back(make(gpu));need(!gpu.BeginPackedRequestWeightResidency(many)&&stats(gpu)["reason"]==4&&stats(gpu)["attempted_count"]==2049,"Unique actual two-buffer cap not enforced");need(stats(gpu)["attempted_bytes"].get<uint64_t>()<uint64_t(8)*1024*1024*1024,"Count case hit byte cap");gpu.EndPackedRequestWeightResidency();clean(gpu);}
#endif
 {MetalContext temporary;auto owned=make(temporary);need(temporary.BeginPackedRequestWeightResidency({owned}),"Destructor admission failed");}clean(gpu);again=MetalBF16Weight();need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights,"Final packed owner leak");
#endif
 need(bool(output)&&bool(leased),"Cannot write complete native outputs");std::cout<<"Packed payload+metadata dedup/actual set/owners/projection/bounds/partialAbort/coverage/fault/global-budget/cleanup PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
