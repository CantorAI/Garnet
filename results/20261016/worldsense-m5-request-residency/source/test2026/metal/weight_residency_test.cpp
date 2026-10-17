#include "metal_context.h"
#include "weight_residency_budget.h"
#include <vector>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <cstdint>
using namespace Garnet;
static void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static uint64_t field(MetalContext& g,const char* name){auto s=g.WeightResidencyStatsJSON();auto k=s.find(std::string("\"")+name+"\":");check(k!=std::string::npos,"Missing stats");return std::stoull(s.substr(s.find(':',k)+1));}
int main(){try{
 WeightResidencyBudget budget(16);check(!budget.Acquire(1,17)&&!budget.Acquire(1,UINT64_MAX)&&!budget.Acquire(1,0),"Budget bound");check(budget.Acquire(1,12)&&!budget.Acquire(1,5),"Budget contention");check(budget.Acquire(2,16),"Device isolation");budget.Release(1,12);budget.Release(2,16);std::vector<std::thread> workers;for(int i=0;i<8;i++)workers.emplace_back([&]{for(int j=0;j<1000;j++)if(budget.Acquire(1,7))budget.Release(1,7);});for(auto& t:workers)t.join();check(budget.Used(1)==0&&budget.Used(2)==0,"Budget leak");
 MetalContext g;g.BeginWeightResidency({});check(field(g,"active_bytes")==0,"Empty active");g.Abort();
 std::vector<uint16_t> ones(32*128,0x3f80);auto w=g.UploadBF16(ones.data(),ones.size()*2,32,128);auto a=g.Upload(std::vector<float>(128,1),1,128);auto o=g.Allocate(1,32);
 for(int pass=0;pass<3;pass++){
  g.BeginWeightResidency({w,w});
#ifdef GARNET_METAL_WEIGHT_RESIDENCY
  check(field(g,"reason")==1&&field(g,"last_count")==1&&field(g,"active_bytes")>0,"Native residency not admitted/deduplicated");
#else
  check(field(g,"admitted")==0&&field(g,"active_bytes")==0,"Disabled activity");
#endif
  g.Begin();bool rejected=false;try{g.BeginWeightResidency({w});}catch(...){rejected=true;}check(rejected,"Open batch accepted residency");rejected=false;try{g.EndWeightResidency();}catch(...){rejected=true;}check(rejected,"Premature residency end accepted");g.EncodeBF16MatVec(a,w,o);
  if(pass==1){g.Flush();g.Abort();}else{g.SubmitAndWait();for(float v:g.Download(o))check(v==128,"Residency changed values");}
  check(field(g,"active_bytes")==0,"Completion or drained Abort leak");
 }
#ifdef GARNET_METAL_WEIGHT_RESIDENCY
 bool rejected=false;try{g.BeginWeightResidency({MetalBF16Weight()});}catch(...){rejected=true;}check(rejected&&field(g,"active_bytes")==0,"Invalid ownership");
 std::vector<MetalBF16Weight> many;uint16_t one=0x3f80;for(int i=0;i<2049;i++)many.push_back(g.UploadBF16(&one,2,1,1));g.BeginWeightResidency(many);check(field(g,"reason")==4&&field(g,"active_bytes")==0,"Allocation cap fallback");
#endif
 {auto temporary=g.UploadBF16(ones.data(),ones.size()*2,32,128);g.BeginWeightResidency({temporary});g.Begin();g.EncodeBF16MatVec(a,temporary,o);temporary=MetalBF16Weight();g.SubmitAndWait();for(float v:g.Download(o))check(v==128,"Owner release changed result");}
 g.BeginWeightResidency({w});g.Abort();check(field(g,"active_bytes")==0,"BeforeBegin Abort leak");
 std::cout<<"Bounded residency numerical/dedup/allocationcap/owner/partialAbort/retry/budget isolation+concurrency PASS "<<g.WeightResidencyStatsJSON()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
