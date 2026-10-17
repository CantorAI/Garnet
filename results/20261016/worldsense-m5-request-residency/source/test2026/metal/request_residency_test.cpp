#include "metal_context.h"
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
static void ck(bool v,const char*m){if(!v)throw std::runtime_error(m);}
static uint64_t n(MetalContext& g,const char*k){auto s=g.WeightResidencyStatsJSON();auto p=s.find(std::string("\"")+k+"\":");ck(p!=std::string::npos,"Missing metadata");return std::stoull(s.substr(s.find(':',p)+1));}
int main(){try{MetalContext g;bool enabled=g.BeginWeightResidencyRequest();
#ifndef GARNET_METAL_REQUEST_WEIGHT_RESIDENCY
 ck(!enabled&&n(g,"request_active")==0,"Disabled request");g.EndWeightResidencyRequest();std::cout<<"Request residency disabled PASS\n";return 0;
#else
 ck(enabled&&n(g,"request_active")==1,"Request begin");bool rejected=false;try{g.BeginWeightResidencyRequest();}catch(...){rejected=true;}ck(rejected,"Nested scope");std::atomic<bool> wrong{false};std::thread t([&]{try{g.EndWeightResidencyRequest();}catch(...){wrong=true;}});t.join();ck(wrong&&n(g,"request_active")==1,"Thread ownership");
 std::vector<uint16_t> bits(32*128,0x3f80);auto w=g.UploadBF16(bits.data(),bits.size()*2,32,128);auto a=g.Upload(std::vector<float>(128,1),1,128);auto o=g.Allocate(1,32);
 auto run=[&](const MetalBF16Weight& weight,float expected){g.Begin();g.EncodeBF16MatVec(a,weight,o);g.SubmitAndWait();for(float v:g.Download(o))ck(v==expected,"Numerical change");};
 g.BeginWeightResidency({w});run(w,128);ck(n(g,"active_bytes")>0&&n(g,"admitted")==1,"Submit did not retain request");g.BeginWeightResidency({w,w});run(w,128);ck(n(g,"admitted")==1&&n(g,"request_reuses")==1,"Identical set not reused");
 std::fill(bits.begin(),bits.end(),0x4000);auto other=g.UploadBF16(bits.data(),bits.size()*2,32,128);g.BeginWeightResidency({other});run(other,256);ck(n(g,"admitted")==2&&n(g,"request_reuses")==1,"Changed set reused stale owners");w=MetalBF16Weight();
 g.BeginWeightResidency({other});g.Begin();g.EncodeBF16MatVec(a,other,o);g.Flush();g.Abort();ck(n(g,"request_active")==1&&n(g,"active_bytes")==0,"Abort retained set or lost logical scope");auto before=n(g,"admitted");g.BeginWeightResidency({other});run(other,256);ck(n(g,"admitted")==before+1,"Abort stale fingerprint");
 g.EndWeightResidency();ck(n(g,"active_bytes")==0&&n(g,"request_active")==1,"Reset-like cache clear ended scope");g.BeginWeightResidency({other});run(other,256);g.EndWeightResidencyRequest();ck(n(g,"active_bytes")==0&&n(g,"request_active")==0&&n(g,"request_starts")==1&&n(g,"request_ends")==1,"End cleanup");
 g.BeginWeightResidency({other});run(other,256);ck(n(g,"active_bytes")==0,"Outside-request batch retained resources");
 std::cout<<"Request residency reuse/changed set/nesting/thread/partialAbort/retry/cache clear/end/perbatch fallback numerical PASS "<<g.WeightResidencyStatsJSON()<<'\n';return 0;
#endif
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
