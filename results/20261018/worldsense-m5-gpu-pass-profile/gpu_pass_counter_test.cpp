// Native counter metadata/transaction proof. Not an inference speed test.
#include "metal_context.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
using namespace Garnet;
static void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv){try{
 Require(argc==2,"trace file argument required");std::remove(argv[1]);
 Require(MetalContext::CounterTraceTimestampPairValid(1,1),"equal valid timestamp");
 Require(!MetalContext::CounterTraceTimestampPairValid(0,2)&&!MetalContext::CounterTraceTimestampPairValid(3,2)&&!MetalContext::CounterTraceTimestampPairValid(1,UINT64_MAX)&&!MetalContext::CounterTraceTimestampPairValid(UINT64_MAX,UINT64_MAX),"invalid timestamps admitted");
#ifndef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
 Require(MetalContext::CounterTraceRequestId()==0,"OFF scope not zero");
 std::cout<<"DefaultOFF GPU counter capability absent; timestamp validator PASS\n";return 0;
#else
 setenv("GARNET_METAL_COUNTER_TRACE","1",1);setenv("GARNET_METAL_COUNTER_TRACE_FILE",argv[1],1);
 bool rejected=false;try{MetalContext::CounterTraceDecodeStep(4);}catch(const std::exception&){rejected=true;}
 Require(rejected,"unscoped step accepted");
 std::vector<nlohmann::json> expected;
 {
  MetalContext gpu;auto a=gpu.Upload(std::vector<float>(64,.25f),1,64);auto b=gpu.Allocate(1,64);auto c=gpu.Allocate(1,64);
  auto run=[&](bool sampled,uint64_t step){gpu.Begin();gpu.EncodeAdd(a,a,b);gpu.EncodeAdd(b,a,c);gpu.SubmitAndWait();Require(gpu.Download(c)==std::vector<float>(64,.75f),"dependent output bits changed");if(sampled)expected.push_back({{"request",MetalContext::CounterTraceRequestId()},{"step",step}});};
  auto unreadable=[&](const MetalMatrix& m){bool no=false;try{gpu.Download(m);}catch(const std::exception&){no=true;}Require(no,"abandoned output readable");};
  {
   MetalCounterTraceRequest request("unit\"outer");auto outer=MetalContext::CounterTraceRequestId();Require(outer>0,"request id missing");MetalContext::CounterTraceDecodeStep(4);run(true,4);
   {MetalCounterTraceRequest nested("unit-nested");Require(MetalContext::CounterTraceRequestId()!=outer,"nested identity reused");MetalContext::CounterTraceDecodeStep(16);run(true,16);}
   Require(MetalContext::CounterTraceRequestId()==outer,"nested scope failed to restore");run(true,4);
   std::thread thread([&](){Require(MetalContext::CounterTraceRequestId()==0,"cross-thread scope leaked");{MetalCounterTraceRequest other("other-thread");Require(MetalContext::CounterTraceRequestId()!=outer,"cross-thread identity reused");}Require(MetalContext::CounterTraceRequestId()==0,"thread scope not cleared");});thread.join();
   MetalContext::CounterTraceDecodeStep(5);run(false,5);
   MetalContext::CounterTraceDecodeStep(4);gpu.Begin();gpu.EncodeAdd(a,a,b);gpu.Abort();unreadable(b);run(true,4);
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
   MetalContext::CounterTraceTestLimit(2);gpu.Begin();gpu.EncodeAdd(a,a,b);rejected=false;
   try{gpu.EncodeAdd(b,a,c);}catch(const std::exception& e){rejected=std::string(e.what()).find("capacity exceeded")!=std::string::npos;}
   Require(rejected,"capacity not rejected");gpu.Abort();MetalContext::CounterTraceTestLimit(4096);unreadable(b);run(true,4);
   MetalContext::CounterTraceTestEmitFailure(true);gpu.Begin();gpu.EncodeAdd(a,a,b);gpu.EncodeAdd(b,a,c);rejected=false;
   try{gpu.SubmitAndWait();}catch(const std::exception& e){rejected=std::string(e.what()).find("resolution failure")!=std::string::npos;}
   Require(rejected,"resolution fault not propagated");gpu.Abort();MetalContext::CounterTraceTestEmitFailure(false);unreadable(b);unreadable(c);run(true,4);
   MetalContext::CounterTraceTestUnavailable(true);rejected=false;
   try{gpu.Begin();}catch(const std::exception& e){rejected=std::string(e.what()).find("unavailable GPU")!=std::string::npos;}
   Require(rejected,"unavailable counters admitted");gpu.Abort();MetalContext::CounterTraceTestUnavailable(false);run(true,4);
#endif
  }
  Require(MetalContext::CounterTraceRequestId()==0,"request scope leaked");
  gpu.Begin();gpu.EncodeAdd(a,a,b);gpu.SubmitAndWait();Require(gpu.Download(b)==std::vector<float>(64,.5f),"unscoped recovery failed");
 }
 std::ifstream input(argv[1]);std::string line;size_t completed=0,records=0;std::vector<nlohmann::json> pending;
 while(std::getline(input,line)){
  const auto split=line.find(' ');Require(split!=std::string::npos,"invalid trace line");auto value=nlohmann::json::parse(line.substr(split+1));
  Require(value["diagnostic"]==true&&value["serialized"]==false,"trace scope metadata wrong");
  if(line.rfind("metal_counter_profile ",0)==0){Require(value["index"]==pending.size(),"noncontiguous samples");Require(MetalContext::CounterTraceTimestampPairValid(value["start_raw"],value["end_raw"]),"bad hardware timestamp");pending.push_back(value);++records;}
  else{
   Require(line.rfind("metal_counter_command ",0)==0&&completed<expected.size(),"unexpected command");Require(value["samples"]==2&&pending.size()==2&&value["partial_submissions"]==0,"command sample count wrong");
   Require(value["request"]==expected[completed]["request"]&&value["decode_step"]==expected[completed]["step"],"request/step association wrong");
   for(const auto& sample:pending)Require(sample["request"]==value["request"]&&sample["model"]==value["model"]&&sample["context"]==value["context"]&&sample["command"]==value["command"],"sample identity mismatch");
   pending.clear();++completed;
  }
 }
 Require(pending.empty()&&completed==expected.size()&&records==completed*2,"missing trace records");
 std::cout<<"Native GPU pass counters exact dependency/scope/nesting/thread/selection/Abort/valid timestamp/identity/recovery PASS; commands="<<completed<<"\n";return 0;
#endif
}catch(const std::exception& e){std::cerr<<"GPU counter guard failed: "<<e.what()<<"\n";return 1;}}
