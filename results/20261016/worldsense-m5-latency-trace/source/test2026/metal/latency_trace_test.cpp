#include "latency_trace.h"
#include <thread>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
using namespace Garnet;
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static void valid(const NativeLatencyTrace&t){need(t.valid&&t.bodyEntered&&t.nativeSeconds>=0&&t.bodySeconds>=0&&t.preBodySeconds>=0&&t.postBodySeconds>=0&&t.samplingSeconds>=0,"Invalid native timing");need(std::fabs(t.nativeSeconds-t.bodySeconds-t.preBodySeconds-t.postBodySeconds)<1e-6,"Native timing partition mismatch");need(t.beforeStatus==0&&t.afterStatus==0&&t.userSeconds>=0&&t.systemSeconds>=0&&t.minorFaults>=0&&t.majorFaults>=0,"Invalid process usage sample");}
int main(){try{
 for(bool scoped:{false,true}){int called=0;TraceNativeSessionRun(scoped,[&]{++called;});auto first=lastNativeLatencyTrace;valid(first);need(first.scoped==scoped&&!first.failed&&called==1,"Trace changed callback");bool caught=false;try{TraceNativeSessionRun(scoped,[]{throw std::runtime_error("trace failure sentinel");});}catch(const std::runtime_error&e){caught=std::string(e.what())=="trace failure sentinel";}valid(lastNativeLatencyTrace);need(caught&&lastNativeLatencyTrace.failed&&lastNativeLatencyTrace.sequence==first.sequence+1,"Trace lost exception/sequence");TraceNativeSessionRun(scoped,[&]{TraceNativeSessionRun(!scoped,[&]{++called;});valid(lastNativeLatencyTrace);});valid(lastNativeLatencyTrace);need(lastNativeLatencyTrace.scoped==scoped&&!lastNativeLatencyTrace.failed&&called==2,"Nested trace did not restore final caller record");}
 const auto saved=lastNativeLatencyTrace.sequence;std::vector<int> passed(2);std::vector<std::thread> threads;
 for(size_t i=0;i<2;++i)threads.emplace_back([&,i]{try{need(nativeLatencySequence==0&&!lastNativeLatencyTrace.valid,"Trace leaked between threads");for(size_t j=0;j<16;++j){TraceNativeSessionRun(j%2,[]{});valid(lastNativeLatencyTrace);need(lastNativeLatencyTrace.sequence==j+1,"Thread trace sequence conflated");}passed[i]=1;}catch(...){passed[i]=0;}});
 for(auto&t:threads)t.join();need(passed[0]&&passed[1]&&lastNativeLatencyTrace.sequence==saved,"Thread isolation failed");
 std::cout<<"Native trace scoped/unscoped callback/exception/nested/thread-isolation/process-sample checks PASS\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
