// SPDX-License-Identifier: Apache-2.0
// Independent CPU oracle + full-cache checks before component-only A/B timing.
#include "qwen35_attention_core.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
 float* base=nullptr;size_t count;static constexpr size_t guard=64;
 explicit Buffer(size_t n):count(n){Check(cudaMalloc(reinterpret_cast<void**>(&base),(n+2*guard)*4));std::vector<float> v(n+2*guard,1234567.f);Check(cudaMemcpy(base,v.data(),v.size()*4,cudaMemcpyHostToDevice));}
 ~Buffer(){if(base)cudaFree(base);}float* data(){return base+guard;}
 void upload(const std::vector<float>& v){if(v.size()!=count)throw std::runtime_error("Input size mismatch");Check(cudaMemcpy(data(),v.data(),count*4,cudaMemcpyHostToDevice));}
 std::vector<float> read(){std::vector<float> v(count);Check(cudaMemcpy(v.data(),data(),count*4,cudaMemcpyDeviceToHost));return v;}
 void guards(){float v[128];Check(cudaMemcpy(v,base,256,cudaMemcpyDeviceToHost));Check(cudaMemcpy(v+64,data()+count,256,cudaMemcpyDeviceToHost));for(float x:v)if(x!=1234567.f)throw std::runtime_error("Buffer overrun");}
};
struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}};
bool Equal(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4);}
void Save(const std::string& p,const std::vector<float>& v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*4);if(!f)throw std::runtime_error("Cannot retain raw context");}
float Value(size_t i,int salt){return (int((i*17+salt)%97)-48)/256.f;}
struct Case {int batch,tokens,heads,kv,capacity,dim,prefix;bool all;};
int main(int argc,char** argv){try{
 if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="--timing"))throw std::runtime_error("Raw output prefix and optional --timing required");
 const bool timing=argc==3;
 std::vector<Case> cases={{1,16,24,4,4096,256,32,false},{1,16,24,4,4096,256,128,false},{1,16,24,4,4096,256,512,false},{1,16,24,4,4096,256,4096,false},{1,16,24,4,4096,256,32,true},{2,4,24,4,33,64,32,true},{1,16,24,4,8192,256,8192,false}};
 cudaStream_t stream;Check(cudaStreamCreate(&stream));cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
 for(size_t ci=0;ci<cases.size();++ci){const auto a=cases[ci];Garnet::Qwen35AttentionCoreConfig c{a.batch,a.tokens,a.heads,a.kv,a.capacity,a.dim};
  size_t qsize=size_t(c.batch)*c.tokens*c.heads*c.dim,kvsize=size_t(c.batch)*c.tokens*c.kv*c.dim,cache=size_t(c.batch)*c.kv*c.capacity*c.dim;
  std::vector<std::vector<float>> host(8);host[0].resize(qsize);host[1].resize(kvsize);host[2].resize(kvsize);host[3].resize(qsize);host[4].resize(cache);host[5].resize(cache);host[6].resize(c.batch*c.tokens);host[7].resize(c.batch*c.tokens);
  for(int i=0;i<6;++i)for(size_t j=0;j<host[i].size();++j)host[i][j]=Value(j,11+i*19)*(i==3?16.f:1.f);
  for(int b=0;b<c.batch;++b){int count=c.batch>1?2:(a.all?c.tokens:1),slot=std::min(c.capacity-count,a.prefix-1);
   for(int t=0;t<c.tokens;++t){bool active=(c.batch==1||b==0)&&(c.batch>1?t%2==0:(a.all||t==0));host[6][b*c.tokens+t]=float(slot);host[7][b*c.tokens+t]=active?1.f:0.f;if(active)++slot;}
  }
  auto nextKey=host[4],nextValue=host[5];
  for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)if(host[7][b*c.tokens+t])for(int h=0;h<c.kv;++h)for(int d=0;d<c.dim;++d){size_t from=((size_t(b)*c.tokens+t)*c.kv+h)*c.dim+d,to=((size_t(b)*c.kv+h)*c.capacity+int(host[6][b*c.tokens+t]))*c.dim+d;nextKey[to]=host[1][from];nextValue[to]=host[2][from];}
  std::vector<float> expected(qsize,0);
  for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)if(host[7][b*c.tokens+t])for(int h=0;h<c.heads;++h){
   int end=int(host[6][b*c.tokens+t])+1,kh=h/(c.heads/c.kv);size_t row=(size_t(b)*c.tokens+t)*c.heads+h;std::vector<double> scores(end);double maximum=-std::numeric_limits<double>::infinity(),total=0;
   for(int p=0;p<end;++p){double sum=0;size_t base=((size_t(b)*c.kv+kh)*c.capacity+p)*c.dim;for(int d=0;d<c.dim;++d)sum+=double(host[0][row*c.dim+d])*nextKey[base+d];scores[p]=sum/std::sqrt(double(c.dim));maximum=std::max(maximum,scores[p]);}
   for(double& s:scores){s=std::exp(s-maximum);total+=s;}
   for(int d=0;d<c.dim;++d){double sum=0;for(int p=0;p<end;++p)sum+=(scores[p]/total)*nextValue[((size_t(b)*c.kv+kh)*c.capacity+p)*c.dim+d];expected[row*c.dim+d]=float(sum/(1+std::exp(-double(host[3][row*c.dim+d]))));}
  }
  Save(std::string(argv[1])+"-case"+std::to_string(ci)+"-cpu.f32",expected);
  std::vector<std::unique_ptr<Buffer>> inputs;const void* in[8];for(int i=0;i<8;++i){inputs.emplace_back(new Buffer(host[i].size()));inputs.back()->upload(host[i]);in[i]=inputs.back()->data();}
  std::vector<std::unique_ptr<Buffer>> outputs;void* out[3];for(size_t n:{qsize,cache,cache})outputs.emplace_back(new Buffer(n));for(int i=0;i<3;++i)out[i]=outputs[i]->data();Buffer workspace(Garnet::Qwen35AttentionCoreWorkspace(c)/4),splitWorkspace(Garnet::Qwen35AttentionSplitWorkspace(c)/4);
  auto run=[&](int mode){return mode==3?Garnet::RunQwen35AttentionSplit(c,in,out,splitWorkspace.data(),stream):mode==2?Garnet::RunQwen35AttentionCompactTiled(c,in,out,nullptr,stream):mode==1?Garnet::RunQwen35AttentionCompact(c,in,out,nullptr,stream):Garnet::RunQwen35AttentionCore(c,in,out,workspace.data(),stream);};
  Graph graphs[4];std::vector<float> baseline;float maxError=0,splitDelta=0;
  for(int mode=0;mode<4;++mode){
   if(run(mode))throw std::runtime_error("Direct attention enqueue failed");Check(cudaStreamSynchronize(stream));auto actual=outputs[0]->read();
   Save(std::string(argv[1])+"-case"+std::to_string(ci)+(mode==0?"-baseline.f32":mode==1?"-compact.f32":mode==2?"-tiled.f32":"-split.f32"),actual);
   for(size_t i=0;i<actual.size();++i){float error=std::fabs(actual[i]-expected[i]);maxError=std::max(maxError,error);if(!std::isfinite(actual[i])||error>2e-5f+2e-4f*std::fabs(expected[i])){fprintf(stderr,"case=%zu mode=%d index=%zu actual=%g expected=%g error=%g\n",ci,mode,i,actual[i],expected[i],error);throw std::runtime_error("Independent CPU attention oracle failed");}}
   if(mode==0)baseline=actual;
   else if(mode<3){if(!Equal(actual,baseline))throw std::runtime_error("Compact differs bitwise from baseline");}
   else for(size_t i=0;i<actual.size();++i){float error=std::fabs(actual[i]-baseline[i]);splitDelta=std::max(splitDelta,error);if(error>2e-5f+2e-4f*std::fabs(baseline[i]))throw std::runtime_error("Split exceeds predeclared legacy numerical tolerance");}
   if(!Equal(outputs[1]->read(),nextKey)||!Equal(outputs[2]->read(),nextValue))throw std::runtime_error("Full KV output mismatch");
   Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(run(mode))throw std::runtime_error("Capture failed");Check(cudaStreamEndCapture(stream,&graphs[mode].graph));Check(cudaGraphInstantiate(&graphs[mode].executable,graphs[mode].graph,nullptr,nullptr,0));
   for(int replay=0;replay<3;++replay){Check(cudaGraphLaunch(graphs[mode].executable,stream));Check(cudaStreamSynchronize(stream));if(!Equal(outputs[0]->read(),actual)||!Equal(outputs[1]->read(),nextKey)||!Equal(outputs[2]->read(),nextValue))throw std::runtime_error("Graph replay output mismatch");}
  }
  for(int i=0;i<8;++i){inputs[i]->guards();if(!Equal(inputs[i]->read(),host[i]))throw std::runtime_error("Input mutation");}for(auto& p:outputs)p->guards();workspace.guards();splitWorkspace.guards();
  const auto beforeInvalid=outputs[0]->read();auto invalid=c;invalid.dim=0;
  if(Garnet::RunQwen35AttentionCompact(invalid,in,out,nullptr,stream)==0||Garnet::RunQwen35AttentionCompactTiled(invalid,in,out,nullptr,stream)==0||Garnet::Qwen35AttentionSplitWorkspace(invalid)!=0||Garnet::RunQwen35AttentionSplit(invalid,in,out,splitWorkspace.data(),stream)==0||!Equal(outputs[0]->read(),beforeInvalid))throw std::runtime_error("Invalid host profile accepted or changed output");
  if(!timing){printf("{\"case\":%zu,\"status\":\"correctness_pass\",\"variants_checked\":4,\"cpu_max_abs\":%.9g,\"row_tiled_bit_exact\":true,\"split_within_tolerance\":true,\"full_kv_bit_exact\":true,\"graph_replays\":3}\n",ci,maxError);fflush(stdout);continue;}
  for(int mode=0;mode<4;++mode)for(int i=0;i<5;++i)Check(cudaGraphLaunch(graphs[mode].executable,stream));Check(cudaStreamSynchronize(stream));
  std::vector<float> ms[4];for(int round=0;round<3;++round)for(int mode:{0,1,2,3,3,2,1,0}){Check(cudaEventRecord(start,stream));for(int i=0;i<20;++i)Check(cudaGraphLaunch(graphs[mode].executable,stream));Check(cudaEventRecord(stop,stream));Check(cudaEventSynchronize(stop));float elapsed=0;Check(cudaEventElapsedTime(&elapsed,start,stop));ms[mode].push_back(elapsed/20);}
  printf("{\"case\":%zu,\"batch\":%d,\"tokens\":%d,\"heads\":%d,\"kv\":%d,\"capacity\":%d,\"dim\":%d,\"first_prefix\":%d,\"all_tokens_active\":%s,\"cpu_max_abs\":%.9g,\"row_tiled_bit_exact\":true,\"split_within_tolerance\":true,\"full_kv_exact\":true,\"graph_replays_checked\":3,\"iterations_per_sample\":20,\"order\":\"ABCDDCBAx3\",\"baseline_ms\":[",ci,c.batch,c.tokens,c.heads,c.kv,c.capacity,c.dim,a.prefix,std::all_of(host[7].begin(),host[7].end(),[](float v){return v==1.f;})?"true":"false",maxError);
  for(size_t i=0;i<ms[0].size();++i)printf("%s%.9g",i?",":"",ms[0][i]);printf("],\"compact_ms\":[");for(size_t i=0;i<ms[1].size();++i)printf("%s%.9g",i?",":"",ms[1][i]);printf("],\"tiled_ms\":[");for(size_t i=0;i<ms[2].size();++i)printf("%s%.9g",i?",":"",ms[2][i]);printf("],\"split_ms\":[");for(size_t i=0;i<ms[3].size();++i)printf("%s%.9g",i?",":"",ms[3][i]);printf("],\"split_max_abs_vs_baseline\":%.9g,\"split_workspace_bytes\":%zu",splitDelta,Garnet::Qwen35AttentionSplitWorkspace(c));printf(",\"scope\":\"Warm attention-core CUDA graph only, includes KV copies; not model TTFT/decode/throughput or2x result\"}\n");fflush(stdout);
 }
 Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));Check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"COMPACT_ATTENTION_FAILURE %s\n",e.what());return 1;}}
