// SPDX-License-Identifier: Apache-2.0
// Private, finite integer corpus. No production dispatch or serving claim.
#include "tensor_router_geometry_native.cuh"
#include <cuda_runtime.h>
#include <zlib.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace GarnetRouterPrivate;
namespace F=std::filesystem;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
 void* p=nullptr; explicit Buffer(size_t n){check(cudaMalloc(&p,n));}
 ~Buffer(){if(p)cudaFree(p);} Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
struct Graph {
 cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
 ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
 Graph()=default;Graph(const Graph&)=delete;
};
int xv(int q,int k,int family){return ((q%17)*23+k*11+family*7)%65-32;}
int wv(int e,int k,int family){return ((e%37)*19+k*7+family*13)%65-32;}
int bv(int e,int family){return (e*5+family*3)%33-16;}
uint32_t bits(float v){uint32_t x;std::memcpy(&x,&v,4);return x;}
float bfRound(float v){uint32_t x=bits(v);x=(x+0x7fffu+((x>>16)&1u))&0xffff0000u;std::memcpy(&v,&x,4);return v;}
void save(const F::path& path,const std::vector<float>& data){
 if(F::exists(path))throw std::runtime_error("Evidence exists");
 gzFile f=gzopen(path.string().c_str(),"wb1");if(!f)throw std::runtime_error("gzip open");
 const auto bytes=data.size()*4;const auto* p=reinterpret_cast<const unsigned char*>(data.data());
 size_t done=0;while(done<bytes){const auto n=std::min(bytes-done,size_t(1)<<28);const int wrote=gzwrite(f,p+done,unsigned(n));if(wrote!=int(n)){gzclose(f);throw std::runtime_error("gzip write");}done+=n;}
 if(gzclose(f)!=Z_OK)throw std::runtime_error("gzip close");
}
void launch(int arm,int rows,const GptOssOptions& o,const Buffer& x,const Buffer& w,const Buffer& b,Buffer& y,cudaStream_t stream){
 if(arm==0)originalRouter<<<dim3((rows+15)/16,(o.experts+63)/64),128,0,stream>>>((float*)x.p,(float*)w.p,(float*)b.p,(float*)y.p,rows,o);
 else if(arm==4)candidateRouter<4><<<dim3((rows+15)/16,(o.experts+63)/64),128,0,stream>>>((float*)x.p,(float*)w.p,(float*)b.p,(float*)y.p,rows,o);
 else if(arm==2)candidateRouter<2><<<dim3((rows+15)/16,(o.experts+31)/32),64,0,stream>>>((float*)x.p,(float*)w.p,(float*)b.p,(float*)y.p,rows,o);
 else if(arm==1)candidateRouter<1><<<dim3((rows+15)/16,(o.experts+15)/16),32,0,stream>>>((float*)x.p,(float*)w.p,(float*)b.p,(float*)y.p,rows,o);
 else throw std::runtime_error("arm");
 check(cudaGetLastError());
}
__global__ void flushL2(unsigned* p,size_t n){const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]+=unsigned(i)+1;}
int main(int argc,char** argv){try{
 if(argc!=6)return 2;
 const int rows=std::stoi(argv[1]),hidden=std::stoi(argv[2]),experts=std::stoi(argv[3]);
 const std::string mode=argv[4];const F::path out=argv[5];
 if(!((rows==19&&hidden==33&&experts==37)||(rows==512&&hidden==2880&&experts==128)||(rows==4096&&hidden==2880&&experts==128)))return 2;
 if(mode!="parity"&&mode!="cost")return 2;
 if(!F::is_directory(out)||!F::is_empty(out))throw std::runtime_error("New empty directory required");
 cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major<8)throw std::runtime_error("WMMA BF16 architecture required");
 GptOssOptions o{};o.hidden=hidden;o.experts=experts;o.topK=8;
 std::vector<float> hx(size_t(rows)*hidden),hw(size_t(experts)*hidden),hb(experts),readX(hx.size()),readW(hw.size()),readB(hb.size()),hy(size_t(rows)*experts);
 Buffer x(hx.size()*4),w(hw.size()*4),b(hb.size()*4),y(hy.size()*4);
 cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 const std::array<int,4> arms{0,4,2,1};std::array<Graph,4> graphs;
 for(size_t a=0;a<arms.size();++a){
  check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(arms[a],rows,o,x,w,b,y,stream);
  check(cudaStreamEndCapture(stream,&graphs[a].graph));check(cudaGraphInstantiate(&graphs[a].exec,graphs[a].graph,nullptr,nullptr,0));
 }
 std::ofstream metadata(out/"dataset.json");metadata<<"{\"rows\":"<<rows<<",\"hidden\":"<<hidden<<",\"experts\":"<<experts<<",\"mode\":\""<<mode<<"\",\"families\":"<<(mode=="cost"?1:3)<<",\"arms\":[0,4,2,1]}\n";metadata.close();if(!metadata)throw std::runtime_error("metadata");
 for(int family=0;family<(mode=="cost"?1:3);++family){
  for(int q=0;q<rows;++q)for(int k=0;k<hidden;++k)hx[size_t(q)*hidden+k]=float(xv(q,k,family))/128.f;
  for(int e=0;e<experts;++e){hb[e]=float(bv(e,family))/4096.f;for(int k=0;k<hidden;++k)hw[size_t(e)*hidden+k]=float(wv(e,k,family))/4096.f;}
  const std::string stem="family"+std::to_string(family);
  save(out/(stem+"-x.f32.gz"),hx);save(out/(stem+"-w.f32.gz"),hw);save(out/(stem+"-b.f32.gz"),hb);
  check(cudaMemcpyAsync(x.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(w.p,hw.data(),hw.size()*4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(b.p,hb.data(),hb.size()*4,cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
  std::array<std::array<int64_t,37>,17> dots{};
  for(int q=0;q<17;++q)for(int e=0;e<37;++e)for(int k=0;k<hidden;++k)dots[q][e]+=int64_t(xv(q,k,family))*wv(e,k,family);
  std::vector<float> reference;
  for(size_t a=0;a<arms.size();++a){
   const std::string armStem=stem+"-arm"+std::to_string(arms[a]);
   for(int replay=0;replay<2;++replay){
    if(!replay)launch(arms[a],rows,o,x,w,b,y,stream);else check(cudaGraphLaunch(graphs[a].exec,stream));
    check(cudaMemcpyAsync(hy.data(),y.p,hy.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    save(out/(armStem+(replay?"-graph":"-eager")+".f32.gz"),hy);
    if(!a&&!replay)reference=hy;
    if(hy.size()!=reference.size()||std::memcmp(hy.data(),reference.data(),hy.size()*4))throw std::runtime_error("Exact original/replay score mismatch");
    for(int q=0;q<rows;++q)for(int e=0;e<experts;++e){
     const auto integer=dots[q%17][e%37]+int64_t(bv(e,family))*128;
     const float expected=bfRound(float(integer)/524288.f);
     if(bits(expected)!=bits(hy[size_t(q)*experts+e]))throw std::runtime_error("Exact integer oracle score mismatch");
    }
   }
  }
  check(cudaMemcpyAsync(readX.data(),x.p,hx.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaMemcpyAsync(readW.data(),w.p,hw.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaMemcpyAsync(readB.data(),b.p,hb.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
  save(out/(stem+"-x.after.f32.gz"),readX);save(out/(stem+"-w.after.f32.gz"),readW);save(out/(stem+"-b.after.f32.gz"),readB);
  if(std::memcmp(hx.data(),readX.data(),hx.size()*4)||std::memcmp(hw.data(),readW.data(),hw.size()*4)||std::memcmp(hb.data(),readB.data(),hb.size()*4))throw std::runtime_error("Inputs changed");
 }
 if(mode=="cost"){
  std::array<std::array<int64_t,37>,17> costDots{};
  for(int q=0;q<17;++q)for(int e=0;e<37;++e)for(int k=0;k<hidden;++k)costDots[q][e]+=int64_t(xv(q,k,0))*wv(e,k,0);
  const size_t flushBytes=std::max(size_t(256)<<20,size_t(prop.l2CacheSize)*4);Buffer flush(flushBytes);check(cudaMemsetAsync(flush.p,0,flushBytes,stream));
  cudaEvent_t start=nullptr,end=nullptr;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
  std::ofstream cost(out/"cost.json");cost<<"{\"warmups\":5,\"trials\":9,\"flush_bytes\":"<<flushBytes<<",\"arms\":[";
  for(size_t a=0;a<arms.size();++a){
   for(int warm=0;warm<5;++warm)check(cudaGraphLaunch(graphs[a].exec,stream));check(cudaStreamSynchronize(stream));
   if(a)cost<<",";cost<<"{\"arm\":"<<arms[a]<<",\"samples_ms\":[";
   for(int trial=0;trial<9;++trial){
    flushL2<<<(flushBytes/4+255)/256,256,0,stream>>>((unsigned*)flush.p,flushBytes/4);check(cudaGetLastError());
    check(cudaEventRecord(start,stream));check(cudaGraphLaunch(graphs[a].exec,stream));check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
    float ms=0;check(cudaEventElapsedTime(&ms,start,end));if(trial)cost<<",";cost.precision(9);cost<<ms;
    check(cudaMemcpyAsync(hy.data(),y.p,hy.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    save(out/("cost-arm"+std::to_string(arms[a])+"-trial"+std::to_string(trial)+".f32.gz"),hy);
    for(int q=0;q<rows;++q)for(int e=0;e<experts;++e){const float expected=bfRound(float(costDots[q%17][e%37]+int64_t(bv(e,0))*128)/524288.f);if(bits(expected)!=bits(hy[size_t(q)*experts+e]))throw std::runtime_error("Cost oracle mismatch");}
   }
   cost<<"]}";
  }
  cost<<"]}\n";cost.close();if(!cost)throw std::runtime_error("cost metadata");check(cudaEventDestroy(start));check(cudaEventDestroy(end));
  check(cudaMemcpyAsync(readX.data(),x.p,hx.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaMemcpyAsync(readW.data(),w.p,hw.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaMemcpyAsync(readB.data(),b.p,hb.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
  save(out/"cost-x.after.f32.gz",readX);save(out/"cost-w.after.f32.gz",readW);save(out/"cost-b.after.f32.gz",readB);
  if(std::memcmp(hx.data(),readX.data(),hx.size()*4)||std::memcmp(hw.data(),readW.data(),hw.size()*4)||std::memcmp(hb.data(),readB.data(),hb.size()*4))throw std::runtime_error("Cost inputs changed");
 }
 check(cudaStreamSynchronize(stream));check(cudaStreamDestroy(stream));
 std::cout<<"ROUTER_PRIVATE_COMPLETE rows="<<rows<<" hidden="<<hidden<<" experts="<<experts<<" mode="<<mode<<" exact_integer_original_replay_inputs=PASS\n";
 return 0;
}catch(const std::exception& e){std::cerr<<"ROUTER_PRIVATE_FAILURE "<<e.what()<<"\n";return 1;}}
