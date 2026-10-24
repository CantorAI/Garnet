// SPDX-License-Identifier: Apache-2.0
// Independent logical INT4 oracle; arithmetic diagnostic only, no timing.
#include "ct_int4.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {void* p=nullptr;explicit Buffer(size_t n){Check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
int main(){try{
 for(int k:{256,6144})for(int m:{1,16}){
  const int n=128;
  std::vector<int> logical(n*k);std::vector<unsigned char> packed(n*k/2,0);
  std::vector<__nv_bfloat16> input(m*k),scales(n*k/128),output(m*n);
  for(int row=0;row<n;++row){
   for(int g=0;g<k/128;++g)scales[row*(k/128)+g]=__float2bfloat16((1+(row+g)%4)/16.f);
   for(int col=0;col<k;++col){int q=(row*7+col*3+col/16)%16-8;logical[row*k+col]=q;
    int i=row*k+col;packed[i/2]|=static_cast<unsigned char>((q+8)<<((i%2)*4));}
  }
  for(int i=0;i<m*k;++i)input[i]=__float2bfloat16(((i*5+i/k)%7-3)/2.f);
  Buffer x(input.size()*2),w(packed.size()),s(scales.size()*2),y(output.size()*2);
  Check(cudaMemcpy(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
  Check(cudaMemcpy(w.p,packed.data(),packed.size(),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(s.p,scales.data(),scales.size()*2,cudaMemcpyHostToDevice));
  cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
  Check(cudaStreamCreate(&stream));Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  Check(Garnet::ProjectCTInt4G128(x.p,static_cast<const unsigned char*>(w.p),s.p,y.p,m,n,k,stream));
  Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
  for(int replay=0;replay<3;++replay){
   if(replay==0)Check(Garnet::ProjectCTInt4G128(x.p,static_cast<const unsigned char*>(w.p),s.p,y.p,m,n,k,stream));
   else Check(cudaGraphLaunch(executable,stream));
   Check(cudaStreamSynchronize(stream));
   Check(cudaMemcpy(output.data(),y.p,output.size()*2,cudaMemcpyDeviceToHost));
   for(int row=0;row<m;++row)for(int col=0;col<n;++col){
    double sum=0;for(int j=0;j<k;++j)sum+=double(__bfloat162float(input[row*k+j]))*logical[col*k+j]*__bfloat162float(scales[col*(k/128)+j/128]);
    float expected=__bfloat162float(__float2bfloat16(float(sum))),actual=__bfloat162float(output[row*n+col]);
    if(actual!=expected){fprintf(stderr,"m=%d k=%d row=%d col=%d actual=%g expected=%g\n",m,k,row,col,actual,expected);throw std::runtime_error("Native INT4 exact oracle mismatch");}
   }
  }
  Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));Check(cudaStreamDestroy(stream));
  printf("CT_INT4_G128_EXACT_PASS m=%d n=%d k=%d outputs=%d direct=1 graph_replays=2 workspace=0; no benchmark\n",m,n,k,m*n);
 }
 return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
