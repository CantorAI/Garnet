// SPDX-License-Identifier: Apache-2.0
#include "ct_int4.cuh"
#include <cuda_bf16.h>
#include <cstdint>
namespace Garnet {
namespace {
__global__ void Project(const __nv_bfloat16* x,const unsigned char* w,
 const __nv_bfloat16* scales,__nv_bfloat16* y,int n,int k){
 const int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32,batch=blockIdx.y;
 if(row>=n)return;
 const auto base=static_cast<int64_t>(row)*k;
 float sum=0;
 for(int col=lane;col<k;col+=32){
  unsigned char byte=w[(base+col)/2];
  int code=int((byte>>((col%2)*4))&15)-8;
  float scale=__bfloat162float(scales[static_cast<int64_t>(row)*(k/128)+col/128]);
  sum=fmaf(__bfloat162float(x[static_cast<int64_t>(batch)*k+col]),code*scale,sum);
 }
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if(lane==0)y[static_cast<int64_t>(batch)*n+row]=__float2bfloat16_rn(sum);
}
}
cudaError_t ProjectCTInt4G128(const void* input,const unsigned char* packed,
 const void* scales,void* output,int m,int n,int k,cudaStream_t stream){
 if(!input||!packed||!scales||!output||m<1||m>128||n<1||n>1048576||k<128||k>65536||k%128)
  return cudaErrorInvalidValue;
 Project<<<dim3((n+3)/4,m),128,0,stream>>>(static_cast<const __nv_bfloat16*>(input),packed,
  static_cast<const __nv_bfloat16*>(scales),static_cast<__nv_bfloat16*>(output),n,k);
 return cudaGetLastError();
}
}
