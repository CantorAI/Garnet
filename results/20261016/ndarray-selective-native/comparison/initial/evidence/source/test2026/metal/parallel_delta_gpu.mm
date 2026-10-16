// Isolated FP32 recurrent-chunk algorithm comparison, not serving inference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct P {uint64_t batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
struct Matrix {id<MTLBuffer> buffer;MPSMatrix* view;uint64_t rows,cols;};
static Matrix matrix(id<MTLDevice> d,uint64_t batches,uint64_t rows,uint64_t cols,id<MTLBuffer> b=nil) {
 if(!b)b=[d newBufferWithLength:batches*rows*cols*4 options:MTLResourceStorageModeShared];
 auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols matrices:batches rowBytes:cols*4 matrixBytes:rows*cols*4 dataType:MPSDataTypeFloat32];
 auto view=[[MPSMatrix alloc] initWithBuffer:b descriptor:desc];
 if(!b||!view)throw std::runtime_error("Matrix allocation failed");return {b,view,rows,cols};
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> lib,NSString* name) {
 NSError* e=nil;auto p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
 if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);return p;
}
struct Product {MPSMatrixMultiplication* op;MPSMatrix* a;MPSMatrix* b;MPSMatrix* out;};
static Product product(id<MTLDevice> d,uint64_t batches,const Matrix& a,const Matrix& b,const Matrix& out,bool ta=false,bool tb=false) {
 auto op=[[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:ta transposeRight:tb resultRows:out.rows resultColumns:out.cols interiorColumns:ta?a.rows:a.cols alpha:1 beta:0];
 op.batchSize=batches;return {op,a.view,b.view,out.view};
}
static void finish(id<MTLCommandBuffer> cb) {
 [cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.localizedDescription.UTF8String);
}
static id<MTLBuffer> load(id<MTLDevice> d,NSString* folder,NSString* file,size_t bytes) {
 NSData* data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:file]];
 if(data.length!=bytes)throw std::runtime_error("Input byte count mismatch");
 return [d newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
}
int main(int argc,char** argv) {@autoreleasepool {try {
 if(argc!=3)throw std::runtime_error("Expected shader and assets");
 auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];NSError* e=nil;
 NSString* text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&e];
 auto lib=[d newLibraryWithSource:text options:nil error:&e];if(!lib)throw std::runtime_error(e.localizedDescription.UTF8String);
 auto control=pipeline(d,lib,@"gated_delta_chunk"),pack=pipeline(d,lib,@"pack_delta"),decay=pipeline(d,lib,@"decay_products"),coef=pipeline(d,lib,@"coefficients"),rhs=pipeline(d,lib,@"rhs_delta"),solve=pipeline(d,lib,@"solve_delta"),scale=pipeline(d,lib,@"scale_updates"),output=pipeline(d,lib,@"finish_output"),state=pipeline(d,lib,@"finish_state");
 if(control.threadExecutionWidth!=32||control.maxTotalThreadsPerThreadgroup<128||pack.threadExecutionWidth!=32||pack.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("128-thread/32-lane pipeline unavailable");
 NSString* folder=[NSString stringWithUTF8String:argv[2]];NSData* meta=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
 NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:meta options:0 error:&e];
 for(NSDictionary* item in manifest[@"cases"]) {@autoreleasepool {
  NSString* name=item[@"name"];uint64_t b=[item[@"batch"] unsignedLongLongValue],c=[item[@"chunk"] unsignedLongLongValue],h=[item[@"heads"] unsignedLongLongValue],k=128,v=128,n=b*h;
  if(!b||!c||c>80||!h)throw std::runtime_error("Invalid component profile");
  P p{b,c,h,k,v,3*h*128+2*h,h*k*v,c,h*v,h*k*v};
  auto prepared=load(d,folder,item[@"prepared"],b*c*p.pStride*4),prior=load(d,folder,item[@"state"],b*p.sStride*4),active=load(d,folder,item[@"active"],b*c*4);
  auto out=[d newBufferWithLength:b*c*p.oStride*4 options:MTLResourceStorageModeShared],next=[d newBufferWithLength:b*p.nextStride*4 options:MTLResourceStorageModeShared];
  auto q=matrix(d,n,c,k),key=matrix(d,n,c,k),value=matrix(d,n,c,v),s=matrix(d,n,k,v,prior);
  auto gram=matrix(d,n,c,c),qk=matrix(d,n,c,c),ks=matrix(d,n,c,v),qs=matrix(d,n,c,v),lower=matrix(d,n,c,c),m=matrix(d,n,c,c),r=matrix(d,n,c,v),u=matrix(d,n,c,v),us=matrix(d,n,c,v),weighted=matrix(d,n,c,v),updates=matrix(d,n,k,v);
  auto factors=[d newBufferWithLength:n*c*4 options:MTLResourceStorageModeShared],beta=[d newBufferWithLength:n*c*4 options:MTLResourceStorageModeShared],dr=[d newBufferWithLength:n*c*c*4 options:MTLResourceStorageModeShared],cum=[d newBufferWithLength:n*c*4 options:MTLResourceStorageModeShared];
  std::vector<Product> products={product(d,n,key,key,gram,false,true),product(d,n,q,key,qk,false,true),product(d,n,key,s,ks),product(d,n,q,s,qs),product(d,n,m,u,weighted),product(d,n,key,us,updates,true,false)};
  auto run=[&](bool candidate) {
   auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];
   auto encode=[&](id<MTLComputePipelineState> kernel,std::vector<id<MTLBuffer>> buffers,uint64_t count,bool groups=false) {
    auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:kernel];
    for(size_t i=0;i<buffers.size();++i)[enc setBuffer:buffers[i] offset:0 atIndex:i];[enc setBytes:&p length:sizeof(p) atIndex:buffers.size()];
    if(groups)[enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    else [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];
   };
   auto mm=[&](size_t i){auto& x=products[i];[x.op encodeToCommandBuffer:cb leftMatrix:x.a rightMatrix:x.b resultMatrix:x.out];};
   if(!candidate)encode(control,{prepared,prior,active,out,next},n*v/16,true);
   else {
    encode(pack,{prepared,active,q.buffer,key.buffer,value.buffer,factors,beta},n*c,true);
    encode(decay,{factors,dr,cum},n*c);
    mm(0);mm(1);mm(2);mm(3);
    encode(coef,{gram.buffer,qk.buffer,dr,beta,lower.buffer,m.buffer},n*c*c);
    encode(rhs,{value.buffer,ks.buffer,cum,beta,r.buffer},n*c*v);
    encode(solve,{lower.buffer,r.buffer,u.buffer},n*v);
    encode(scale,{u.buffer,dr,us.buffer},n*c*v);
    mm(4);mm(5);
    encode(output,{qs.buffer,weighted.buffer,cum,active,out},n*c*v);
    encode(state,{prior,updates.buffer,cum,next},n*k*v);
   }
   finish(cb);return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
  };
  run(false);std::vector<float> expected(b*c*p.oStride),expectedState(b*p.nextStride);std::memcpy(expected.data(),out.contents,expected.size()*4);std::memcpy(expectedState.data(),next.contents,expectedState.size()*4);
  run(true);double worst=0,maxError=0;
  for(auto pair:{std::pair<float*,std::vector<float>*>{static_cast<float*>(out.contents),&expected},std::pair<float*,std::vector<float>*>{static_cast<float*>(next.contents),&expectedState}}) {
   for(size_t i=0;i<pair.second->size();++i){float a=pair.first[i],x=(*pair.second)[i];if(!std::isfinite(a))throw std::runtime_error("Nonfinite parallel result");double diff=std::abs(double(a)-x);worst=std::max(worst,diff/(5e-6+5e-5*std::abs(double(x))));maxError=std::max(maxError,diff);}
  }
  std::printf("{\"kind\":\"check\",\"name\":\"%s\",\"worst_ratio\":%.9g,\"max_abs_error\":%.9g,\"passed\":%s}\n",name.UTF8String,worst,maxError,worst<=1?"true":"false");std::fflush(stdout);
  if(worst>1)throw std::runtime_error("Parallel recurrence exceeds unchanged bound");
  for(int round=0;round<2;++round)for(bool candidate:{false,true,true,false}) {
   for(int warm=0;warm<3;++warm)run(candidate);std::vector<double> gpu,wall;
   for(int i=0;i<12;++i){auto t=run(candidate);gpu.push_back(t.first);wall.push_back(t.second);}std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
   std::printf("{\"kind\":\"timing\",\"name\":\"%s\",\"round\":%d,\"candidate\":%s,\"gpu_ms\":%.9g,\"wall_ms\":%.9g}\n",name.UTF8String,round,candidate?"true":"false",(gpu[5]+gpu[6])*.5,(wall[5]+wall[6])*.5);std::fflush(stdout);
  }
 }}return 0;
 }catch(const std::exception& x){std::fprintf(stderr,"parallel_delta_probe_failed: %s\n",x.what());return 1;}}}
