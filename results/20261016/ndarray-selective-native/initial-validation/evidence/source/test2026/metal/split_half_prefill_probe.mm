// Isolated lossless activation decomposition, not serving or comparator inference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../../src/backends/metal/exact_half_codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct P {uint64_t rows,inner,columns,aStride,hStride,oStride;};
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> lib,NSString* name){
 NSError* e=nil;auto p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
 if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);return p;
}
struct Matrix {id<MTLBuffer> buffer;MPSMatrix* view;uint64_t stride;};
static Matrix matrix(id<MTLDevice> d,size_t rows,size_t cols,MPSDataType type){
 auto stride=[MPSMatrixDescriptor rowBytesFromColumns:cols dataType:type];
 auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:stride dataType:type];
 auto buffer=[d newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];
 return {buffer,[[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc],stride/(type==MPSDataTypeFloat16?2:4)};
}
int main(int argc,char** argv){@autoreleasepool{try{
 bool rejection=argc==4&&std::strcmp(argv[3],"admission")==0;
 if(argc!=3&&!rejection)throw std::runtime_error("Expected shader, assets and optional admission probe");
 auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];NSError* e=nil;
 auto shader=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&e];
 MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
 auto lib=[d newLibraryWithSource:shader options:options error:&e];if(!lib)throw std::runtime_error(e.localizedDescription.UTF8String);
 auto split=pipeline(d,lib,@"split_activation"),combine=pipeline(d,lib,@"combine_split"),rescale=pipeline(d,lib,@"combine_control");
 NSString* folder=[NSString stringWithUTF8String:argv[2]];
 auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
 auto manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:&e];
 for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
  NSString* name=item[@"name"];uint64_t k=[item[@"inner"] unsignedLongLongValue],n=[item[@"columns"] unsignedLongLongValue];
  auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
  if(bytes.length!=k*n*2)throw std::runtime_error("BF16 byte count mismatch");
  auto weights=matrix(d,n,k,MPSDataTypeFloat16);std::vector<int32_t> scales(n);auto raw=static_cast<const uint16_t*>(bytes.bytes);
  for(uint64_t row=0;row<n;++row){uint16_t largest=0;
   for(uint64_t j=0;j<k;++j)largest=std::max(largest,uint16_t(raw[row*k+j]&0x7fff));
   if(!Garnet::ExactHalfRowScale(largest,scales[row]))throw std::runtime_error("Weight row cannot use exact half");
   for(uint64_t j=0;j<k;++j){uint16_t encoded;if(!Garnet::EncodeExactScaledHalf(raw[row*k+j],scales[row],encoded))throw std::runtime_error("Weight cannot use exact half");
    static_cast<uint16_t*>(weights.buffer.contents)[row*weights.stride+j]=encoded;}
  }
  auto colScales=[d newBufferWithBytes:scales.data() length:n*4 options:MTLResourceStorageModeShared];
  for(uint64_t rows:{24,32,64,80}){
   auto input=matrix(d,rows,k,MPSDataTypeFloat32),hi=matrix(d,rows,k,MPSDataTypeFloat16),lo=matrix(d,rows,k,MPSDataTypeFloat16),tail=matrix(d,rows,k,MPSDataTypeFloat16);
   auto y0=matrix(d,rows,n,MPSDataTypeFloat32),y1=matrix(d,rows,n,MPSDataTypeFloat32),y2=matrix(d,rows,n,MPSDataTypeFloat32),out=matrix(d,rows,n,MPSDataTypeFloat32);
   auto rowScales=[d newBufferWithLength:rows*4 options:MTLResourceStorageModeShared],failures=[d newBufferWithLength:4 options:MTLResourceStorageModeShared];
   P p{rows,k,n,input.stride,hi.stride,out.stride};
   for(uint64_t row=0;row<rows;++row)for(uint64_t j=0;j<k;++j){float x=std::sin(float(row*k+j)*.013f)*.3f;
    if(rejection&&j%101==0)x=std::ldexp(x,-30);if(j%211==0)x=-0.0f;static_cast<float*>(input.buffer.contents)[row*input.stride+j]=x;}
   auto product=[[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:n interiorColumns:k alpha:1 beta:0];
   auto run=[&](bool candidate){auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];
    auto encode=[&](id<MTLComputePipelineState> kernel,std::vector<id<MTLBuffer>> buffers,uint64_t count,bool groups){
     auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:kernel];
     for(size_t i=0;i<buffers.size();++i)[enc setBuffer:buffers[i] offset:0 atIndex:i];[enc setBytes:&p length:sizeof(p) atIndex:buffers.size()];
     if(groups)[enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
     else [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];};
    if(candidate){*static_cast<uint32_t*>(failures.contents)=0;
     encode(split,{input.buffer,hi.buffer,lo.buffer,tail.buffer,rowScales,failures},rows,true);
     [product encodeToCommandBuffer:cb leftMatrix:hi.view rightMatrix:weights.view resultMatrix:y0.view];
     [product encodeToCommandBuffer:cb leftMatrix:lo.view rightMatrix:weights.view resultMatrix:y1.view];
     [product encodeToCommandBuffer:cb leftMatrix:tail.view rightMatrix:weights.view resultMatrix:y2.view];
     encode(combine,{y0.buffer,y1.buffer,y2.buffer,rowScales,colScales,out.buffer},rows*n,false);
    }else{[product encodeToCommandBuffer:cb leftMatrix:input.view rightMatrix:weights.view resultMatrix:y0.view];
     encode(rescale,{y0.buffer,colScales,out.buffer},rows*n,false);}
    [cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.localizedDescription.UTF8String);
    if(candidate&&*static_cast<uint32_t*>(failures.contents))throw std::runtime_error("Activation split is not bit-exact; fallback required");
    return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};};
   if(rejection){try{run(true);}catch(const std::runtime_error& error){
    if(std::strcmp(error.what(),"Activation split is not bit-exact; fallback required")==0){printf("{\"kind\":\"admission\",\"unsafe_input_rejected\":true}\n");return 0;}throw;}
    throw std::runtime_error("Wide-exponent input unexpectedly accepted");}
   run(false);std::vector<float> expected(rows*out.stride);std::memcpy(expected.data(),out.buffer.contents,expected.size()*4);run(true);
   double worst=0,error=0;for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<n;++col){uint64_t i=row*out.stride+col;float x=static_cast<float*>(out.buffer.contents)[i],ref=expected[i];
    if(!std::isfinite(x))throw std::runtime_error("Nonfinite result");double diff=std::abs(double(x)-ref);error=std::max(error,diff);worst=std::max(worst,diff/(5e-6+5e-5*std::abs(double(ref))));}
   printf("{\"kind\":\"check\",\"name\":\"%s\",\"rows\":%llu,\"activation_bits_exact\":true,\"worst_ratio\":%.9g,\"max_abs_error\":%.9g}\n",name.UTF8String,(unsigned long long)rows,worst,error);fflush(stdout);
   if(worst>1)throw std::runtime_error("Three-product output exceeds unchanged bound");
   for(int round=0;round<2;++round)for(bool candidate:{false,true,true,false}){
    for(int warm=0;warm<3;++warm)run(candidate);std::vector<double> gpu,wall;
    for(int sample=0;sample<12;++sample){auto t=run(candidate);gpu.push_back(t.first);wall.push_back(t.second);}std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
    printf("{\"kind\":\"timing\",\"name\":\"%s\",\"rows\":%llu,\"round\":%d,\"candidate\":%s,\"gpu_ms\":%.9g,\"wall_ms\":%.9g}\n",name.UTF8String,(unsigned long long)rows,round,candidate?"true":"false",(gpu[5]+gpu[6])*.5,(wall[5]+wall[6])*.5);fflush(stdout);
   }
  }
 }}return 0;
 }catch(const std::exception& x){fprintf(stderr,"split_half_prefill_probe_failed: %s\n",x.what());return 1;}}}
