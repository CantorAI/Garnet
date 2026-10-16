// Native BF16 matvec versus one MPS product of three exact input components.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../../src/backends/metal/exact_half_codec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
struct P{uint64_t rows,inner,columns,aStride,hStride,oStride;};
struct BF16Params{uint64_t rows,inner,columns,aStride,oStride;};
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void emit(NSDictionary* object){auto d=[NSJSONSerialization dataWithJSONObject:object options:0 error:nil];need(d!=nil,"JSON failed");puts([[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString* name){NSError* e=nil;auto p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:name] error:&e];if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);return p;}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==4||argc==5,"Expected native shader, candidate shader, assets and optional rejection gate");bool rejection=argc==5;need(!rejection||std::strcmp(argv[4],"reject")==0,"Unknown gate");
 auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];need(d&&queue,"Metal unavailable");NSError* error=nil;
 auto read=[&](const char* p){return [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:p] encoding:NSUTF8StringEncoding error:&error];};
 auto controlLibrary=[d newLibraryWithSource:read(argv[1]) options:nil error:&error];if(!controlLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto options=[MTLCompileOptions new];options.fastMathEnabled=NO;auto candidateLibrary=[d newLibraryWithSource:read(argv[2]) options:options error:&error];if(!candidateLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
 id<MTLComputePipelineState> control[]={pipeline(d,controlLibrary,@"bf16_decode"),pipeline(d,controlLibrary,@"bf16_decode_small")};
 auto split=pipeline(d,candidateLibrary,@"joined_split"),combine=pipeline(d,candidateLibrary,@"joined_combine");
 auto folder=[NSString stringWithUTF8String:argv[3]];auto manifest=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]] options:0 error:nil];need(manifest!=nil,"Assets unavailable");
 emit(@{@"kind":@"hardware",@"device":d.name,@"control":@"Original BF16 shader/default compile options",@"candidate":@"One MPS half product with lossless three-component input and FP32 result/recombination"});
 for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
  auto name=item[@"name"];const uint64_t k=[item[@"inner"] unsignedLongLongValue],n=[item[@"columns"] unsignedLongLongValue];
  auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];need(bytes.length==k*n*2,"BF16 length mismatch");auto raw=static_cast<const uint16_t*>(bytes.bytes);
  uint64_t wStride=[MPSMatrixDescriptor rowBytesFromColumns:k dataType:MPSDataTypeFloat16]/2;
  std::vector<uint16_t> halfBits(n*wStride,0);std::vector<int32_t> colScale(n);bool eligible=true;
  for(uint64_t col=0;col<n&&eligible;++col){uint16_t largest=0;for(uint64_t j=0;j<k;++j)largest=std::max(largest,uint16_t(raw[col*k+j]&0x7fff));eligible=Garnet::ExactHalfRowScale(largest,colScale[col]);
   for(uint64_t j=0;j<k&&eligible;++j){uint16_t encoded;eligible=Garnet::EncodeExactScaledHalf(raw[col*k+j],colScale[col],encoded);if(!eligible)break;halfBits[col*wStride+j]=encoded;
    __fp16 h;std::memcpy(&h,&encoded,2);float restored=std::ldexp(float(h),-colScale[col]);uint32_t bits;std::memcpy(&bits,&restored,4);need(bits==(uint32_t(raw[col*k+j])<<16),"Learned value changed");}
  }
  if(!eligible){emit(@{@"kind":@"excluded",@"matrix":name,@"reason":@"Exact half weight admission failed; original native fallback required"});continue;}
  auto weight=[d newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];auto halfWeight=[d newBufferWithBytes:halfBits.data() length:halfBits.size()*2 options:MTLResourceStorageModeShared];
  auto colScales=[d newBufferWithBytes:colScale.data() length:n*4 options:MTLResourceStorageModeShared];
  auto wMatrix=[[MPSMatrix alloc] initWithBuffer:halfWeight descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:wStride*2 dataType:MPSDataTypeFloat16]];need(weight&&halfWeight&&colScales&&wMatrix,"Weights unavailable");
  for(uint64_t rows:{1,2}){@autoreleasepool{
   uint64_t aStride=((k+15)/16)*16+16,hStride=[MPSMatrixDescriptor rowBytesFromColumns:k dataType:MPSDataTypeFloat16]/2,oStride=((n+15)/16)*16+16;
   P p{rows,k,n,aStride,hStride,oStride};BF16Params bp{rows,k,n,aStride,oStride};bool small=n<=256;need(control[small].threadExecutionWidth==32,"SIMD width changed");
   auto a=[d newBufferWithLength:rows*aStride*4 options:MTLResourceStorageModeShared],joined=[d newBufferWithLength:rows*3*hStride*2 options:MTLResourceStorageModeShared],parts=[d newBufferWithLength:rows*3*oStride*4 options:MTLResourceStorageModeShared],out=[d newBufferWithLength:rows*oStride*4 options:MTLResourceStorageModeShared];
   auto rowScales=[d newBufferWithLength:rows*4 options:MTLResourceStorageModeShared],failures=[d newBufferWithLength:4 options:MTLResourceStorageModeShared];need(a&&joined&&parts&&out&&rowScales&&failures,"Work allocation failed");
   auto left=[[MPSMatrix alloc] initWithBuffer:joined descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows*3 columns:k rowBytes:hStride*2 dataType:MPSDataTypeFloat16]];
   auto result=[[MPSMatrix alloc] initWithBuffer:parts descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows*3 columns:n rowBytes:oStride*4 dataType:MPSDataTypeFloat32]];
   auto product=[[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:NO transposeRight:YES resultRows:rows*3 resultColumns:n interiorColumns:k alpha:1 beta:0];need(left&&result&&product,"MPS setup failed");
   std::vector<float> base(rows*aStride,0);for(uint64_t r=0;r<rows;++r)for(uint64_t j=0;j<k;++j){float x=std::sin(float(r*k+j)*.013f)*.3f;if(rejection&&j==1)x=std::ldexp(1.0f,-60);if(j%211==0)x=-0.0f;base[r*aStride+j]=x;}
   const float variants[]={1,-1,.5f,2};size_t validated=0;double comparisonWorst=0,controlWorst=0,candidateWorst=0;
   std::vector<double> exact(rows*n),sumabs(rows*n);for(uint64_t r=0;r<rows;++r)for(uint64_t col=0;col<n;++col)for(uint64_t j=0;j<k;++j){uint32_t bits=uint32_t(raw[col*k+j])<<16;float w;std::memcpy(&w,&bits,4);double v=double(base[r*aStride+j])*double(w);exact[r*n+col]+=v;sumabs[r*n+col]+=std::abs(v);}
   std::vector<float> reference[4];double eps=std::ldexp(1.0,-23),operations=std::ceil(double(k)/32)+32,gamma=operations*eps/(1-operations*eps);
   auto run=[&](bool candidate,int variant,bool validate){
    auto values=static_cast<float*>(a.contents);for(size_t i=0;i<base.size();++i)values[i]=base[i]*variants[variant];std::fill_n(static_cast<float*>(out.contents),rows*oStride,99.5f);*static_cast<uint32_t*>(failures.contents)=0;
    auto begin=Clock::now();auto cb=[queue commandBuffer];need(cb!=nil,"Command unavailable");
    if(candidate){auto e=[cb computeCommandEncoder];need(e!=nil,"Split unavailable");[e setComputePipelineState:split];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:joined offset:0 atIndex:1];[e setBuffer:rowScales offset:0 atIndex:2];[e setBuffer:failures offset:0 atIndex:3];[e setBytes:&p length:sizeof(p) atIndex:4];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
     [product encodeToCommandBuffer:cb leftMatrix:left rightMatrix:wMatrix resultMatrix:result];
     e=[cb computeCommandEncoder];need(e!=nil,"Combine unavailable");[e setComputePipelineState:combine];[e setBuffer:parts offset:0 atIndex:0];[e setBuffer:rowScales offset:0 atIndex:1];[e setBuffer:colScales offset:0 atIndex:2];[e setBuffer:out offset:0 atIndex:3];[e setBytes:&p length:sizeof(p) atIndex:4];[e dispatchThreads:MTLSizeMake(rows*n,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
    }else{auto e=[cb computeCommandEncoder];need(e!=nil,"Control unavailable");[e setComputePipelineState:control[small]];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:weight offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e setBytes:&bp length:sizeof(bp) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((n+(small?3:15))/(small?4:16),1,1) threadsPerThreadgroup:MTLSizeMake(small?64:128,1,1)];[e endEncoding];}
    auto encoded=Clock::now();[cb commit];[cb waitUntilCompleted];auto end=Clock::now();need(cb.status==MTLCommandBufferStatusCompleted,"Projection failed");
    if(candidate&&*static_cast<uint32_t*>(failures.contents))throw std::runtime_error("Activation reconstruction rejected; original native fallback required");
    if(validate){auto values=static_cast<const float*>(out.contents);
     for(uint64_t r=0;r<rows;++r){for(uint64_t col=0;col<n;++col){auto index=r*oStride+col,refIndex=r*n+col;double v=values[index],expected=exact[refIndex]*variants[variant],bound=gamma*sumabs[refIndex]*std::abs(variants[variant])+1e-7;need(std::isfinite(v),"Nonfinite output");
      if(candidate){comparisonWorst=std::max(comparisonWorst,std::abs(v-reference[variant][index])/(5e-6+5e-5*std::abs(double(reference[variant][index]))));candidateWorst=std::max(candidateWorst,std::abs(v-expected)/bound);}
      else controlWorst=std::max(controlWorst,std::abs(v-expected)/bound);
     }for(uint64_t col=n;col<oStride;++col)need(values[r*oStride+col]==99.5f,"Output padding changed");}
     need(comparisonWorst<=1&&controlWorst<=1&&candidateWorst<=1,"Unchanged comparison/FP64 envelope failed");++validated;
    }
    return std::array<double,3>{ms(begin,encoded),ms(begin,end),(cb.GPUEndTime-cb.GPUStartTime)*1000};
   };
   if(rejection){try{run(true,0,false);}catch(const std::runtime_error& e){if(std::strcmp(e.what(),"Activation reconstruction rejected; original native fallback required")==0){emit(@{@"kind":@"admission",@"unsafe_input_rejected":@YES});return 0;}throw;}throw std::runtime_error("Wide-exponent input was admitted");}
   for(int v=0;v<4;++v){run(false,v,true);reference[v].resize(rows*oStride);std::memcpy(reference[v].data(),out.contents,reference[v].size()*4);run(true,v,true);}
   size_t epoch=0;auto warm=Clock::now();do{run(false,int(epoch%4),true);run(true,int(epoch++%4),true);}while(ms(warm,Clock::now())<250);
   emit(@{@"kind":@"warmup",@"matrix":name,@"rows":@(rows),@"elapsed_ms":@(ms(warm,Clock::now()))});
   for(int round=0;round<2;++round){int window=0;for(bool candidate:{false,true,true,false}){for(int w=0;w<3;++w)run(candidate,int(epoch++%4),true);auto enc=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];for(int sample=0;sample<12;++sample){auto t=run(candidate,int(epoch++%4),true);[enc addObject:@(t[0])];[wall addObject:@(t[1])];[gpu addObject:@(t[2])];}
    emit(@{@"kind":@"timing",@"matrix":name,@"rows":@(rows),@"round":@(round),@"window":@(window++),@"candidate":@(candidate),@"encode_ms":enc,@"wall_ms":wall,@"gpu_ms":gpu});}}
   emit(@{@"kind":@"check",@"matrix":name,@"rows":@(rows),@"weight_bits_exact":@YES,@"activation_bits_exact":@YES,@"output_padding_preserved":@YES,@"input_variants":@4,@"validated_executions":@(validated),@"comparison_worst_ratio":@(comparisonWorst),@"control_fp64_worst_ratio":@(controlWorst),@"candidate_fp64_worst_ratio":@(candidateWorst)});
  }}
 }}return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}}
