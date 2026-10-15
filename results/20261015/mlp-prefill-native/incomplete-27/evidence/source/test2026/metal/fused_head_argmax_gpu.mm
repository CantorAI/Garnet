// Isolated head component; original dot products and exact greedy semantics.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
struct P{uint64_t rows,inner,columns,aStride,oStride;};
struct A{uint64_t rows,columns,iStride,oStride;};
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void finish(id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");}
static id<MTLComputePipelineState> pipe(id<MTLDevice> d,id<MTLLibrary> l,NSString* name){NSError* error=nil;auto p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:name] error:&error];need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"Unsupported pipeline");return p;}
static id<MTLBuffer> load(id<MTLDevice> d,NSString* root,NSString* name){auto data=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:name]];need(data&&data.length,"Missing asset");return [d newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==3,"Expected shader and assets");auto d=MTLCreateSystemDefaultDevice();auto q=[d newCommandQueue];NSError* error=nil;
 auto code=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];auto l=[d newLibraryWithSource:code options:nil error:&error];if(!l)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto raw=pipe(d,l,@"bf16_decode"),fused=pipe(d,l,@"fused_head"),reduce=pipe(d,l,@"fused_head_reduce"),argmax=pipe(d,l,@"argmax_rows");
 NSString* folder=[NSString stringWithUTF8String:argv[2]];auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];NSDictionary* meta=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];need(meta!=nil,"Missing manifest");
 uint64_t n=[meta[@"rows"] unsignedLongLongValue],k=[meta[@"inner"] unsignedLongLongValue];auto weight=load(d,folder,@"raw.bin");need(weight.length==n*k*2,"Weight size mismatch");
 auto test=[&](NSString* name,uint64_t b,uint64_t inner,uint64_t columns,id<MTLBuffer> w,id<MTLBuffer> input,id<MTLBuffer> expected,bool timed){
  const uint64_t groups=(columns+15)/16,stride=columns+3;P p{b,inner,columns,inner,stride};A ap{b,columns,stride,2},rp{b,groups,groups*3,2};
  auto logits=[d newBufferWithLength:b*stride*4 options:MTLResourceStorageModeShared],debug=[d newBufferWithLength:b*stride*4 options:MTLResourceStorageModeShared],parts=[d newBufferWithLength:(b*groups*3+3)*4 options:MTLResourceStorageModeShared];
  auto pred=[d newBufferWithLength:b*2*4 options:MTLResourceStorageModeShared];
  std::fill_n(static_cast<float*>(logits.contents),b*stride,12345.0f);std::fill_n(static_cast<float*>(debug.contents),b*stride,12345.0f);std::fill_n(static_cast<float*>(parts.contents),b*groups*3+3,12345.0f);
  auto run=[&](bool candidate,uint32_t emit){auto begin=std::chrono::steady_clock::now();auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:candidate?fused:raw];[e setBuffer:input offset:0 atIndex:0];[e setBuffer:w offset:0 atIndex:1];[e setBuffer:candidate?parts:logits offset:0 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];
   if(candidate){[e setBuffer:debug offset:0 atIndex:4];[e setBytes:&emit length:sizeof(emit) atIndex:5];}[e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
   e=[cb computeCommandEncoder];[e setComputePipelineState:candidate?reduce:argmax];[e setBuffer:candidate?parts:logits offset:0 atIndex:0];[e setBuffer:pred offset:0 atIndex:1];[e setBytes:candidate?&rp:&ap length:sizeof(A) atIndex:2];[e dispatchThreadgroups:MTLSizeMake(b,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];finish(cb);
   return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};};
  run(false,0);std::vector<float> reference(static_cast<float*>(logits.contents),static_cast<float*>(logits.contents)+b*stride),prediction(static_cast<float*>(pred.contents),static_cast<float*>(pred.contents)+b*2);
  if(expected){need(expected.length==b*columns*4,"Reference size mismatch");for(uint64_t row=0;row<b;++row)need(std::memcmp(reference.data()+row*stride,static_cast<float*>(expected.contents)+row*columns,columns*4)==0,"Native changed captured logits");}
  run(true,1);need(std::memcmp(pred.contents,prediction.data(),b*2*4)==0,"Greedy mismatch with debug enabled");
  for(uint64_t row=0;row<b;++row)need(std::memcmp(static_cast<float*>(debug.contents)+row*stride,reference.data()+row*stride,columns*4)==0,"Fused dot products changed bits");
  for(uint64_t row=0;row<b;++row)for(uint64_t col=columns;col<stride;++col)need(static_cast<float*>(logits.contents)[row*stride+col]==12345.0f&&static_cast<float*>(debug.contents)[row*stride+col]==12345.0f,"Logit padding overwritten");
  for(uint64_t col=b*groups*3;col<b*groups*3+3;++col)need(static_cast<float*>(parts.contents)[col]==12345.0f,"Partial padding overwritten");
  run(true,0);need(std::memcmp(pred.contents,prediction.data(),b*2*4)==0,"Timed greedy mismatch");
  std::printf("{\"kind\":\"check\",\"query\":\"%s\",\"batch\":%llu,\"columns\":%llu,\"inner\":%llu,\"projection_bits_exact\":true,\"greedy_exact\":true,\"padding_preserved\":true,\"captured_reference_checked\":%s,\"timed\":%s}\n",name.UTF8String,(unsigned long long)b,(unsigned long long)columns,(unsigned long long)inner,expected?"true":"false",timed?"true":"false");std::fflush(stdout);
  if(!timed)return;
  auto warm=std::chrono::steady_clock::now();while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<.25){run(false,0);run(true,0);}
  for(int round=0;round<2;++round)for(int window=0;window<4;++window){bool candidate=window==1||window==2;std::vector<double> gpu,wall;
   for(int trial=0;trial<8;++trial){auto times=run(candidate,0);need(std::memcmp(pred.contents,prediction.data(),b*2*4)==0,"Timed prediction changed");gpu.push_back(times.first);wall.push_back(times.second);}
   std::printf("{\"kind\":\"timing\",\"query\":\"%s\",\"batch\":%llu,\"candidate\":%s,\"round\":%d,\"window\":%d,\"gpu_ms\":[",name.UTF8String,(unsigned long long)b,candidate?"true":"false",round,window);for(size_t i=0;i<gpu.size();++i)std::printf("%s%.17g",i?",":"",gpu[i]);std::printf("],\"wall_ms\":[");for(size_t i=0;i<wall.size();++i)std::printf("%s%.17g",i?",":"",wall[i]);std::puts("]}");std::fflush(stdout);}
 };
 for(NSDictionary* query in meta[@"queries"]){@autoreleasepool{NSString* name=query[@"name"];uint64_t b=[query[@"batch"] unsignedLongLongValue];need(b==1||b==2,"Unsupported batch");auto input=load(d,folder,[name stringByAppendingString:@".input.bin"]),expected=load(d,folder,[name stringByAppendingString:@".reference.bin"]);need(input.length==b*k*4,"Input size mismatch");test(name,b,k,n,weight,input,expected,true);}}
 // Irregular dimensions, all-equal ties, signed zeros and nonfinite detection.
 for(uint64_t columns:{1ull,3ull,17ull,33ull})for(uint64_t b:{1ull,2ull})for(unsigned fixture=0;fixture<4;++fixture){
  uint64_t inner=7;std::vector<uint16_t> bits(columns*inner,0);std::vector<float> input(b*inner,fixture==1?-0.0f:1.0f);
  if(fixture==2)bits[(columns-1)*inner]=0x7f80;if(fixture==3)bits[(columns-1)*inner]=0x7fc0;
  auto w=[d newBufferWithBytes:bits.data() length:bits.size()*2 options:MTLResourceStorageModeShared],a=[d newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
  test([NSString stringWithFormat:@"edge-%llu-b%llu-f%u",(unsigned long long)columns,(unsigned long long)b,fixture],b,inner,columns,w,a,nil,false);
 }
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"fused_head_argmax_failed: %s\n",e.what());return 1;}}}
