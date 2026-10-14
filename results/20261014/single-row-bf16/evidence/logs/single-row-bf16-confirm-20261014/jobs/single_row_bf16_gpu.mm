// Real-weight component gate; original shader math and fresh input/output data.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
static double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void emit(NSDictionary* object){NSError* error=nil;auto data=[NSJSONSerialization dataWithJSONObject:object options:0 error:&error];require(data!=nil,"JSON serialization failed");std::printf("%s\n",[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String);std::fflush(stdout);}
static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,id<MTLLibrary> library,NSString* name,bool indirect){
 auto descriptor=[MTLComputePipelineDescriptor new];descriptor.computeFunction=[library newFunctionWithName:name];descriptor.supportIndirectCommandBuffers=indirect;
 NSError* error=nil;auto result=[device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
 if(!result)throw std::runtime_error(error.localizedDescription.UTF8String);return result;
}
int main(int argc,char** argv){@autoreleasepool{try{
 require(argc==3,"Expected shader and existing verified assets");auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
 auto library=[device newLibraryWithSource:source options:nil error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 id<MTLComputePipelineState> normal[2]={makePipeline(device,library,@"bf16_decode",false),makePipeline(device,library,@"bf16_decode_small",false)};
 id<MTLComputePipelineState> recordable[2]={makePipeline(device,library,@"single_bf16_decode",false),makePipeline(device,library,@"single_bf16_decode_small",false)};
 emit(@{@"kind":@"device",@"device":device.name,@"small_output_policy":@"N <= 256",@"row_specialization":@"single row; batch two retains original kernel"});
 auto folder=[NSString stringWithUTF8String:argv[2]];auto manifestBytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
 auto manifest=[NSJSONSerialization JSONObjectWithData:manifestBytes options:0 error:&error];require(manifest!=nil,"Manifest unavailable");
 size_t profiles=0,executions=0;
 for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
  NSString* name=item[@"name"];uint64_t columns=[item[@"columns"] unsignedLongLongValue],inner=[item[@"inner"] unsignedLongLongValue];
  auto bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
  require(bytes.length==inner*columns*2,"Incomplete BF16 matrix");
  auto weight=[device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];require(weight!=nil,"Weight allocation failed");
  for(uint64_t rows:{1,2}){@autoreleasepool{
   const uint64_t aStride=((inner+15)/16)*16+16,oStride=((columns+15)/16)*16+16;
   const bool small=columns<=256;const uint64_t threads=small?64:128,tile=small?4:16;
   require(normal[small].threadExecutionWidth==32&&recordable[small].threadExecutionWidth==32,"SIMD width changed");
   std::vector<float> base(rows*aStride,0);for(uint64_t row=0;row<rows;++row)for(uint64_t k=0;k<inner;++k)base[row*aStride+k]=std::sin(float(row*inner+k)*.013f)*.3f;
   auto a=[device newBufferWithLength:base.size()*4 options:MTLResourceStorageModeShared];auto out=[device newBufferWithLength:rows*oStride*4 options:MTLResourceStorageModeShared];
   Params p{rows,inner,columns,aStride,oStride};auto parameter=[device newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];require(a&&out&&parameter,"Matrix buffer allocation failed");
   std::vector<double> exact(rows*columns),absolute(exact.size());auto raw=static_cast<const uint16_t*>(bytes.bytes);
   for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<columns;++col){double sum=0,absSum=0;
    for(uint64_t k=0;k<inner;++k){uint32_t bits=uint32_t(raw[col*inner+k])<<16;float w;std::memcpy(&w,&bits,4);double v=double(w)*base[row*aStride+k];sum+=v;absSum+=std::abs(v);}
    exact[row*columns+col]=sum;absolute[row*columns+col]=absSum;
   }
   const double epsilon=std::ldexp(1.0,-23),operations=std::ceil(double(inner)/32)+32,gamma=operations*epsilon/(1-operations*epsilon);
   const float scales[4]={1,-1,.5f,2};std::vector<float> reference[4];double worst=0;
   auto run=[&](int mode,int variant,bool validate){
    auto input=static_cast<float*>(a.contents);for(size_t i=0;i<base.size();++i)input[i]=base[i]*scales[variant];std::fill_n(static_cast<float*>(out.contents),rows*oStride,99.5f);
    auto begin=Clock::now();auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];require(command&&encoder,"Encoder unavailable");
    [encoder setComputePipelineState:mode==2&&rows==1?recordable[small]:normal[small]];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:weight offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder dispatchThreadgroups:MTLSizeMake((columns+tile-1)/tile,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [encoder endEncoding];auto encoded=Clock::now();[command commit];[command waitUntilCompleted];auto end=Clock::now();
    require(command.status==MTLCommandBufferStatusCompleted,"Projection command failed");
    if(validate)require(std::memcmp(out.contents,reference[variant].data(),rows*oStride*4)==0,"Projection or padding changed bits");
    ++executions;return std::array<double,3>{elapsed(begin,encoded),elapsed(begin,end),(command.GPUEndTime-command.GPUStartTime)*1000};
   };
   for(int variant=0;variant<4;++variant){run(0,variant,false);reference[variant].resize(rows*oStride);std::memcpy(reference[variant].data(),out.contents,rows*oStride*4);
    for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<oStride;++col){const auto value=reference[variant][row*oStride+col];if(col>=columns){require(value==99.5f,"Reference overwrote padding");continue;}
     const size_t index=row*columns+col;const double bound=gamma*absolute[index]*std::abs(scales[variant])+1e-7;
     require(std::isfinite(value)&&std::isfinite(bound),"Nonfinite reference");worst=std::max(worst,std::abs(double(value)-exact[index]*scales[variant])/bound);
    }
   }require(worst<=1,"Independent FP64 envelope exceeded");
   size_t epoch=0;
   size_t commonWarm=0;auto warmBegin=Clock::now();
   do{run(commonWarm%2?2:0,int(epoch++%4),true);++commonWarm;}while(elapsed(warmBegin,Clock::now())<250);
   emit(@{@"kind":@"warmup",@"matrix":name,@"rows":@(rows),@"validated_executions":@(commonWarm),@"elapsed_ms":@(elapsed(warmBegin,Clock::now()))});
   for(int pair=0;pair<1;++pair)for(int round=0;round<2;++round)for(bool candidate:{false,true,true,false}){
    const int mode=candidate?2:pair;for(int warm=0;warm<3;++warm)run(mode,int(epoch++%4),true);
    auto encoding=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];
    for(int sample=0;sample<12;++sample){auto timing=run(mode,int(epoch++%4),true);[encoding addObject:@(timing[0])];[wall addObject:@(timing[1])];[gpu addObject:@(timing[2])];}
    emit(@{@"kind":@"timing",@"matrix":name,@"rows":@(rows),@"pair":pair?@"direct_indirect_flag":@"native",@"round":@(round),@"policy":candidate?@"candidate":@"control",@"encode_ms":encoding,@"wall_ms":wall,@"gpu_ms":gpu,@"bits_exact":@YES});
   }
   emit(@{@"kind":@"check",@"matrix":name,@"rows":@(rows),@"bits_exact":@YES,@"input_variants":@4,@"fp64_worst_ratio":@(worst),@"padding_preserved":@YES,@"validated_executions":@(120+commonWarm)});++profiles;
  }}
 }}emit(@{@"kind":@"completed",@"profiles":@(profiles),@"executions":@(executions)});return 0;
 }catch(const std::exception& error){std::fprintf(stderr,"indirect_projection_failed: %s\n",error.what());return 1;}}}
