// Isolated original-kernel bit checks and component timing; no model speed claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
struct P {uint64_t batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void emit(NSDictionary* value){NSError* e=nil;auto raw=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingSortedKeys error:&e];need(raw!=nil,"JSON serialization failed");std::printf("%.*s\n",int(raw.length),static_cast<const char*>(raw.bytes));std::fflush(stdout);}
static std::vector<float> load(NSString* folder,NSString* file,size_t words){auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:file]];need(data.length==words*4,"Asset bytes differ");std::vector<float> v(words);std::memcpy(v.data(),data.bytes,data.length);return v;}
static id<MTLBuffer> buffer(id<MTLDevice> device,const std::vector<float>& values){auto b=[device newBufferWithBytes:values.data() length:values.size()*4 options:MTLResourceStorageModeShared];need(b!=nil,"Allocation failed");return b;}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==4,"Expected shader, assets and numeric/timing mode");bool timing=std::strcmp(argv[3],"timing")==0;need(timing||std::strcmp(argv[3],"numeric")==0,"Unknown mode");
 auto device=MTLCreateSystemDefaultDevice();need(device!=nil,"No Metal device");auto queue=[device newCommandQueue];NSError* error=nil;
 auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];need(text!=nil,"Shader missing");
 MTLCompileOptions* options=nil; // Matches accepted d52 main shader: native-bfloat-body is disabled.
 auto library=[device newLibraryWithSource:text options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 NSMutableDictionary<NSNumber*,id<MTLComputePipelineState>>* pipelines=[NSMutableDictionary new];
 for(int width:{16,32,64}){auto name=[NSString stringWithFormat:@"gated_delta_chunk_value%d",width];auto f=[library newFunctionWithName:name];need(f!=nil,"Missing value tile kernel");auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"128 threads / 32 lanes unavailable");pipelines[@(width)]=p;emit(@{@"kind":@"pipeline",@"width":@(width),@"execution_width":@(p.threadExecutionWidth),@"maximum_threads":@(p.maxTotalThreadsPerThreadgroup),@"static_threadgroup_bytes":@(p.staticThreadgroupMemoryLength)});}
 NSString* folder=[NSString stringWithUTF8String:argv[2]];auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];need(data!=nil,"Manifest missing");auto manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];need([manifest isKindOfClass:[NSDictionary class]],"Invalid manifest");size_t profiles=0,checks=0,checked=0,samples=0;
 for(NSDictionary* item in manifest[@"cases"]){@autoreleasepool{
  NSString* name=item[@"name"];bool learned=[name hasPrefix:@"real-layer"];if(timing&&!learned)continue;
  uint64_t b=[item[@"batch"] unsignedLongLongValue],c=[item[@"chunk"] unsignedLongLongValue],h=[item[@"heads"] unsignedLongLongValue],k=128,v=128;need(b&&c&&c<=80&&h,"Invalid profile");
  auto preparedValues=load(folder,item[@"prepared"],b*c*(3*h*128+2*h)),stateValues=load(folder,item[@"state"],b*h*k*v),activeValues=load(folder,item[@"active"],b*c);
  for(uint64_t padding:{0,4}){
   if(timing&&padding)continue;P p{b,c,h,k,v,3*h*128+2*h,h*k*v+padding,c,h*v+padding,h*k*v+2*padding};
   constexpr float guard=-767.125f;std::vector<float> state(b*p.sStride,guard),out(b*c*p.oStride,guard),next(b*p.nextStride,guard);
   for(size_t row=0;row<b;++row)std::memcpy(state.data()+row*p.sStride,stateValues.data()+row*h*k*v,h*k*v*4);
   auto prepared=buffer(device,preparedValues),prior=buffer(device,state),active=buffer(device,activeValues),output=buffer(device,out),nextState=buffer(device,next);
   auto run=[&](int width,int repeat){
    auto begin=std::chrono::steady_clock::now();auto command=[queue commandBuffer];
    for(int iteration=0;iteration<repeat;++iteration){auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipelines[@(width)]];[encoder setBuffer:prepared offset:0 atIndex:0];[encoder setBuffer:prior offset:0 atIndex:1];[encoder setBuffer:active offset:0 atIndex:2];[encoder setBuffer:output offset:0 atIndex:3];[encoder setBuffer:nextState offset:0 atIndex:4];[encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder dispatchThreadgroups:MTLSizeMake(b*h*v/width,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];}
    [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);double gpu=(command.GPUEndTime-command.GPUStartTime)*1000,wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Invalid clocks");return std::pair<double,double>{gpu,wall};
   };
   if(!timing){
    for(int step=0;step<3;++step){
     auto changed=preparedValues;for(size_t row=0;row<b*c;++row)for(size_t i=0;i<h*k;++i)changed[row*p.pStride+i]*=1.f+.01f*step;std::memcpy(prepared.contents,changed.data(),changed.size()*4);
     std::memcpy(output.contents,out.data(),out.size()*4);std::memcpy(nextState.contents,next.data(),next.size()*4);run(16,1);std::vector<float> expected(out.size()),expectedState(next.size());std::memcpy(expected.data(),output.contents,expected.size()*4);std::memcpy(expectedState.data(),nextState.contents,expectedState.size()*4);
     for(size_t row=0;row<b;++row){for(size_t i=h*k*v;i<p.nextStride;++i)need(expectedState[row*p.nextStride+i]==guard,"Original state padding overwritten");for(size_t token=0;token<c;++token){size_t base=(row*c+token)*p.oStride;for(size_t i=h*v;i<p.oStride;++i)need(expected[base+i]==guard,"Original output padding overwritten");if(activeValues[row*c+token]==0)for(size_t i=0;i<h*v;++i)need(expected[base+i]==0,"Inactive original output nonzero");}}
     for(int width:{16,32,64}){std::memcpy(output.contents,out.data(),out.size()*4);std::memcpy(nextState.contents,next.data(),next.size()*4);run(width,1);bool equal=std::memcmp(output.contents,expected.data(),expected.size()*4)==0&&std::memcmp(nextState.contents,expectedState.data(),expectedState.size()*4)==0;emit(@{@"kind":@"check",@"name":name,@"padding":@(padding),@"step":@(step),@"width":@(width),@"output_words":@(expected.size()),@"state_words":@(expectedState.size()),@"original_bits_equal":@(equal)});need(equal,"Original output/state bits differ");for(float x:expected)need(std::isfinite(x),"Nonfinite original output");for(float x:expectedState)need(std::isfinite(x),"Nonfinite original state");need(std::memcmp(prepared.contents,changed.data(),changed.size()*4)==0&&std::memcmp(prior.contents,state.data(),state.size()*4)==0&&std::memcmp(active.contents,activeValues.data(),activeValues.size()*4)==0,"Read-only input changed");checked+=expected.size()+expectedState.size();++checks;}
     for(size_t row=0;row<b;++row)std::memcpy(state.data()+row*p.sStride,expectedState.data()+row*p.nextStride,h*k*v*4);std::memcpy(prior.contents,state.data(),state.size()*4);
    }
   }else{
    // Fixed repeated primitive inputs are a component calibration, not request reuse in a model comparison.
    constexpr int repeats=16;
    for(int width:{32,64}){
     auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);do{run(16,repeats);run(width,repeats);}while(std::chrono::steady_clock::now()<until);
     for(bool null:{true,false})for(int round=0;round<10;++round)for(int window=0;window<4;++window){bool candidate=window==1||window==2;int actual=(!null&&candidate)?width:16;for(int warm=0;warm<3;++warm)run(actual,repeats);for(int sample=0;sample<5;++sample){auto value=run(actual,repeats);emit(@{@"kind":@"timing",@"name":name,@"width":@(width),@"null":@(null),@"candidate":@(candidate),@"actual_width":@(actual),@"round":@(round),@"window":@(window),@"sample":@(sample),@"dispatches":@(repeats),@"gpu_ms":@(value.first),@"wall_ms":@(value.second)});++samples;}}
    }
   }
   ++profiles;
  }
 }}emit(@{@"kind":@"complete",@"mode":timing?@"timing":@"numeric",@"profiles":@(profiles),@"checks":@(checks),@"fp32_words_checked":@(checked),@"timing_samples":@(samples),@"model_inference":@NO,@"full_goal_met":@NO});return 0;
}catch(const std::exception& error){std::fprintf(stderr,"delta_value_tiles_failed: %s\n",error.what());return 1;}}}
