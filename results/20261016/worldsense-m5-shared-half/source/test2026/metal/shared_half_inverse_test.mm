// Exhaustive correctness test of the GPU inverse; not a timing benchmark.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "shared_half_shader.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
int main(){@autoreleasepool{try{
 auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
 NSString* prefix=@"#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\nconstant uint fixed_rows [[function_constant(0)]];constant uint fixed_inner [[function_constant(1)]];constant uint fixed_columns [[function_constant(2)]];constant uint fixed_aStride [[function_constant(3)]];constant uint fixed_oStride [[function_constant(4)]];constant bool fixed_no_tail [[function_constant(5)]];\n";
 auto options=[MTLCompileOptions new];options.fastMathEnabled=NO;NSError* error=nil;
 auto library=[device newLibraryWithSource:[prefix stringByAppendingString:[NSString stringWithUTF8String:Garnet::kSharedHalfShader]] options:options error:&error];
 if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"shared_half_inverse_test"] error:&error];if(!pipeline)throw std::runtime_error("Inverse test pipeline unavailable");
 auto buffer=[device newBufferWithLength:65536*4 options:MTLResourceStorageModeShared];size_t checked=0;
 for(int scale=-64;scale<=64;++scale){auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:pipeline];[e setBuffer:buffer offset:0 atIndex:0];[e setBytes:&scale length:4 atIndex:1];[e dispatchThreads:MTLSizeMake(65536,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("GPU inverse command failed");
  const auto* actual=static_cast<const float*>(buffer.contents);
  for(uint32_t i=0;i<65536;++i){if(((i>>10)&31)==31)continue;uint16_t word=uint16_t(i);__fp16 half;std::memcpy(&half,&word,2);float expected=std::ldexp(float(half),-scale);if(std::memcmp(&expected,actual+i,4))throw std::runtime_error("GPU inverse changed FP32 bits");++checked;}
 }
 std::cout<<"Exhaustive GPU finite-half inverse and signed-zero bit equality PASS "<<checked<<" values across 129 scales\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
