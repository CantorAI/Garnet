// SPDX-License-Identifier: Apache-2.0
// Toy mixed-storage compatibility/numerical probe, not model inference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstring>
#include <vector>
static float checkpointValue(float value) {
    uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);bits&=0xffff0000;
    std::memcpy(&value,&bits,4);return value;
}
static void emit(NSDictionary* value) {
    NSData* data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    printf("mps_exact_half_probe %s\n",[[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] UTF8String]);fflush(stdout);
}
int main() {
 @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    if(!device||!queue)return 2;
    @try {
     for(size_t shape=0;shape<2;++shape) {
      const size_t m=shape?16:3,k=shape?1024:7,n=shape?64:5;
      auto ad=[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:[MPSMatrixDescriptor rowBytesFromColumns:k dataType:MPSDataTypeFloat32] dataType:MPSDataTypeFloat32];
      auto bd=[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:[MPSMatrixDescriptor rowBytesFromColumns:k dataType:MPSDataTypeFloat16] dataType:MPSDataTypeFloat16];
      auto cd=[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:[MPSMatrixDescriptor rowBytesFromColumns:n dataType:MPSDataTypeFloat32] dataType:MPSDataTypeFloat32];
      id<MTLBuffer> ab=[device newBufferWithLength:m*ad.rowBytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> bb=[device newBufferWithLength:n*bd.rowBytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> cb=[device newBufferWithLength:m*cd.rowBytes options:MTLResourceStorageModeShared];
      std::vector<float> a(m*k),b(n*k);
      for(size_t row=0;row<m;++row)for(size_t col=0;col<k;++col) {
       float value=std::sin(float(row*k+col)*0.37f)*0.125f;a[row*k+col]=value;
       std::memcpy((char*)ab.contents+row*ad.rowBytes+col*4,&value,4);
      }
      for(size_t row=0;row<n;++row)for(size_t col=0;col<k;++col) {
       const float value=checkpointValue(std::cos(float(row*k+col)*0.17f)*0.25f);b[row*k+col]=value;
       const __fp16 encoded=__fp16(std::ldexp(value,16));float recovered=std::ldexp(float(encoded),-16);
       if(std::memcmp(&value,&recovered,4))throw "Toy weight does not round-trip exactly";
       std::memcpy((char*)bb.contents+row*bd.rowBytes+col*2,&encoded,2);
      }
      auto am=[[MPSMatrix alloc] initWithBuffer:ab descriptor:ad],bm=[[MPSMatrix alloc] initWithBuffer:bb descriptor:bd],cm=[[MPSMatrix alloc] initWithBuffer:cb descriptor:cd];
      auto product=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:1 beta:0];
      id<MTLCommandBuffer> command=[queue commandBuffer];[product encodeToCommandBuffer:command leftMatrix:am rightMatrix:bm resultMatrix:cm];[command commit];[command waitUntilCompleted];
      if(command.status!=MTLCommandBufferStatusCompleted){emit(@{@"supported":@NO,@"error":command.error.localizedDescription?:@"GPU command failed"});return 1;}
      double worst=0;bool passed=true;
      for(size_t row=0;row<m;++row)for(size_t col=0;col<n;++col) {
       double expected=0;for(size_t inner=0;inner<k;++inner)expected+=double(a[row*k+inner])*b[col*k+inner];
       float actual;std::memcpy(&actual,(char*)cb.contents+row*cd.rowBytes+col*4,4);actual=std::ldexp(actual,-16);
       double error=std::abs(actual-expected);worst=std::max(worst,error);
       if(!std::isfinite(actual)||error>0.0002*(1+std::abs(expected)))passed=false;
      }
      emit(@{@"device":device.name,@"supported":@YES,@"numerical_pass":@(passed),@"weight_bits_roundtrip_exact":@YES,@"rows":@(m),@"inner":@(k),@"columns":@(n),@"max_abs_error":@(worst),@"fp32_activations":@YES,@"fp32_output":@YES,@"weight_scale_exponent":@16});
      if(!passed)return 1;
     }
    } @catch(NSException* error) {emit(@{@"supported":@NO,@"exception":error.name,@"reason":error.reason?:@""});return 1;}
 }
 return 0;
}
