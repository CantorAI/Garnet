// SPDX-License-Identifier: Apache-2.0
// Standalone compatibility probe; never an inference speed benchmark.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
static uint16_t bf16(float value) {
    uint32_t bits;std::memcpy(&bits,&value,4);
    bits+=0x7fff+((bits>>16)&1);return uint16_t(bits>>16);
}
static float expand(uint16_t value) {uint32_t bits=uint32_t(value)<<16;float result;std::memcpy(&result,&bits,4);return result;}
static void emit(NSDictionary* value) {
    NSData* data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    printf("mps_bf16_probe %s\n",[[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] UTF8String]);fflush(stdout);
}
int main(int argc,const char** argv) {
    @autoreleasepool {
        if(argc!=2)return 2;
        NSString* mode=[NSString stringWithUTF8String:argv[1]];
        const bool inputBF=[mode isEqualToString:@"bf16-input"]||[mode isEqualToString:@"bf16-all"];
        const bool outputBF=[mode isEqualToString:@"bf16-all"],weightBF=![mode isEqualToString:@"fp32-control"];
        if(!inputBF&&!weightBF&&![mode isEqualToString:@"fp32-control"])return 2;
        if(![mode isEqualToString:@"mixed"]&&!inputBF&&weightBF)return 2;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
        if(!device||!queue)return 2;
        @try {
            for(size_t shape=0;shape<2;++shape) {
                const size_t m=shape?16:3,k=shape?1024:7,n=shape?64:5;
                const MPSDataType at=inputBF?MPSDataTypeBFloat16:MPSDataTypeFloat32;
                const MPSDataType bt=weightBF?MPSDataTypeBFloat16:MPSDataTypeFloat32;
                const MPSDataType ct=outputBF?MPSDataTypeBFloat16:MPSDataTypeFloat32;
                auto ad=[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:[MPSMatrixDescriptor rowBytesFromColumns:k dataType:at] dataType:at];
                auto bd=[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:[MPSMatrixDescriptor rowBytesFromColumns:k dataType:bt] dataType:bt];
                auto cd=[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:[MPSMatrixDescriptor rowBytesFromColumns:n dataType:ct] dataType:ct];
                id<MTLBuffer> ab=[device newBufferWithLength:m*ad.rowBytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bb=[device newBufferWithLength:n*bd.rowBytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> cb=[device newBufferWithLength:m*cd.rowBytes options:MTLResourceStorageModeShared];
                std::vector<float> a(m*k),b(n*k);
                for(size_t row=0;row<m;++row)for(size_t col=0;col<k;++col) {
                    float value=std::sin(float(row*k+col)*0.37f)*0.125f;
                    if(inputBF){uint16_t bits=bf16(value);std::memcpy((char*)ab.contents+row*ad.rowBytes+col*2,&bits,2);value=expand(bits);}
                    else std::memcpy((char*)ab.contents+row*ad.rowBytes+col*4,&value,4);
                    a[row*k+col]=value;
                }
                for(size_t row=0;row<n;++row)for(size_t col=0;col<k;++col) {
                    float value=expand(bf16(std::cos(float(row*k+col)*0.17f)*0.25f));b[row*k+col]=value;
                    if(weightBF){uint16_t bits=bf16(value);std::memcpy((char*)bb.contents+row*bd.rowBytes+col*2,&bits,2);}
                    else std::memcpy((char*)bb.contents+row*bd.rowBytes+col*4,&value,4);
                }
                auto am=[[MPSMatrix alloc] initWithBuffer:ab descriptor:ad],bm=[[MPSMatrix alloc] initWithBuffer:bb descriptor:bd],cm=[[MPSMatrix alloc] initWithBuffer:cb descriptor:cd];
                auto product=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:1 beta:0];
                id<MTLCommandBuffer> command=[queue commandBuffer];[product encodeToCommandBuffer:command leftMatrix:am rightMatrix:bm resultMatrix:cm];[command commit];[command waitUntilCompleted];
                if(command.status!=MTLCommandBufferStatusCompleted){emit(@{@"mode":mode,@"supported":@NO,@"error":command.error.localizedDescription?:@"GPU command failed"});return 1;}
                double worst=0;bool passed=true;
                for(size_t row=0;row<m;++row)for(size_t col=0;col<n;++col) {
                    double expected=0;for(size_t inner=0;inner<k;++inner)expected+=double(a[row*k+inner])*b[col*k+inner];
                    float actual;if(outputBF){uint16_t bits;std::memcpy(&bits,(char*)cb.contents+row*cd.rowBytes+col*2,2);actual=expand(bits);}else std::memcpy(&actual,(char*)cb.contents+row*cd.rowBytes+col*4,4);
                    double error=std::abs(actual-expected);worst=std::max(worst,error);
                    if(!std::isfinite(actual)||error>(outputBF?0.008:0.0002)*(1+std::abs(expected)))passed=false;
                }
                emit(@{@"mode":mode,@"device":device.name,@"supported":@YES,@"numerical_pass":@(passed),@"rows":@(m),@"inner":@(k),@"columns":@(n),@"max_abs_error":@(worst),@"fp32_activations":@(!inputBF),@"fp32_output":@(!outputBF)});
                if(!passed)return 1;
            }
        } @catch(NSException* error) {emit(@{@"mode":mode,@"supported":@NO,@"exception":error.name,@"reason":error.reason?:@""});return 0;}
    }
    return 0;
}
