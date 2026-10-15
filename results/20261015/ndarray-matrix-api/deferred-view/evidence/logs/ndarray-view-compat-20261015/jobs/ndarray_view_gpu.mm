// API gate only: MPS NDArray versus Garnet's mixed-storage MPS matrix policy.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "exact_half_codec.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"JSON failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static NSArray* floats(const float* data,size_t count){auto result=[NSMutableArray arrayWithCapacity:count];for(size_t i=0;i<count;++i)[result addObject:@(data[i])];return result;}
static NSArray* words(const std::vector<uint16_t>& data){auto result=[NSMutableArray arrayWithCapacity:data.size()];for(auto v:data)[result addObject:@(v)];return result;}
static float expand(uint16_t raw){uint32_t w=uint32_t(raw)<<16;float v;std::memcpy(&v,&w,4);return v;}
static id<MTLBuffer> buffer(id<MTLDevice> device,const void* data,size_t bytes){auto result=[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];need(result!=nil,"Buffer allocation failed");return result;}
static MPSNDArray* array(id<MTLBuffer> storage,size_t rows,size_t cols,MPSDataType type){
    auto d=[MPSNDArrayDescriptor descriptorWithDataType:type shape:@[@1,@1,@(rows),@(cols)]];d.preferPackedRows=YES;
    auto result=[[MPSNDArray alloc] initWithBuffer:storage offset:0 descriptor:d];
    need(result!=nil&&result.userBuffer==storage&&result.dataType==type,"NDArray dtype or alias changed");return result;
}
static MPSMatrix* matrix(id<MTLBuffer> storage,size_t rows,size_t cols,MPSDataType type){
    size_t bytes=type==MPSDataTypeFloat16?2:4;
    auto d=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:cols*bytes dataType:type];
    auto result=[[MPSMatrix alloc] initWithBuffer:storage descriptor:d];need(result!=nil,"Matrix view failed");return result;
}
int main(int argc,char** argv){@autoreleasepool{@try{try{
    need(argc==3,"Expected storage and alpha path");std::string mode=argv[1];need(mode=="mixed"||mode=="f32","Unknown mode");
    std::string variant=argv[2];need(variant=="unit"||variant=="three","Unknown alpha path");int scale=variant=="unit"?0:8;NSUInteger sources=variant=="three"?3:2;
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];need(queue!=nil,"No Metal queue");
    const size_t shapes[][3]={{3,8,16},{16,1024,64},{80,64,32},{160,64,32}};
    size_t checked=0;
    for(size_t shape=0;shape<4;++shape){@autoreleasepool{
        size_t m=shapes[shape][0],k=shapes[shape][1],n=shapes[shape][2];
        std::vector<uint16_t> raw(n*k),half(n*k),transposed(k*n);std::vector<float> wide(k*n);
        for(size_t c=0;c<n;++c)for(size_t j=0;j<k;++j){
            float value=float(int((c*7+j*13)%17)-8)/128.f;uint32_t bits;std::memcpy(&bits,&value,4);raw[c*k+j]=uint16_t(bits>>16);
            need(Garnet::EncodeExactScaledHalf(raw[c*k+j],scale,half[c*k+j]),"Exact half encoding rejected fixture");
            transposed[j*n+c]=half[c*k+j];wide[j*n+c]=std::ldexp(expand(raw[c*k+j]),scale);
            _Float16 restored;std::memcpy(&restored,&transposed[j*n+c],2);
            need(std::ldexp(float(restored),-scale)==expand(raw[c*k+j]),"Original weight value changed");
        }
        auto original=buffer(device,half.data(),half.size()*2);
        auto right=mode=="mixed"?buffer(device,transposed.data(),transposed.size()*2):buffer(device,wide.data(),wide.size()*4);
        std::vector<float> rowWide(n*k);for(size_t i=0;i<n*k;++i)rowWide[i]=std::ldexp(expand(raw[i]),scale);
        auto viewStorage=mode=="mixed"?original:buffer(device,rowWide.data(),rowWide.size()*4);
        auto viewBase=array(viewStorage,n,k,mode=="mixed"?MPSDataTypeFloat16:MPSDataTypeFloat32);
        MPSNDArray* rhs=nil;
        for(bool packed:{true,false}){
            auto desc=[viewBase descriptor];desc.preferPackedRows=packed;[desc transposeDimension:0 withDimension:1];
            auto v=[viewBase arrayViewWithDescriptor:desc];auto logical=v?[v descriptor]:nil;
            emit(@{@"kind":@"view",@"shape":@(shape),@"packed_requested":@(packed),@"available":@(v!=nil),
                @"dtype":@(v?v.dataType:0),@"user_buffer_alias":@(v&&v.userBuffer==viewStorage),
                @"physical_dim0":@(v?[v lengthOfDimension:0]:0),@"physical_dim1":@(v?[v lengthOfDimension:1]:0),
                @"logical_shape":logical?[logical getShape]:@[]});
            if(!packed)rhs=v;
        }
        need(rhs!=nil,"Deferred transpose view unavailable");auto logical=[rhs descriptor];
        need([logical lengthOfDimension:0]==n&&[logical lengthOfDimension:1]==k,"Deferred logical shape mismatch");
        auto controlRhs=matrix(original,n,k,MPSDataTypeFloat16);
        auto control=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:std::ldexp(1.,-scale) beta:0];
        auto product=[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:sources];
        need(control!=nil&&product!=nil,"Product allocation failed");product.alpha=std::ldexp(1.,-scale);product.beta=0;
        emit(@{@"kind":@"fixture",@"mode":[NSString stringWithUTF8String:mode.c_str()],@"shape":@(shape),@"rows":@(m),@"inner":@(k),@"columns":@(n),@"scale":@(scale),@"variant":[NSString stringWithUTF8String:variant.c_str()],@"source_count":@(sources),@"alpha_requested":@(std::ldexp(1.,-scale)),@"alpha_reported":@(product.alpha),@"beta_reported":@(product.beta),@"bf16":words(raw),@"half":words(half),@"transposed_half":words(transposed),@"wide":floats(wide.data(),wide.size())});
        for(int pattern=0;pattern<4;++pattern){
            std::vector<float> a(m*k),sentinel(m*n+16,99.5f);
            for(size_t r=0;r<m;++r)for(size_t j=0;j<k;++j){float v=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;if(pattern==1)v*=float(j%2?-16:16);if(pattern==2)v=j==((r*53+17)%k)?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(j%9)-4);a[r*k+j]=v;}
            auto left=buffer(device,a.data(),a.size()*4),nativeOut=buffer(device,sentinel.data(),sentinel.size()*4),ndOut=buffer(device,sentinel.data(),sentinel.size()*4);
            auto lhs=array(left,m,k,MPSDataTypeFloat32),result=array(ndOut,m,n,MPSDataTypeFloat32);
            std::vector<float> zeros(m*n,0.f);auto addendStorage=buffer(device,zeros.data(),zeros.size()*4);
            auto addend=array(addendStorage,m,n,MPSDataTypeFloat32);NSArray* operands=sources==3?@[lhs,rhs,addend]:@[lhs,rhs];
            auto nativeLhs=matrix(left,m,k,MPSDataTypeFloat32),nativeResult=matrix(nativeOut,m,n,MPSDataTypeFloat32);
            auto command=[queue commandBuffer];[control encodeToCommandBuffer:command leftMatrix:nativeLhs rightMatrix:controlRhs resultMatrix:nativeResult];
            [command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"Native command failed");
            command=[queue commandBuffer];[product encodeToCommandBuffer:command sourceArrays:operands destinationArray:result];
            [command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"NDArray command failed");
            need(!std::memcmp(left.contents,a.data(),a.size()*4)&&!std::memcmp(original.contents,half.data(),half.size()*2),"Immutable input changed");
            need(!std::memcmp(addendStorage.contents,zeros.data(),zeros.size()*4),"Zero addend changed");
            need(!std::memcmp(right.contents,mode=="mixed"?static_cast<const void*>(transposed.data()):static_cast<const void*>(wide.data()),mode=="mixed"?transposed.size()*2:wide.size()*4),"NDArray weights changed");
            need(!std::memcmp(viewStorage.contents,mode=="mixed"?static_cast<const void*>(half.data()):static_cast<const void*>(rowWide.data()),mode=="mixed"?half.size()*2:rowWide.size()*4),"View storage changed");
            auto actual=static_cast<const float*>(ndOut.contents),expected=static_cast<const float*>(nativeOut.contents);
            for(size_t i=m*n;i<m*n+16;++i)need(actual[i]==99.5f&&expected[i]==99.5f,"Output guard overwritten");
            double worst=0,nativeWorst=0,comparison=0,eps=std::ldexp(1.,-23),terms=std::ceil(double(k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<m;++r)for(size_t c=0;c<n;++c){double sum=0,absolute=0;for(size_t j=0;j<k;++j){double v=double(a[r*k+j])*expand(raw[c*k+j]);sum+=v;absolute+=std::abs(v);}size_t i=r*n+c;
                need(std::isfinite(actual[i])&&std::isfinite(expected[i]),"Nonfinite product");double bound=gamma*absolute+1e-7;
                worst=std::max(worst,std::abs(double(actual[i])-sum)/bound);nativeWorst=std::max(nativeWorst,std::abs(double(expected[i])-sum)/bound);
                comparison=std::max(comparison,std::abs(double(actual[i])-expected[i])/(5e-6+5e-5*std::abs(double(expected[i]))));}
            bool passed=worst<=1&&nativeWorst<=1&&comparison<=1;
            emit(@{@"kind":@"check",@"shape":@(shape),@"pattern":@(pattern),@"left_fp32":@(lhs.dataType==MPSDataTypeFloat32),@"right_fp16":@(rhs.dataType==MPSDataTypeFloat16),@"right_fp32":@(rhs.dataType==MPSDataTypeFloat32),@"output_fp32":@(result.dataType==MPSDataTypeFloat32),@"guard_preserved":@YES,@"immutable_inputs":@YES,@"inputs":floats(a.data(),a.size()),@"control":floats(expected,m*n),@"candidate":floats(actual,m*n),@"native_fp64_worst":@(nativeWorst),@"fp64_worst":@(worst),@"comparison_worst":@(comparison),@"passed":@(passed)});
            need(passed,"NDArray exceeded original numerical envelope");++checked;
        }
    }}emit(@{@"kind":@"complete",@"checks":@(checked)});return 0;
}catch(const std::exception& e){emit(@{@"kind":@"error",@"reason":[NSString stringWithUTF8String:e.what()]});return 1;}}
@catch(NSException* e){emit(@{@"kind":@"exception",@"reason":e.reason?:@"Unknown Objective-C error"});return 1;}}}
