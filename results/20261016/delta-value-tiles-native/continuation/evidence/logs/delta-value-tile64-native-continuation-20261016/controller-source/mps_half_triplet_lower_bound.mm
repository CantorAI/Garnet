// Synthetic feasibility only: precomputed exact activation planes, not learned-model admission.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void emit(NSDictionary* value){auto data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];need(data!=nil,"JSON failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
static MPSMatrix* matrix(id<MTLDevice> d,size_t rows,size_t cols,MPSDataType type){
    size_t bytes=type==MPSDataTypeFloat32?4:2,stride=[MPSMatrixDescriptor rowBytesFromColumns:cols dataType:type]+64;
    auto b=[d newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];need(b!=nil,"Matrix allocation failed");
    auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:stride dataType:type];
    auto result=[[MPSMatrix alloc] initWithBuffer:b descriptor:desc];need(result!=nil&&result.rowBytes>=cols*bytes,"Matrix creation failed");return result;
}
static MPSMatrixMultiplication* product(id<MTLDevice> d,size_t m,size_t n,size_t k,double alpha,double beta){
    auto p=[[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];need(p!=nil,"MPS product unavailable");return p;
}
static void fill_output(MPSMatrix* o){std::fill_n(static_cast<float*>(o.data.contents),o.rows*o.rowBytes/4,99.5f);}
int main(){@autoreleasepool{@try{try{
    auto d=MTLCreateSystemDefaultDevice();auto queue=[d newCommandQueue];need(d&&queue,"Metal unavailable");
    emit(@{@"kind":@"hardware",@"device":d.name,@"scope":@"Synthetic shape feasibility only; MPS internal accumulator dtype not established from output dtype"});
    // Distinguish a rounded-half product/result from the observed float result.
    // These finite fixtures do not prove every internal MPS accumulation instruction.
    for(unsigned fixture=0;fixture<3;++fixture){
        size_t k=fixture==2?1024:1;float v=fixture?std::ldexp(1.f,-14):1.f+std::ldexp(1.f,-10);
        auto a=matrix(d,1,k,MPSDataTypeFloat16),w=matrix(d,8,k,MPSDataTypeFloat16),o=matrix(d,1,8,MPSDataTypeFloat32);
        for(size_t j=0;j<k;++j)static_cast<__fp16*>(a.data.contents)[j]=__fp16(v);
        for(size_t c=0;c<8;++c)for(size_t j=0;j<k;++j)reinterpret_cast<__fp16*>(static_cast<char*>(w.data.contents)+c*w.rowBytes)[j]=__fp16(v);
        auto p=product(d,1,8,k,1,0);auto cb=[queue commandBuffer];[p encodeToCommandBuffer:cb leftMatrix:a rightMatrix:w resultMatrix:o];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Diagnostic command failed");
        float expected=float(double(k)*double(v)*double(v)),rounded=float(k)*float(__fp16(v*v));need(expected!=rounded,"Non-distinguishing diagnostic");
        for(size_t c=0;c<8;++c)need(static_cast<float*>(o.data.contents)[c]==expected,"MPS diagnostic differs from exact float product");
        emit(@{@"kind":@"precision_diagnostic",@"fixture":@(fixture),@"inner":@(k),@"half_operand":@(v),@"expected_float":@(expected),@"actual_float":@(static_cast<float*>(o.data.contents)[0]),@"half_rounded_product_control":@(rounded),@"exact_outputs":@8,@"passed":@YES});
    }
    {
        auto a=matrix(d,1,1,MPSDataTypeFloat16),w=matrix(d,8,1,MPSDataTypeFloat16),o=matrix(d,1,8,MPSDataTypeFloat32);
        static_cast<__fp16*>(a.data.contents)[0]=__fp16(1);for(size_t c=0;c<8;++c)*reinterpret_cast<__fp16*>(static_cast<char*>(w.data.contents)+c*w.rowBytes)=__fp16(1);
        auto cb=[queue commandBuffer];for(unsigned part=0;part<3;++part){auto p=product(d,1,8,1,std::ldexp(1.,-11*int(part)),part?1:0);[p encodeToCommandBuffer:cb leftMatrix:a rightMatrix:w resultMatrix:o];}
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Triplet scale diagnostic failed");
        float expected=1.f+std::ldexp(1.f,-11)+std::ldexp(1.f,-22);for(size_t c=0;c<8;++c)need(static_cast<float*>(o.data.contents)[c]==expected,"Triplet alpha/beta lost a source plane");
        emit(@{@"kind":@"scale_diagnostic",@"alpha0":@1,@"alpha1":@(std::ldexp(1.,-11)),@"alpha2":@(std::ldexp(1.,-22)),@"expected_float":@(expected),@"actual_float":@(static_cast<float*>(o.data.contents)[0]),@"exact_outputs":@8,@"passed":@YES});
    }
    const size_t widths[][2]={{1024,3072},{896,4864},{1024,3584}};const char* names[]={"Qwen3-0.6B","Qwen2.5-Coder-0.5B-Instruct","Qwen3.5-0.8B"};
    for(size_t model=0;model<3;++model)for(size_t rows:{size_t(80),size_t(160)}){@autoreleasepool{
        size_t k=widths[model][0],n=widths[model][1];auto setup=Clock::now();
        auto a=matrix(d,rows,k,MPSDataTypeFloat32),w=matrix(d,n,k,MPSDataTypeFloat16),o=matrix(d,rows,n,MPSDataTypeFloat32);
        MPSMatrix* planes[]={matrix(d,rows,k,MPSDataTypeFloat16),matrix(d,rows,k,MPSDataTypeFloat16),matrix(d,rows,k,MPSDataTypeFloat16)};
        auto control=product(d,rows,n,k,1,0);MPSMatrixMultiplication* parts[]={product(d,rows,n,k,1,0),product(d,rows,n,k,std::ldexp(1.,-11),1),product(d,rows,n,k,std::ldexp(1.,-22),1)};
        for(size_t c=0;c<n;++c)for(size_t j=0;j<k;++j){float f=float(int(j%7)+1)/8.f*(j%2?-1.f:1.f);reinterpret_cast<__fp16*>(static_cast<char*>(w.data.contents)+c*w.rowBytes)[j]=__fp16(f*float(c%17+1)/16.f);}
        size_t reconstructed=0,validated=0,epoch=0;double worst=0,comparison=0;const float scales[]={1,-1,.5f,2};std::vector<float> baseline(rows*n);std::vector<double> exact(rows*n),bound(rows*n);
        auto inputs=[&](unsigned pattern){
            double eps=std::ldexp(1.,-23),terms=std::ceil(double(k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<rows;++r){double sum=0,absolute=0;for(size_t j=0;j<k;++j){uint32_t bits=0x3f800000u|uint32_t((j*104729+r*15485863+17)&0x7fffffu);float x;std::memcpy(&x,&bits,4);x*=scales[pattern];
                reinterpret_cast<float*>(static_cast<char*>(a.data.contents)+r*a.rowBytes)[j]=x;
                __fp16 h0=__fp16(x);float residual=x-float(h0);__fp16 h1=__fp16(std::ldexp(residual,11));residual-=std::ldexp(float(h1),-11);__fp16 h2=__fp16(std::ldexp(residual,22));
                double recovered=double(h0)+std::ldexp(double(h1),-11)+std::ldexp(double(h2),-22);float restored=float(recovered);need(std::memcmp(&x,&restored,4)==0,"Three-plane source bit reconstruction failed");++reconstructed;
                __fp16 values[]={h0,h1,h2};for(size_t p=0;p<3;++p)reinterpret_cast<__fp16*>(static_cast<char*>(planes[p].data.contents)+r*planes[p].rowBytes)[j]=values[p];
                double z=double(x)*double(float(int(j%7)+1)/8.f*(j%2?-1.f:1.f));sum+=z;absolute+=std::abs(z);
            }for(size_t c=0;c<n;++c){exact[r*n+c]=sum*double(c%17+1)/16.;bound[r*n+c]=gamma*absolute*double(c%17+1)/16.+1e-7;}}
        };
        auto run=[&](int mode){fill_output(o);auto begin=Clock::now();auto cb=[queue commandBuffer];need(cb!=nil,"Command unavailable");
            if(mode==1){for(size_t p=0;p<3;++p)[parts[p] encodeToCommandBuffer:cb leftMatrix:planes[p] rightMatrix:w resultMatrix:o];}
            else [control encodeToCommandBuffer:cb leftMatrix:a rightMatrix:w resultMatrix:o];
            auto encoded=Clock::now();[cb commit];[cb waitUntilCompleted];auto end=Clock::now();need(cb.status==MTLCommandBufferStatusCompleted,"MPS triplet command failed");
            return std::array<double,3>{ms(begin,encoded),ms(begin,end),(cb.GPUEndTime-cb.GPUStartTime)*1000};
        };
        for(unsigned pattern=0;pattern<4;++pattern){inputs(pattern);for(int mode=0;mode<3;++mode){run(mode);for(size_t r=0;r<rows;++r){auto values=reinterpret_cast<const float*>(static_cast<const char*>(o.data.contents)+r*o.rowBytes);
            for(size_t c=0;c<n;++c){float v=values[c];size_t at=r*n+c;need(std::isfinite(v),"Nonfinite product");double error=std::abs(double(v)-exact[at]);worst=std::max(worst,error/bound[at]);need(error<=bound[at]&&error<=5e-6+5e-5*std::abs(exact[at]),"Original FP64 bound exceeded");
                if(mode==0)baseline[at]=v;else comparison=std::max(comparison,std::abs(double(v)-baseline[at])/(5e-6+5e-5*std::abs(double(baseline[at]))));}
            for(size_t c=n;c<o.rowBytes/4;++c)need(values[c]==99.5f,"MPS output padding changed");
        }++validated;}}need(comparison<=1,"Original comparison bound exceeded");
        emit(@{@"kind":@"check",@"model":[NSString stringWithUTF8String:names[model]],@"rows":@(rows),@"inner":@(k),@"columns":@(n),@"patterns":@4,@"reconstructed_inputs":@(reconstructed),@"validated_executions":@(validated),@"original_fp64_worst":@(worst),@"original_comparison_worst":@(comparison),@"setup_and_checks_ms":@(ms(setup,Clock::now())),@"source_bits_exact":@YES,@"padding_preserved":@YES,@"passed":@YES});
        inputs(0);auto warm=Clock::now();do{run(int(epoch++%3));}while(ms(warm,Clock::now())<2000);
        emit(@{@"kind":@"warmup",@"model":[NSString stringWithUTF8String:names[model]],@"rows":@(rows),@"elapsed_ms":@(ms(warm,Clock::now())),@"float_input_bytes":@(a.data.length),@"half_planes_bytes":@(planes[0].data.length+planes[1].data.length+planes[2].data.length),@"half_weight_bytes":@(w.data.length),@"float_output_bytes":@(o.data.length)});
        for(int candidate:{2,1})for(int round=0;round<10;++round){int position=0;for(bool selected:{false,true,true,false}){int mode=selected?candidate:0;
            for(int i=0;i<3;++i)run(mode);auto enc=[NSMutableArray new],wall=[NSMutableArray new],gpu=[NSMutableArray new];
            for(int i=0;i<5;++i){auto t=run(mode);need(t[1]>0&&t[2]>0,"Invalid timing");[enc addObject:@(t[0])];[wall addObject:@(t[1])];[gpu addObject:@(t[2])];}
            emit(@{@"kind":@"timing",@"model":[NSString stringWithUTF8String:names[model]],@"rows":@(rows),@"candidate":@(candidate),@"round":@(round),@"position":@(position++),@"mode":@(mode),@"encode_ms":enc,@"wall_ms":wall,@"gpu_ms":gpu});
        }}
    }}
    return 0;
}catch(const std::exception& e){fprintf(stderr,"mps_half_triplet_lower_bound_failed: %s\n",e.what());return 1;}
}@catch(NSException* error){emit(@{@"kind":@"unsupported_or_exception",@"name":error.name,@"reason":error.reason?:@"",@"scope":@"No timing admitted; exception does not prove precision capability"});return 2;}}}
