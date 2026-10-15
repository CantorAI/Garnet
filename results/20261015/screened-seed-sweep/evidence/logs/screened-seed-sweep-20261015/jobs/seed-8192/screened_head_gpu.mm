// Native Metal greedy-head experiment; no production backend integration.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
struct Params {uint64_t rows,inner,columns,aStride,oStride;};
struct ArgmaxParams {uint64_t rows,columns,iStride,oStride;};
static id<MTLBuffer> load(id<MTLDevice> device,NSString* folder,NSString* file) {
    NSData* bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:file]];
    if(!bytes||!bytes.length)throw std::runtime_error("Missing input bytes");
    auto result=[device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
    if(!result)throw std::runtime_error("Buffer allocation failed");return result;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> library,NSString* name) {
    NSError* error=nil;auto result=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
    if(!result)throw std::runtime_error(error.localizedDescription.UTF8String);
    if(result.threadExecutionWidth!=32||result.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Unsupported SIMD group");return result;
}
static void finish(id<MTLCommandBuffer> command) {
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
}
int main(int argc,char** argv) {
    @autoreleasepool {try {
        if(argc!=3&&argc!=4)throw std::runtime_error("Expected shader path, assets and optional wide geometry");
        bool wide=argc==4&&std::string(argv[3])=="wide";
        if(argc==4&&!wide)throw std::runtime_error("Unknown head geometry");
        auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
        NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
        auto library=[device newLibraryWithSource:source options:nil error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto strictOptions=[MTLCompileOptions new];strictOptions.fastMathEnabled=NO;
        auto strictLibrary=[device newLibraryWithSource:source options:strictOptions error:&error];
        if(!strictLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto rawPipeline=pipeline(device,library,@"bf16_decode"),preparePipeline=pipeline(device,strictLibrary,@"prepare_head_bounds");
        auto prunePipeline=pipeline(device,strictLibrary,@"screened_head"),argmaxPipeline=pipeline(device,library,@"argmax_rows");
        NSString* folder=[NSString stringWithUTF8String:argv[2]];
        NSData* bytes=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error];
        uint64_t columns=[manifest[@"rows"] unsignedLongLongValue],inner=[manifest[@"inner"] unsignedLongLongValue];
        uint64_t seedRows=std::min<uint64_t>(8192,columns),blocks=inner/128;
        if(inner%128||blocks<1||blocks>64||columns>16777216)throw std::runtime_error("Unsupported head dimensions");
        auto weight=load(device,folder,@"raw.bin"),remaining=load(device,folder,@"remaining.bin");
        if(weight.length!=columns*inner*2||remaining.length!=columns*4*4)throw std::runtime_error("Head buffer size mismatch");
        auto approximate=load(device,folder,@"approximate.bin");if(approximate.length!=columns*inner)throw std::runtime_error("Index byte mismatch");
        auto metadata=static_cast<const float*>(remaining.contents);
        for(uint64_t row=0;row<columns;++row){
            for(int j=0;j<4;++j)if(!std::isfinite(metadata[row*4+j])||metadata[row*4+j]<0||metadata[row*4+j]>1e6f)throw std::runtime_error("Screen metadata outside admitted finite range");
            if(!(metadata[row*4+3]>0))throw std::runtime_error("Screen scale must be positive");
        }
        for(NSDictionary* query in manifest[@"queries"]) {
            @autoreleasepool {
            NSString* name=query[@"name"];uint64_t batch=[query[@"batch"] unsignedLongLongValue];
            if(batch<1||batch>2)throw std::runtime_error("Unsupported batch");
            auto a=load(device,folder,[name stringByAppendingString:@".input.bin"]);
            auto reference=load(device,folder,[name stringByAppendingString:@".reference.bin"]);
            if(a.length!=batch*inner*4||reference.length!=batch*columns*4)throw std::runtime_error("Query buffer size mismatch");
            for(uint64_t i=0;i<batch*inner;++i)if(!std::isfinite(static_cast<float*>(a.contents)[i]))throw std::runtime_error("Nonfinite input");
            auto output=[device newBufferWithLength:batch*columns*4 options:MTLResourceStorageModeShared];
            auto seed=[device newBufferWithLength:batch*seedRows*4 options:MTLResourceStorageModeShared];
            auto stats=[device newBufferWithLength:batch*(blocks+2)*4 options:MTLResourceStorageModeShared];
            auto visits=[device newBufferWithLength:columns*2*4 options:MTLResourceStorageModeShared];
            auto prediction=[device newBufferWithLength:batch*2*4 options:MTLResourceStorageModeShared];
            Params p{batch,inner,columns,inner,columns},seedParams{batch,inner,seedRows,inner,seedRows};
            ArgmaxParams ap{batch,columns,columns,2};
            auto encodeRaw=[&](id<MTLCommandBuffer> cb,const Params& params,id<MTLBuffer> target) {
                auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:rawPipeline];
                [encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:weight offset:0 atIndex:1];
                [encoder setBuffer:target offset:0 atIndex:2];[encoder setBytes:&params length:sizeof(params) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake((params.columns+15)/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
            };
            auto run=[&](bool pruned) {
                auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];
                if(pruned) {
                    encodeRaw(cb,seedParams,seed);
                    auto prep=[cb computeCommandEncoder];[prep setComputePipelineState:preparePipeline];
                    [prep setBuffer:a offset:0 atIndex:0];[prep setBuffer:seed offset:0 atIndex:1];[prep setBuffer:stats offset:0 atIndex:2];[prep setBytes:&p length:sizeof(p) atIndex:3];
                    [prep dispatchThreadgroups:MTLSizeMake(batch,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[prep endEncoding];
                    auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:prunePipeline];
                    [encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:weight offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
                    [encoder setBuffer:stats offset:0 atIndex:4];[encoder setBuffer:remaining offset:0 atIndex:5];[encoder setBuffer:visits offset:0 atIndex:6];[encoder setBuffer:seed offset:0 atIndex:7];[encoder setBuffer:approximate offset:0 atIndex:8];
                    uint64_t groupColumns=16;
                    [encoder dispatchThreadgroups:MTLSizeMake((columns+groupColumns-1)/groupColumns,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
                }else encodeRaw(cb,p,output);
                auto argmax=[cb computeCommandEncoder];[argmax setComputePipelineState:argmaxPipeline];
                [argmax setBuffer:output offset:0 atIndex:0];[argmax setBuffer:prediction offset:0 atIndex:1];[argmax setBytes:&ap length:sizeof(ap) atIndex:2];
                [argmax dispatchThreadgroups:MTLSizeMake(batch,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[argmax endEncoding];finish(cb);
                return std::pair<double,double>{(cb.GPUEndTime-cb.GPUStartTime)*1000,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
            };
            run(false);if(std::memcmp(reference.contents,output.contents,batch*columns*4))throw std::runtime_error("Raw kernel differs from actual native full logits");
            run(true);auto actual=static_cast<float*>(output.contents),expected=static_cast<float*>(reference.contents);
            auto counts=static_cast<uint32_t*>(visits.contents);
            auto maximum=static_cast<float*>(stats.contents),selected=static_cast<float*>(prediction.contents);
            uint64_t reads=seedRows*inner,prunedRows=0,retainedRows=0;
            for(uint64_t column=0;column<columns;++column) {
                reads+=uint64_t(counts[column])*128;
                for(uint64_t row=0;row<batch;++row) {
                    uint64_t index=row*columns+column;
                    if(counts[columns+column]&(1u<<row)) {
                        if(std::memcmp(actual+index,expected+index,4))throw std::runtime_error("Retained logit bits changed");++retainedRows;
                    }else {
                        if(!(expected[index]<maximum[row*(blocks+2)]))throw std::runtime_error("Pruned logit can beat the seed");++prunedRows;
                    }
                }
            }
            for(uint64_t row=0;row<batch;++row)if(selected[row*2]!=[query[@"predictions"][row] floatValue]||selected[row*2+1]!=0)throw std::runtime_error("Pruned greedy prediction differs or is invalid");
            std::printf("{\"kind\":\"check\",\"query\":\"%s\",\"batch\":%llu,\"raw_native_bits_exact\":true,\"retained_bits_exact\":true,\"greedy_exact\":true,\"pruned_rows\":%llu,\"retained_rows\":%llu,\"counted_weight_byte_fraction\":%.9g}\n",name.UTF8String,(unsigned long long)batch,(unsigned long long)prunedRows,(unsigned long long)retainedRows,double((columns-seedRows)*inner+2*reads)/double(2*columns*inner));std::fflush(stdout);
            if([name isEqualToString:@"b1-step0"]||[name isEqualToString:@"b2-step0"]){
                std::vector<float> saved(batch*inner);std::memcpy(saved.data(),a.contents,saved.size()*4);
                for(int pattern=0;pattern<(batch==2?6:5);++pattern){
                    auto input=static_cast<float*>(a.contents);
                    for(uint64_t r=0;r<batch;++r)for(uint64_t j=0;j<inner;++j){
                        float v=pattern==0?0.f:pattern==1?1e-20f:pattern==2?1e20f:pattern==3?NAN:INFINITY;
                        input[r*inner+j]=(pattern==5&&r==1)?saved[r*inner+j]:v;
                    }
                    run(false);std::vector<float> full(batch*columns),rawPrediction(batch*2);
                    std::memcpy(full.data(),output.contents,full.size()*4);std::memcpy(rawPrediction.data(),prediction.contents,rawPrediction.size()*4);
                    run(true);auto actual=static_cast<const float*>(output.contents);auto flags=static_cast<const uint32_t*>(visits.contents);
                    auto bound=static_cast<const float*>(stats.contents);auto selected=static_cast<const float*>(prediction.contents);
                    for(uint64_t r=0;r<batch;++r){bool fallback=pattern!=5||r==0;
                        if(fallback&&bound[r*(blocks+2)+2]!=0)throw std::runtime_error("Unsafe input did not disable screening");
                        if(!fallback&&bound[r*(blocks+2)+2]!=1)throw std::runtime_error("Safe mixed row unexpectedly disabled screening");
                        for(uint64_t c=0;c<columns;++c){uint64_t i=r*columns+c;bool kept=flags[columns+c]&(1u<<r);
                            if(fallback&&!kept)throw std::runtime_error("Fallback dropped a vocabulary row");
                            if(kept){if(std::isfinite(full[i])){if(std::memcmp(actual+i,full.data()+i,4))throw std::runtime_error("Finite fallback/refinement changed bits");}
                                else if(std::isnan(full[i])?!std::isnan(actual[i]):actual[i]!=full[i])throw std::runtime_error("Nonfinite fallback changed classification");
                            }else if(!(full[i]<bound[r*(blocks+2)]))throw std::runtime_error("Mixed-row rejection can beat seed");
                        }
                        if(selected[r*2+1]!=rawPrediction[r*2+1]||(selected[r*2+1]==0&&selected[r*2]!=rawPrediction[r*2]))throw std::runtime_error("Fallback greedy/error flag mismatch");
                    }
                    printf("{\"kind\":\"fallback_check\",\"batch\":%llu,\"pattern\":%d,\"all_values_checked\":true,\"greedy_and_error_flags_match\":true}\n",batch,pattern);fflush(stdout);
                }
                std::memcpy(a.contents,saved.data(),saved.size()*4);run(true);
            }
            auto commonWarm=std::chrono::steady_clock::now();size_t warmEpoch=0;
            do{run((warmEpoch++%2)!=0);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-commonWarm).count()<.25);
            for(int round=0;round<2;++round)for(bool pruned:{false,true,true,false}) {
                for(int warm=0;warm<2;++warm)run(pruned);std::vector<double> gpu,wall;
                for(int trial=0;trial<8;++trial){auto timing=run(pruned);gpu.push_back(timing.first);wall.push_back(timing.second);}
                auto rawGpu=gpu,rawWall=wall;
                printf("{\"kind\":\"samples\",\"query\":\"%s\",\"batch\":%llu,\"round\":%d,\"candidate\":%s,\"gpu_ms\":[",name.UTF8String,batch,round,pruned?"true":"false");
                for(size_t i=0;i<rawGpu.size();++i)printf("%s%.12g",i?",":"",rawGpu[i]);printf("],\"wall_ms\":[");
                for(size_t i=0;i<rawWall.size();++i)printf("%s%.12g",i?",":"",rawWall[i]);puts("]}");
                std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
                std::printf("{\"kind\":\"timing\",\"query\":\"%s\",\"batch\":%llu,\"round\":%d,\"policy\":\"%s\",\"samples\":8,\"gpu_ms_median\":%.9g,\"wall_ms_median\":%.9g}\n",name.UTF8String,(unsigned long long)batch,round,pruned?"pruned":"raw",(gpu[3]+gpu[4])*0.5,(wall[3]+wall[4])*0.5);std::fflush(stdout);
            }
            }
        }
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"pruned_head_probe_failed: %s\n",error.what());return 1;}}
}
