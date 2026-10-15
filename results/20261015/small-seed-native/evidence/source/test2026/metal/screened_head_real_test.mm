// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include "screened_head_index.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"Expected verified captured head assets");auto folder=[NSString stringWithUTF8String:argv[1]];
    auto load=[&](NSString* name){auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:name]];need(data!=nil,"Missing trained head asset");return data;};
    auto manifest=[NSJSONSerialization JSONObjectWithData:load(@"manifest.json") options:0 error:nil];need(manifest!=nil,"Invalid manifest");
    size_t n=[manifest[@"rows"] unsignedLongLongValue],k=[manifest[@"inner"] unsignedLongLongValue];auto bytes=load(@"raw.bin");need(bytes.length==n*k*2,"Wrong head bytes");
    Garnet::MetalContext gpu;auto weight=gpu.UploadBF16(bytes.bytes,bytes.length,n,k);need(gpu.PrepareScreenedHead(weight),"Actual trained head did not admit screening");size_t checks=0;
    for(NSDictionary* q in manifest[@"queries"]){@autoreleasepool{
        NSString* name=q[@"name"];size_t batch=[q[@"batch"] unsignedLongLongValue];auto inputBytes=load([name stringByAppendingString:@".input.bin"]),referenceBytes=load([name stringByAppendingString:@".reference.bin"]);
        need(inputBytes.length==batch*k*4&&referenceBytes.length==batch*n*4,"Wrong trained query shape");
        std::vector<float> input(batch*k),reference(batch*n);std::memcpy(input.data(),inputBytes.bytes,inputBytes.length);std::memcpy(reference.data(),referenceBytes.bytes,referenceBytes.length);
        auto a=gpu.Upload(input,batch,k),raw=gpu.Allocate(batch,n),control=gpu.Allocate(batch,2),prediction=gpu.Allocate(batch,2);
        gpu.Begin();gpu.EncodeBF16MatVec(a,weight,raw);gpu.EncodeArgmax(raw,control);gpu.SubmitAndWait();auto actualRaw=gpu.Download(raw),expected=gpu.Download(control);
        need(std::memcmp(actualRaw.data(),reference.data(),reference.size()*4)==0,"Integrated raw head differs from captured original GPU bits");
        gpu.Begin();need(gpu.EncodeScreenedArgmax(a,weight,prediction),"Integrated trained head did not screen");gpu.SubmitAndWait();auto selected=gpu.Download(prediction);auto d=gpu.ScreenedDiagnostics();
        need(gpu.LastScreenedHeadDispatches()==1,"Integrated actual route count differs");need(gpu.LastScreenedHeadSeedRows()==Garnet::kScreenedHeadSeedRows,"Actual seed rows differ");size_t rejected=0,retained=0;
        for(size_t row=0;row<batch;++row){need(d.querySafe[row]==1,"Captured query unexpectedly outside admitted range");
            need(selected[row*2]==[q[@"predictions"][row] floatValue]&&selected[row*2+1]==0&&selected[row*2]==expected[row*2],"Integrated trained greedy prediction differs");
            float threshold=-std::numeric_limits<float>::infinity();for(size_t col=0;col<Garnet::kScreenedHeadSeedRows;++col)threshold=std::max(threshold,reference[row*n+col]);
            for(size_t col=0;col<n;++col){size_t i=row*n+col;if(d.kept[i]){need(std::memcmp(&d.maskedLogits[i],&reference[i],4)==0,"Integrated retained bits differ");++retained;}
                else {need(reference[i]<threshold,"Integrated rejected logit can beat seed");++rejected;}
            }
        }
        printf("{\"kind\":\"integrated_head_check\",\"query\":\"%s\",\"batch\":%zu,\"values\":%zu,\"raw_bits_exact\":true,\"retained_bits_exact\":true,\"every_rejection_checked\":true,\"greedy_exact\":true,\"dispatches\":1,\"seed_rows\":%zu,\"retained\":%zu,\"rejected\":%zu}\n",name.UTF8String,batch,batch*n,gpu.LastScreenedHeadSeedRows(),retained,rejected);fflush(stdout);++checks;
    }}
    need(checks==17,"Wrong trained query coverage");return 0;
}catch(const std::exception& e){fprintf(stderr,"integrated_screened_head_failed: %s\n",e.what());return 1;}}}
