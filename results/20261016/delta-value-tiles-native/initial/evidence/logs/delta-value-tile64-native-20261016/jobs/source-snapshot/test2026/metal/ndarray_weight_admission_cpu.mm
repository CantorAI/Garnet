// Independent original-codec audit of every exported non-common matrix. No Metal/GPU use.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "exact_half_codec.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static uint16_t word(const char* p,size_t i){uint16_t b;std::memcpy(&b,p+2*i,2);return b;}
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"Expected completed Python audit folder");NSString* folder=[NSString stringWithUTF8String:argv[1]];
    auto data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"report.json"]];need(data!=nil,"Missing report");
    NSDictionary* report=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];need(report!=nil,"Bad report");size_t checked=0,totalValues=0;
    for(NSDictionary* model in report[@"models"])for(NSDictionary* item in model[@"cases"]){@autoreleasepool{
        if([item[@"global_half_ready"] boolValue])continue;
        NSString* path=[[[[folder stringByAppendingPathComponent:@"assets"] stringByAppendingPathComponent:model[@"model"]] stringByAppendingPathComponent:item[@"name"]] stringByAppendingPathComponent:@"raw.bin"];
        auto raw=[NSData dataWithContentsOfFile:path options:NSDataReadingMappedAlways error:nil];
        size_t n=[item[@"rows"] unsignedLongLongValue],k=[item[@"inner"] unsignedLongLongValue];need(raw.length==n*k*2,"Raw size mismatch");const char* p=static_cast<const char*>(raw.bytes);
        unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(raw.bytes,CC_LONG(raw.length),hash);char hex[65];for(size_t i=0;i<32;++i)std::snprintf(hex+2*i,3,"%02x",hash[i]);
        need([item[@"raw_sha256"] isEqualToString:[NSString stringWithUTF8String:hex]],"Raw digest mismatch");
        uint16_t largest=0;for(size_t i=0;i<n*k;++i)largest=std::max(largest,uint16_t(word(p,i)&32767));int32_t scale=0;bool common=Garnet::ExactHalfRowScale(largest,scale);
        for(size_t i=0;i<n*k&&common;++i){uint16_t h;common=Garnet::EncodeExactScaledHalf(word(p,i),scale,h);}
        need(!common,"Export unexpectedly common-half ready");
        bool exact=true;size_t invalidRows=0,roundtrips=0;NSMutableArray* bad=[NSMutableArray array];
        for(size_t r=0;r<n;++r){uint16_t maximum=0;for(size_t j=0;j<k;++j)maximum=std::max(maximum,uint16_t(word(p,r*k+j)&32767));
            int32_t rowScale=0;bool valid=Garnet::ExactHalfRowScale(maximum,rowScale);std::vector<uint16_t> half(k);
            for(size_t j=0;j<k&&valid;++j)valid=Garnet::EncodeExactScaledHalf(word(p,r*k+j),rowScale,half[j]);
            if(!valid){exact=false;++invalidRows;[bad addObject:@{@"row":@(r),@"max_bf16_exponent":@((maximum>>7)&255),@"requested_scale":@(maximum?142-int((maximum>>7)&255):0)}];continue;}
            for(size_t j=0;j<k;++j){_Float16 h;std::memcpy(&h,&half[j],2);float v=std::ldexp(float(h),-rowScale);uint32_t b;std::memcpy(&b,&v,4);
                need(b==(uint32_t(word(p,r*k+j))<<16),"Original-codec FP32 bit roundtrip failed");++roundtrips;}
        }
        need(exact==[item[@"exact_half_ready"] boolValue],"NumPy/original-codec admission disagreement");
        need(invalidRows==[item[@"rowwise"][@"unrepresentable_rows"] unsignedLongLongValue],"Invalid-row count mismatch");
        NSDictionary* value=@{@"kind":@"matrix",@"model":model[@"model"],@"tensor":item[@"tensor"],@"global_half_ready":@(common),@"exact_half_ready":@(exact),@"bad_rows":bad,@"bitwise_roundtrip_values":@(roundtrips),@"raw_sha256":item[@"raw_sha256"]};
        auto text=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];fwrite(text.bytes,1,text.length,stdout);putchar('\n');fflush(stdout);++checked;totalValues+=roundtrips;
    }}
    printf("{\"kind\":\"complete\",\"matrices\":%zu,\"bitwise_roundtrip_values\":%zu}\n",checked,totalValues);return 0;
}catch(const std::exception& e){fprintf(stderr,"weight_admission_cpu_failed: %s\n",e.what());return 1;}}}
