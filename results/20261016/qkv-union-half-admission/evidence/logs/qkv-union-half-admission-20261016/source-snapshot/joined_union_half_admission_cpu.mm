// Independent original-codec full-value check; Foundation/CommonCrypto only, no GPU.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "exact_half_codec.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
static void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static std::string sha(const void* bytes,size_t length){unsigned char result[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes,CC_LONG(length),result);std::ostringstream s;for(auto v:result)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v);return s.str();}
static void emit(NSDictionary* object){auto data=[NSJSONSerialization dataWithJSONObject:object options:0 error:nil];need(data!=nil,"JSON failed");fwrite(data.bytes,1,data.length,stdout);putchar('\n');fflush(stdout);}
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"Expected captured CPU union report");auto raw=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];need(raw!=nil,"Missing report");auto report=[NSJSONSerialization JSONObjectWithData:raw options:0 error:nil];need(report!=nil,"Report parse failed");size_t pairs=0,matrices=0,restored=0,admittedPairs=0;
    for(NSDictionary* model in report[@"models"]){@autoreleasepool{
        NSError* error=nil;auto file=[NSData dataWithContentsOfFile:model[@"checkpoint"] options:NSDataReadingMappedIfSafe error:&error];need(file!=nil,"Checkpoint unavailable");
        for(NSDictionary* pair in model[@"pairs"]){int32_t scale=[pair[@"union_scale"] intValue];size_t totalBad=0,totalRestored=0;uint16_t largest=0;NSMutableArray* sides=[NSMutableArray new];
            for(size_t side=0;side<2;++side){NSDictionary* fixture=pair[@"sides"][side];size_t count=[fixture[@"original_bytes"] unsignedLongLongValue]/2,offset=[fixture[@"data_offset"] unsignedLongLongValue];need(offset<=file.length&&count<=(file.length-offset)/2,"Checkpoint slice bounds failed");
                const auto* source=static_cast<const unsigned char*>(file.bytes)+offset;need(sha(source,count*2)==std::string([fixture[@"raw_sha256"] UTF8String]),"Learned weight hash changed");size_t bad=0,good=0;
                for(size_t i=0;i<count;++i){uint16_t bits,encoded;std::memcpy(&bits,source+2*i,2);largest=std::max(largest,uint16_t(bits&32767));
                    if(!Garnet::EncodeExactScaledHalf(bits,scale,encoded)){++bad;continue;}
                    __fp16 h;std::memcpy(&h,&encoded,2);float value=std::ldexp(float(h),-scale);uint32_t actual;std::memcpy(&actual,&value,4);need(actual==(uint32_t(bits)<<16),"Original codec BF16 roundtrip changed bits");++good;
                }
                NSDictionary* expected=pair[@"checks"][side];need(bad==[expected[@"unrepresentable_values"] unsignedLongLongValue]&&(bad==0)==[expected[@"admitted"] boolValue],"NumPy/original-codec eligibility differs");
                [sides addObject:@{@"tensor":fixture[@"tensor"],@"raw_sha256":fixture[@"raw_sha256"],@"encoded_roundtrip_values":@(good),@"rejected_values":@(bad)}];totalBad+=bad;totalRestored+=good;++matrices;
            }
            int32_t requested=0;bool scaleOk=Garnet::ExactHalfRowScale(largest,requested);need(requested==scale&&[pair[@"union_admitted"] boolValue]==(scaleOk&&totalBad==0),"Union max/scale admission differs");
            ++pairs;restored+=totalRestored;admittedPairs+=totalBad==0;
            emit(@{@"kind":@"pair",@"model":model[@"model"],@"layer":pair[@"layer"],@"union_scale":@(scale),@"admitted":@(totalBad==0),@"sides":sides,@"bitwise_roundtrip_values":@(totalRestored)});
        }
    }}need(pairs==76&&matrices==152,"Full-layer coverage differs");emit(@{@"kind":@"complete",@"pairs":@(pairs),@"matrices":@(matrices),@"admitted_pairs":@(admittedPairs),@"bitwise_roundtrip_values":@(restored),@"scope":@"Original CPU codec only; no native allocation/GPU/inference timing"});return 0;
}catch(const std::exception& error){fprintf(stderr,"joined_union_cpu_failed: %s\n",error.what());return 1;}}}
