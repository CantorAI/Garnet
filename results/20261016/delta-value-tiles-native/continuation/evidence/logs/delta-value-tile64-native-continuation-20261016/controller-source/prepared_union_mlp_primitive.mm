// Private prototype controls: exact original weights, explicit prepare, activation only.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(){@autoreleasepool{try{
    Garnet::MetalContext gpu,other;const size_t k=8,n=16;size_t cases=0;std::vector<uint16_t> gb(n*k,0x3e00),ub(n*k,0x3d00);
    auto gate=gpu.UploadBF16(gb.data(),gb.size()*2,n,k),up=gpu.UploadBF16(ub.data(),ub.size()*2,n,k);
    need(gate.GlobalHalfReady()&&up.GlobalHalfReady(),"Original common-half fixture unavailable");size_t weightBytes=0;
    for(size_t rows:{24,44,26,52,80,160}){
        need(gpu.PrepareUnionMlp(gate,up,rows),"Unequal-scale union not prepared");
        if(!weightBytes)weightBytes=gpu.PreparedUnionWeightBytes();need(gpu.PreparedUnionWeightBytes()==weightBytes&&weightBytes>0,"Shape change duplicated prepared weight payload");
        auto x=gpu.Allocate(rows,k),g=gpu.Allocate(rows,n),u=gpu.Allocate(rows,n),out=gpu.Allocate(rows,n);
        for(float value:{.25f,-.5f,1.f,2.f}){
            gpu.Update(x,std::vector<float>(rows*k,value));gpu.Begin();gpu.EncodeBF16MatVec(x,gate,g);gpu.EncodeBF16MatVec(x,up,u);gpu.EncodeSiluMul(g,u,out);gpu.SubmitAndWait();auto expected=gpu.Download(out);
            gpu.Begin();need(gpu.LastPreparedUnionMlpDispatches()==0,"Begin retained counter");need(gpu.EncodePreparedUnionMlp(x,gate,up,out),"Prepared union encode failed");need(gpu.LastPreparedUnionMlpDispatches()==1,"Actual union count differs");
            bool rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending activation readable");
            rejected=false;try{gpu.DownloadPreparedUnion(gate,up,rows);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending joined scratch readable");
            gpu.SubmitAndWait();auto actual=gpu.Download(out);need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"Dyadic control activation bits differ");
            auto joined=gpu.DownloadPreparedUnion(gate,up,rows);need(joined.size()==rows*n*2,"Joined diagnostic shape differs");
            for(size_t row=0;row<rows;++row)for(size_t col=0;col<n;++col){need(joined[row*2*n+col]==value,"Gate product changed");need(joined[row*2*n+n+col]==value*.25f,"Up product changed");}
            ++cases;
        }
        gpu.Begin();need(gpu.EncodePreparedUnionMlp(x,gate,up,out),"Abort precondition failed");gpu.Abort();need(gpu.LastPreparedUnionMlpDispatches()==0,"Abort retained counter");
        bool rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted activation readable");
        gpu.Begin();need(gpu.EncodePreparedUnionMlp(x,gate,up,out),"Recovery encode failed");gpu.SubmitAndWait();need(gpu.Download(out).size()==rows*n,"Recovery output unavailable");
        gpu.Begin();rejected=false;try{gpu.PrepareUnionMlp(gate,up,rows);}catch(const std::logic_error&){rejected=true;}need(rejected,"Preparation allowed inside active batch");gpu.Abort();
        auto missing=gpu.Allocate(rows,k);auto preserved=gpu.Download(out);gpu.Begin();rejected=false;try{gpu.EncodePreparedUnionMlp(missing,gate,up,out);}catch(const std::logic_error&){rejected=true;}need(rejected&&gpu.LastPreparedUnionMlpDispatches()==0,"Uninitialized source accepted");gpu.Abort();need(gpu.Download(out)==preserved,"Rejected encode changed old output");
        need(other.PrepareUnionMlp(gate,up,rows),"Second owner preparation failed");gpu.Begin();need(gpu.EncodePreparedUnionMlp(x,gate,up,out),"Foreign ownership precondition failed");
        other.Begin();rejected=false;try{other.EncodePreparedUnionMlp(x,gate,up,out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Foreign pending writer accepted");other.Abort();gpu.SubmitAndWait();
        gpu.Begin();rejected=false;try{gpu.EncodePreparedUnionMlp(x,gate,up,x);}catch(const std::invalid_argument&){rejected=true;}need(rejected&&gpu.LastPreparedUnionMlpDispatches()==0,"Aliased/invalid output accepted");gpu.Abort();
    }
    need(cases==24,"Original control case count differs");
    auto squareGate=other.UploadBF16(gb.data(),8*8*2,8,8),squareUp=other.UploadBF16(ub.data(),8*8*2,8,8);
    need(other.PrepareUnionMlp(squareGate,squareUp,24),"Valid alias fixture not prepared");auto alias=other.Upload(std::vector<float>(24*8,1),24,8);
    other.Begin();bool aliasRejected=false;try{other.EncodePreparedUnionMlp(alias,squareGate,squareUp,alias);}catch(const std::invalid_argument&){aliasRejected=true;}need(aliasRejected&&other.LastPreparedUnionMlpDispatches()==0,"Valid-shape input/output alias accepted");other.Abort();
    for(float value:other.Download(alias))need(value==1.f,"Rejected alias modified input");
    need(!gpu.PrepareUnionMlp(gate,up,2)&&!gpu.PrepareUnionMlp(gate,gate,24),"Unsupported profile admitted");
    auto small=gpu.Upload(std::vector<float>(2*k,1),2,k),o=gpu.Allocate(2,n);gpu.Begin();need(!gpu.EncodePreparedUnionMlp(small,gate,up,o),"Unprepared shape encoded");need(gpu.LastPreparedUnionMlpDispatches()==0,"Fallback counter changed");gpu.Abort();
    auto tiny=gb;tiny[0]=0x0200;auto bad=gpu.UploadBF16(tiny.data(),tiny.size()*2,n,k);auto before=gpu.PreparedUnionWeightBytes();need(!gpu.PrepareUnionMlp(bad,up,24)&&gpu.PreparedUnionWeightBytes()==before,"Nonexact union mutated prepared cache");
    size_t admitted=1,rejected=0;for(size_t i=0;i<129;++i){auto g=gpu.UploadBF16(gb.data(),gb.size()*2,n,k),u=gpu.UploadBF16(ub.data(),ub.size()*2,n,k);bool selected=gpu.PrepareUnionMlp(g,u,24);need(selected==(i<127),"Bounded owner cache differs");admitted+=selected;rejected+=!selected;}
    need(admitted==128&&rejected==2&&gpu.PrepareUnionMlp(gate,up,24),"Cached owners lost after cap");
    printf("Prepared union native unequal-scale/dyadic24/pending/ownership/alias/abort-recovery/fallback/cache128-reject2 PASS weight_bytes=%zu\n",weightBytes);return 0;
}catch(const std::exception& error){fprintf(stderr,"prepared_union_primitive_failed: %s\n",error.what());return 1;}}}
