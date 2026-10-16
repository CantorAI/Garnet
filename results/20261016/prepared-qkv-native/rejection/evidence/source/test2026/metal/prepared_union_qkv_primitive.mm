// Actual private-context controls: unequal output widths, FP32 key baseline, no source weight mutation.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){@autoreleasepool{try{
    Garnet::MetalContext gpu,other;const size_t inner=8;std::vector<uint16_t> qb(13*inner,0x3e00),kb(5*inner,0x3d00),vb(3*inner,0xbd80);
    auto q=gpu.UploadBF16(qb.data(),qb.size()*2,13,inner),k=gpu.UploadBF16(kb.data(),kb.size()*2,5,inner,true),v=gpu.UploadBF16(vb.data(),vb.size()*2,3,inner);
    need(q.GlobalHalfReady()&&!k.GlobalHalfReady()&&v.GlobalHalfReady(),"Original FP32-key/half query-value fixture differs");size_t cases=0,weightBytes=0;
    for(size_t rows:{24,44,26,52,80,160}){
        need(gpu.PrepareUnionQkv(q,k,v,rows),"Heterogeneous QKV preparation failed");if(!weightBytes)weightBytes=gpu.PreparedUnionQkvWeightBytes();need(weightBytes>0&&gpu.PreparedUnionQkvWeightBytes()==weightBytes,"Extra weight payload duplicated across shapes");
        auto input=gpu.Allocate(rows,inner),qo=gpu.Allocate(rows,13),ko=gpu.Allocate(rows,5),vo=gpu.Allocate(rows,3);
        for(float value:{.25f,-.5f,1.f,2.f}){
            gpu.Update(input,std::vector<float>(rows*inner,value));gpu.Begin();gpu.EncodeBF16MatVec(input,q,qo);gpu.EncodeBF16MatVec(input,k,ko);gpu.EncodeBF16MatVec(input,v,vo);gpu.SubmitAndWait();std::array<std::vector<float>,3> expected{gpu.Download(qo),gpu.Download(ko),gpu.Download(vo)};
            gpu.Begin();need(gpu.LastPreparedUnionQkvDispatches()==0,"Begin retained QKV counter");need(gpu.EncodePreparedUnionQkv(input,q,k,v,qo,ko,vo)&&gpu.LastPreparedUnionQkvDispatches()==1,"Actual QKV encode/counter differs");
            for(const auto& output:{qo,ko,vo}){bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending QKV output readable");}
            bool rejected=false;try{gpu.DownloadPreparedUnionQkv(q,k,v,rows);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending joined QKV scratch readable");
            gpu.SubmitAndWait();std::array<std::vector<float>,3> actual{gpu.Download(qo),gpu.Download(ko),gpu.Download(vo)};
            for(size_t side=0;side<3;++side)need(actual[side].size()==expected[side].size()&&!std::memcmp(actual[side].data(),expected[side].data(),actual[side].size()*4),"Dyadic original control bits differ");
            auto joined=gpu.DownloadPreparedUnionQkv(q,k,v,rows);need(joined.size()==rows*21,"Joined QKV shape differs");for(size_t row=0;row<rows;++row)for(size_t col=0;col<21;++col)need(joined[row*21+col]==value*(col<13?1.f:(col<18?.25f:-.5f)),"Joined QKV intermediate changed");++cases;
        }
        gpu.Begin();need(gpu.EncodePreparedUnionQkv(input,q,k,v,qo,ko,vo),"Abort precondition failed");gpu.Abort();need(gpu.LastPreparedUnionQkvDispatches()==0,"Abort retained QKV counter");
        for(const auto& output:{qo,ko,vo}){bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted QKV output readable");}
        gpu.Begin();need(gpu.EncodePreparedUnionQkv(input,q,k,v,qo,ko,vo),"Recovery encode failed");gpu.SubmitAndWait();auto preserved=gpu.Download(qo);
        gpu.Begin();bool rejected=false;try{gpu.PrepareUnionQkv(q,k,v,rows);}catch(const std::logic_error&){rejected=true;}need(rejected,"QKV prepare allowed within Begin");gpu.Abort();
        auto missing=gpu.Allocate(rows,inner);gpu.Begin();rejected=false;try{gpu.EncodePreparedUnionQkv(missing,q,k,v,qo,ko,vo);}catch(const std::logic_error&){rejected=true;}need(rejected&&gpu.LastPreparedUnionQkvDispatches()==0,"Uninitialized QKV source accepted");gpu.Abort();need(gpu.Download(qo)==preserved,"Rejected encode changed old query output");
        need(other.PrepareUnionQkv(q,k,v,rows),"Other owner QKV prepare failed");gpu.Begin();need(gpu.EncodePreparedUnionQkv(input,q,k,v,qo,ko,vo),"Foreign writer precondition failed");other.Begin();rejected=false;try{other.EncodePreparedUnionQkv(input,q,k,v,qo,ko,vo);}catch(const std::logic_error&){rejected=true;}need(rejected&&other.LastPreparedUnionQkvDispatches()==0,"Foreign QKV output writer accepted");other.Abort();gpu.SubmitAndWait();
    }
    need(cases==24,"Original dyadic case count differs");need(!gpu.PrepareUnionQkv(q,k,v,2)&&!gpu.PrepareUnionQkv(q,q,v,24),"Unsupported/duplicate QKV prepared");
    auto small=gpu.Upload(std::vector<float>(2*inner,1),2,inner),sq=gpu.Allocate(2,13),sk=gpu.Allocate(2,5),sv=gpu.Allocate(2,3);gpu.Begin();need(!gpu.EncodePreparedUnionQkv(small,q,k,v,sq,sk,sv)&&gpu.LastPreparedUnionQkvDispatches()==0,"Unprepared QKV row count encoded");gpu.Abort();
    auto tiny=vb;tiny[0]=0x0200;auto bad=gpu.UploadBF16(tiny.data(),tiny.size()*2,3,inner);size_t bytesBefore=gpu.PreparedUnionQkvWeightBytes();need(!gpu.PrepareUnionQkv(q,k,bad,24)&&gpu.PreparedUnionQkvWeightBytes()==bytesBefore,"Rejected tiny value changed cache");
    std::vector<uint16_t> square(8*8,0x3e00);auto aq=other.UploadBF16(square.data(),square.size()*2,8,8),ak=other.UploadBF16(square.data(),square.size()*2,8,8,true),av=other.UploadBF16(square.data(),square.size()*2,8,8);need(other.PrepareUnionQkv(aq,ak,av,24),"Alias fixture prepare failed");
    auto a=other.Upload(std::vector<float>(24*8,1),24,8),b=other.Allocate(24,8),c=other.Allocate(24,8);
    for(bool inputAlias:{false,true}){other.Begin();bool rejected=false;try{other.EncodePreparedUnionQkv(a,aq,ak,av,inputAlias?a:b,b,c);}catch(const std::invalid_argument&){rejected=true;}need(rejected&&other.LastPreparedUnionQkvDispatches()==0,"Valid-shape QKV alias accepted");other.Abort();for(float value:other.Download(a))need(value==1,"Rejected alias changed input");}
    size_t admitted=1,rejected=0;for(size_t i=0;i<65;++i){auto wq=gpu.UploadBF16(qb.data(),qb.size()*2,13,inner),wk=gpu.UploadBF16(kb.data(),kb.size()*2,5,inner,true),wv=gpu.UploadBF16(vb.data(),vb.size()*2,3,inner);bool ok=gpu.PrepareUnionQkv(wq,wk,wv,24);need(ok==(i<63),"Bounded QKV owner cache differs");admitted+=ok;rejected+=!ok;}
    need(admitted==64&&rejected==2&&gpu.PrepareUnionQkv(q,k,v,24),"QKV strong ownership/cap differs");
    printf("Prepared QKV heterogeneous24/fp32-key/pending/ownership/alias/abort-recovery/fallback/cache64-reject2 PASS weight_bytes=%zu\n",weightBytes);return 0;
}catch(const std::exception& error){fprintf(stderr,"prepared_qkv_primitive_failed: %s\n",error.what());return 1;}}}
