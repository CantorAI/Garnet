// Private canonical output views. Download is the only adapted consumer; no native admission.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){@autoreleasepool{try{
    Garnet::MetalContext gpu,other;const size_t inner=8;
    std::vector<uint16_t> qb(13*inner,0x3e00),kb(5*inner,0x3d00),vb(3*inner,0xbd80);
    auto q=gpu.UploadBF16(qb.data(),qb.size()*2,13,inner),k=gpu.UploadBF16(kb.data(),kb.size()*2,5,inner,true),v=gpu.UploadBF16(vb.data(),vb.size()*2,3,inner);
    need(q.GlobalHalfReady()&&!k.GlobalHalfReady()&&v.GlobalHalfReady(),"Original key fixture differs");
    size_t cases=0,weightBytes=0;
    for(size_t rows:{24,44,26,52,80,160}){
        need(gpu.PrepareUnionQvViewsOriginalK(q,k,v,rows),"View preparation failed");
        auto views=gpu.PreparedUnionQvViewsOriginalKViews(q,k,v,rows);auto qo=views.first,vo=views.second;
        need(qo.SharesStorage(vo),"Canonical QV views must share storage");
        auto input=gpu.Allocate(rows,inner),bq=gpu.Allocate(rows,13),ko=gpu.Allocate(rows,5),bv=gpu.Allocate(rows,3);
        if(!weightBytes)weightBytes=gpu.PreparedUnionQvViewsOriginalKWeightBytes();need(weightBytes==256&&gpu.PreparedUnionQvViewsOriginalKWeightBytes()==weightBytes,"Weight payload duplicated across shapes");
        for(float value:{.25f,-.5f,1.f,2.f}){
            gpu.Update(input,std::vector<float>(rows*inner,value));gpu.Begin();gpu.EncodeBF16MatVec(input,q,bq);gpu.EncodeBF16MatVec(input,k,ko);gpu.EncodeBF16MatVec(input,v,bv);gpu.SubmitAndWait();
            std::array<std::vector<float>,3> expected{gpu.Download(bq),gpu.Download(ko),gpu.Download(bv)};
            gpu.Begin();need(gpu.LastPreparedUnionQvViewsOriginalKDispatches()==0,"Counter not reset");need(gpu.EncodePreparedUnionQvViewsOriginalK(input,q,k,v,qo,ko,vo)&&gpu.LastPreparedUnionQvViewsOriginalKDispatches()==1,"View encode/counter failed");
            for(auto output:{qo,ko,vo}){bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending view readable");}
            bool rejected=false;try{gpu.PreparedUnionQvViewsOriginalKGuard(q,k,v,rows);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending guard readable");
            gpu.SubmitAndWait();need(gpu.PreparedUnionQvViewsOriginalKGuard(q,k,v,rows),"Guard row overwritten");
            std::array<std::vector<float>,3> actual{gpu.Download(qo),gpu.Download(ko),gpu.Download(vo)};
            for(size_t side=0;side<3;++side)need(actual[side].size()==expected[side].size()&&!std::memcmp(actual[side].data(),expected[side].data(),actual[side].size()*sizeof(float)),"Original output bits differ");
            auto joined=gpu.DownloadPreparedUnionQvViewsOriginalK(q,k,v,rows);need(joined.size()==rows*16,"Joined shape differs");
            for(size_t row=0;row<rows;++row)for(size_t col=0;col<16;++col)need(joined[row*16+col]==value*(col<13?1.f:-.5f),"Joined output differs");++cases;
        }
        gpu.Begin();need(gpu.EncodePreparedUnionQvViewsOriginalK(input,q,k,v,qo,ko,vo),"Abort precondition failed");gpu.Abort();need(gpu.LastPreparedUnionQvViewsOriginalKDispatches()==0,"Abort retained counter");
        for(auto output:{qo,ko,vo}){bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted view readable");}
        gpu.Begin();need(gpu.EncodePreparedUnionQvViewsOriginalK(input,q,k,v,qo,ko,vo),"Recovery failed");gpu.SubmitAndWait();auto preserved=gpu.Download(qo);
        auto missing=gpu.Allocate(rows,inner);gpu.Begin();bool rejected=false;try{gpu.EncodePreparedUnionQvViewsOriginalK(missing,q,k,v,qo,ko,vo);}catch(const std::logic_error&){rejected=true;}need(rejected&&gpu.LastPreparedUnionQvViewsOriginalKDispatches()==0,"Missing input accepted");gpu.Abort();need(gpu.Download(qo)==preserved,"Rejected input changed old output");
        gpu.Begin();rejected=false;try{gpu.EncodePreparedUnionQvViewsOriginalK(input,q,k,v,bq,ko,vo);}catch(const std::invalid_argument&){rejected=true;}need(rejected&&gpu.LastPreparedUnionQvViewsOriginalKDispatches()==0,"Independent output accepted as canonical view");gpu.Abort();
        gpu.Begin();rejected=false;try{gpu.PrepareUnionQvViewsOriginalK(q,k,v,rows);}catch(const std::logic_error&){rejected=true;}need(rejected,"Preparation during Begin accepted");gpu.Abort();
        need(other.PrepareUnionQvViewsOriginalK(q,k,v,rows),"Other context preparation failed");auto ov=other.PreparedUnionQvViewsOriginalKViews(q,k,v,rows);
        gpu.Begin();need(gpu.EncodePreparedUnionQvViewsOriginalK(input,q,k,v,qo,ko,vo),"Foreign key writer precondition failed");other.Begin();rejected=false;try{other.EncodePreparedUnionQvViewsOriginalK(input,q,k,v,ov.first,ko,ov.second);}catch(const std::logic_error&){rejected=true;}need(rejected&&other.LastPreparedUnionQvViewsOriginalKDispatches()==0,"Foreign key writer accepted");other.Abort();gpu.SubmitAndWait();
    }
    need(cases==24&&!gpu.PrepareUnionQvViewsOriginalK(q,k,v,2)&&!gpu.PrepareUnionQvViewsOriginalK(q,q,v,24),"Coverage/fallback differs");
    auto tiny=vb;tiny[0]=0x0200;auto bad=gpu.UploadBF16(tiny.data(),tiny.size()*2,3,inner);need(!gpu.PrepareUnionQvViewsOriginalK(q,k,bad,24)&&gpu.PreparedUnionQvViewsOriginalKWeightBytes()==weightBytes,"Rejected group entered cache");
    size_t admitted=1,rejected=0;
    for(size_t i=0;i<65;++i){auto a=gpu.UploadBF16(qb.data(),qb.size()*2,13,inner),b=gpu.UploadBF16(kb.data(),kb.size()*2,5,inner,true),c=gpu.UploadBF16(vb.data(),vb.size()*2,3,inner);bool ok=gpu.PrepareUnionQvViewsOriginalK(a,b,c,24);need(ok==(i<63),"Bounded cache differs");admitted+=ok;rejected+=!ok;}
    need(admitted==64&&rejected==2&&gpu.PrepareUnionQvViewsOriginalK(q,k,v,24),"Strong owners/cap differs");
    printf("Prepared QV views24/original-key/canonical-offset/guard/pending/foreign/abort-recovery/fallback/cache64-reject2 PASS weight_bytes=256\n");return 0;
}catch(const std::exception& error){fprintf(stderr,"prepared_qv_views_primitive_failed: %s\n",error.what());return 1;}}}
