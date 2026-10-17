#include "metal_context.h"
#include <iostream>
#include <stdexcept>
#include <utility>
using namespace Garnet;
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void audit(const MetalMatrix& matrix){need(MetalContext::MatrixMetadataMatchesLive(matrix),"Cached metadata differs from live MPS properties");}
int main(){try{
    MetalContext gpu;MetalMatrix empty;
    audit(empty);need(!empty.SharesStorage(empty)&&!empty.CompatibleLayout(empty),"Empty matrix comparison changed");
    need(!gpu.IsDefaultLayout(empty,1,1)&&!MetalContext::MatrixBufferInfo(empty).identity,"Empty layout or identity changed");
    for(auto shape:{std::pair<size_t,size_t>{1,1},{2,17},{3,257},{2,2048}}){
        const auto rows=shape.first,cols=shape.second;
        auto a=gpu.Allocate(rows,cols),b=gpu.Allocate(rows,cols),different=gpu.Allocate(rows,cols+1);
        audit(a);audit(b);audit(different);
        need(a.CompatibleLayout(b)&&!a.SharesStorage(b),"Independent same-layout matrices aliased");
        need(!a.CompatibleLayout(different)&&!a.SharesStorage(different),"Different logical width accepted");
        need(gpu.IsDefaultLayout(a,rows,cols)&&!gpu.IsDefaultLayout(a,rows+1,cols)&&!gpu.IsDefaultLayout(a,rows,cols+1),"Default layout descriptor mismatch");
        auto alias=a;auto id=MetalContext::MatrixBufferInfo(a);need(id.identity&&id.allocatedBytes>=id.lengthBytes,"Invalid cached buffer lengths");
        need(alias.SharesStorage(a)&&alias.CompatibleLayout(a),"Copy lost storage identity");
        auto moved=std::move(alias);audit(moved);need(!alias.Valid()&&moved.SharesStorage(a),"Move lost immutable owner");
        gpu.Update(a,std::vector<float>(rows*cols,3.25f));audit(a);
        gpu.Begin();gpu.EncodeAdd(a,a,b);audit(a);audit(b);need(a.CompatibleLayout(b),"Pending output layout changed");gpu.SubmitAndWait();audit(b);
        for(float value:gpu.Download(b))need(value==6.5f,"Metadata changed encoded output");
        gpu.Begin();gpu.EncodeAdd(a,a,b);gpu.Abort();audit(b);gpu.Zero(b);audit(b);
        for(float value:gpu.Download(b))need(value==0,"Abort/zero recovery changed");
        a=MetalMatrix();audit(moved);need(MetalContext::MatrixBufferInfo(moved).identity==id.identity,"Live alias lost buffer");
        moved=b;a=gpu.Allocate(rows,cols);audit(moved);audit(a);need(moved.SharesStorage(b)&&!moved.SharesStorage(a),"Reassignment retained stale identity");
    }
    MetalMatrix survivor;{MetalContext other;survivor=other.Upload(std::vector<float>(34,7),2,17);audit(survivor);}
    audit(survivor);need(gpu.IsDefaultLayout(survivor,2,17),"Matrix did not survive original context");
    for(float value:gpu.Download(survivor))need(value==7,"Surviving matrix data changed");
    std::cout<<"Matrix metadata/live-MPS parity empty/copy/move/reassign/padding/pending/Submit/Abort/host-zero/context-lifetime PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
