#include "resident_workspace_plan.h"
#include "metal_context.h"
#include <iostream>
#include <stdexcept>
#include <map>
#include <cmath>
#include <limits>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){try{using namespace Garnet;auto p=ColorWorkspace({{0,1,7,0,2},{1,1,7,1,3},{2,1,7,3,4},{3,2,7,4,5}});need(p.size()==4&&p[0].color!=p[1].color&&p[2].color==p[0].color&&p[2].color!=p[1].color,"Strict overlap/layout coloring failed");
for(auto bad:{std::vector<WorkspaceInterval>{{0,1,1,2,1}},std::vector<WorkspaceInterval>{{0,0,1,0,1}},std::vector<WorkspaceInterval>{{0,1,1,0,1},{0,2,1,2,3}}}){bool rejected=false;try{ColorWorkspace(bad);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"Invalid intervals admitted");}
for(size_t rows:{size_t(1),size_t(3)})for(size_t cols:{size_t(7),size_t(32),size_t(129)}){MetalContext gpu;auto a=gpu.Upload(std::vector<float>(rows*cols,.25f),rows,cols),b=gpu.Upload(std::vector<float>(rows*cols,.5f),rows,cols);auto x=gpu.Allocate(rows,cols),y=gpu.Allocate(rows,cols),out=gpu.Allocate(rows,cols);need(gpu.HazardTracked(x)&&gpu.HazardTracked(y)&&gpu.HazardTracked(out),"Scratch buffers are not hazard tracked");
// x is written twice with an intervening read, all in one batch: queued-pending reuse must preserve order and owners.
gpu.Begin();gpu.EncodeAdd(a,b,x);gpu.EncodeAdd(x,b,y);gpu.EncodeAdd(y,a,x);gpu.EncodeAdd(x,y,out);gpu.SubmitAndWait();for(float v:gpu.Download(out))need(v==2.75f,"Pending scratch reuse changed GPU result");gpu.Update(x,std::vector<float>(rows*cols,std::numeric_limits<float>::quiet_NaN()));auto ma=gpu.Upload(std::vector<float>(rows*3,.25f),rows,3),mb=gpu.Upload(std::vector<float>(cols*3,.5f),cols,3);gpu.Begin();gpu.EncodeMatMul(ma,mb,x,true);gpu.SubmitAndWait();for(float v:gpu.Download(x))need(v==.375f,"beta-zero matmul read stale/NaN scratch");}
for(size_t dim:{size_t(3),size_t(4)})for(float enabled:{0.f,1.f}){MetalContext gpu;const size_t heads=2,capacity=8,chunk=2,width=heads*capacity*dim;auto token=gpu.Upload(std::vector<float>(chunk*heads*dim,.5f),chunk,heads*dim),cache=gpu.Upload(std::vector<float>(width,.25f),1,width),slots=gpu.Upload({1.f,7.f},1,2),active=gpu.Upload({enabled,0.f},1,2),out=gpu.Upload(std::vector<float>(width,std::numeric_limits<float>::quiet_NaN()),1,width);gpu.Begin();gpu.EncodeKVChunk(token,cache,slots,active,out,heads,dim,capacity,chunk,false);gpu.SubmitAndWait();auto values=gpu.Download(out);for(size_t i=0;i<width;++i)need(values[i]==(enabled!=0&&((i/dim)%capacity)==1?.5f:.25f),"KV chunk failed full logical overwrite including inactive/tails");}
std::cout<<"Exact-layout strict coloring/invalid interval rejection and pending GPU scratch reuse tails PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
