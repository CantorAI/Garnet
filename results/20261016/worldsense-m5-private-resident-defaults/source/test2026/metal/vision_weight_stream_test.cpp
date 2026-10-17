#include "metal_context.h"
#include <vector>
#include <limits>
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static std::vector<float> expand(const std::vector<uint16_t>& raw){std::vector<float> result(raw.size());for(size_t i=0;i<raw.size();++i){uint32_t bits=uint32_t(raw[i])<<16;std::memcpy(&result[i],&bits,4);}return result;}
static void same(const std::vector<float>&a,const std::vector<float>&b,const char*m){need(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0,m);}
int main(){try{MetalContext gpu;
#ifndef GARNET_METAL_VISION_WEIGHT_STREAM
 bool rejected=false;gpu.Begin();try{gpu.EncodeExpandBF16Weight(MetalBF16Weight(),MetalMatrix());}catch(const std::logic_error&){rejected=true;}gpu.Abort();need(rejected,"Disabled expansion route accepted");std::cout<<"Vision weight stream disabled route correctly rejected\n";return 0;
#else
 std::vector<uint16_t> all;for(uint32_t bits=0;bits<65536;++bits)if((bits&0x7f80)!=0x7f80)all.push_back(uint16_t(bits));need(all.size()==65280,"Finite BF16 enumeration mismatch");
 auto master=gpu.UploadBF16(all.data(),all.size()*2,255,256,false,false,true);auto infos=MetalContext::WeightBufferInfo(master);need(infos.size()==1&&infos[0].lengthBytes==all.size()*2,"Raw master created extra representation");auto out=gpu.Upload(std::vector<float>(all.size(),std::numeric_limits<float>::quiet_NaN()),255,256);
 gpu.Begin();gpu.EncodeExpandBF16Weight(master,out);gpu.SubmitAndWait();same(gpu.Download(out),expand(all),"GPU expansion changed a finite BF16 bit pattern");
 size_t cases=0;
 for(size_t rows:{size_t(3),size_t(7),size_t(17)})for(size_t columns:{size_t(3),size_t(7),size_t(33),size_t(129)})for(size_t batch:{size_t(1),size_t(3)}){
  std::vector<uint16_t> a(rows*columns),b(a.size());for(size_t i=0;i<a.size();++i){uint32_t seed=uint32_t(0x9e3779b9u*(i+1));a[i]=uint16_t(((seed>>16)&0x8000u)|((120u+uint32_t(i%13))<<7)|((seed>>9)&127u));b[i]=uint16_t(((seed>>12)&0x8000u)|((121u+uint32_t(i%11))<<7)|((seed>>3)&127u));}
  auto wa=gpu.UploadBF16(a.data(),a.size()*2,rows,columns,false,false,true),wb=gpu.UploadBF16(b.data(),b.size()*2,rows,columns,false,false,true);auto fa=gpu.Upload(expand(a),rows,columns),fb=gpu.Upload(expand(b),rows,columns);std::vector<float> activations(batch*columns);for(size_t i=0;i<activations.size();++i){uint32_t seed=uint32_t(0x85ebca6bu*(i+3)),bits=(seed&0x807fffffu)|((121u+uint32_t(i%10))<<23);std::memcpy(&activations[i],&bits,4);}auto input=gpu.Upload(activations,batch,columns);
  auto ra=gpu.Allocate(batch,rows),rb=gpu.Allocate(batch,rows),oa=gpu.Allocate(batch,rows),ob=gpu.Allocate(batch,rows),scratch=gpu.Upload(std::vector<float>(rows*columns,std::numeric_limits<float>::quiet_NaN()),rows,columns);need(gpu.HazardTracked(scratch),"Expanded weight is not hazard tracked");
  gpu.Begin();gpu.EncodeMatMul(input,fa,ra,true);gpu.EncodeMatMul(input,fb,rb,true);gpu.SubmitAndWait();auto wantA=gpu.Download(ra),wantB=gpu.Download(rb);
  gpu.Begin();gpu.EncodeExpandBF16Weight(wa,scratch);gpu.EncodeMatMul(input,scratch,oa,true);gpu.EncodeExpandBF16Weight(wb,scratch);gpu.EncodeMatMul(input,scratch,ob,true);gpu.SubmitAndWait();same(gpu.Download(oa),wantA,"First queued MPS product differs from CPU-expanded control");same(gpu.Download(ob),wantB,"Reused-weight MPS product differs from CPU-expanded control");same(gpu.Download(scratch),expand(b),"Tail expansion left stale/poisoned weights");
  // The command buffer must retain a source dropped before submission completes.
  gpu.Begin();{auto temporary=gpu.UploadBF16(a.data(),a.size()*2,rows,columns,false,false,true);gpu.EncodeExpandBF16Weight(temporary,scratch);}gpu.SubmitAndWait();same(gpu.Download(scratch),expand(a),"Queued source lifetime failed");
  auto wrong=gpu.Allocate(rows+1,columns);bool rejected=false;gpu.Begin();try{gpu.EncodeExpandBF16Weight(wa,wrong);}catch(const std::invalid_argument&){rejected=true;}gpu.Abort();need(rejected,"Expansion accepted mismatched output dimensions");
  MetalContext foreign;gpu.Begin();gpu.EncodeExpandBF16Weight(wa,scratch);rejected=false;foreign.Begin();try{foreign.EncodeExpandBF16Weight(wb,scratch);}catch(const std::exception&){rejected=true;}foreign.Abort();need(rejected,"Expansion accepted another pending batch output");gpu.Abort();bool unreadable=false;try{gpu.Download(scratch);}catch(const std::logic_error&){unreadable=true;}need(unreadable,"Aborted expansion exposed completed output");gpu.Begin();gpu.EncodeExpandBF16Weight(wb,scratch);gpu.SubmitAndWait();same(gpu.Download(scratch),expand(b),"Expansion failed exact recovery after abort");++cases;
 }
 bool rejected=false;try{gpu.UploadBF16(all.data(),all.size()*2,255,256,true,false,true);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"Raw storage accepted incompatible conversion policy");
 std::cout<<"All65280finite BF16 patterns and "<<cases<<" tail/batch/queued MPS exact/owner/invalid/foreign/abort checks PASS\n";return 0;
#endif
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
