from pathlib import Path
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-gpu/source');old=Path('/Users/shawn/CantorAI/out/research/worldsense-request-residency/source/test2026/metal/dense_decode1024_test.cpp')
t=old.read_text().replace('using namespace Garnet;','using namespace Garnet;\nstatic uint64_t Encoded(){auto s=MetalContext::PrimaryStats();return s.decodeEncoded+s.headEncoded+s.prefillEncoded;}')
t=t.replace('prepared==(columns<100000)','prepared==(rows<=2&&columns<100000)')
a=t.index('  bool eligible=false;');b=t.index('  std::vector<float> saved;',a)
t=t[:a]+'''  bool eligible=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
  eligible=columns>256&&columns%4==0&&inner%128==0;
#endif
  require(w.PrimaryReady()==eligible,"Primary admission mismatch");
'''+t[b:]
t=t.replace('gpu.DenseDecode1024EncodedTotal()','Encoded()')
a=t.index(' require(cases==77');b=t.index(' if(binary.is_open())',a)
t=t[:a]+''' for(size_t rows:{3,16,32,64})for(size_t inner:{2048,2560})for(size_t columns:{272,768})run(rows,inner,columns);
 require(cases==93&&passes==186,"Case coverage mismatch");
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 require(widePasses>0,"Primary route not executed");
#else
 require(widePasses==0,"Disabled policy routed primary");
#endif
'''+t[b:]
t=t.replace('Dense1024 numerical/reference','Primary numerical/reference')
(s/'test2026/metal/primary_projection_test.cpp').write_text(t)
# The GPU bit decoder checks every finite BF16 encoding and compact/raw mixtures.
(s/'test2026/metal/primary_storage_test.cpp').write_text(r'''#include "metal_context.h"
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
using namespace Garnet;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){try{
 MetalContext gpu;auto initial=MetalContext::PrimaryStats();
 {
 std::vector<uint16_t> bits;for(uint32_t i=0;i<65536;++i)if(((i>>7)&255)!=255)bits.push_back(uint16_t(i));bits.resize(512*128,0x8000);
 auto w=gpu.UploadBF16(bits.data(),bits.size()*2,512,128);auto shared=w;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(w.PrimaryReady(),"Finite-pattern matrix not packed");
 auto now=MetalContext::PrimaryStats();need(now.activeWeights==initial.activeWeights+1&&now.originalBytes==initial.originalBytes+bits.size()*2,"Owner accounting mismatch");
 auto info=MetalContext::WeightBufferInfo(w);need(info.size()==2&&info[0].representation=="primary_bf16_payload"&&info[1].representation=="primary_bf16_metadata","Packed weight retained a raw duplicate");
 auto actual=gpu.PrimaryReadWords(w,0,bits.size());for(size_t i=0;i<bits.size();++i)need(actual[i]==uint32_t(bits[i])<<16,"Finite GPU bits mismatch");
 w=MetalBF16Weight();need(MetalContext::PrimaryStats().activeWeights==now.activeWeights,"Shared owner was freed");
 need(gpu.PrimaryReadWords(shared,7,100)[0]==uint32_t(bits[7])<<16,"Nonzero read offset mismatch");
 bool rejected=false;try{gpu.PrimaryReadWords(shared,bits.size()-1,2);}catch(...){rejected=true;}need(rejected,"Invalid read range accepted");
#else
 need(!w.PrimaryReady(),"Disabled packed admission");
#endif
 }
 need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights&&MetalContext::PrimaryStats().allocatedBytes==initial.allocatedBytes,"Released owners still charged");
 // Alternating wide-exponent/raw and compact blocks, plus unsupported dimensions.
 std::vector<uint16_t> mix(260*256);
 for(size_t r=0;r<260;++r)for(size_t k=0;k<256;++k){const size_t block=(r/4)*2+k/128;const unsigned e=block%4==0?(k%2?96:130):120+unsigned(k%4);mix[r*256+k]=uint16_t((e<<7)|((r*17+k*5)&127)|((k&1)<<15));}
 {
 auto w=gpu.UploadBF16(mix.data(),mix.size()*2,260,256);
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(w.PrimaryReady(),"Mixed matrix not packed");auto actual=gpu.PrimaryReadWords(w,0,mix.size());for(size_t i=0;i<mix.size();++i)need(actual[i]==uint32_t(mix[i])<<16,"Mixed raw/compact GPU bits mismatch");
#endif
 }
 std::vector<uint16_t> raw(257*129,0x3f80);auto fallback=gpu.UploadBF16(raw.data(),raw.size()*2,257,129);need(!fallback.PrimaryReady(),"Unsupported dimensions packed");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
 for(int stage:{1,2}){auto before=MetalContext::PrimaryStats();MetalContext::PrimaryTestFailAfter(stage);bool rejected=false;try{gpu.UploadBF16(mix.data(),mix.size()*2,260,256);}catch(...){rejected=true;}auto after=MetalContext::PrimaryStats();need(rejected&&after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Partial primary allocation failed cleanup");}
 auto recovered=gpu.UploadBF16(mix.data(),mix.size()*2,260,256);need(recovered.PrimaryReady(),"Primary allocation did not recover");
#endif
 std::cout<<"Primary storage finite65536-filtered/mixed/address/owner/dedup/no-master/fallback/allocation-recovery PASS
";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'
';return 1;}}
''')
p=s/'cmake/GarnetMetal.cmake';t=p.read_text()+'''
add_executable(garnet_primary_storage_test test2026/metal/primary_storage_test.cpp)
target_link_libraries(garnet_primary_storage_test PRIVATE garnet_metal)
add_test(NAME metal_primary_storage_reference COMMAND garnet_primary_storage_test)
add_executable(garnet_primary_projection_test test2026/metal/primary_projection_test.cpp)
target_link_libraries(garnet_primary_projection_test PRIVATE garnet_metal)
add_test(NAME metal_primary_projection_reference COMMAND garnet_primary_projection_test ${CMAKE_BINARY_DIR}/primary_projection_words.bin)
''';p.write_text(t)
# Model class is unchanged: wrapper telemetry/accounting only.
import ast,json,hashlib
a=ast.parse((s/'runtime/metal_serving.py').read_text());b=ast.parse(Path('/Users/shawn/CantorAI/out/research/worldsense-cached-matrix-metadata/source/runtime/metal_serving.py').read_text())
model=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Model')
assert ast.dump(model(a),include_attributes=False)==ast.dump(model(b),include_attributes=False)
(w/'model-body-unchanged.json').write_text(json.dumps({'entire_Model_class_AST_identical':True,'wrapper_stats_inclusive_TTFT_total_only':True},indent=2))
print('Prepared native storage/GPUbit and93case186FP32word+FP64 controls; no GPU execution yet')
