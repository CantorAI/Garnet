"""Implement only the prospective WS77 selected-buffer ownership change."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI')
w=b/'work/worldsense-owned-storage-local';old=b/'work/worldsense-minimal-selective-defaults'
s=r/'out/research/worldsense-owned-storage-local/source';seed=r/'out/research/worldsense-minimal-selective-defaults/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'protocol.json');m=load(w/'seed-source-manifest.json')
assert p['state']=='DESIGN_BEFORE_SOURCE' and p['design_sha256']==sha(w/'native-design-before-implementation.txt')
assert not s.exists() and len(m)==1721 and all(sha(seed/k)==v for k,v in m.items())
shutil.copytree(seed,s,ignore=shutil.ignore_patterns('__pycache__','.git'));ops=[]
def change(rel,before,after,count=1,append=False):
 path=s/rel;t=path.read_text()
 if append:assert not before;new=t+after
 else:assert t.count(before)==count,(rel,t.count(before),count);new=t.replace(before,after)
 path.write_text(new);ops.append(dict(path=rel,before=before,after=after,count=count,append=append,seed_sha256=m[rel],result_sha256=sha(path)))
before='''        id<MTLBuffer> storage=w->buffer;bool primary=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        primary=w->primaryCharged;if(primary)storage=w->primaryPayload;
#endif
'''
after='''#ifdef GARNET_METAL_OWNED_STORAGE_LOCAL
        // w owns immutable strong buffer fields throughout this scope.
        // Ordinary command buffers retain bound resources after Encode returns.
        bool primary=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        primary=w->primaryCharged;
        id<MTLBuffer> storage __attribute__((objc_externally_retained))=primary?w->primaryPayload:w->buffer;
#else
        id<MTLBuffer> storage __attribute__((objc_externally_retained))=w->buffer;
#endif
#else
'''+before+'''#endif
'''
change('src/backends/metal/metal_context.mm',before,after,count=2)
guard='''
#if defined(GARNET_METAL_OWNED_STORAGE_LOCAL) && !__has_attribute(objc_externally_retained)
#error "Owned storage locals require Clang objc_externally_retained support"
#endif
'''
change('src/backends/metal/metal_context.mm','',guard,append=True)
change('src/entry/metal_main.cpp','        return info;\n','''#ifdef GARNET_METAL_OWNED_STORAGE_LOCAL
        info+="; owned_storage_local";
#endif
        return info;
''')
change('cmake/GarnetMetal.cmake','','''
option(GARNET_METAL_OWNED_STORAGE_LOCAL "Omit redundant selected-buffer local ARC operations while weight owns storage" OFF)
if(GARNET_METAL_OWNED_STORAGE_LOCAL)
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_OWNED_STORAGE_LOCAL=1)
endif()
add_executable(garnet_owned_storage_local_test test2026/metal/owned_storage_local_test.cpp)
target_link_libraries(garnet_owned_storage_local_test PRIVATE garnet_metal)
target_include_directories(garnet_owned_storage_local_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
add_test(NAME metal_owned_storage_local_lifetime COMMAND garnet_owned_storage_local_test owned_storage_projection_words.bin)
''',append=True)
native=r'''#include "metal_context.h"
#include "nlohmann/json.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static auto live(){auto v=MetalContext::PrimaryStats();return std::array<uint64_t,5>{v.activeWeights,v.originalBytes,v.payloadBytes,v.metadataBytes,v.allocatedBytes};}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: owned_storage_local_test words.bin");MetalContext gpu;
 std::ofstream stream(argv[1],std::ios::binary);need(bool(stream),"Cannot open complete lifetime words");
 const auto baseline=live();size_t records=0,totalWords=0,rawCases=0,packedCases=0,rejections=0,preparedCases=0;
 auto run=[&](size_t rows,size_t columns,bool forceRaw,size_t mode){
  constexpr size_t inner=128;std::vector<uint16_t> bits(columns*inner,0x3f80);
  std::vector<float> input(rows*inner);for(size_t row=0;row<rows;++row)for(size_t k=0;k<inner;++k)input[row*inner+k]=float(row+1)/16;
  auto a=gpu.Upload(input,rows,inner),out=gpu.Allocate(rows,columns);
  gpu.Update(out,std::vector<float>(rows*columns,std::numeric_limits<float>::quiet_NaN()));
  auto make=[&](){return gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,!forceRaw);};
  MetalBF16Weight original;
  if(mode==2){MetalContext creator;original=creator.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,!forceRaw);}
  else original=make();
  auto alias=original;original=MetalBF16Weight();need(alias.Valid(),"Copy lost storage owner");
  bool packed=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
  packed=!forceRaw;
#endif
  need(alias.PrimaryReady()==packed,"Lifetime case representation differs");
  if(packed)++packedCases;else ++rawCases;
  const bool expectedPrepare=rows<=2&&columns<100000;
  need(gpu.PrepareBF16BodyShape(a,alias,out)==expectedPrepare,"Alias/creator-close prepare changed");preparedCases+=expectedPrepare;
  if(mode==3){
   auto badout=gpu.Allocate(rows,columns+1);need(!gpu.PrepareBF16BodyShape(a,alias,badout),"Invalid output prepared");
   gpu.Begin();bool caught=false;try{gpu.PrepareBF16BodyShape(a,alias,out);}catch(const std::exception&){caught=true;}
   need(caught,"Active batch prepare did not reject");++rejections;caught=false;
   try{gpu.EncodeBF16MatVec(a,alias,badout);}catch(const std::exception&){caught=true;}
   need(caught,"Invalid shape encoding accepted");++rejections;alias=MetalBF16Weight();gpu.Abort();
   need(live()==baseline,"Exception/Abort retained weight owner");alias=make();
  }
  gpu.Begin();gpu.EncodeBF16MatVec(a,alias,out);
  if(mode==1||mode==4)gpu.Flush();
  alias=MetalBF16Weight();need(live()==baseline,"Last host weight owner was not released");
  if(mode==4){
   gpu.Abort();bool caught=false;try{gpu.Download(out);}catch(const std::exception&){caught=true;}
   need(caught,"Flushed aborted output readable");++rejections;
   alias=make();gpu.Begin();gpu.EncodeBF16MatVec(a,alias,out);alias=MetalBF16Weight();
  }
  gpu.SubmitAndWait();auto actual=gpu.Download(out);need(actual.size()==rows*columns,"Lifetime output truncated");
  for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col)need(actual[row*columns+col]==float(inner*(row+1))/16,"Queued released-owner FP32 word changed");
  need(gpu.Download(a)==input,"Input mutated");need(live()==baseline,"Completed case retained logical weight");
  const uint64_t header[]={rows,inner,columns,uint64_t(forceRaw),mode,actual.size()};
  stream.write(reinterpret_cast<const char*>(header),sizeof(header));stream.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);
  need(bool(stream),"Cannot write complete lifetime output");++records;totalWords+=actual.size();
 };
 for(bool raw:{false,true})for(size_t mode=0;mode<5;++mode){
  for(size_t rows:{1,2,16,32,64})for(size_t columns:{260,1024})run(rows,columns,raw,mode);
  for(size_t rows:{1,2})run(rows,100352,raw,mode);
 }
 need(records==120&&rawCases+packedCases==120&&rejections==72&&preparedCases==40,"Lifetime coverage incomplete");
 stream.flush();need(bool(stream)&&live()==baseline,"Final output/accounting invalid");
#ifdef GARNET_METAL_OWNED_STORAGE_LOCAL
 constexpr bool enabled=true;
#else
 constexpr bool enabled=false;
#endif
 json report={{"owned_storage_local_enabled",enabled},{"complete_records",records},{"complete_FP32_words",totalWords},{"raw_cases",rawCases},{"packed_cases",packedCases},{"invalid_cases_rejected",rejections},{"prepared_cases",preparedCases},{"live_logical_accounting_restored",true},{"original_alias_release_before_prepare",true},{"last_owner_release_before_commit",true},{"last_owner_release_after_flush",true},{"creator_context_closed_before_use",true},{"flushed_abort_owner_release_and_recovery",true},{"no_retainCount_or_timing_claim",true}};
 std::ofstream summary(std::string(argv[1])+".summary.json");summary<<report.dump(2);need(bool(summary),"Cannot write lifetime summary");
 std::cout<<"Selected storage alias/queued/Flush/Abort/creator-close lifetime PASS "<<report.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
'''
(s/'test2026/metal/owned_storage_local_test.cpp').write_text(native)
actual={str(p.relative_to(s)):sha(p)for p in s.rglob('*')if p.is_file()}
expected=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/owned_storage_local_test.cpp']
assert len(actual)==1722 and sorted(k for k in actual if actual[k]!=m.get(k))==expected
for rel in {x['path']for x in ops}:
 text=(s/rel).read_text()
 for op in reversed([x for x in ops if x['path']==rel]):
  if op['append']:assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert hashlib.sha256(text.encode()).hexdigest()==m[rel]
(w/'correctness-source-manifest.json').write_text(json.dumps(actual,indent=2)+'\n')
(w/'source-transformations.json').write_text(json.dumps(ops,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1721,source_files=1722,changed_or_added=expected,new_files=[expected[-1]],all_existing_changes_inverse_exact_to74=True,all_other1718files_byte_exact=True,shader_math_codec_weights_graph_session_CUDA_engine_original_tests_unchanged=True),indent=2)+'\n')
shutil.copytree(old/'build-profile',w/'build-profile')
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','primary_core_isolation_test.cpp']:shutil.copy2(old/name,w/name)
t=(old/'build.py').read_text().replace('worldsense-minimal-selective-defaults','worldsense-owned-storage-local').replace('len(manifest)==1721','len(manifest)==1722')
t=t.replace("roles=['off','primary','on','fault','compat56','compat56-fault']","roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']")
t=t.replace("primary='OFF'if role=='off'else'ON'","profile={'control':'on','raw-borrow':'off'}.get(role,role);primary='OFF'if profile=='off'else'ON'")
t=t.replace("compat=role.startswith('compat56')","compat=profile.startswith('compat56')")
t=t.replace("role in ['on','fault']","profile in ['on','fault']").replace("role in ['primary','on','fault']","profile in ['primary','on','fault']")
anchor=" flags['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']='OFF'";assert t.count(anchor)==1
t=t.replace(anchor,anchor+"\n flags['GARNET_METAL_OWNED_STORAGE_LOCAL']='ON'if role in ['primary','on','fault','raw-borrow']else'OFF'")
ast.parse(t);(w/'build.py').write_text(t)
d=load(b/'work/worldsense-next.json');d.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='SOURCE_READY_BUILD_NEXT',source_implemented=True,source_files=1722,inference_source_delta=4)
(b/'work/worldsense-next.json').write_text(json.dumps(d,indent=2)+'\n')
print('WS77 source1722/four-path delta; source inverse exact, shaders/CUDA/engine/original tests unchanged. Eight-role build and complete lifetime test ready; no build/model yet.')
