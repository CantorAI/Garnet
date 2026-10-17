from pathlib import Path
import ast,datetime,hashlib,json,re,shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI')
old=r/'out/research/worldsense-primary-kmajor/source';s=r/'out/research/worldsense-primary-u32/source'
assert not s.exists()
assert json.loads((w/'design-identity.json').read_text())['design_sha256']==hashlib.sha256((w/'native-design-before-implementation.txt').read_bytes()).hexdigest()
seed=json.loads((w.parent/'worldsense-primary-kmajor/starting-source-manifest.json').read_text())
assert all(hashlib.sha256((old/rel).read_bytes()).hexdigest()==h for rel,h in seed.items())
shutil.copytree(old,s,ignore=shutil.ignore_patterns('__pycache__','.git','results'))
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-40','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':'Frozen WS39 plus checked uint32 metadata/addressing; all398 target matrices preserved. Larger valid tensors use original raw fallback.','hypothesis':'Narrower checked integer arithmetic and halved metadata may reduce decode overhead; no speed prediction.','gates':'Original native/fault/dualGPUcorpus/full32VLM28text gates; NEW8resource10%Metal5%footprint; NEW20speedbothmodels2%TTFTtotaldecodeSkillwall preservation; NEW48fulltwoindependentpairs beforeintegration.','fixed':'Payload bytes,FP32math,Korder,128threads,ownership,profiles,budgets,Model/wrapper/counters andCUDA.','accounting':'Native C++/Metal/XLang3 inference/images. All first/warm/failures,source/build/model/SDK/host/input identities. No competingGPU/build/archivecopyduringtiming;restoreWS10normalfourfiles/noCam/404.'},indent=2))

(s/'src/backends/metal/primary_32_contract.h').write_text(r'''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace Primary32 {
constexpr uint64_t MaxRawBytes=uint64_t{1}<<31;
inline bool ShapeFits(uint64_t rows,uint64_t inner,uint64_t bytes) {
    if(rows<=256||!inner||rows%4||inner%128||rows>UINT32_MAX||inner>UINT32_MAX)return false;
    // Divide before multiplying: even invalid caller dimensions cannot overflow.
    if(rows>(MaxRawBytes/2)/inner)return false;
    return bytes==rows*inner*2;
}
inline uint32_t NarrowMetadata(uint64_t value) {
    if(value>UINT32_MAX)throw std::overflow_error("Primary metadata exceeds uint32 range");
    return static_cast<uint32_t>(value);
}
}
''')
p=s/'src/backends/metal/metal_context.mm';t=p.read_text()
t=t.replace('#include "primary_block_codec.h"','#include "primary_block_codec.h"\n#include "primary_32_contract.h"',1)
needle='if(!floatPrefill&&!visionPrefill&&rows>256&&PrimaryBlocks::Eligible(rows,columns))';assert t.count(needle)==1;t=t.replace(needle,'if(!floatPrefill&&!visionPrefill&&Primary32::ShapeFits(rows,columns,bytes))',1)
for a,b in [('if(count>SIZE_MAX/8)','if(count>SIZE_MAX/4)'),('std::vector<uint64_t> metadata;metadata.reserve(count);','std::vector<uint32_t> metadata;metadata.reserve(count);'),('metadata.push_back(encoded.metadata);','metadata.push_back(Primary32::NarrowMetadata(encoded.metadata));'),('const auto metaBytes=metadata.size()*sizeof(uint64_t);','const auto metaBytes=metadata.size()*sizeof(uint32_t);')]:
    assert t.count(a)==1,a;t=t.replace(a,b,1)
p.write_text(t)

p=s/'src/backends/metal/primary_block_shader.h';before=p.read_text();t=before
substitutions=[('device const ulong* metadata','device const uint* metadata'),('ulong row,ulong k,ulong inner','uint row,uint k,uint inner'),('ulong block=(row/4)','uint block=(row/4)'),('ulong meta=metadata[','uint meta=metadata['),('meta&256ul','meta&256u')]
audit=[]
for a,b in substitutions:
    n=t.count(a);assert n,a;t=t.replace(a,b);audit.append({'old':a,'new':b,'count':n})
recovered=t
for a,b in reversed(substitutions):
    # No uint metadata existed in the seed primary shader.
    recovered=recovered.replace(b,a)
assert recovered==before
assert 'ulong row,ulong k,ulong inner' not in t and 'ulong meta=metadata[' not in t
p.write_text(t)
(w/'shader-width-substitution-audit.json').write_text(json.dumps({'substitutions':audit,'inverse_recovers_entire_WS39_primary_shader_exactly':True,'scope':'Only checked metadata pointer and helper integer widths. All kernel FP32 arithmetic/dispatch/control bodies unchanged.'},indent=2))
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='; primary_kmajor_decode";';assert t.count(needle)==1;p.write_text(t.replace(needle,'; primary_kmajor_decode; primary_u32_metadata";',1))

p=s/'test2026/metal/inventory_test.cpp';t=p.read_text();needle='w[1].lengthBytes==1024*1024/512*8';assert needle in t;p.write_text(t.replace(needle,'w[1].lengthBytes==1024*1024/512*4',1))
p=s/'test2026/metal/primary_storage_test.cpp';t=p.read_text();t=t.replace('#include "metal_context.h"','#include "metal_context.h"\n#include "primary_32_contract.h"\n#include "primary_block_codec.h"',1)
needle=' MetalContext gpu;auto initial=MetalContext::PrimaryStats();';assert needle in t
checks=r''' // Pure integer boundary checks require no multi-GiB allocation.
 need(Primary32::ShapeFits(260,256,260ULL*256*2),"Valid primary32 shape rejected");
 need(Primary32::ShapeFits(8388608,128,Primary32::MaxRawBytes),"Exact2GiB bound rejected");
 need(!Primary32::ShapeFits(8388612,128,Primary32::MaxRawBytes+1024),"Over2GiB admitted");
 need(!Primary32::ShapeFits(UINT64_MAX,128,1024)&&!Primary32::ShapeFits(260,UINT64_MAX,1024),"Overflow dimensions admitted");
 need(!Primary32::ShapeFits(uint64_t(UINT32_MAX)+1,128,1024)&&!Primary32::ShapeFits(260,uint64_t(UINT32_MAX)+1,1024),"Overuint32 dimensions admitted");
 need(!Primary32::ShapeFits(0,128,0)&&!Primary32::ShapeFits(260,0,0)&&!Primary32::ShapeFits(256,128,65536)&&!Primary32::ShapeFits(257,128,65792)&&!Primary32::ShapeFits(260,129,67080),"Unsupported shape admitted");
 need(!Primary32::ShapeFits(260,256,260ULL*256*2-1),"Mismatched byte count admitted");
 need(Primary32::NarrowMetadata(UINT32_MAX)==UINT32_MAX,"Maximum uint32 metadata changed");
 {bool bad=false;try{Primary32::NarrowMetadata(uint64_t(UINT32_MAX)+1);}catch(...){bad=true;}need(bad,"Metadata truncation accepted");}
 {std::array<uint16_t,PrimaryBlocks::Words> words;words.fill(0x3f80);
  auto edge=PrimaryBlocks::Encode(words,Primary32::MaxRawBytes-256);
  need(Primary32::NarrowMetadata(edge.metadata)==edge.metadata,"Representable high offset changed");
  auto tooHigh=PrimaryBlocks::Encode(words,Primary32::MaxRawBytes);bool bad=false;try{Primary32::NarrowMetadata(tooHigh.metadata);}catch(...){bad=true;}need(bad,"Unrepresentable encoded offset accepted");}
'''
t=t.replace(needle,checks+needle,1);p.write_text(t)

drivers=['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py']
for name in drivers:(w/name).write_text((w.parent/'worldsense-primary-kmajor'/name).read_text().replace('worldsense-primary-kmajor','worldsense-primary-u32'))
for name in ['fixture-1-8','fixture-2-64']:shutil.copytree(w.parent/'worldsense-primary-kmajor'/name,w/name)
p=w/'primary_corpus_test.cpp';t=p.read_text();assert 'primaryBytes==9079938304ULL' in t;p.write_text(t.replace('primaryBytes==9079938304ULL','primaryBytes==9033678080ULL',1))
p=w/'api_state_guard.py';t=p.read_text();needle="assert ('primary_kmajor_decode' in call.get('backend',''))==bool(expected);";assert needle in t;p.write_text(t.replace(needle,needle+"assert ('primary_u32_metadata' in call.get('backend',''))==bool(expected);",1))
p=w/'full_correctness.py';t=p.read_text();needle="'all398actualpacked_weights_Kmajor_GPU_bit_reconstruction':True";assert needle in t;p.write_text(t.replace(needle,needle+",'all398actualpacked_weights_u32_GPU_bit_reconstruction':True",1))

changed=[]
for p in s.rglob('*'):
    if not p.is_file()or '__pycache__'in p.parts:continue
    q=old/p.relative_to(s)
    if not q.exists()or p.read_bytes()!=q.read_bytes():changed.append(str(p.relative_to(s)))
expected=['src/backends/metal/primary_32_contract.h','src/backends/metal/metal_context.mm','src/backends/metal/primary_block_shader.h','src/entry/metal_main.cpp','test2026/metal/inventory_test.cpp','test2026/metal/primary_storage_test.cpp']
assert sorted(changed)==sorted(expected),changed
cuda=[p for p in old.rglob('*')if p.is_file()and('cuda'in str(p.relative_to(old)).lower()or p.suffix=='.cu')];assert all(p.read_bytes()==(s/p.relative_to(old)).read_bytes()for p in cuda)
raw=lambda p:re.findall(r'R"METAL\((.*?)\)METAL"',p.read_text(),re.S)
assert raw(s/'src/backends/metal/metal_context.mm')==raw(old/'src/backends/metal/metal_context.mm')
(w/'implementation-audit.json').write_text(json.dumps({'changed_files_from_WS39':changed,'CUDA_unchanged_files':len(cuda),'all_other_source_Model_wrapper_counters_codec_geometry_identical':True,'entire_primary_shader_inverse_integer_width_substitution_exact':True,'inventory_schema_only_change':'metadata length8→4 bytes per block','scope':'Static invariants, not runtime correctness/performance.'},indent=2))
(w/'correctness-source-manifest.json').write_text(json.dumps({str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()and '__pycache__'not in p.parts},indent=2))
(w/'seed-source-verification.json').write_text(json.dumps({'all_WS39_seed_files_match_frozen_manifest':True,'count':len(seed),'seed_archive':'Garnet/results/20261016/worldsense-m5-primary-kmajor/starting-source-manifest.json'},indent=2))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('WS40 checked uint32 metadata/addressing six-file delta prepared; static proof and parsing PASS',flush=True)
