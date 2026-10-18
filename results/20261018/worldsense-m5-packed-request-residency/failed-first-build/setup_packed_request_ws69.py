"""Isolated packed-only residency APIs; preserve old raw policy and all math."""
from pathlib import Path
import json,hashlib,shutil,re,ast,textwrap
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-packed-request-residency';old=r/'out/research/worldsense-request-weight-residency/source';s=r/'out/research/worldsense-packed-request-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();protocol=load(w/'protocol.json');audit=load(w/'read-only-packed-residency-consumer-audit.json');seed=audit['seed_manifest'];assert sha(w/'native-design-before-implementation.txt')==protocol['design_sha256'];assert len(seed)==1720 and all(sha(old/k)==v for k,v in seed.items());s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
ops={}
def change(rel,x,y,count=1):
 p=s/rel;t=p.read_text();assert t.count(x)==count,(rel,x,t.count(x));p.write_text(t.replace(x,y));ops.setdefault(rel,[]).append(dict(before=x,after=y,count=count))
def append(rel,t):
 p=s/rel;p.write_text(p.read_text()+t);ops.setdefault(rel,[]).append(dict(before='',after=t,append=True,count=1))
oldFlag='GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY';newFlag='GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY';oldFault='GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS';newFault='GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS'
def names(text):
 text=text.replace(oldFlag,newFlag).replace(oldFault,newFault)
 # Longest explicit public names first, then member/internal identifiers.
 for x,y in [('BeginRequestWeightResidency','BeginPackedRequestWeightResidency'),('EndRequestWeightResidency','EndPackedRequestWeightResidency'),('RequestWeightResidencyActive','PackedRequestWeightResidencyActive'),('RequestWeightsCovered','PackedRequestWeightsCovered'),('CancelRequestWeightResidency','CancelPackedRequestWeightResidency'),('RequestWeightResidencyStatsJSON','PackedRequestWeightResidencyStatsJSON'),('RequestResidencyTestFailAfter','PackedRequestResidencyTestFailAfter'),('BeginRequestResidency','BeginPackedRequestResidency'),('EndRequestResidency','EndPackedRequestResidency'),('RequestResidencyStatsJSON','PackedRequestResidencyStatsJSON'),('CancelRequestUnlocked','CancelPackedRequestUnlocked'),('CheckRequestCoverage','CheckPackedRequestCoverage'),('kRequestResidencyLimit','kPackedRequestResidencyLimit'),('RequestMaybeFail','PackedRequestMaybeFail'),('RequestBudget','PackedRequestBudget')]:text=text.replace(x,y)
 return re.sub(r'\brequest([A-Z]\w*)',lambda m:'packedRequest'+m.group(1),text)
def block(text,start):
 pos=text.index(start);depth=0;end=pos
 for line in text[pos:].splitlines(keepends=True):
  token=line.lstrip()
  if token.startswith(('#if ','#ifdef ','#ifndef ')):depth+=1
  elif token.startswith('#endif'):depth-=1
  end+=len(line)
  if depth==0:return text[pos:end]
 raise AssertionError('Unclosed original preprocessor block')
m='src/backends/metal/metal_context.mm';h='src/backends/metal/metal_context.h';dc='src/backends/metal/decoder_session.cpp';dh='src/backends/metal/decoder_session.h';e='src/entry/metal_main.cpp';py='runtime/metal_serving.py'
# Keep the old budget header exactly, broaden only its include guard.
t=(s/m).read_text();needle='#ifdef '+oldFlag+'\n#include "weight_residency_budget.h"';change(m,needle,'#if defined('+oldFlag+') || defined('+newFlag+')\n#include "weight_residency_budget.h"')
originalFields=block(t,'#ifdef '+oldFlag+'\n    id<MTLResidencySet>');change(m,originalFields,originalFields+names(originalFields))
public='''    bool BeginRequestWeightResidency(const std::vector<MetalBF16Weight>&);
    void EndRequestWeightResidency();
    bool RequestWeightResidencyActive() const;
    bool RequestWeightsCovered(const std::vector<MetalBF16Weight>&) const;
    void CancelRequestWeightResidency(uint64_t reason);
    std::string RequestWeightResidencyStatsJSON() const;
#ifdef GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS
    static void RequestResidencyTestFailAfter(int);
#endif
''';change(h,public,public+names(public))
# Duplicate original Context lifecycle as separate state/API, then change only representation/bounds.
context=(old/m).read_text();start=context.index('\nnamespace Garnet {\n#ifdef '+oldFlag+'\nnamespace {');originalContext=context[start:];assert originalContext.endswith('}\n');packedContext=names(originalContext);contextChanges=[]
def pc(x,y,count=1):
 global packedContext
 assert packedContext.count(x)==count,(x,packedContext.count(x));packedContext=packedContext.replace(x,y);contextChanges.append(dict(before=x,after=y,count=count))
pc('constexpr uint64_t kPackedRequestResidencyLimit=uint64_t(4)*1024*1024*1024;','constexpr uint64_t kPackedRequestResidencyLimit=uint64_t(8)*1024*1024*1024;')
pc('constexpr uint64_t kPackedRequestResidencyLimit=64*1024;','constexpr uint64_t kPackedRequestResidencyLimit=128*1024;')
startOld='''     if(weight.PrimaryReady()){fallback(7);return false;}
     if(!weight.impl_||!weight.impl_->buffer)throw std::invalid_argument("Invalid raw BF16 request weight");
     auto buffer=weight.impl_->buffer;if(buffer.device!=impl_->device)throw std::invalid_argument("Request weight device mismatch");
     auto identity=(__bridge const void*)buffer;if(!seen.insert(identity).second)continue;
     auto n=buffer.allocatedSize;if(n>UINT64_MAX-bytes)throw std::overflow_error("Request residency byte overflow");bytes+=n;
     impl_->packedRequestAttemptedBytes=bytes;impl_->packedRequestAttemptedCount=seen.size();
     if(seen.size()>2048||bytes>kPackedRequestResidencyLimit){fallback(4);return false;}
     [allocations addObject:buffer];owners.push_back(weight);'''
startNew='''     if(!weight.impl_)throw std::invalid_argument("Invalid packed request weight owner");
     if(!weight.PrimaryReady()){
      if(!weight.impl_->buffer||weight.impl_->buffer.device!=impl_->device)throw std::invalid_argument("Invalid automatic raw request weight");
      continue;
     }
     auto owner=weight.impl_;
     if(!owner->primaryPayload||!owner->primaryMetadata||owner->primaryPayload.device!=impl_->device||owner->primaryMetadata.device!=impl_->device)throw std::invalid_argument("Packed request allocation/device mismatch");
     bool retained=false;
     for(id<MTLBuffer> buffer in @[owner->primaryPayload,owner->primaryMetadata]){
      auto identity=(__bridge const void*)buffer;if(!seen.insert(identity).second)continue;
      auto n=buffer.allocatedSize;if(n>UINT64_MAX-bytes)throw std::overflow_error("Packed request residency byte overflow");bytes+=n;
      impl_->packedRequestAttemptedBytes=bytes;impl_->packedRequestAttemptedCount=seen.size();
      if(seen.size()>2048||bytes>kPackedRequestResidencyLimit){fallback(4);return false;}
      [allocations addObject:buffer];retained=true;
     }
     if(retained)owners.push_back(weight);'''
pc(startOld,startNew)
coverageOld=' for(const auto& weight:weights){if(!weight.impl_||weight.PrimaryReady()||!weight.impl_->buffer||weight.impl_->buffer.device!=impl_->device||!impl_->packedRequestIdentities.count((__bridge const void*)weight.impl_->buffer))return false;}'
coverageNew=''' for(const auto& weight:weights){
  if(!weight.impl_)return false;
  if(!weight.PrimaryReady()){if(!weight.impl_->buffer||weight.impl_->buffer.device!=impl_->device)return false;continue;}
  if(!weight.impl_->primaryPayload||!weight.impl_->primaryMetadata||weight.impl_->primaryPayload.device!=impl_->device||weight.impl_->primaryMetadata.device!=impl_->device)return false;
  if(!impl_->packedRequestIdentities.count((__bridge const void*)weight.impl_->primaryPayload)||!impl_->packedRequestIdentities.count((__bridge const void*)weight.impl_->primaryMetadata))return false;
 }'''
pc(coverageOld,coverageNew);pc('Garnet bounded request BF16 weights','Garnet bounded packed payload and metadata')
# Actual native lifecycle inspection is telemetry, included in request total.
head='std::string MetalContext::PackedRequestWeightResidencyStatsJSON() const {\n#ifdef '+newFlag
pc(head,head+'\n uint64_t packedSetAllocations=0;if(@available(macOS 15.0,*)){if(impl_->packedRequestResidencySet)packedSetAllocations=impl_->packedRequestResidencySet.allocationCount;}')
part=r'std::to_string(impl_->packedRequestReason)+",\"scope_open\":"'
replacement=r'std::to_string(impl_->packedRequestReason)+",\"set_allocation_count\":"+std::to_string(packedSetAllocations)+",\"owner_count\":"+std::to_string(impl_->packedRequestOwners.size())+",\"attached\":"+(impl_->packedRequestAttached?"true":"false")+",\"requested\":"+(impl_->packedRequestRequested?"true":"false")+",\"scope_open\":"'
pc(part,replacement)
inverse=packedContext
for op in reversed(contextChanges):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==names(originalContext);append(m,packedContext)
abort='#ifdef '+oldFlag+'\n    if(impl_->requestScopeOpen)CancelRequestWeightResidency(9);\n#endif';change(m,abort,abort+'\n'+names(abort))
# Session duplicates keep original lock/validation/owner/generation/cleanup order.
session=(old/dc).read_text();start=session.index('\nnamespace Garnet {\n#ifdef '+oldFlag+'\nvoid MetalDecoderSession::CancelRequestUnlocked()');originalSession=session[start:];assert originalSession.endswith('}\n');append(dc,names(originalSession))
private=block((old/dh).read_text(),'#ifdef '+oldFlag+'\n    bool requestScopeOpen_');change(dh,private,private+names(private))
publicSession='''    bool BeginRequestResidency(X::Value graphs);
    bool EndRequestResidency();
    std::string RequestResidencyStatsJSON() const;''';change(dh,publicSession,publicSession+'\n'+names(publicSession))
exports='''        APISET().AddFunc<1>("begin_request_residency",&MetalDecoderSession::BeginRequestResidency);
        APISET().AddFunc<0>("end_request_residency",&MetalDecoderSession::EndRequestResidency);
        APISET().AddProp("request_residency_stats_json",&MetalDecoderSession::RequestResidencyStatsJSON);''';newExports=names(exports).replace('"begin_request_residency"','"begin_packed_request_residency"').replace('"end_request_residency"','"end_packed_request_residency"').replace('"request_residency_stats_json"','"packed_request_residency_stats_json"');change(dh,exports,exports+'\n'+newExports)
# Instrument all original lifecycle hook positions without moving old behavior.
for marker,count in [('#ifdef '+oldFlag+'\n    CancelRequestUnlocked();\n#endif',2),('#ifdef '+oldFlag+'\n    CheckRequestCoverage(locked,graphs);\n#endif',1)]:change(dc,marker,marker+'\n'+names(marker),count)
guard=block(session,'#ifdef '+oldFlag+'\n    const int requestExceptions=');change(dc,guard,guard+names(guard))
marker='#ifdef '+oldFlag+'\n        info+="; request_scope_weight_residency";\n#endif';change(e,marker,marker+'\n#ifdef '+newFlag+'\n        info+="; packed_request_weight_residency";\n#endif')
# Actual compiled graph metadata selects eligibility at Model construction; no ID routing.
change(py,"_request_scope_residency = 'request_scope_weight_residency' in garnet.backend_info()", "_request_scope_residency = 'request_scope_weight_residency' in garnet.backend_info()\n_packed_request_scope_residency = 'packed_request_weight_residency' in garnet.backend_info()")
needle="        self.initialization_memory['after_decoder_ready'] = json.loads(garnet.memory_stats_json())";change(py,needle,"        self._packed_request_scope = _packed_request_scope_residency and len(self.decoder.layers) > 0 and bool(self.decoder.layers[0]['graph'].wide_primary_eligible)\n"+needle)
text=(s/py).read_text();begin=text.index("        def append_text(text):",text.index('    def infer('));end=text.index('\n\ndef serve_model(',begin);body=text[begin:end];assert body.endswith('\n')
wrapped='''        packed_request_started = False
        result = None
        try:
            if self._packed_request_scope:
                self.decoder.session.begin_packed_request_residency([layer['graph'] for layer in self.decoder.layers])
                packed_request_started = True
'''+textwrap.indent(body,'    ')+'''        finally:
            if packed_request_started:
                self.decoder.session.end_packed_request_residency()
            if _packed_request_scope_residency:
                stages['packed_request_eligible'] = bool(self._packed_request_scope)
                stages['packed_request_weight_residency'] = json.loads(self.decoder.session.packed_request_residency_stats_json)
                stages['memory_after_packed_request_cleanup'] = json.loads(garnet.memory_stats_json())
                if result is not None:
                    result['total_seconds'] = time.monotonic() - start
'''
change(py,body,wrapped)
append('cmake/GarnetMetal.cmake','''
option(GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY "Bound packed VLM payload/metadata residency across an inference request" OFF)
option(GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS "Correctness-only packed 128KiB budget and reserve/attach failure hooks" OFF)
if(GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_WIDE_PRIMARY_STORAGE OR NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS OR GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY)
  message(FATAL_ERROR "Packed request residency requires primary/wide/lazy and excludes raw request residency")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY=1)
endif()
if(GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS)
 if(NOT GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY)
  message(FATAL_ERROR "Packed request test hooks require packed request residency")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS=1)
endif()
add_executable(garnet_packed_request_residency_test test2026/metal/packed_request_residency_test.cpp)
target_link_libraries(garnet_packed_request_residency_test PRIVATE garnet_metal)
target_include_directories(garnet_packed_request_residency_test PRIVATE third_party)
add_test(NAME metal_packed_request_residency_reference COMMAND garnet_packed_request_residency_test)
''')
# No test/helper is allowed to alter production limits or original test math.
shutil.copy2(w/'packed_request_residency_test.cpp',s/'test2026/metal/packed_request_residency_test.cpp')
for rel,records in ops.items():
 t=(s/rel).read_text()
 for op in reversed(records):
  if op.get('append'):assert t.endswith(op['after']);t=t[:-len(op['after'])]
  else:assert t.count(op['after'])==op['count'];t=t.replace(op['after'],op['before'])
 assert t==(old/rel).read_text(),rel
 if rel.endswith('.py'):ast.parse((s/rel).read_text())
pattern=r'R"([A-Za-z_]+)\((.*?)\)\1"';assert re.findall(pattern,(s/m).read_text(),re.S)==re.findall(pattern,(old/m).read_text(),re.S)
current={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};assert len(current)==1721 and all(current[k]==v for k,v in seed.items()if k not in ops)
for name,value in [('seed-source-manifest.json',seed),('correctness-source-manifest.json',current),('source-transformations.json',ops),('packed-context-adaptation-proof.json',dict(original=originalContext,renamed_original=names(originalContext),new=packedContext,operations=contextChanges,all_old_lifecycle_code_retained_after_explicit_representation_name_budget_normalization=True)),('packed-session-adaptation-proof.json',dict(original=originalSession,new=names(originalSession),all_original_locks_generation_cache_lifetimes_errors_inverse_exact_after_names=True))]: (w/name).write_text(json.dumps(value,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1720,current_files=1721,changed_or_added=sorted(k for k,v in current.items()if seed.get(k)!=v),old_raw_residency_body_API_budget_and_tests_unchanged=True,original_all_shaders_math_graph_xModel_image_CUDA_engine_and_original_tests_unchanged_except_explicit_native_lifecycle_hooks=True,serving_original_body_inverse_exact_after_outer_lease_scope=True,separate_packed_payload_metadata_allocation_policy=True),indent=2)+'\n');shutil.copytree(b/'work/worldsense-primary-simd-metadata/build-profile',w/'build-profile');print('WS69 source1721; old raw policy and math preserved, separate packed lifecycle/native test added; no build/model yet.')
