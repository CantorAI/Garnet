"""Pre-native diagnostic-contract amendment; retain first builds, no original test edits."""
from pathlib import Path
import shutil,json,hashlib,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-core-isolation';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-core-isolation/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault']
assert(w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+v for v in roles]
m=load(w/'correctness-source-manifest.json');assert all(sha(s/k)==v for k,v in m.items());saved=w/'pre-native-program-contract-review';saved.mkdir(exist_ok=False)
for name in ['correctness-source-manifest.json','source-inverse-proof.json','source-transformations.json','build-controller.log','build-plan.json','build.py','protocol.json','primary_core_isolation_test.cpp','native-design-before-implementation.txt']:
 shutil.copy2(w/name,saved/name)
for name in ['setup_core_isolation_ws70.py','design_core_isolation_ws70.py']:shutil.copy2(b/'work/worldsense-opt'/name,saved/name)
for rel in load(w/'source-inverse-proof.json')['changed_or_added']:
 p=saved/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,p)
ids={}
for role in roles:
 shutil.copy2(w/('build-'+role+'.log'),saved/('build-'+role+'.log'));build=r/('out/build/garnet-metal-worldsense-primary-core-isolation-'+role)
 ids[role]={name:sha(build/name)for name in ['libgarnet.dylib','libgarnet_metal.a','garnet_primary_core_isolation_test','CMakeCache.txt']}
(saved/'initial-build-identities.json').write_text(json.dumps(ids,indent=2)+'\n')
transforms=load(w/'source-transformations.json');mm='src/backends/metal/metal_context.mm'
for op in transforms[mm]:
 op['after']=op['after'].replace('id<MTLLibrary> primaryCoreLibrary=nil;','id<MTLLibrary> primaryCoreLibrary=nil;\n    std::string primaryCoreProgram;').replace('impl_->kernels->prefillProgram=coreSource.UTF8String;','impl_->kernels->primaryCoreProgram=coreSource.UTF8String;')
marker=' return {impl_->kernels->mainProgram,impl_->kernels->prefillProgram};\n}\n}'
replacement=''' return {impl_->kernels->mainProgram,impl_->kernels->prefillProgram};
}
std::string MetalContext::PrimaryCoreProgramSource() const {
 if(!impl_->kernels)throw std::logic_error("Core program source requires ready kernels");
#ifdef GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY
 return impl_->kernels->primaryCoreProgram;
#else
 return {};
#endif
}
}'''
assert (old/mm).read_text().count(marker)==1;transforms[mm].append(dict(before=marker,after=replacement,count=1))
header='src/backends/metal/metal_context.h';marker='    std::vector<std::string> ShaderProgramSources() const;';transforms[header].append(dict(before=marker,after=marker+'\n    std::string PrimaryCoreProgramSource() const;',count=1))
for rel,ops in transforms.items():
 text=(old/rel).read_text()
 for op in ops:
  if op.get('append'):text+=op['after']
  else:assert text.count(op['before'])==op['count'];text=text.replace(op['before'],op['after'])
 inverse=text
 for op in reversed(ops):
  if op.get('append'):assert inverse.endswith(op['after']);inverse=inverse[:-len(op['after'])]
  else:assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse==(old/rel).read_text();(s/rel).write_text(text)
p=w/'primary_core_isolation_test.cpp';t=p.read_text();needle='constexpr bool core=true;need(programs[1]==std::string(kPrimaryIsolatedPreamble)+kPrimaryBlockShader,';assert t.count(needle)==1;t=t.replace(needle,'constexpr bool core=true;need(programs[1].empty(),"Original vector-prefill program contract changed");need(gpu.PrimaryCoreProgramSource()==std::string(kPrimaryIsolatedPreamble)+kPrimaryBlockShader,')
needle=' auto initial=MetalContext::PrimaryStats();';assert t.count(needle)==1;t=t.replace(needle,' if(!core)need(gpu.PrimaryCoreProgramSource().empty(),"Disabled core program dump nonempty");\n {std::ofstream program("primary_core_shader_program.metal",std::ios::binary);program<<gpu.PrimaryCoreProgramSource();need(bool(program),"Cannot save actual core program");}\n'+needle);p.write_text(t);shutil.copy2(p,s/'test2026/metal/primary_core_isolation_test.cpp')
(w/'source-transformations.json').write_text(json.dumps(transforms,indent=2)+'\n');manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};seed=load(w/'seed-source-manifest.json');assert len(manifest)==1721 and all(manifest[k]==v for k,v in seed.items()if k not in transforms);(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
proof=load(w/'source-inverse-proof.json');proof.update(original_two_program_API_semantics_preserved=True,new_separate_CPP_core_program_getter=True);(w/'source-inverse-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
amendment='''WS70 diagnostic API amendment BEFORE native/model/timing execution
Initial six builds78008exit0. Static review of unchanged original primary_storage_test.cpp reveals ShaderProgramSources()[1] must be empty unless original vector-prefill-isolation flag is enabled. The first implementation reused that slot for ordinary core source; it would violate the original diagnostic contract. No native/model test was dispatched.
Preserve initial design/source/transforms/generator/flags/binary hashes/build logs. Keep original two-string API and its original vector-prefill semantics unchanged. Add separate C++ PrimaryCoreProgramSource() getter (empty when disabled), retained core-source string and native dump primary_core_shader_program.metal. Original tests remain byte-identical. Library/PSO/compute/dispatch/weights/math/runtime behavior is unchanged; source-string storage simply moves to a dedicated diagnostic field. This narrowly amends design item2, not any gate/math/model/input policy. Rebuild allsix before native. No performance or correctness rerun, since none occurred yet.
''';(w/'diagnostic-api-amendment-before-native.txt').write_text(amendment)
protocol=load(w/'protocol.json');protocol['diagnostic_api_amendment_sha256']=sha(w/'diagnostic-api-amendment-before-native.txt');protocol['amendment_before_native']=True;(w/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
p=b/'outputs/Garnet-4B-WS70-Design.txt';p.write_text(p.read_text()+'\n'+amendment)
state=load(b/'work/worldsense-next.json');state.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='DIAGNOSTIC_CONTRACT_AMENDED_REBUILD_NEXT',initial_build_session=78008,initial_build_exit_code=0,initial_six_builds_pass=True,no_native_or_model_dispatched=True,diagnostic_api_amendment_before_native=True,live_job=None);(b/'work/worldsense-next.json').write_text(json.dumps(state,indent=2)+'\n');print('Preserved six initial builds, repaired diagnostic API only before any native/model execution; original test bytes preserved, source inverse refreshed, rebuild allsix next.')
