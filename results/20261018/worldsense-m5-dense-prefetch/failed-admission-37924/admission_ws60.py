"""Post-terminal WS60 configuration/native/source admission; no inference."""
from pathlib import Path
import json,hashlib,subprocess,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-dense-prefetch'
r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-dense-prefetch/source';old=w.parent/'worldsense-isolated-primary'
assert all(json.loads((w/'full-proof-complete.json').read_text()).values())
assert json.loads((w/'correctness-terminal.json').read_text())['controller_session']==68305
m=json.loads((w/'correctness-source-manifest.json').read_text());assert len(m)==1720 and all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items())
base=['/opt/homebrew/bin/cmake','-S',str(s),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]
cases=[
 ('k32',['-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=OFF','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY=ON'],'K32-only primary vector prefill requires primary vector prefill'),
 ('cached',['-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF','-DGARNET_METAL_CACHED_MATRIX_METADATA=ON'],'Cached matrix metadata cohort requires lazy default storage'),
 ('wide-primary',['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=ON','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=ON'],'Wide primary storage profile requires primary weights and global lazy defaults'),
 ('wide-lazy',['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=ON','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF'],'Wide primary storage profile requires primary weights and global lazy defaults'),
 ('full-primary',['-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=ON','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF'],'Full primary isolation requires primary weights and isolated vector prefill'),
 ('full-secondary',['-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=ON','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=ON','-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=OFF'],'Full primary isolation requires primary weights and isolated vector prefill'),
 ('prefetch-body',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_DENSE_PREFETCH=ON','-DGARNET_METAL_BODY_SHAPE=OFF'],'Dense prefetch requires body-shape specialization'),
 ('prefetch-native',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_DENSE_PREFETCH=ON','-DGARNET_METAL_NATIVE_BFLOAT_BODY=ON'],'Dense prefetch excludes later raw body override: GARNET_METAL_NATIVE_BFLOAT_BODY'),
 ('prefetch-interleaved',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_DENSE_PREFETCH=ON','-DGARNET_METAL_INTERLEAVED_BF16_BODY=ON'],'Dense prefetch excludes later raw body override: GARNET_METAL_INTERLEAVED_BF16_BODY'),
 ('prefetch-coalesced',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_DENSE_PREFETCH=ON','-DGARNET_METAL_INTERLEAVED_BF16_BODY=ON','-DGARNET_METAL_COALESCED_BF16_BODY=ON'],'Dense prefetch excludes later raw body override: GARNET_METAL_INTERLEAVED_BF16_BODY')]
results={}
for name,flags,expected in cases:
 command=base+['-B',str(w/('invalid-'+name+'-config'))]+flags
 p=subprocess.run(command,capture_output=True,text=True);text=p.stdout+p.stderr
 (w/('invalid-'+name+'-config.log')).write_text(text)
 assert p.returncode!=0 and expected in' '.join(text.split()),(name,p.returncode,text[-1000:])
 results[name]=dict(command=command,returncode=p.returncode,expected_rejection=True,expected_diagnostic=expected)
(w/'flag-dependency-proof.json').write_text(json.dumps(results,indent=2)+'\n')
for mode in ['on','fault']:
 b=r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode)
 log=(b/'Testing/Temporary/LastTest.log').read_text()
 assert 'Matrix metadata/live-MPS parity empty/copy/move/reassign/padding/pending/Submit/Abort/host-zero/context-lifetime PASS'in log
 assert 'Forced-raw dense prefetch numerical/reference/poison/input/tail272/fallback/small/vocab/queued owner/Abort recovery PASS'in log
 flags=(b/'CMakeFiles/garnet_dense_prefetch_test.dir/flags.make').read_text()
 assert '-DGARNET_METAL_DENSE_PREFETCH=1'in flags
inherited={}
for name in ['wide-scope-proof.json','independent-direct-upload-proof.json','cpu-loop-proof.json']:
 p=old/name;inherited[name]=dict(original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Immutable WS54 source proof. Wide-admission and loader bodies are byte-identical; new dense decode source/scope and fresh native/full model execution separately proved.',original_proof=json.loads(p.read_text()))
(w/'inherited-source-proofs.json').write_text(json.dumps(inherited,indent=2)+'\n')
assert json.loads((old/'wide-scope-proof.json').read_text())['wide_admission_scope_verified']
assert json.loads((w/'direct-upload-corpus.json').read_text())['totals']=={'weights':546,'FP32_words_checked':334609152}
(w/'admission-complete.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ten_expected_config_rejections=True,current_native_cached_live_properties_ON_fault_pass=True,current_forced_raw_prefetch_bounds_fallback_bits_abort_guards_pass=True,unchanged_source_proofs_explicitly_inherited=True,original_full_correctness_prerequisites_pass=True,no_performance_result=True),indent=2)+'\n')
print('Original6/newprefetch4 configuration cases and native/source/fullmodel admission PASS')
