"""Only after terminal full-model proof: preserve old policy rejection and close new readers."""
from pathlib import Path
import json,hashlib,subprocess,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-dual-decode';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-sole-dual-decode/source'
assert json.loads((w/'learned-independent-audit.json').read_text())['all_original_model_word_state_token_vision_gates_pass']
base=['/opt/homebrew/bin/cmake','-S',str(s),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL=ON','-DGARNET_METAL_SOLE_DUAL_PROJECTION_DECODE=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]
cases=[
 ('k32',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=OFF','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY=ON'],'K32-only primary vector prefill requires primary vector prefill'),
 ('cached',['-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF','-DGARNET_METAL_CACHED_MATRIX_METADATA=ON'],'Cached matrix metadata cohort requires lazy default storage'),
 ('wide-primary',['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=ON','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=ON'],'Wide primary storage profile requires primary weights and global lazy defaults'),
 ('wide-lazy',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DGARNET_METAL_WIDE_PRIMARY_STORAGE=ON','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF'],'Wide primary storage profile requires primary weights and global lazy defaults'),
 ('full-primary',['-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=ON','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF'],'Full primary isolation requires primary weights and isolated vector prefill'),
 ('full-secondary',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=ON','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=ON','-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=OFF'],'Full primary isolation requires primary weights and isolated vector prefill'),
 ('sole-body',['-DGARNET_METAL_SOLE_INTERLEAVED_MLP=ON','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_BODY_SHAPE=OFF'],'Sole interleaved MLP requires body-shape support')]
policies=['EXACT_HALF_PREFILL','GLOBAL_HALF_ALPHA','NATIVE_BFLOAT_BODY','CONSTANT_BF16_HEAD','SINGLE_CODER_HEAD','INTERLEAVED_BF16_BODY','COALESCED_BF16_BODY','FUSED_HEAD_ARGMAX','SCREENED_HEAD','FLOAT_DENSE_KEY_PREFILL','JOINED_PREFILL','JOINED_GLOBAL_MLP','MLP_PREFILL_GRAPH']
for policy in policies:
 flags=['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_SOLE_INTERLEAVED_MLP=ON','-DGARNET_METAL_'+policy+'=ON'];expected='Sole interleaved MLP excludes row-major-only auxiliary policy: GARNET_METAL_'+policy
 if policy=='GLOBAL_HALF_ALPHA':flags+=['-DGARNET_METAL_EXACT_HALF_PREFILL=ON'];expected='Sole interleaved MLP excludes row-major-only auxiliary policy: GARNET_METAL_EXACT_HALF_PREFILL'
 if policy=='COALESCED_BF16_BODY':flags+=['-DGARNET_METAL_INTERLEAVED_BF16_BODY=ON'];expected='Sole interleaved MLP excludes row-major-only auxiliary policy: GARNET_METAL_INTERLEAVED_BF16_BODY'
 if policy=='JOINED_GLOBAL_MLP':flags+=['-DGARNET_METAL_GLOBAL_HALF_ALPHA=ON','-DGARNET_METAL_EXACT_HALF_PREFILL=ON'];expected='Sole interleaved MLP excludes row-major-only auxiliary policy: GARNET_METAL_EXACT_HALF_PREFILL'
 if policy=='JOINED_PREFILL':flags+=['-DGARNET_METAL_EXACT_HALF_PREFILL=ON'];expected='Sole interleaved MLP excludes row-major-only auxiliary policy: GARNET_METAL_EXACT_HALF_PREFILL'
 if policy=='NATIVE_BFLOAT_BODY':expected='Body shape and native bfloat are separate decode policies'
 cases.append(('sole-'+policy.lower(),flags,expected))
cases.append(('direct-storage',['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_SOLE_INTERLEAVED_MLP=OFF'],'Direct sole prefill requires sole interleaved MLP storage'))
cases.extend([('dual-direct',['-DGARNET_METAL_SOLE_INTERLEAVED_MLP=ON','-DGARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL=OFF'],'Sole dual decode requires sole MLP storage and direct prefill'),('dual-storage',['-DGARNET_METAL_SOLE_INTERLEAVED_MLP=OFF','-DGARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL=OFF'],'Sole dual decode requires sole MLP storage and direct prefill')])
results={}
for name,flags,expected in cases:
 command=base+['-B',str(w/('invalid-'+name+'-config'))]+flags;p=subprocess.run(command,capture_output=True,text=True);text=p.stdout+p.stderr;(w/('invalid-'+name+'-config.log')).write_text(text)
 assert p.returncode!=0 and expected in ' '.join(text.split()),(name,p.returncode,text[-1200:]);results[name]=dict(command=command,returncode=p.returncode,expected_diagnostic=expected)
(w/'flag-dependency-proof.json').write_text(json.dumps(results,indent=2)+'\n')
(w/'admission-complete.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),twentythree_expected_config_rejections=len(results)==23,original_full_learned_correctness_pass=True,new_layout_all_direct_readers_closed_by_explicit_projection_readback_and_policy_rejection=True,no_performance_result=True),indent=2)+'\n');print('Six original/seventeen new configuration rejection cases PASS')
