"""Prepare native proof orchestration; no GPU work or source mutation."""
from pathlib import Path
import json,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved';old=b/'work/worldsense-isolated-primary'
for name in ['restore_normal.py','wait_cantor.py','cases.json']:shutil.copy2(old/name,w/name)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
t=(old/'native_controller.py').read_text().replace('worldsense-isolated-primary','worldsense-sole-interleaved')
t=t.replace("assert json.loads((w/'independent-shader-proof.json').read_text())['CXX_materialized_OFF_primary_fragment_exact']","assert json.loads((w/'source-inverse-proof.json').read_text())['original_shader_literals_graph_session_xModel_image_CUDA_tests_unchanged']")
t=t.replace("p.name!='native-driver-identities-before-dispatch.json'","p.name!='native-driver-identities-before-dispatch.json'")
t=t.replace("'Native7tests'","'Native8tests'")
t=t.replace("paths={mode:","paths={mode:",1)
needle=" assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())\n programs={}"
extra=""" sole_paths={mode:r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)/'sole_projection_words.bin'for mode in ['off','primary','on','fault']}
 sole_words=sole_paths['off'].read_bytes();assert all(p.read_bytes()==sole_words for p in sole_paths.values())
 (w/'sole-word-comparison.json').write_text(json.dumps(dict(records=28,all_four_builds_FP32_words_identical=True,bytes=len(sole_words),scope='Two exact MLP shapes, seven row counts, two changed inputs; full outputs and original FP64 bounds. Separate from model/speed proof.'),indent=2)+'\\n')
 actual_sole={mode:(r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)/'sole_projection_words.bin.metal').read_bytes()for mode in ['off','primary','on','fault']}
 assert not actual_sole['off']and not actual_sole['primary']and actual_sole['on']==actual_sole['fault']
 declared=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode();assert actual_sole['on']==declared
 for mode,value in actual_sole.items():(w/('actual-'+mode+'-sole-program.metal')).write_bytes(value)
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
 programs={} """.rstrip()
assert t.count(needle)==1;t=t.replace(needle,extra)
needle=" assert programs['off']['main']==programs['on']['main']==programs['fault']['main']"
t=t.replace(needle,needle+"\n assert all(programs[mode][kind]==(w.parent/'worldsense-isolated-primary'/('actual-'+mode+'-'+kind+'-program.metal')).read_bytes()for mode in programs for kind in ['main','prefill'])")
t=t.replace("'scope':'Actual constructor-produced", "'all_four_original_MAIN_and_prefill_exact_seed':True,'new_sole_program_exact_declared_separate_library':True,'scope':'Actual constructor-produced")
t=t.replace('ALL4BUILDS7NATIVE_TESTS_AND186_COMPLETE_PROJECTION_RECORDS_EXACT','ALL4BUILDS8NATIVE_TESTS_186ORIGINAL_28SOLE_PROJECTION_RECORDS_EXACT')
# Freeze actual native executables/static+dylib/runtime in addition to driver/source closure.
needle="subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False)"
freeze="""native_binaries=[n/'xlang3',n/'libxlang3_runtime.dylib']
for mode in ['off','primary','on','fault']:
 build=r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)
 native_binaries += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
frozen_binaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in native_binaries}
(w/'native-binaries-before.json').write_text(json.dumps(frozen_binaries,indent=2)+'\\n')
"""
assert t.count(needle)==1;t=t.replace(needle,freeze+needle)
t=t.replace(" print('ACTUAL_MAIN_CANDIDATE_EQUALS_RAW_WITH_ALL_PRIMARY_KERNELS_IN_SECONDARY',flush=True)"," assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen_binaries.items())\n print('ACTUAL_MAIN_CANDIDATE_EQUALS_RAW_WITH_ALL_PRIMARY_KERNELS_IN_SECONDARY',flush=True)")
ast.parse(t);(w/'native_controller.py').write_text(t)
print('Prepared original native gates + complete sole outputs/separate source/identity admission; no native run dispatched')
