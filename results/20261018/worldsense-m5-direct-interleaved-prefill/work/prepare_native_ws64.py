"""Preserve original native references and add direct/restore compatibility proof."""
from pathlib import Path
import json,ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-direct-interleaved-prefill';old=b/'work/worldsense-sole-interleaved'
for name in ['restore_normal.py','wait_cantor.py','cases.json']:shutil.copy2(old/name,w/name)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
t=(old/'native_controller.py').read_text().replace('worldsense-sole-interleaved','worldsense-direct-interleaved-prefill').replace("['off','primary','on','fault']","['off','primary','on','fault','restore']")
t=t.replace("['original_shader_literals_graph_session_xModel_image_CUDA_tests_unchanged']","['original_shader_literals_and_63_decode_graph_session_xModel_image_CUDA_original_tests_unchanged']")
needle=" declared=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R\"SOLEMLP(',1)[1].split(')SOLEMLP\"',1)[0].encode();assert actual_sole['on']==declared"
assert t.count(needle)==1
z=""" original_sole=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode()
 direct=(s/'src/backends/metal/direct_interleaved_prefill_shader.h').read_text().split('R"DIRECTMLP(',1)[1].split(')DIRECTMLP"',1)[0].encode()
 assert actual_sole['on']==original_sole+direct and actual_sole['restore']==original_sole
 assert original_sole==(w.parent/'worldsense-sole-interleaved/actual-on-sole-program.metal').read_bytes()
""".rstrip();t=t.replace(needle,z)
needle="('actual-'+mode+'-'+kind+'-program.metal')).read_bytes()for mode in programs"
assert t.count(needle)==1;t=t.replace(needle,"('actual-'+('on'if mode=='restore'else mode)+'-'+kind+'-program.metal')).read_bytes()for mode in programs")
t=t.replace('ALL4BUILDS8NATIVE_TESTS_186ORIGINAL_28SOLE_PROJECTION_RECORDS_EXACT','ALL5BUILDS8NATIVE_TESTS_186ORIGINAL_28SOLE_PROJECTION_RECORDS_EXACT')
t=t.replace("'all_four_builds_FP32_words_identical':True","'all_four_builds_FP32_words_identical':True,'all_five_builds_FP32_words_identical':True")
t=t.replace('all_four_builds_FP32_words_identical=True','all_four_builds_FP32_words_identical=True,all_five_builds_FP32_words_identical=True')
ast.parse(t);(w/'native_controller.py').write_text(t)
print('Five-role native proof prepared; original words/MAIN/prefill and63sole defaultOFF compatibility preserved')
