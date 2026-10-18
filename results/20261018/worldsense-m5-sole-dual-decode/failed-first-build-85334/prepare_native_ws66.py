from pathlib import Path
import json,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-dual-decode';old=b/'work/worldsense-direct-interleaved-prefill';roles="['off','primary','on','fault','restore','direct64']"
for name in ['restore_normal.py','wait_cantor.py','cases.json']:shutil.copy2(old/name,w/name)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
t=(old/'build.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-dual-decode').replace("['off','primary','on','fault','restore']",roles).replace("['on','fault','restore']","['on','fault','restore','direct64']").replace("['on','fault']else'OFF'","['on','fault','direct64']else'OFF'")
a="'-DGARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL='+('ON'if mode in ['on','fault','direct64']else'OFF'),";assert t.count(a)==1;t=t.replace(a,a+"'-DGARNET_METAL_SOLE_DUAL_PROJECTION_DECODE='+('ON'if mode in ['on','fault']else'OFF'),");ast.parse(t);(w/'build.py').write_text(t)
t=(old/'native_controller.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-dual-decode').replace("['off','primary','on','fault','restore']",roles).replace("('on'if mode=='restore'else mode)","('on'if mode in ['restore','direct64']else mode)").replace("['original_shader_literals_and_63_decode_graph_session_xModel_image_CUDA_original_tests_unchanged']","['original_shader_literals_and_all_original_eight_tests_unchanged']")
t=t.replace("assert actual_sole['on']==original_sole+direct and actual_sole['restore']==original_sole","assert actual_sole['on']==original_sole+direct and actual_sole['direct64']==actual_sole['on'] and actual_sole['restore']==original_sole")
t=t.replace("'all_five_builds_FP32_words_identical':True","'all_five_builds_FP32_words_identical':True,'all_six_builds_FP32_words_identical':True").replace('all_five_builds_FP32_words_identical=True','all_five_builds_FP32_words_identical=True,all_six_builds_FP32_words_identical=True')
needle=' actual_sole={mode:';assert t.count(needle)==1
extra=''' import struct
 paths={mode:r/('out/build/garnet-metal-worldsense-sole-dual-decode-'+mode)/'sole_dual_projection_words.bin'for mode in ['off','primary','on','fault','restore','direct64']}
 reference=paths['off'].read_bytes();assert all(path.read_bytes()==reference for path in paths.values())
 expected=[(rows,2048,6144,passid,projection,pair)for pair in [0,1]for rows in [1,2]for passid in range(5 if pair==0 else 4)for projection in [0,1]]
 position=0;records=[]
 for shape in expected:
  assert struct.unpack('6Q',reference[position:position+48])==shape;position+=48;count=shape[0]*shape[2]*4;data=reference[position:position+count];assert len(data)==count;position+=count;records.append(dict(shape=shape,bytes=count,sha256=hashlib.sha256(data).hexdigest()))
 assert position==len(reference)and len(records)==36
 (w/'dual-word-comparison.json').write_text(json.dumps(dict(record_count=36,records=records,all_six_builds_complete_FP32_words_identical=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest(),scope='Complete separate gate/up outputs versus original kernel bits plus original independent FP64 bounds. Not learned/model/speed evidence.'),indent=2)+'\\n')
 declared=original_program=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode()+(s/'src/backends/metal/direct_interleaved_prefill_shader.h').read_text().split('R"DIRECTMLP(',1)[1].split(')DIRECTMLP"',1)[0].encode()
 declared+=(s/'src/backends/metal/sole_dual_projection_shader.h').read_text().split('R"SOLEDUAL(',1)[1].split(')SOLEDUAL"',1)[0].encode()
 for mode,path in paths.items():
  program=Path(str(path)+'.metal').read_bytes();assert program==(declared if mode in ['on','fault']else b'');(w/('actual-'+mode+'-dual-program.metal')).write_bytes(program)
'''
t=t.replace(needle,extra+needle).replace('Native8tests','Native9tests').replace('ALL5BUILDS8NATIVE_TESTS_186ORIGINAL_28SOLE_PROJECTION_RECORDS_EXACT','ALL6BUILDS9NATIVE_TESTS_186ORIGINAL_28MLP_36DUAL_RECORDS_EXACT');ast.parse(t);(w/'native_controller.py').write_text(t)
print('WS66 six build roles and native54test/186+28+36complete record proof prepared')
