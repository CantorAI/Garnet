from pathlib import Path
import shutil,json,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-cooperative-prefill';old=b/'work/worldsense-sole-dual-decode'
for name in ['restore_normal.py','wait_cantor.py','cases.json']:shutil.copy2(old/name,w/name)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
roles="['off','primary','on','fault','restore','direct64','dual66']"
t=(old/'build.py').read_text().replace('worldsense-sole-dual-decode','worldsense-cooperative-prefill').replace("['off','primary','on','fault','restore','direct64']",roles).replace("['on','fault','restore','direct64']","['on','fault','restore','direct64','dual66']").replace("['on','fault','direct64']","['on','fault','direct64','dual66']").replace("['on','fault']else'OFF'","['on','fault','dual66']else'OFF'")
a="'-DGARNET_METAL_SOLE_DUAL_PROJECTION_DECODE='+('ON'if mode in ['on','fault','dual66']else'OFF'),";assert t.count(a)==1;t=t.replace(a,a+"'-DGARNET_METAL_SOLE_COOPERATIVE_PREFILL='+('ON'if mode in ['on','fault']else'OFF'),");ast.parse(t);(w/'build.py').write_text(t)
t=(old/'native_controller.py').read_text().replace('worldsense-sole-dual-decode','worldsense-cooperative-prefill').replace("['off','primary','on','fault','restore','direct64']",roles).replace("['restore','direct64']","['restore','direct64','dual66']").replace("['original_shader_literals_and_all_original_eight_tests_unchanged']","['all_original_nine_native_mathematics_limits_and_complete_records_unchanged']")
t=t.replace("'all_six_builds_FP32_words_identical':True","'all_six_builds_FP32_words_identical':True,'all_seven_builds_FP32_words_identical':True").replace('all_six_builds_FP32_words_identical=True','all_six_builds_FP32_words_identical=True,all_seven_builds_FP32_words_identical=True').replace('all_six_builds_complete_FP32_words_identical=True','all_six_builds_complete_FP32_words_identical=True,all_seven_builds_complete_FP32_words_identical=True')
a=" declared=original_program=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R\"SOLEMLP(',1)[1].split(')SOLEMLP\"',1)[0].encode()+(s/'src/backends/metal/direct_interleaved_prefill_shader.h').read_text().split('R\"DIRECTMLP(',1)[1].split(')DIRECTMLP\"',1)[0].encode()\n declared+=(s/'src/backends/metal/sole_dual_projection_shader.h').read_text().split('R\"SOLEDUAL(',1)[1].split(')SOLEDUAL\"',1)[0].encode()"
z=""" original_program=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode()
 direct_program=(s/'src/backends/metal/direct_interleaved_prefill_shader.h').read_text().split('R"DIRECTMLP(',1)[1].split(')DIRECTMLP"',1)[0].encode()
 cooperative_program=(s/'src/backends/metal/cooperative_sole_prefill_shader.h').read_text().split('R"COOPMLP(',1)[1].split(')COOPMLP"',1)[0].encode()
 dual_program=(s/'src/backends/metal/sole_dual_projection_shader.h').read_text().split('R"SOLEDUAL(',1)[1].split(')SOLEDUAL"',1)[0].encode()
 declared=original_program+cooperative_program+dual_program
""".rstrip();assert t.count(a)==1;t=t.replace(a,z)
a="program==(declared if mode in ['on','fault']else b'')";assert t.count(a)==1;t=t.replace(a,"program==(declared if mode in ['on','fault']else original_program+direct_program+dual_program if mode=='dual66'else b'')")
a="assert actual_sole['on']==original_sole+direct and actual_sole['direct64']==actual_sole['on'] and actual_sole['restore']==original_sole";assert t.count(a)==1;t=t.replace(a,"assert actual_sole['on']==original_sole+cooperative_program and actual_sole['direct64']==actual_sole['dual66']==original_sole+direct and actual_sole['restore']==original_sole")
needle=' for mode,value in actual_sole.items():';assert t.count(needle)==1
extra=''' capabilities={}
 for mode in ['off','primary','on','fault','restore','direct64','dual66']:
  path=r/('out/build/garnet-metal-worldsense-cooperative-prefill-'+mode)/'sole_projection_words.bin.prefill-pipelines.tsv';lines=path.read_text().splitlines();rows=[]
  for line in lines:
   name,cooperative,width,threads,memory,limit=line.split('\\t');row=dict(name=name,cooperative=bool(int(cooperative)),thread_width=int(width),max_threads=int(threads),static_threadgroup_bytes=int(memory),device_max_threadgroup_bytes=int(limit));rows.append(row)
   assert row['cooperative']==(mode in ['on','fault'])and row['thread_width']==32 and row['max_threads']>=(128 if name=='base32x32'else256)and row['static_threadgroup_bytes']<=row['device_max_threadgroup_bytes']
  assert len(rows)==(4 if mode in ['on','fault','direct64','dual66']else0)
  if rows:assert {row['name']for row in rows}=={'base32x32','wide16_64x64','wide32_64x32','wordprobe'}
  capabilities[mode]=rows
 (w/'actual-prefill-pipeline-capabilities.json').write_text(json.dumps(dict(roles=capabilities,scope='Actual native Metal PSO properties before learned proof; staticbytes are compiler/device values, not occupancy/bank/performance evidence.'),indent=2)+'\\n')
''';t=t.replace(needle,extra+needle).replace('ALL6BUILDS9NATIVE_TESTS_186ORIGINAL_28MLP_36DUAL_RECORDS_EXACT','ALL7BUILDS9NATIVE_TESTS_186ORIGINAL_28MLP_36DUAL_RECORDS_EXACT');ast.parse(t);(w/'native_controller.py').write_text(t)
print('Seven build roles/native63tests/186+28+36word records plus actual PSO capabilities prepared')
