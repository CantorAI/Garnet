from pathlib import Path
import json,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved-text';old=b/'work/worldsense-direct-interleaved-prefill'
for name in ['restore_normal.py','wait_cantor.py','cases.json']:shutil.copy2(old/name,w/name)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
roles="['off','primary','on','fault','restore','direct64']"
t=(old/'build.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved-text').replace("['off','primary','on','fault','restore']",roles).replace("['on','fault','restore']","['on','fault','restore','direct64']").replace("['on','fault']else'OFF'","['on','fault','direct64']else'OFF'")
needle="'-DGARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL='+('ON'if mode in ['on','fault','direct64']else'OFF'),";assert t.count(needle)==1;t=t.replace(needle,needle+"'-DGARNET_METAL_SOLE_INTERLEAVED_TEXT='+('ON'if mode in ['on','fault']else'OFF'),");ast.parse(t);(w/'build.py').write_text(t)
t=(old/'native_controller.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved-text').replace("['off','primary','on','fault','restore']",roles).replace("('on'if mode=='restore'else mode)","('on'if mode in ['restore','direct64']else mode)")
t=t.replace("assert actual_sole['on']==original_sole+direct and actual_sole['restore']==original_sole","assert actual_sole['on']==original_sole+direct and actual_sole['direct64']==actual_sole['on'] and actual_sole['restore']==original_sole")
t=t.replace("'all_five_builds_FP32_words_identical':True","'all_five_builds_FP32_words_identical':True,'all_six_builds_FP32_words_identical':True").replace('all_five_builds_FP32_words_identical=True','all_five_builds_FP32_words_identical=True,all_six_builds_FP32_words_identical=True')
needle=' actual_sole={mode:';assert t.count(needle)==1
extra=''' # Independently parse and compare EVERY new projection record; retain lossless compressed complete bytes.
 import struct,gzip
 text_paths={mode:r/('out/build/garnet-metal-worldsense-sole-interleaved-text-'+mode)/'sole_text_projection_words.bin'for mode in ['off','primary','on','fault','restore','direct64']}
 def sha_file(path):
  h=hashlib.sha256()
  with path.open('rb')as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
  return h.hexdigest()
 reference=text_paths['off'];expected_shapes=[(rows,2048,columns,passid)for columns in [2048,1024,151936]for rows in [1,2,3,16,32,64,128]for passid in [0,1]]
 records=[]
 with reference.open('rb')as f:
  for shape in expected_shapes:
   assert struct.unpack('4Q',f.read(32))==shape
   count=shape[0]*shape[2]*4;data=f.read(count);assert len(data)==count;records.append(dict(shape=shape,bytes=count,sha256=hashlib.sha256(data).hexdigest()))
  assert not f.read(1)
 for mode,path in text_paths.items():
  with path.open('rb')as f,reference.open('rb')as g:
   while True:
    one=f.read(4*1024*1024);two=g.read(4*1024*1024);assert one==two,(mode,'new full output bytes differ')
    if not one:break
  packed=w/('native-'+mode+'-sole-text-projection-words.bin.gz')
  with path.open('rb')as f,gzip.GzipFile(filename=str(packed),mode='wb',mtime=0)as z:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):z.write(chunk)
  with gzip.open(packed,'rb')as f,path.open('rb')as g:
   while True:
    one=f.read(4*1024*1024);two=g.read(4*1024*1024);assert one==two
    if not one:break
 (w/'sole-text-word-comparison.json').write_text(json.dumps(dict(records=records,record_count=42,all_six_builds_complete_FP32_words_identical=True,uncompressed_bytes=reference.stat().st_size,uncompressed_sha256=sha_file(reference),archives={mode:dict(bytes=(w/('native-'+mode+'-sole-text-projection-words.bin.gz')).stat().st_size,sha256=sha_file(w/('native-'+mode+'-sole-text-projection-words.bin.gz')))for mode in text_paths},scope='Complete output comparisons plus original FP64 per-element bounds; gzip lossless audit verified. Not learned inference or timing evidence.'),indent=2)+'\\n')
'''
t=t.replace(needle,extra+needle).replace('Native8tests','Native9tests').replace('ALL5BUILDS8NATIVE_TESTS_186ORIGINAL_28SOLE_PROJECTION_RECORDS_EXACT','ALL6BUILDS9NATIVE_TESTS_186ORIGINAL_28MLP_42NEW_TEXT_RECORDS_EXACT');ast.parse(t);(w/'native_controller.py').write_text(t)
print('WS65 six build roles and nine native tests prepared; original186+28 and complete new42 output records')
