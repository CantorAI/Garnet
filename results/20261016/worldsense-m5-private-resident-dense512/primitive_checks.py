from pathlib import Path
import subprocess,json,hashlib,struct,shutil
w=Path(__file__).resolve().parent;results={};payloads={}
for mode in ['off','on']:
 b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-dense512-'+mode)
 with (w/('native-'+mode+'.log')).open('w')as stream:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--output-on-failure'],stdout=stream,stderr=subprocess.STDOUT)
 results[mode]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2));assert p.returncode==0,mode+' native suite failed; log retained'
 data=(b/'dense_decode1024_words.bin').read_bytes();payloads[mode]=data;shutil.copy2(b/'dense_decode1024_words.bin',w/('dense-words-'+mode+'.bin'));offset=0;records=0
 while offset<len(data):
  rows,inner,cols,iteration=struct.unpack_from('<QQQQ',data,offset);offset+=32;assert rows in [1,2]and iteration in [0,1];offset+=rows*cols*4;records+=1
 assert offset==len(data)and records==154
assert payloads['off']==payloads['on'],'Complete output words differ between OFF/ON'
(w/'dense-bit-proof.json').write_text(json.dumps(dict(all154_full_output_records_bitwise_identical=True,shape_cases=77,bytes=len(payloads['off']),sha256=hashlib.sha256(payloads['off']).hexdigest(),scope='Native correctness only, no performance evidence; each record four little-endian uint64 shape/pass values followed by logicalFP32 words'),indent=2))
print('Both native12 suites and all154complete output records bitwise identical PASS',flush=True)
