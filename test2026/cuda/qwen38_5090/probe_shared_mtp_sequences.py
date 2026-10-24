"""Untimed diagnosis after historical cross-profile VLM gate rejection."""
import argparse,json,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('study',type=Path);parser.add_argument('output',type=Path)
parser.add_argument('--reuse-verified-head',action='store_true');parser.add_argument('--joined-mlp',action='store_true');a=parser.parse_args();root,out=a.study,a.output
sys.path.insert(0,str(root/'Garnet/test2026/cuda/qwen38_5090'))
from validate_native_mtp_http import post,compare_greedy_pair,validate_worker_profile
from benchmark_stream import payload
from urllib.request import urlopen
out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous diagnostic')
url='http://127.0.0.1:18082'
with urlopen(url+'/health',timeout=5) as response:health=json.load(response)
validate_worker_profile(health,3,4,'direct',4,128,True,True if a.reuse_verified_head else False if a.joined_mlp else None,True if a.joined_mlp else None)
report={'status':'checking','health':health,'rows':[],
 'scope':'Untimed same-profile target versus MTP full token comparisons and VLM semantic diagnosis after historical T16 regression rejection. Does not waive historical gate, qualify broad precision or admit benchmarking.'}
def save():path.write_text(json.dumps(report,indent=2))
save()
try:
 for key,want in [('red','red'),('green','green'),('blue','blue'),('bottom-blue','blue')]:
  body=json.loads((root/'results/native-mtp3-http'/(key+'-request.json')).read_text())
  teacher=post(url,dict(body,study_target_only=True),out,key+'-target');trial=post(url,body,out,key+'-mtp')
  row=compare_greedy_pair(key,teacher,trial)
  texts=[r['choices'][0]['message']['content'].strip().lower().rstrip('.') for r in (teacher,trial)]
  row['semantic_pass']=texts==[want,want]
  report['rows'].append(row);save()
  assert row['passed'] and row['semantic_pass'],'Same-profile VLM MTP mismatch: '+key
 for index,case in enumerate(('text','image')):
  for i in range(8):
   key='s3-'+case+'-c1-r'+str(i)
   body,_=payload(case,key,300000+index*20000+1000+i,'qwen38-27b-nvfp4',128,'jpeg','image-first','binary-grid');body['stream']=False
   teacher=post(url,dict(body,study_target_only=True),out,key+'-target');trial=post(url,body,out,key+'-mtp')
   row=compare_greedy_pair(key,teacher,trial);report['rows'].append(row);save()
   print('SHARED_MTP_DIAGNOSTIC',key,row['passed'],flush=True)
   assert row['passed'] and row['actual_generated_tokens']==128,'Same-profile full128 MTP mismatch: '+key
 report['status']='diagnostic_passed';print('SHARED_MTP_TWENTY_SAME_PROFILE_PAIRS_PASS; historical gate remains unresolved; no timing',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:save()
