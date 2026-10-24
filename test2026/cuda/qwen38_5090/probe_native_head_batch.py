"""Replay saved B2 normalized hidden through B1/B2 heads; no decoder/timing."""
import hashlib,json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.getcwd())
import garnet
from test2026.cuda.qwen38_5090.run_logged import source_identity

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous head diagnostic')
saved=root/'results/native-batch2-prefill-controls'
prior=json.loads((saved/'report.json').read_text(encoding='utf8'))
assert prior['within_batch_isolation_pass']
report={'status':'starting','source':source_identity(),'cases':[],
 'prior_report_sha256':hashlib.sha256((saved/'report.json').read_bytes()).hexdigest(),
 'scope':'Head-only replay of identical saved normalized hidden at B1/B2, native W4A4 and existing dequantized packed-head path. No decoder quality, kernel oracle, batch admission or timing claim.'}
io=garnet.cuda_tensor_io()
try:
 io.open(str(root/'models/Qwen3.8-27B-NVFP4'))
 for group in (0,4):
  raw=(saved/('group-'+str(group)+'-repeat-0-normalized.f32')).read_bytes()
  assert len(raw)==40960
  values=list(struct.unpack('<10240f',raw));full=io.from_host(values,[2,5120])
  rows=[io.from_host(values[i*5120:(i+1)*5120],[1,5120]) for i in range(2)]
  entry={'group':group,'normalized_sha256':hashlib.sha256(raw).hexdigest()}
  for native in (True,False):
   io.enable_nvfp4_head(native)
   full_ids=list(io.predict(full));single_ids=[list(io.predict(row))[0] for row in rows]
   repeated_ids=list(io.predict(full))
   entry['native' if native else 'dequantized']={'full_ids':full_ids,'single_ids':single_ids,
       'repeat_ids':repeated_ids,'same_input_shape_invariant':full_ids==single_ids,'repeat_exact':full_ids==repeated_ids}
  original=next(c for c in prior['cases'] if c['group']==group and c['repeat']==0)
  entry['original_full_ids']=original['first_prediction_ids']
  entry['native_replays_original']=entry['native']['full_ids']==entry['original_full_ids']
  report['cases'].append(entry);path.write_text(json.dumps(report,indent=2));print('HEAD_BATCH_DIAGNOSTIC',json.dumps(entry),flush=True)
  assert entry['native_replays_original'],'Saved hidden did not reproduce original head predictions'
  assert all(entry[m]['repeat_exact'] for m in ('native','dequantized')),'Head repeat instability'
 report['status']='diagnostic_completed';report['io_memory']=json.loads(io.memory_json())
except Exception as error:
 report['status']='failed';report['error']=repr(error);raise
finally:
 path.write_text(json.dumps(report,indent=2));io.close()
