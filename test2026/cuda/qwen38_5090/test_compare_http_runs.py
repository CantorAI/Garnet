"""Corruption/matching fixtures; no synthetic record is a measured model result."""
import copy,hashlib,json,unittest
from compare_http_runs import verify_run,compare
from metrics import summarize

class ComparisonEvidenceTests(unittest.TestCase):
 def fixture(self):
  key='run-s3-text-c1-r0';wire=b'{"max_tokens":2}';row={'status':'completed','submitted':1.,'first_token':2.,'last_token':3.,'completed':4.,'generated_tokens':2,'prompt_tokens':3,'text':'ab','finish_reason':'length','request_sha256':hashlib.sha256(wire).hexdigest()}
  row.update(first_arrival_batch_tokens=1,largest_arrival_batch_tokens=1,single_token_arrivals=True)
  data={key+'-request.json':wire,key+'-result.json':json.dumps(row).encode()};events=[]
  for n,t in [(1,2.),(2,3.)]:events.append({'monotonic':t,'data':json.dumps({'choices':[{'delta':{'content':'a' if n==1 else 'b'},'finish_reason':'length' if n==2 else None}],'usage':{'prompt_tokens':3,'completion_tokens':n}})})
  events.append({'monotonic':3.1,'data':'[DONE]'});data[key+'-events.jsonl']='\n'.join(json.dumps(e) for e in events).encode()
  cohort=summarize([row]);cohort.update(case='text',concurrency=1,request_keys=[key])
  manifest={'status':'completed','run_id':'run','engine':'garnet','cohorts':[cohort]}
  return manifest,data,key
 def test_replay_rejects_corrupted_request_and_token_counts(self):
  m,d,k=self.fixture();verified=verify_run(m,d.__getitem__);self.assertEqual(verified['cohorts']['text-c1']['metrics']['generated_tokens'],2)
  bad=copy.deepcopy(d);bad[k+'-request.json']=b'{"max_tokens":3}'
  with self.assertRaises(ValueError):verify_run(m,bad.__getitem__)
  saved=json.loads(d[k+'-result.json']);saved['generated_tokens']=3;d[k+'-result.json']=json.dumps(saved).encode()
  with self.assertRaises(ValueError):verify_run(m,d.__getitem__)
 def test_ratios_do_not_imply_quality_or_both_references(self):
  m,d,k=self.fixture();n=verify_run(m,d.__getitem__);r=copy.deepcopy(n);r['engine']='sglang';report=compare(n,[r])
  self.assertTrue(report['cohorts'][0]['identical_request_bytes']);self.assertFalse(report['two_x_established']);self.assertFalse(report['has_vllm_and_sglang'])
  r['cohorts']['text-c1']['requests']['s3-text-c1-r0']['sha256']='different'
  self.assertFalse(compare(n,[r])['cohorts'][0]['identical_request_bytes'])
if __name__=='__main__':unittest.main()
