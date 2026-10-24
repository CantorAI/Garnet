"""Explicit idle-only cache reset evidence for a loopback research server."""
import json,math,re,time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen

FAMILIES={
 'running':('vllm:num_requests_running',),
 'waiting':('vllm:num_requests_waiting',),
 'queries':('vllm:mm_cache_queries_total','vllm:mm_cache_queries'),
 'hits':('vllm:mm_cache_hits_total','vllm:mm_cache_hits'),
 'process_start':('process_start_time_seconds',),
}

def parse_metrics(raw):
 values={}
 for line in raw.decode('utf8').splitlines():
  if not line or line.startswith('#'):continue
  match=re.fullmatch(r'([a-zA-Z_:][a-zA-Z_0-9:]*)(?:\{.*\})?\s+(\S+)(?:\s+\S+)?',line)
  if match:
   try:value=float(match[2])
   except ValueError:continue
   values[match[1]]=values.get(match[1],0)+value
 result={}
 for field,names in FAMILIES.items():
  found=[name for name in names if name in values]
  if len(found)!=1:raise ValueError('Missing or ambiguous metric: '+field)
  value=values[found[0]]
  if not math.isfinite(value) or value<0:raise ValueError('Invalid metric: '+field)
  result[field]=value
 return result

def check_url(url):
 parsed=urlsplit(url)
 if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
  raise ValueError('Literal loopback research URL required')
 return url.rstrip('/')

def snapshot(url,path):
 if Path(path).exists():raise FileExistsError('Preserve prior metrics snapshot')
 with urlopen(check_url(url)+'/metrics',timeout=15) as response:
  if response.status!=200:raise ValueError('Metrics endpoint failed')
  raw=response.read()
 Path(path).write_bytes(raw)
 return parse_metrics(raw)

def reset_idle_caches(url,output,timeout=120,model='qwen38-27b-int4'):
 url=check_url(url);out=Path(output);out.mkdir(exist_ok=False)
 report={'status':'waiting_idle','reset_responses':[],
  'scope':'Idle metric observation, installed processor/transport and encoder reset acknowledgements, raw counters. HTTP acknowledgement alone is not a universal cache-success proof.'}
 path=out/'reset.json'
 def save():path.write_text(json.dumps(report,indent=2))
 try:
  with urlopen(url+'/v1/models',timeout=15) as response:models=response.read()
  (out/'models.json').write_bytes(models)
  if not any(row.get('id')==model for row in json.loads(models).get('data',[])):raise ValueError('Actual server model mismatch')
  deadline=time.monotonic()+timeout;attempt=0
  while True:
   before=snapshot(url,out/('idle-'+str(attempt)+'.prom'));attempt+=1
   if before['running']==before['waiting']==0:break
   if time.monotonic()>=deadline:raise TimeoutError('Server did not become idle; no reset issued')
   time.sleep(1)
  report['idle_before']=before;save()
  for endpoint in ('/reset_mm_cache','/reset_encoder_cache'):
   with urlopen(Request(url+endpoint,data=b'',method='POST'),timeout=60) as response:
    body=response.read();status=response.status
   (out/(endpoint[1:]+'-response.bin')).write_bytes(body)
   report['reset_responses'].append({'endpoint':endpoint,'status':status,'monotonic':time.monotonic()});save()
   if status!=200:raise ValueError('Reset endpoint did not acknowledge')
  report['after_reset']=snapshot(url,out/'after-reset.prom')
  if report['after_reset']['running'] or report['after_reset']['waiting']:raise ValueError('Requests appeared during reset')
  if report['after_reset']['process_start']!=before['process_start']:raise ValueError('Server process changed')
  report['status']='reset_acknowledged';save();return report
 except Exception as error:
  report['status']='failed';report['error']=repr(error);save();raise

def verify_after_run(url,output,before):
 out=Path(output);after=snapshot(url,out/'after-run.prom')
 result={'before':before,'after':after,'query_delta':after['queries']-before['queries'],'hit_delta':after['hits']-before['hits'],
  'scope':'Observed processor-cache counter change plus separate encoder-reset acknowledgement and unique fixture evidence. Not a hardware-cache residency claim.'}
 (out/'counter-delta.json').write_text(json.dumps(result,indent=2))
 if after['process_start']!=before['process_start']:raise ValueError('Server restarted during trial')
 if result['query_delta']<=0 or result['hit_delta']!=0:raise ValueError('Expected measured/warmup queries and zero new processor-cache hits')
 return result
