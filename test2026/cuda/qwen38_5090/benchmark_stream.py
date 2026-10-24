"""Real loopback streaming benchmark. Counts server usage tokens, never SSE chunks."""
import argparse,base64,concurrent.futures,datetime,hashlib,json,math,platform,sys,time,uuid
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from urllib.parse import urlsplit
from probe_reference_http import image_png
from metrics import summarize
from run_logged import source_identity
from sample_vram import Sampler

class TokenEvents:
    def __init__(self):
        self.count=0;self.prompt=None;self.first=None;self.last=None;self.first_batch=None;self.max_batch=0;self.finish_reason=None;self.done=False;self.text=[]
    def feed(self,raw,now):
        if self.done:raise ValueError('Data after completed stream')
        if raw=='[DONE]':self.done=True;return
        data=json.loads(raw)
        if data.get('error'):raise ValueError('Server stream error: '+str(data['error']))
        choices=data.get('choices',[])
        if len(choices)>1:raise ValueError('Exactly one output choice required')
        for choice in choices:
            if choice.get('index',0)!=0:raise ValueError('Unexpected choice index')
            delta=choice.get('delta',{})
            if delta.get('reasoning_content'):raise ValueError('Thinking output is outside this benchmark profile')
            if delta.get('content'):self.text.append(delta['content'])
            if choice.get('finish_reason'):self.finish_reason=choice['finish_reason']
        usage=data.get('usage')
        if usage:
            n=usage.get('completion_tokens');prompt=usage.get('prompt_tokens')
            if isinstance(n,bool) or not isinstance(n,int) or n<self.count:raise ValueError('Nonmonotonic/invalid cumulative token usage')
            if isinstance(prompt,bool) or not isinstance(prompt,int) or prompt<1:raise ValueError('Invalid prompt token usage')
            if self.prompt is not None and prompt!=self.prompt:raise ValueError('Prompt token count changed')
            self.prompt=prompt
            if (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0):raise ValueError('Cached prompt invalidates uncached profile')
            if n>self.count:
                batch=n-self.count;self.max_batch=max(self.max_batch,batch)
                if self.first is None:self.first=now;self.first_batch=batch
                self.last=now;self.count=n
    def result(self,submitted,completed,maximum):
        if not self.done or self.finish_reason not in ('stop','length') or not self.count or self.first is None:
            raise ValueError('Incomplete stream or missing continuous token usage')
        if self.prompt+self.count>4096 or self.count>maximum:raise ValueError('Context/output capacity exceeded')
        if self.count>1 and self.last<=self.first:raise ValueError('No measurable decode interval; usage arrived only at final chunk')
        return {'status':'completed','submitted':submitted,'first_token':self.first,'last_token':self.last,'completed':completed,
            'generated_tokens':self.count,'prompt_tokens':self.prompt,'finish_reason':self.finish_reason,
            'first_arrival_batch_tokens':self.first_batch,'largest_arrival_batch_tokens':self.max_batch,
            'single_token_arrivals':self.max_batch==1,'token_count_source':'server cumulative completion_tokens with continuous_usage_stats',
            'text':''.join(self.text),'scope':'Client-observed token arrival times over loopback HTTP; batched arrivals are retained explicitly'}

def payload(case,key,seed,model,maximum,image_format='png',image_order='text-first'):
    if image_format not in ('png','jpeg') or image_order not in ('text-first','image-first'):
        raise ValueError('Unsupported image format/order')
    prompt='Explain in detail how rain forms, step by step, with enough detail for a science lesson. Request '+key+'.'
    image=None;content=prompt
    if case=='image':
        image=image_png(seed)
        if image_format=='jpeg':
            from io import BytesIO
            from PIL import Image
            encoded=BytesIO()
            with Image.open(BytesIO(image)) as source:source.convert('RGB').save(encoded,format='JPEG',quality=100,subsampling=0)
            image=encoded.getvalue()
        content=[{'type':'text','text':'Describe the colors and shapes in this image in detail, then explain their spatial relationships. Request '+key+'.'},
            {'type':'image_url','image_url':{'url':'data:image/'+image_format+';base64,'+base64.b64encode(image).decode()}}]
        if image_order=='image-first':content.reverse()
    return {'model':model,'messages':[{'role':'user','content':content}],'max_tokens':maximum,'temperature':0,'stream':True,
        'stream_options':{'include_usage':True,'continuous_usage_stats':True},'chat_template_kwargs':{'enable_thinking':False}},image

def request(url,body,root,key,maximum):
    wire=json.dumps(body,separators=(',',':')).encode();(root/(key+'-request.json')).write_bytes(wire)
    state=TokenEvents();submitted=time.perf_counter();result=None
    with (root/(key+'-events.jsonl')).open('x',encoding='utf-8') as events:
        try:
            req=Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=300) as response:
                data=[]
                for line in response:
                    now=time.perf_counter();text=line.decode('utf-8').rstrip('\r\n')
                    if text.startswith('data:'):data.append(text[5:].lstrip(' '))
                    elif not text and data:
                        raw='\n'.join(data);data=[];events.write(json.dumps({'monotonic':now,'data':raw})+'\n');events.flush();state.feed(raw,now)
            result=state.result(submitted,time.perf_counter(),maximum)
        except Exception as error:
            if isinstance(error,HTTPError):
                events.write(json.dumps({'monotonic':time.perf_counter(),'http_status':error.code,'http_error_body':error.read().decode('utf-8',errors='replace')})+'\n');events.flush()
            result={'status':'failed','submitted':submitted,'completed':time.perf_counter(),'error':repr(error)}
        result.update(request_sha256=hashlib.sha256(wire).hexdigest(),key=key)
        (root/(key+'-result.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',required=True);p.add_argument('--engine',required=True,choices=['vllm','sglang','garnet'])
    p.add_argument('--model',default='qwen38-27b-nvfp4');p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1]);p.add_argument('--requests',type=int,default=4);p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--image-format',choices=['png','jpeg'],default='png');p.add_argument('--image-order',choices=['text-first','image-first'],default='text-first')
    p.add_argument('--workload-seed',type=int,default=1);p.add_argument('--max-tokens',type=int,default=128);p.add_argument('--cases',nargs='+',choices=['text','image'],default=['text','image']);a=p.parse_args()
    parsed=urlsplit(a.url)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.path not in ('','/') or parsed.username:raise ValueError('Study requires loopback HTTP')
    if not 1<=a.workload_seed<=100 or not 1<=a.requests<=256 or not 2<=a.max_tokens<=1024 or a.warmups<1 or any(c not in (1,2,4,8) or c>a.requests for c in a.concurrency):raise ValueError('Invalid study profile')
    a.output.mkdir(parents=True,exist_ok=True);run=uuid.uuid4().hex;root=a.output;prefix=run+'-';cohorts=[]
    metadata=json.loads(a.metadata.read_text());manifest={'run_id':run,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'engine':a.engine,
        'status':'running','argv':sys.argv,'source':source_identity(),'engine_metadata':metadata,'engine_metadata_sha256':hashlib.sha256(a.metadata.read_bytes()).hexdigest(),
        'python':platform.python_version(),'platform':platform.platform(),'context_capacity':4096,
        'image_format':a.image_format,'image_order':a.image_order,
        'workload':'Deterministic per-request nonce and one changed image pixel avoid prefix/vision result reuse; matching keys yield identical requests across engines',
        'measurement_scope':'Exploratory unless engine precision/workload/quality/configuration equivalence is independently established; no automatic2x claim'}
    (root/(prefix+'manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sampler=None
    try:
        with urlopen(a.url.rstrip('/')+'/health',timeout=10) as response:
            if response.status!=200:raise ValueError('Engine is not healthy')
        sampler=Sampler(root/(prefix+'vram.jsonl'));manifest['gpu']=sampler.device;sampler.start()
        for case_index,case in enumerate(a.cases):
            for c in a.concurrency:
                sampler.phase='warmup-'+case+'-c'+str(c)
                for wave in range(a.warmups):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                        warm=[]
                        for i in range(c):
                            key='s'+str(a.workload_seed)+'-'+case+'-c'+str(c)+'-warmup-'+str(wave)+'-'+str(i)
                            seed=a.workload_seed*100000+50000+c*1000+wave*10+i
                            body,image=payload(case,key,seed,a.model,a.max_tokens,a.image_format,a.image_order)
                            warm.append(pool.submit(request,a.url.rstrip('/'),body,root,prefix+key,a.max_tokens))
                        for future in warm:
                            result=future.result()
                            if result['status']!='completed':raise RuntimeError('Warmup failed: '+str(result))
                phase=case+'-c'+str(c);sampler.phase=phase;sampler.sample();jobs=[]
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    for i in range(a.requests):
                        key='s'+str(a.workload_seed)+'-'+phase+'-r'+str(i);body,image=payload(case,key,a.workload_seed*100000+case_index*20000+c*1000+i,a.model,a.max_tokens,a.image_format,a.image_order)
                        jobs.append(pool.submit(request,a.url.rstrip('/'),body,root,prefix+key,a.max_tokens))
                    rows=[f.result() for f in jobs]
                sampler.sample();summary=summarize(rows);summary.update(sampler.peak(phase));summary.update(case=case,concurrency=c,
                    single_token_arrivals=all(r['single_token_arrivals'] for r in rows),prompt_tokens=[r['prompt_tokens'] for r in rows],
                    request_keys=[r['key'] for r in rows]);cohorts.append(summary);print(json.dumps(summary),flush=True)
        manifest['status']='completed'
    except Exception as error:
        manifest['status']='failed';manifest['error']=repr(error);raise
    finally:
        try:
            if sampler is not None:sampler.close()
        except Exception as error:
            manifest['status']='failed';manifest['error']=repr(error);raise
        finally:
            manifest['cohorts']=cohorts;manifest['sampling_errors']=sampler.errors if sampler is not None else []
            (root/(prefix+'manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'run_id':run,'cohorts':len(cohorts),'scope':manifest['measurement_scope']}))
if __name__=='__main__':main()
