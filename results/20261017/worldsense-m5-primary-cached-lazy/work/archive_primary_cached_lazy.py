"""WS52 evidence packaging only, AFTER all dispatched workers are terminal."""
from pathlib import Path
import csv,hashlib,json,shutil,subprocess
r=Path('/Users/shawn/CantorAI');repo=r/'Garnet'
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-cached-lazy')
s=r/'out/research/worldsense-primary-cached-lazy/source'
d=repo/'results/20261017'/('worldsense-m5-'+w.name.removeprefix('worldsense-'))
base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def copy(p,t):t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
assert d==repo/'results/20261017/worldsense-m5-primary-cached-lazy'
assert d.name=='worldsense-m5-'+w.name.removeprefix('worldsense-')
assert load(w/'protocol.json')['experiment']=='WS-M5-52'
assert (d/'README.md').read_text().splitlines()[0].startswith('# WS-M5-52 ')
assert (w/'decision.json').exists(),'Terminal decision required'
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
assert (d/'README.md').exists(),'Write reviewed report first'
for p in w.iterdir():
    if p.is_file()and p.suffix in ['.json','.log','.py','.txt','.make','.csv','.cpp','.metal','.ll','.cmake']:
        copy(p,d/p.name)
for name in ['fixture-1-8','fixture-2-64','build-profile','primary-speed','speed','production-speed','confirmation']:
    for p in (w/name).rglob('*'):
        if p.is_file()and '__pycache__'not in p.parts and p.suffix in ['.json','.log','.py','.txt','.csv','.f32','.cmake']:
            copy(p,d/name/p.relative_to(w/name))
for mode in ['off','primary','on','fault']:
    copy(r/('out/build/garnet-metal-worldsense-primary-cached-lazy-'+mode)/'primary_projection_words.bin',d/('native-'+mode+'-projection-words.bin'))

manifest=load(w/'starting-source-manifest.json')
assert manifest==load(w/'correctness-source-manifest.json')
refs=[base+':'+rel for rel in manifest]
raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode())
offset=0;overlay=[]
for rel,digest in manifest.items():
    end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
    if header[-1]==b'missing':old=None
    else:
        assert header[1]==b'blob';size=int(header[2]);old=raw[offset:offset+size];offset+=size+1
    assert sha(s/rel)==digest,rel
    if old!=(s/rel).read_bytes():copy(s/rel,d/'source'/rel);overlay.append(rel)
assert offset==len(raw)
(d/'source-reconstruction.json').write_text(json.dumps(dict(base_commit=base,base_scope='Actual canonical f21c1a258 seed verified before WS52; later repository archive/docs commits had identical consumed source.',overlay_files=overlay,source_manifest='starting-source-manifest.json',full_closure_files=len(manifest)),indent=2))

calls=[];routes=[];requests=[];stage_counts={}
for stage,folder in [('resource',d),('primary-speed',d/'primary-speed'),('speed',d/'speed'),('production-speed',d/'production-speed'),('confirmation',d/'confirmation')]:
    count=0
    for p in sorted(folder.glob('*.json')):
        if not p.name.startswith(('control-','candidate-')):continue
        data=load(p)
        if not isinstance(data,dict)or 'items'not in data or'search_plans'not in data:continue
        for category,items in [('frame',data['items']),('query',data['search_plans'])]:
            for index,item in enumerate(items):
                common=dict(artifact=str(p.relative_to(d)),artifact_sha256=sha(p),stage=stage,worker=data['label'],category=category,sample_index=index,warmup=item.get('warmup',''),repetition=item.get('repetition',''))
                requests.append(dict(common,native_call_seconds=item.get('native_call_seconds',''),observation_id=item.get('observation_id',''),query=item.get('query','')))
                responses=item['result'].get('model_response',{})if category=='frame'else{'query_plan':item['result'].get('benchmark_model_response',{})}
                for phase,c in responses.items():
                    if not isinstance(c,dict)or'model_id'not in c:continue
                    t=c.get('timings',{});total,first,n=(c.get(k)for k in ['total_seconds','ttft_seconds','generated_tokens'])
                    rate=(n-1)/(total-first)if all(isinstance(v,(float,int))for v in [n,total,first])and total>first and n>1 else''
                    identity=dict(common,phase=phase,model_id=c['model_id'])
                    calls.append(dict(identity,status=c.get('status',''),input_tokens=t.get('input_tokens',''),generated_tokens=n,ttft_seconds=first,total_seconds=total,derived_decode_tokens_per_second=rate,input_prompt_sha256=c.get('input_prompt_sha256',''),image_sha256=item.get('image_sha256',''),skill_sha256=item.get('skill_sha256',''),eligible_for_speed=stage!='resource'and not item.get('warmup',True)))
                    route=dict(identity,backend=c.get('backend',''),serving_source=t.get('serving_source',''))
                    for key in ['primary_storage_after','primary_encoded_delta','memory_after_decode']:
                        route[key+'_available']=key in t
                        route.update({key+'_'+k:v for k,v in t.get(key,{}).items()})
                    routes.append(route);count+=1
    if count:stage_counts[stage]=count
for name,rows in [('calls.csv',calls),('execution-routes.csv',routes),('skill-requests.csv',requests)]:
    with(d/name).open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
(d/'export-scope.json').write_text(json.dumps(dict(stage_calls=stage_counts,raw_calls=len(calls),unique_native_requests=len(requests),coverage='Every returned WorldSense benchmark model call, first/warm/failure, exactly once in each call/routeCSV. One row per parent native frame/query request in skill-requestsCSV; never sum parent Skill wall time across model phases.',decode='(generated-1)/(total-TTFT): host/stop/post-loop and possible extraEOS work included; not isolated GPU throughput.',memory='Post-decode sampled maxima, not peaks. Native owner accounting is not physical residency; encoded deltas are not completed GPU counts. ActualWS10 old memory telemetry remains unavailable.',statistics='Retain original stage/worker/pair grouping; Matched primary-only stage adds prospective5%VLM TTFT/2%memory preservation, other original gates unchanged. Exported rows are not independent confirmation. Resource timings ineligible for speed.'),indent=2))
for name in ['archive_primary_cached_lazy.py','verify_primary_cached_lazy.py']:
    p=Path(__file__).parent/name
    if p.exists():copy(p,d/'work'/name)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2))
print(json.dumps(dict(archive=str(d),files=len(load(d/'file-manifest.json')),source_overlay=len(overlay),stage_calls=stage_counts,requests=len(requests))))
