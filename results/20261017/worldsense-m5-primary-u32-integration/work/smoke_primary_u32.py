"""Normal WorldSense smoke after WS40 selection, using persisted video evidence."""
from pathlib import Path
import datetime, hashlib, json, runpy, time
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-u32-integration')
r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
selection=json.loads((w/'production-selection.json').read_text());assert selection['promoted']
start=int(datetime.datetime.fromisoformat(selection['selection_utc']).timestamp()*1000)
api=runpy.run_path(str(w.parent/'worldsense-http.py'));token=api['login']()
def request(path,data=None,method='GET'):
    return api['request'](path,data=data,method=method,token=token)
def model_calls(value):
    result=[]
    if isinstance(value,dict):
        if 'model_id'in value and 'backend'in value:result.append(value)
        else:
            for v in value.values():result.extend(model_calls(v))
    elif isinstance(value,list):
        for v in value:result.extend(model_calls(v))
    return result
record=dict(start_ms=start,scope='Normal production API integration smoke, not paired performance evidence. Observation metrics cover only successful persisted calls; exclude loading/retries/queueing.')
record['mounted']=request('/api/devices/mounted')
assert len(record['mounted'])==4 and all(v['online']for v in record['mounted'].values())and all('/test-media/'in k for k in record['mounted'])
record['cantor_status']=request('/api/system/cantor/status')
for attempt in range(24):
    end=int(time.time()*1000)
    observations=request('/api/skill-lab/observations?start_ms='+str(start)+'&end_ms='+str(end)+'&limit=200')
    calls=model_calls([i.get('diagnostic',{})for i in observations['items']])
    native=[c for c in calls if c.get('status')=='ok' and 'primary_u32_metadata'in c.get('backend','')]
    if {'Qwen3.5-4B','Qwen3-1.7B'}<={c['model_id']for c in native}:break
    time.sleep(10)
else:raise RuntimeError('New normal observations did not prove both selected native models within four minutes')
assert all(Path(c['timings']['serving_source']).resolve()==r/'Garnet/runtime/metal_serving.py'for c in native)
record['observations']=observations
record['native_calls']=len(native)
record['states']=request('/api/skill-lab/states?limit=100')
record['events']=request('/api/events/list?pending_only=false')
record['searches']=[]
for query in ['a car','a purple elephant flying over a swimming pool']:
    before=time.perf_counter();result=request('/api/search',dict(query=query,limit=10),method='POST')
    record['searches'].append(dict(query=query,wall_seconds=time.perf_counter()-before,result=result))
    assert result['rerank_status']=='complete'
    assert bool(result['items'])==(query=='a car')
assert hashlib.sha256((n/'libgarnet.dylib').read_bytes()).hexdigest()==selection['selected_sha256']
record.update(passed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),both_native_models_in_new_normal_observations=True,positive_search_returns_results=True,absent_elephant_returns_zero=True)
(w/'normal-app-smoke.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items()if k not in ['observations','states','events','searches','mounted','cantor_status']},indent=2))
