"""Audit every numeric/clock record and regenerate source/fixture transforms without GPU execution."""
from pathlib import Path
import hashlib,json,math,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text());assert len(proof['jobs'])==1
trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==0 and not proof['full_goal_met'];jobs=root/'logs'/trial/'jobs'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256'],name
meta=json.loads((jobs/'metadata.json').read_text());sources={}
for absolute,sha in meta['source_sha256'].items():
 name=Path(absolute).name;matches=[p for p in (jobs/'source-snapshot').rglob(name) if p.is_file() and digest(p)==sha];assert len(matches)==1;assert name not in sources;sources[name]=matches[0]
factory=runpy.run_path(str(sources['make_delta_value_tiles.py']));context=sources['metal_context.mm'].read_text();assert factory['shader'](context).encode()==(jobs/'delta-value-tiles.metal').read_bytes() and digest(jobs/'delta-value-tiles.metal')==meta['generated_shader_sha256']
assert '-DGARNET_METAL_NATIVE_BFLOAT_BODY' not in sources['flags.make'].read_text()
prepare=sources['prepare_parallel_delta_probe.py'].read_text()
for old,new in [('for chunk in (16,32,64,80):','for chunk in (1,2,16,32,64,80):'),('((1,80),(2,32))','((1,1),(1,80),(2,1),(2,80))'),('active[:,-1]=0','active[:,-1]=0 if chunk>1 else 1')]:assert prepare.count(old)==1;prepare=prepare.replace(old,new)
prefix=(jobs/'prepare.executed.py').read_text().splitlines()[:2];assert prefix[0]=='import sys' and prefix[1].startswith('sys.path.insert(0,')
assert ('\n'.join(prefix)+'\n'+prepare).encode()==(jobs/'prepare.executed.py').read_bytes() and digest(jobs/'prepare.executed.py')==meta['generated_prepare_sha256']
stages=json.loads((jobs/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile','prepare','numeric','timing'] and all(s['exit_code']==0 for s in stages)
binary='logs/'+trial+'/jobs/delta_value_tiles_gpu';assert proof['omitted'][binary]['sha256']==meta['binary_sha256']
assets=json.loads((jobs/'assets/manifest.json').read_text());cases=assets['cases'];assert len(cases)==104 and assets['checkpoint_sha256']==meta['checkpoint_sha256'];learned=[c for c in cases if c['name'].startswith('real-layer')];assert len(learned)==8
assert {(c['batch'],c['chunk']) for c in learned}=={(b,c) for b in (1,2) for c in (1,80)}
for case in cases:
 for key in ('prepared','state','active'):
  name='logs/'+trial+'/jobs/assets/'+case[key];assert proof['omitted'][name]['sha256']==case[key+'_sha256']
numeric=[json.loads(line) for line in (jobs/'numeric.log').read_text().splitlines()];timing=[json.loads(line) for line in (jobs/'timing.log').read_text().splitlines()]
for records in (numeric,timing):
 pipelines=[r for r in records if r['kind']=='pipeline'];assert len(pipelines)==3 and {r['width'] for r in pipelines}=={16,32,64};assert all(r['execution_width']==32 and r['maximum_threads']>=128 for r in pipelines)
checks=[r for r in numeric if r['kind']=='check'];assert len(checks)==1872;seen=set();words=0
byname={c['name']:c for c in cases};assert len(byname)==104
for row in checks:
 case=byname[row['name']];b,c,h=case['batch'],case['chunk'],case['heads'];padding=row['padding'];assert padding in (0,4) and row['step'] in (0,1,2) and row['width'] in (16,32,64) and row['original_bits_equal']
 key=(row['name'],padding,row['step'],row['width']);assert key not in seen;seen.add(key)
 assert row['output_words']==b*c*(h*128+padding) and row['state_words']==b*(h*128*128+2*padding);words+=row['output_words']+row['state_words']
assert words==553986648 and numeric[-1]==json.loads((jobs/'numeric-summary.json').read_text()) and numeric[-1]['fp32_words_checked']==words and numeric[-1]['profiles']==208 and numeric[-1]['checks']==1872
samples=[r for r in timing if r['kind']=='timing'];assert len(samples)==6400;seen=set()
for row in samples:
 assert row['name'] in {c['name'] for c in learned} and row['width'] in (32,64) and isinstance(row['null'],bool) and row['round'] in range(10) and row['window'] in range(4) and row['sample'] in range(5) and row['dispatches']==16
 assert row['candidate']==(row['window'] in (1,2)) and row['actual_width']==(row['width'] if row['candidate'] and not row['null'] else 16)
 assert all(math.isfinite(row[k]) and row[k]>0 for k in ('gpu_ms','wall_ms'));key=(row['name'],row['width'],row['null'],row['round'],row['window'],row['sample']);assert key not in seen;seen.add(key)
assert timing[-1]==json.loads((jobs/'timing-summary.json').read_text()) and timing[-1]['profiles']==8 and timing[-1]['timing_samples']==6400
result=dict(verified_files=len(proof['files']),trial=trial,source_files=len(sources),numeric_profiles=208,original_word_comparisons=words,numeric_checks=1872,learned_numeric_profiles=16,carried_steps=3,timing_profiles=8,timing_samples=6400,generated_shader_and_preparation_reproduced=True,all_original_bits_equal=True,scope='Component source/record audit only; individual original/width32/64 GPU execution recorded, no native/model speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
