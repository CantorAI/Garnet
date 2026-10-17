from pathlib import Path
import json,shutil
w=Path(__file__).resolve().parent;d=json.loads((w/'memory-analysis.json').read_text());pairs={}
for model in ['Qwen3.5-4B','Qwen3-1.7B']:
 a=next(o for o in d['observations']if o['worker']=='control-1'and o['model']==model and not o['warmup']);b=next(o for o in d['observations']if o['worker']=='candidate-1'and o['model']==model and not o['warmup'])
 changes={k:a['logical_native_categories'].get(k,0)-b['logical_native_categories'].get(k,0)for k in set(a['logical_native_categories'])|set(b['logical_native_categories'])};assert sum(changes.values())==3036086272
 graphics=lambda o:next(c for c in o['os_categories']if c['category']=='IOAccelerator (graphics)')
 assert graphics(a)['dirty_bytes']==graphics(b)['dirty_bytes']==17829150720
 pairs[model]={'followup_logical_category_control_minus_candidate':changes,'control_graphics_category':graphics(a),'candidate_graphics_category':graphics(b),'scope':'Paused diagnostic snapshots, not performance/causal estimates or per-buffer physical attribution'}
(w/'diagnostic-conclusion.json').write_text(json.dumps(pairs,indent=2));r=Path('/Users/shawn/CantorAI/Garnet');j=r/'docs/research/metal_optimization_research_log.md';heading='## WS-M5-35 — Global ownership and OS graphics accounting (diagnostic)';assert heading not in j.read_text()
with j.open('a')as f:f.write('\n\n'+heading+'\n\n'+(w/'archive-body.txt').read_text()+'\nArchive: results/20261016/worldsense-m5-physical-memory.\n')
old=w.parent/'worldsense-cached-matrix-metadata'
s=(old/'archive_finalize.py').read_text().replace('worldsense-m5-cached-matrix-metadata','worldsense-m5-physical-memory').replace('worldsense-cached-matrix-metadata','worldsense-physical-memory')
a=s.index('fpatch=[];fchanges=[]');b=s.index("for cohort in ['speed','confirmation']",a);s=s[:a]+"shutil.copytree(w/'observations',d/'observations',dirs_exist_ok=True)\nfchanges=[]\n"+s[b:]
s=s.replace("fault=source.parent/'fault-source';",'');(w/'archive_finalize.py').write_text(s)
s=(old/'verify_archive.py').read_text().replace('worldsense-m5-cached-matrix-metadata','worldsense-m5-physical-memory');s+="""
observations=list(root.glob('observations/*/*/observation.json'))
assert len(observations)==8
for p in observations:
 o=json.loads(p.read_text());folder=p.parent;assert hashlib.sha256((folder/'native-inventory.json').read_bytes()).hexdigest()==o['native_boundary']['inventory']['sha256']
 for tool,t in o['tools'].items():
  assert t['returncode']==0
  for stream in ['stdout','stderr']:assert hashlib.sha256((folder/(tool+'.'+stream+'.txt')).read_bytes()).hexdigest()==t[stream+'_sha256']
print('All8global inventories andOSrawtoolcaptures SHA verified')
""";(w/'verify_archive.py').write_text(s)
s=(old/'post_finalize.py').read_text().replace('worldsense-m5-cached-matrix-metadata','worldsense-m5-physical-memory').replace('worldsense-cached-matrix-metadata','worldsense-physical-memory').replace("['off','on','fault']","['off','on']");s+="""
assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain','--untracked-files=all','--','results/20261016/worldsense-m5-cached-matrix-metadata'],text=True)
""";(w/'post_finalize.py').write_text(s)
s=(old/'export_execution_routes.py').read_text();s=s.replace("fields+=['backend'", "fields+=['os_memory_capture_id','diagnostic_paused']\nfields+=['backend'");s=s.replace("row.update(backend=", "row.update(os_memory_capture_id=timings.get('os_memory_observation',{}).get('capture_id',''),diagnostic_paused='os_memory_observation'in timings)\n     row.update(backend=");s=s.replace('All first/warm cohort calls.','All8paused diagnostic calls; no performance claim.');(w/'export_execution_routes.py').write_text(s)
print('Diagnostic conclusion and exact archive destinations prepared')
