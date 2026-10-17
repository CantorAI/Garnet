import sys,json,pathlib
sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-memory')
import garnet
x=json.loads(garnet.memory_stats_json())
assert x['task_info_status']==0 and x['metal_recommended_working_set_bytes']>0
for key,value in x.items():
 if key!='scope':assert isinstance(value,int) and value>=0
pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-memory/api-idle.json').write_text(json.dumps(x,indent=2))
print('Native XLang3 memory JSON contract PASS',flush=True)
