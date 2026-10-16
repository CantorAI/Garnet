"""Summarize separate native initialization/first-use/RSS from completed ABBA jobs.

Arguments: text jobs directory, frame jobs directory, output JSON file.
"""
import json,statistics,sys
from pathlib import Path
text,frame,output=map(Path,sys.argv[1:4]);groups=[]
for path in (text,frame):assert json.loads((path.parent/'exit.json').read_text())['exit_code']==0
records=json.loads((text/'native_summary.json').read_text())['records']
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
 for batch in (1,2):
  group=dict(model=model,batch=batch)
  for backend in ('garnet','garnet-control'):
   values=[v for v in records if v['folder'].startswith(model+'-b'+str(batch)+'-'+backend+'-') and v['request']==0]
   if backend=='garnet':values=[v for v in values if 'control' not in v['folder']]
   assert len(values)==2;rss=[]
   for value in values:
    log=(text/value['folder']/'output.log').read_text()
    process=next(json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('matched_process '));assert process['exit_code']==0
    rss.append(process['peak_rss_bytes']/2**30)
   group[backend]=dict(initialization_seconds=statistics.mean(v['initialization_seconds'] for v in values),first_use_ttft_seconds=statistics.mean(v['ttft_seconds'] for v in values),peak_child_rss_gib=statistics.mean(rss))
  groups.append(group);print(json.dumps(group),flush=True)
records=json.loads((frame/'native_frame_summary.json').read_text())['records']
for batch in (1,2):
 group=dict(model='Qwen3.5-0.8B',batch=batch)
 for backend in ('garnet','garnet-control'):
  values=[v for v in records if v['folder'].startswith('Qwen3.5-0.8B-b'+str(batch)+'-'+backend+'-') and v['request']==0]
  if backend=='garnet':values=[v for v in values if 'control' not in v['folder']]
  assert len(values)==2
  group[backend]=dict(initialization_seconds=statistics.mean(v['initialization_seconds'] for v in values),first_use_ttft_seconds=statistics.mean(v['ttft_seconds'] for v in values))
 groups.append(group);print(json.dumps(group),flush=True)
output.write_bytes((json.dumps(dict(groups=groups,memory_scope='Individual native text/coder child high-water RSS, not unique GPU memory'),indent=2)+'\n').encode())
