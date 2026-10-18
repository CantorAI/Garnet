"""Prepare complete groups8 source/native/counter evidence export, no inference."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt';t=(f/'archive_ws71.py').read_text().replace('WS71','WS72').replace('WS-M5-71','WS-M5-72').replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('worldsense-m5-minimal-full-prefill','worldsense-m5-primary-decode-groups8').replace('ws71','ws72')
t=t.replace("w.parent/'worldsense-primary-core-isolation'","w.parent/'worldsense-minimal-full-prefill'")
t=t.replace("['primary_projection_words.bin','primary_core_projection_words.bin','primary_core_projection_words.bin.summary.json']","['primary_projection_words.bin','primary_core_projection_words.bin','primary_core_projection_words.bin.summary.json','primary_decode_groups8_words.bin','primary_decode_groups8_words.bin.summary.json']")
t=t.replace('calls=[];parents=[];routes=[];memory=[];core_rows=[];attempts=[];counts={}','calls=[];parents=[];routes=[];memory=[];core_rows=[];groups8_rows=[];attempts=[];counts={}')
needle='    core_rows.append(dict(identity,counter_telemetry_available=available,**derived))'
after=needle+"\n    groups8_rows.append(dict(identity,counter_telemetry_available=available,**{key:(t['primary_storage_after'][key]-t['primary_storage_before'][key]if available else None)for key in ['decode_groups8_encoded','head_groups8_encoded']}))";assert t.count(needle)==1;t=t.replace(needle,after)
t=t.replace("('core-dispatch-deltas.csv',core_rows)","('core-dispatch-deltas.csv',core_rows),('groups8-dispatch-deltas.csv',groups8_rows)")
t=t.replace('core_delta_rows=len(core_rows),counter_available_rows','core_delta_rows=len(core_rows),groups8_delta_rows=len(groups8_rows),counter_available_rows')
t=t.replace('source_files=1721','source_files=1723')
t=t.replace('Valid rejection: textTTFT and4Bderiveddecode fail unchanged2%gates despite favorable observed4BTTFT/total/wholeSkill. No later stages/promotion.','Valid rejection: bothmodels TTFT/total/deriveddecode andwholeSkill fail unchanged2%gates; no laterstages/promotion.')
t=t.replace('and len(core_rows)==32 and sum','and len(core_rows)==len(groups8_rows)==32 and sum')
ast.parse(t);p=f/'archive_ws72.py';assert not p.exists();p.write_text(t);print('WS72 archiver ready:1723source/18canonical overlays,native60/full proofs,32pipelinecalls andcore/groups8deltaCSV. No copy/inference yet.')
