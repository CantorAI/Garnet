from pathlib import Path
import json,hashlib,collections,csv
w=Path(__file__).resolve().parent;p=json.loads((w/'inventory-1.json').read_text());base=json.loads((w/'analysis.json').read_text());assert base['all12_full_outputs_prompts_configs_and_inputs_equal_to_WS10_confirmation']
rows=[];buffer_rows=[];reference_rows=[];calls=[]
for item in p['items']:
 for phase,c in item['result']['model_response'].items():
  if isinstance(c,dict)and c.get('status')=='ok':calls.append((item['repetition'],phase,c))
for item in p['search_plans']:calls.append((item['repetition'],'search_plan',item['result']['benchmark_model_response']))
assert len(calls)==12
for index,(window,phase,c)in enumerate(calls):
 info=c['timings']['memory_inventory'];path=Path(info['path']);assert path.is_relative_to(Path('/Users/shawn/CantorAI/out/research/worldsense-inventory/captures'))
 local_capture=w/'captures'/path.name;raw=(local_capture if local_capture.is_file() else path).read_bytes();assert len(raw)==info['bytes']and hashlib.sha256(raw).hexdigest()==info['sha256'];d=json.loads(raw);assert len(d['buffers'])==info['buffer_count']and len(d['references'])==info['reference_count']
 assert [b['id']for b in d['buffers']]==list(range(len(d['buffers'])))
 by=collections.defaultdict(list)
 for ref in d['references']:
  assert 0<=ref['buffer_id']<len(d['buffers'])and 'identity'not in ref;by[ref['buffer_id']].append(ref)
 categories=collections.Counter();total=0
 for b in d['buffers']:
  assert b['allocated_bytes']>=b['length_bytes']>0;rr=by[b['id']];assert rr
  if any(r['weight']for r in rr):category='weights'
  elif any(r['tag']=='session'for r in rr):category='private_session'
  elif any(r['input']for r in rr):category='graph_inputs'
  elif any(r['output_part']for r in rr):category='graph_output_parts'
  elif not any(r['resident_mask_operand']for r in rr)and any(r['any_op_writer']for r in rr):category='nonresident_written_temps_not_safe_free_proof'
  else:category='other_graph_storage'
  total+=b['allocated_bytes'];categories[category]+=b['allocated_bytes']
  buffer_rows.append(dict(call_index=index,model_id=c['model_id'],capture=path.name,buffer_id=b['id'],allocated_bytes=b['allocated_bytes'],length_bytes=b['length_bytes'],recorded_references=len(rr),category=category,tags=';'.join(sorted({r['tag']for r in rr}))))
 for r in d['references']:reference_rows.append(dict(call_index=index,model_id=c['model_id'],capture=path.name,**r))
 assert total==d['unique_allocated_bytes']==info['unique_allocated_bytes']and sum(categories.values())==total
 assert total<=d['device_snapshot']['metal_allocated_bytes'];assert d['device_snapshot']['task_info_status']==0
 rows.append(dict(call_index=index,window=window,phase=phase,model_id=c['model_id'],capture=path.name,sha256=info['sha256'],capture_bytes=len(raw),buffer_count=len(d['buffers']),reference_count=len(d['references']),unique_allocated_bytes=total,device_allocated_bytes=d['device_snapshot']['metal_allocated_bytes'],categories=dict(categories)))
for name,values in [('capture-summaries.csv',[dict(**{k:v for k,v in row.items()if k!='categories'},**row['categories'])for row in rows]),('buffer-allocations.csv',buffer_rows),('buffer-references.csv',reference_rows)]:
 fields=list(dict.fromkeys(k for row in values for k in row))
 with(w/name).open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)
summary=dict(scope='Read-only unique-buffer inventory per model collection. Categories use a priority rule for overlapping owners; references are not ARC counts. Models cannot be unioned by collection-local ids. No safe-free, peak RAM or speed claim.',all12_full_outputs_and_capture_hashes_valid=True,summaries=rows,largest_collection_by_model={model:max((r for r in rows if r['model_id']==model),key=lambda r:r['unique_allocated_bytes'])for model in sorted({r['model_id']for r in rows})})
(w/'inventory-analysis.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['largest_collection_by_model'],indent=2))
