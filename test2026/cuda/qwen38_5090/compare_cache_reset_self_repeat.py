from pathlib import Path
import hashlib,json
r=Path('D:/CantorAI/5090-study-20261023');old=r/'results/vllm-int4-cache-reset-quality/greedy/equivalence.json';new=r/'results/vllm-int4-cache-reset-self-repeat/equivalence.json';a,b=[json.loads(p.read_text()) for p in (old,new)];rows=[]
for x,y in zip(a['cases'],b['cases']):
 assert x['key']==y['key'] and x['semantic_request_sha256']==y['semantic_request_sha256'];left,right=x['generated_ids'],y['generated_ids'];first=next((i for i,(l,r) in enumerate(zip(left,right)) if l!=r),None)
 rows.append({'key':x['key'],'prompt_ids_equal':x['prompt_ids']==y['prompt_ids'],'generated_ids_equal':left==right,'first_difference':first,'old_ids_near_difference':left[max(0,(first or 0)-4):(first or 0)+5] if first is not None else [],'new_ids_near_difference':right[max(0,(first or 0)-4):(first or 0)+5] if first is not None else []})
report={'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'rows':rows,'matching_prompts':sum(c['prompt_ids_equal'] for c in rows),'matching_full_sequences':sum(c['generated_ids_equal'] for c in rows),'scope':'Actual same-live-process ordinary repeat; no timing or attribution of the separate-process drift.'}
out=r/'results/vllm-int4-cache-reset-self-repeat-diagnostic.json';assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
