import sys,json,hashlib
sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache')
import garnet
t=garnet.tokenizer();t.open('/Users/shawn/CantorAI/out/models/Qwen3-1.7B')
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '
rows=[]
for suffix in ['Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.','Observation: people walk across a street. Condition: a passenger vehicle is visible.']:
 ids=t.encode('<|im_start|>user\n')+t.encode(base+suffix+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
 rows.append({'tokens':len(ids),'prefix_sha256':hashlib.sha256(json.dumps(list(ids[:64])).encode()).hexdigest()})
print(json.dumps(rows),flush=True)
