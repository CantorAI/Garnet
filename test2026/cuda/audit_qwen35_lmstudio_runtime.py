"""Extract actual llama.cpp capacity/slots and prompt evaluation evidence."""
import argparse,hashlib,json,re
from pathlib import Path
from qwen35_comparator_speculation import lmstudio_speculation
p=argparse.ArgumentParser();p.add_argument('log');p.add_argument('output');p.add_argument('--identifier',required=True)
p.add_argument('--batch',type=int,required=True);p.add_argument('--context',type=int,default=512)
a=p.parse_args();path=Path(a.log);values={k:[] for k in ('n_ctx','n_ctx_per_seq','n_ctx_slot','n_seq_max','n_parallel','n_slots')}
prompts=[];slots=set();messages=[];unified=[]
for line_number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
    try:record=json.loads(line)
    except json.JSONDecodeError:continue
    data=record.get('data',{})
    if data.get('modelIdentifier')!=a.identifier:continue
    text=data.get('message','');matched=False
    for value in re.findall(r"\bkv_unified\s*=\s*'?(true|false)",text):unified.append(value=='true');matched=True
    for field in values:
        for value in re.findall(r'\b'+field+r'\s*=\s*(\d+)',text):values[field].append(int(value));matched=True
    for count in re.findall(r'prompt eval time\s*=.*?/\s*(\d+) tokens',text):prompts.append(int(count))
    for slot in re.findall(r'slot print_timing:\s*id\s*(\d+)',text):slots.add(int(slot))
    if matched:messages.append({'line':line_number,'message':text,'pid':data.get('pid')})
values={k:sorted(set(v)) for k,v in values.items()}
contexts=[values[k] for k in ('n_ctx_per_seq','n_ctx_slot') if values[k]]
context_verified=bool(contexts) and all(v==[a.context] for v in contexts)
parallel=[values[k] for k in ('n_seq_max','n_parallel','n_slots') if values[k]]
parallel_verified=bool(parallel) and all(v==[a.batch] for v in parallel)
result={'identifier':a.identifier,'requested_batch':a.batch,'requested_context_per_sequence':a.context,
    'raw_log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actual_fields':values,
    'context_per_sequence_verified':context_verified,'configured_parallel_slots_verified':parallel_verified,
    'kv_unified_observed':sorted(set(unified)),
    'independent_slot_capacity_verified':context_verified and parallel_verified and sorted(set(unified))==[False],
    'observed_timed_slots':sorted(slots),'prompt_eval_events':len(prompts),'prompt_eval_token_counts':sorted(set(prompts)),
    'all_expected_prompt_evals_seen':len(prompts)==28*a.batch and set(prompts)=={85},
    'capacity_evidence':messages,'cache_disabled_setting_verified':False,
    'limitation':'Only identifier-attributed lines count. Full85-token evaluations evidence actual prompt work; do not prove every internal cache disabled. HTTP concurrency is not proof of GPU batching.'}
result['speculation']=lmstudio_speculation(path,a.identifier)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
