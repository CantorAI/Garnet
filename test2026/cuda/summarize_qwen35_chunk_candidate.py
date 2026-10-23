"""Qualified candidate-vs-baseline cohort and retained ownership statistics."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1])
candidate_name=sys.argv[2] if len(sys.argv)>2 else 'benchmark-garnet-2b-b1-chunk4-graph.json'
output_name=sys.argv[3] if len(sys.argv)>3 else 'chunk4-graph-comparison.json'
label=sys.argv[4] if len(sys.argv)>4 else 'Qwen3.5-2B batch1 chunk4 + verified CUDA graphs'
candidate_key='fused_chunk4_graph' if 'fused' in candidate_name else 'chunk4_graph'
names=[candidate_name,'benchmark-garnet-2b-b1-variants.json',
    'benchmark-vllm-2b-b1-variants.json','benchmark-lmstudio-2b-b1-rest-variants.json']
if 'fused' in candidate_name:names.append('benchmark-garnet-2b-b1-chunk4-graph.json')
cohorts=[json.loads((root/name).read_text()) for name in names]
candidate=cohorts[0];measured=[r for r in candidate if not r['warmup']]
assert all(len(a)==28 for a in cohorts) and len(measured)==20
assert all(r['vision_compile_ms']==0 for r in candidate)
def key(row):return row['trial'],tuple(row['images'])
indices=[{key(r):r for r in rows} for rows in cohorts]
assert all(set(index)==set(indices[0]) for index in indices)
paired=[]
for row in measured:
    k=key(row);v=indices[2][k];lm=indices[3][k]
    assert row['prompt_token_ids'][0]==v['records'][0]['prompt_token_ids']
    serial=indices[1][k]
    assert serial['prompt_token_ids']==row['prompt_token_ids']
    # The original serial cohort predates explicit stop accounting. Requiring
    # 32 visible tokens proves 32 samples without any inferred EOS work.
    serial_samples=serial.get('sampled_tokens_including_stop',serial['generated_tokens'])[0]
    prior_samples=indices[4][k]['sampled_tokens_including_stop'][0] if len(indices)>4 else 32
    if row['sampled_tokens_including_stop'][0]==serial_samples==prior_samples==len(v['records'][0]['token_ids'])==lm['result']['stats']['total_output_tokens']==32:paired.append(k)
memory=measured[-1]['owned_memory'];graphs=memory['decoder']['graphs']
assert all(g['graph_captures']>0 and g['graph_replays']>0 and not g['graph_capture_failures'] for g in graphs)
weights=json.loads((root/'weight-ownership-2b.json').read_text())['stored_parameter_bytes']
active=sum(n for k,n in weights.items() if k!='unused_mtp')
result={'candidate':label,'measured':len(measured),
    'median_wall_ms':statistics.median(r['wall_seconds'] for r in measured)*1000,
    'median_prefill_ms':statistics.median(r['prefill_seconds'] for r in measured)*1000,
    'median_ttft_ms':statistics.median(r['ttft_seconds'] for r in measured)*1000,
    'median_decode_tps':statistics.median(r['decode_steps']/r['decode_seconds'] for r in measured),
    'paired_equal_32token_cases':len(paired),'paired_wall_ms':{name:statistics.median(index[k]['wall_seconds'] for k in paired)*1000
        for name,index in zip([candidate_key,'serial','vllm','lmstudio','chunk4_graph'],indices)},
    'owned_model_bytes':memory['owned_model_bytes'],
    'owned_model_stable_after_warmup':len({r['owned_memory']['owned_model_bytes'] for r in measured})==1,
    'logical_active_checkpoint_bytes':active,'owned_over_payload_ratio':memory['owned_model_bytes']/active,
    'trt_text_engine_live_bytes':memory['decoder']['trt_runtime_allocator_live_bytes'],'logical_text_layer_bytes':weights['text_layers'],
    'decoder_context_count':sum(g['contexts'] for g in graphs),'graph_captures':sum(g['graph_captures'] for g in graphs),
    'graph_replays':sum(g['graph_replays'] for g in graphs),'capture_failures':sum(g['graph_capture_failures'] for g in graphs),
    'limitations':['Owned bytes exclude driver/external libraries, transient request/build allocations and graph-executable overhead; not process peak.',
        'Small-fixture and one32-token pretrained reference pass do not establish broad quality or full-capacity batch admission.',
        'Chunking and fused reduction order can change text; one tested chunk-policy reference matched32/32 tokens, not broad quality acceptance.',
        'Initial competitor limitations in INITIAL_BASELINE.md still apply.'],
    'goal_2x_vs_competitors_achieved':False,
    'decision':'Keep chunk4/graph as explicit experimental profile; decode regression prevents general default adoption.'}
if len(indices)>4:
    differences=[{'trial':r['trial'],'images':r['images'],'warmup':r['warmup'],
        'previous_generated_tokens':indices[4][key(r)]['generated_tokens'],'candidate_generated_tokens':r['generated_tokens']}
        for r in candidate if r['generated_token_ids']!=indices[4][key(r)]['generated_token_ids']]
    result['prior_profile_output_comparison']={'total':len(candidate),'exact_sequences':len(candidate)-len(differences),
        'measured_exact_sequences':len(measured)-sum(not r['warmup'] for r in differences),'differences':differences}
(root/output_name).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
