"""Audit batch8 candidate independently of historical failed profile files."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);prefix='4b-b8-full-projection-w256'
rows=json.loads((root/f'batch-validation-{prefix}.json').read_text())
assert len(rows)==6
for style in ('uniform','ragged','mixed_images'):
    a,b=[r for r in rows if r['style']==style]
    assert a['generated_token_ids']==b['generated_token_ids'] and a['finish_reasons']==b['finish_reasons']
    assert a['sampled_tokens_including_stop']==b['sampled_tokens_including_stop']
assert len({r['owned_memory']['owned_model_bytes'] for r in rows})==1
for r in rows:
    assert r['decoder_slots']==[p+n-1 for p,n in zip(r['prompt_tokens'],r['sampled_tokens_including_stop'])]
    assert r['arithmetic']['full_attention_projection_layout']=='chunk_batched'
    graphs=r['owned_memory']['decoder']['graphs']
    assert all(g['graph_captures'] and g['graph_replays'] and not g['graph_capture_failures'] for g in graphs)
    assert all(graphs[i]['builder_workspace_limit_bytes']==256<<20 for i in range(3,32,4))
refs=[]
for style in ('uniform','ragged','mixed'):
    path=root/f'batch-reference-{prefix}-{style}.json'
    if not path.exists():continue
    reference=json.loads(path.read_text());q=reference['quantitative_hidden_logit_comparison']
    assert reference['full_attention_projection_layout']=='chunk_batched'
    refs.append({'style':style,'decisions':len(q),
        'exact_ids':sum(r['native_selected_token']==r['reference_top1'] for r in q),
        'maximum_set':sum(r['native_selected_reference_rank']==1 for r in q),
        'head_reconstructions':sum(r['native_selected_token']==r['native_hidden_reconstructed_top1'] for r in q),
        'nonmaximum_choices':[r for r in q if r['native_selected_reference_rank']>1]})
memory=[json.loads(line) for line in (root/f'memory-{prefix}.json').read_text().splitlines()]
samples=[r for r in memory if r['event']=='sample']
target=[]
for sample in samples:
    positive=[r for r in sample['target_rows'] if r['DedicatedUsage'] or r['SharedUsage']]
    assert len(positive)<=1,'Ambiguous positive duplicate WDDM rows'
    target.extend(positive)
result={'profile':prefix,'native_cases':6,'runtime_checks_passed':True,
    'retained_owned_model_bytes':rows[0]['owned_memory']['owned_model_bytes'],
    'decoder_references':refs,'all_reference_styles_present':len(refs)==3,
    'partial_process_capture':{'samples_with_positive_target':len(target),
        'sampled_max_dedicated_bytes':max(r['DedicatedUsage'] for r in target),
        'sampled_shared_bytes_range':[min(r['SharedUsage'] for r in target),max(r['SharedUsage'] for r in target)],
        'scope':'WDDM process counters on verified GPU0. Capture began during loading; no guarded warm idle holds, no true peak claim.'},
    'broad_quality_accepted':False,'comparative_batch_performance_measured':False,
    'limitation':'Native embeddings isolate decoder reference. Synthetic batch8 hidden-check discrepancy remains open on old and candidate builds. Historical batch2/4 use different projection layout.'}
(root/'batch8-full-projection-candidate-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
