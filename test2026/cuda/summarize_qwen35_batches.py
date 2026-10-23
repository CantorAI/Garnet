"""Audit completed batch diagnostics without promoting them to broad acceptance."""
import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--model',choices=['2b','4b'],default='2b')
args=parser.parse_args();root=Path(args.root);profiles=[]
for batch in (2,4,8):
    path=root/f'batch-validation-{args.model}-b{batch}.json'
    if not path.exists():continue
    native=json.loads(path.read_text());styles=sorted({r['style'] for r in native})
    assert all(r['batch']==batch and r['arithmetic']['projection_reduced_precision_reduction'] is False for r in native)
    assert all(r['decoder_slots']==[p+n-1 for p,n in zip(r['prompt_tokens'],r['sampled_tokens_including_stop'])] for r in native)
    for style in styles:
        a,b=[r for r in native if r['style']==style]
        assert a['generated_token_ids']==b['generated_token_ids'] and a['finish_reasons']==b['finish_reasons'] and a['sampled_tokens_including_stop']==b['sampled_tokens_including_stop']
    uniform=next(r for r in native if r['style']=='uniform')
    assert all(row==uniform['generated_token_ids'][0] for row in uniform['generated_token_ids'])
    references=[]
    for style,label in [('uniform','uniform'),('ragged','ragged'),('mixed_images','mixed')]:
        reference=root/f'batch-reference-{args.model}-b{batch}-{label}.json'
        if not reference.exists():continue
        data=json.loads(reference.read_text());rows=data['quantitative_hidden_logit_comparison']
        assert data['batch']==batch and data['style']==style and data['native_input_embeddings'] and data['positions_equal_on_valid_tokens']
        references.append({'style':style,'decisions':len(rows),
            'top1_matches':sum(r['native_selected_token']==r['reference_top1'] for r in rows),
            'maximum_set_matches':sum(r['native_selected_reference_rank']==1 for r in rows),
            'native_head_reconstruction_matches':sum(r['native_selected_token']==r['native_hidden_reconstructed_top1'] for r in rows),
            'max_hidden_relative_l2':max(r['hidden_relative_l2'] for r in rows),
            'nonmaximum_selections':[r for r in rows if r['native_selected_reference_rank']>1],
            'source':reference.name})
    memory=sorted({r['owned_memory']['owned_model_bytes'] for r in native})
    graphs=native[-1]['owned_memory']['decoder']['graphs']
    assert all(g['graph_captures']>0 and g['graph_replays']>0 and not g['graph_capture_failures'] for g in graphs)
    profiles.append({'batch':batch,'profile':native[0]['profile'],'native_cases':len(native),'native_styles':styles,
        'reset_replay_exact':True,'replicated_uniform_rows_equal':True,'slot_accounting_exact':True,
        'mixed_image_isolation_checked':any(r['style']=='mixed_images' for r in native),
        'references':references,'missing_styles':sorted({'uniform','ragged','mixed_images'}-set(styles)),
        'owned_model_bytes':memory,'owned_bytes_stable':len(memory)==1,
        'reference_decisions':sum(r['decisions'] for r in references),
        'reference_top1_matches':sum(r['top1_matches'] for r in references),
        'reference_maximum_set_matches':sum(r['maximum_set_matches'] for r in references),
        'native_head_reconstruction_matches':sum(r['native_head_reconstruction_matches'] for r in references)})
for profile in profiles:
    count=profile['reference_decisions']
    profile['all_recorded_reference_ids_match']=bool(count) and profile['reference_top1_matches']==count
    profile['all_recorded_selections_in_reference_maximum_set']=bool(count) and profile['reference_maximum_set_matches']==count
result={'model':'Qwen3.5-'+args.model.upper(),'profiles':profiles,'recorded_batches':[r['batch'] for r in profiles],
    'all_three_styles_recorded_at_every_batch':len(profiles)==3 and all(not r['missing_styles'] for r in profiles),
    'full_capacity_boundary_verified_on_official_model':False,'broad_quality_accepted':False,
    'limitation':'Diagnostic copies invalidate timing. Decoder oracle uses fixed native embeddings and recorded tokens, bypassing independent vision. One image family set and tested trajectories do not establish broad quality, full cache-boundary coverage or comparative batch performance.'}
(root/f'batch-validation-{args.model}-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
