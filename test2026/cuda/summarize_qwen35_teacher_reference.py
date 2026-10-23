"""Summarize a fixed-trajectory diagnostic without inventing quality gates."""
import hashlib,json,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:3])
data=json.loads(source.read_text());rows=data['quantitative_hidden_logit_comparison']
assert data['teacher_forced'] and rows
result={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'steps':len({r['step'] for r in rows}),'decisions':len(rows),'batch':data.get('batch',1),'style':data.get('style'),
    'deterministic_top1_matches':sum(r['native_selected_token']==r['reference_top1'] for r in rows),
    'reference_maximum_set_matches':sum(r['native_selected_reference_rank']==1 for r in rows),
    'native_head_reconstructed_matches':sum(r['native_selected_token']==r['native_hidden_reconstructed_top1'] for r in rows),
    'max_hidden_relative_l2':max(r['hidden_relative_l2'] for r in rows),
    'min_hidden_cosine':min(r['hidden_cosine'] for r in rows),
    'max_hidden_abs':max(r['hidden_max_abs'] for r in rows),
    'max_logit_abs':max(r['logit_max_abs'] for r in rows),
    'input_comparison':data.get('input_comparison',{}),
    'native_input_embeddings':data.get('native_input_embeddings',False),
    'allow_bf16_reduced_precision_reduction':data.get('allow_bf16_reduced_precision_reduction'),
    'mismatches':[r for r in rows if r['native_selected_token']!=r['reference_top1']],
    'broad_quality_accepted':False,
    'limitation':'One teacher-forced trajectory; tied-logit set agreement differs from deterministic greedy token identity. Native-input mode bypasses independent vision validation. No numerical acceptance threshold has been established for a full model.'}
result['per_row']=[{'row':row,'decisions':sum(r.get('row',0)==row for r in rows),
    'deterministic_top1_matches':sum(r.get('row',0)==row and r['native_selected_token']==r['reference_top1'] for r in rows),
    'maximum_set_matches':sum(r.get('row',0)==row and r['native_selected_reference_rank']==1 for r in rows),
    'native_head_reconstruction_matches':sum(r.get('row',0)==row and r['native_selected_token']==r['native_hidden_reconstructed_top1'] for r in rows)}
    for row in sorted({r.get('row',0) for r in rows})]
output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
