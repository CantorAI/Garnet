"""Compare isolated factory smoke artifacts without claiming broad preservation."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);a=json.loads((root/'metal-factory-baseline.json').read_text());b=json.loads((root/'metal-factory-current.json').read_text())
assert a['mode']=='baseline' and b['mode']=='current'
assert a['backend']==b['backend'] and a['image_sha256']==b['image_sha256'] and a['profile']==b['profile']
assert a['process_id']!=b['process_id'] and len(a['rows'])==len(b['rows'])==2
for old,new in zip(a['rows'],b['rows']):
    assert old['generated_token_ids']==new['generated_token_ids']
    assert old['timings']['input_tokens']==new['timings']['input_tokens']
    assert old['timings']['image_grid']==new['timings']['image_grid']
assert all(r['generated_token_ids']==a['rows'][0]['generated_token_ids'] for r in a['rows']+b['rows'])
result={'same_binary_factory_smoke_passed':True,'requests_per_process':2,
    'generated_token_ids':a['rows'][0]['generated_token_ids'],
    'input_tokens':a['rows'][0]['timings']['input_tokens'],'image_grid':a['rows'][0]['timings']['image_grid'],
    'process_ids':[a['process_id'],b['process_id']],'backend':a['backend'],
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path('Garnet/runtime/metal_serving.py'),Path('Garnet/xModel/qwen35/vl_4b/model.py'),Path(__file__).with_name('probe_qwen35_factory_preservation.py')]},
    'broad_quality_accepted':False,'current_local_native_source_build_verified':False,
    'limitation':'One real image,8-token limit,2 repeats per factory on existing Mac binary. Remote native Metal source differs from Windows base; full source/build preservation and larger profiles remain unproven. No speed comparison.'}
(root/'metal-factory-preservation-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
