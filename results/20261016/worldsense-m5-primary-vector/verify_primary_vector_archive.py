"""Independent archive integrity/coverage checks, not a new timing result."""
from pathlib import Path
import csv, hashlib, json, subprocess
r = Path('/Users/shawn/CantorAI/Garnet')
d = r / 'results/20261016/worldsense-m5-primary-vector'
assert d.name == 'worldsense-m5-primary-vector'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((d/'file-manifest.json').read_text())
assert set(manifest) == {str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p.name != 'file-manifest.json'}
for rel, digest in manifest.items():
    assert sha(d/rel) == digest, rel
reconstruction = json.loads((d/'source-reconstruction.json').read_text())
closure = json.loads((d/'starting-source-manifest.json').read_text())
for rel, digest in closure.items():
    p = d/'source'/rel
    raw = p.read_bytes() if p.exists() else subprocess.check_output(['git','-C',str(r),'show',reconstruction['base_commit']+':'+rel])
    assert hashlib.sha256(raw).hexdigest() == digest, rel
subprocess.run(['git','-C',str(r),'apply','--check',str(d/'source-overlay.patch')], check=True)
expected = []
for cohort in [d, d/'speed', d/'confirmation']:
    for p in sorted(cohort.glob('*.json')):
        if not p.name.startswith(('control','candidate')):
            continue
        data = json.loads(p.read_text())
        if not isinstance(data,dict) or 'items' not in data or 'search_plans' not in data:
            continue
        for category, items in [('frame',data['items']),('query',data['search_plans'])]:
            for index,item in enumerate(items):
                responses = item['result'].get('model_response',{}) if category=='frame' else {'query_plan':item['result'].get('benchmark_model_response',{})}
                for phase,call in responses.items():
                    if isinstance(call,dict) and 'model_id' in call:
                        expected.append((str(p.relative_to(d)),category,str(index),phase,call['model_id']))
for name in ['calls.csv','execution-routes.csv']:
    with (d/name).open() as f:
        rows = list(csv.DictReader(f))
    actual = [tuple(row[k] for k in ['artifact','category','sample_index','phase','model_id']) for row in rows]
    assert sorted(actual) == sorted(expected), name
    for row in rows:
        assert row['artifact_sha256'] == sha(d/row['artifact'])
assert (d/'native-off-projection-words.bin').read_bytes() == (d/'native-on-projection-words.bin').read_bytes()
assert json.loads((d/'api-correctness-result.json').read_text())['all32VLM28text_state_bits_tokens_prefix_capacity_vision_and_layer_isolation']
print(json.dumps({'verified_sha_files':len(manifest),'reconstructed_frozen_source_files':len(closure),'raw_model_calls_exactly_once_in_each_csv':len(expected),'full_projection_output_words_exact':True,'scope':'Artifact/coverage/source consistency; native and WorldSense test scopes remain their original reports.'}))

assert json.loads((d/'api-correctness-result.json').read_text())['all398actualpacked_weights_vector_GPU_bit_reconstruction']
corpus=json.loads((d/'corpus-GPU-bits.json').read_text())['totals']
assert corpus['GPU_vector_original_FP32_words_checked']==corpus['GPU_original_FP32_words_checked']==5921308672
assert json.loads((d/'correctness-source-manifest.json').read_text())==json.loads((d/'starting-source-manifest.json').read_text())
print('Vector-specific GPU corpus and unchanged source between correctness and benchmark PASS')
