"""Archive canonical WS40 and production acceptance after timing is terminal.

CPU-only evidence packaging. Never execute while the production controller runs.
"""
from pathlib import Path
import csv, datetime, hashlib, json, shutil, subprocess

r = Path('/Users/shawn/CantorAI')
repo = r / 'Garnet'
w = Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-u32-integration')
d = repo / 'results/20261017/worldsense-m5-primary-u32-integration'
base = 'f21c1a258'
base = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', base], text=True).strip()
assert d == repo / 'results/20261017/worldsense-m5-primary-u32-integration'
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert (w/'decision.json').exists(), 'Retain an explicit terminal decision before packaging'
assert json.loads((w/'dependencies-after-production.json').read_text())['all_aliases_and_file_bytes_unchanged']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def copy(p, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)

d.mkdir(parents=True, exist_ok=True)
assert (d/'README.md').exists(), 'Write the reviewed report first'
for p in w.rglob('*'):
    if not p.is_file() or any(s in p.relative_to(w).parts for s in ['before', '__pycache__']):
        continue
    if p.suffix in ['.json', '.log', '.py', '.csv', '.txt', '.cpp']:
        copy(p, d/p.relative_to(w))

# Preserve the actual full projection words used for correctness, not a summary alone.
for mode in ['off', 'on', 'fault']:
    copy(r/('out/build/garnet-metal-worldsense-primary-u32-integration-'+mode)/'primary_projection_words.bin',
         d/('native-'+mode+'-projection-words.bin'))

sources = {}
for role, source, manifest_name in [
    ('canonical', repo, 'canonical-source-before-build.json'),
    ('control', Path(json.loads((w/'environment.json').read_text())['control_source']), 'control-source-before-dispatch.json'),
]:
    manifest = json.loads((w/manifest_name).read_text())
    overlay = []
    for rel, expected in manifest.items():
        p = source/rel
        assert sha(p) == expected, (role, rel)
        old = subprocess.run(['git', '-C', str(repo), 'show', base+':'+rel], capture_output=True)
        if old.returncode or old.stdout != p.read_bytes():
            copy(p, d/(role+'-source')/rel)
            overlay.append(rel)
    sources[role] = dict(base_commit=base, source_manifest=manifest_name,
                         overlay_directory=role+'-source', overlay_files=overlay,
                         frozen_source_files=len(manifest))
(d/'source-reconstruction.json').write_text(json.dumps(sources, indent=2))

calls, routes = [], []
for p in sorted((d/'production').glob('*.json')):
    if not p.name.startswith(('control-full-', 'candidate-full-')):
        continue
    data = json.loads(p.read_text())
    if not isinstance(data, dict) or 'items' not in data or 'search_plans' not in data:
        continue
    for category, items in [('frame', data['items']), ('query', data['search_plans'])]:
        for index, item in enumerate(items):
            responses = (item['result'].get('model_response', {}) if category == 'frame'
                         else {'query_plan': item['result'].get('benchmark_model_response', {})})
            for phase, c in responses.items():
                if not isinstance(c, dict) or 'model_id' not in c:
                    continue
                identity = dict(artifact=str(p.relative_to(d)), artifact_sha256=sha(p),
                                worker=data['label'], category=category, sample_index=index,
                                phase=phase, model_id=c['model_id'])
                total, first, count = (c[k] for k in ['total_seconds', 'ttft_seconds', 'generated_tokens'])
                t = c['timings']
                calls.append(dict(identity, warmup=item['warmup'], repetition=item['repetition'],
                                  status=c['status'], generated_tokens=count, input_tokens=t['input_tokens'],
                                  ttft_seconds=first, total_seconds=total,
                                  derived_decode_tokens_per_second=(count-1)/(total-first),
                                  native_call_seconds=item.get('native_call_seconds', ''),
                                  input_prompt_sha256=c['input_prompt_sha256'],
                                  image_sha256=item.get('image_sha256', ''), skill_sha256=item.get('skill_sha256', ''),
                                  observation_id=item.get('observation_id', ''), query=item.get('query', '')))
                route = dict(identity, backend=c['backend'], serving_source=t['serving_source'])
                for name in ['primary_storage_after', 'primary_encoded_delta', 'memory_after_decode']:
                    if name not in t:
                        route[name+'_available'] = False
                    else:
                        route[name+'_available'] = True
                        route.update({name+'_'+k: v for k, v in t[name].items()})
                routes.append(route)
assert len(calls) == len(routes) == 48
for name, rows in [('calls.csv', calls), ('execution-routes.csv', routes)]:
    with (d/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)
(d/'export-scope.json').write_text(json.dumps(dict(
    calls=48, first_and_warm_calls_separate=True,
    decode='(generated_tokens-1)/(total-TTFT); includes host/stop/possible extra EOS work; not pure GPU decode',
    memory='Old actual deployed control has no new memory API: unavailable, not zero. Candidate counters are diagnostic post-request snapshots, not peak or completed-dispatch counts.',
    scope='Additional canonical-production acceptance. Prototype resource and repeated-confirmation gates remain separate prerequisites.'), indent=2))
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)): sha(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name != 'file-manifest.json'}, indent=2))
print(json.dumps(dict(archive=str(d), calls=len(calls), files=len(json.loads((d/'file-manifest.json').read_text())), sources=sources)))
