"""Finalize WS37 after all dispatched timing workers are terminal. No GPU work."""
from pathlib import Path
import csv, difflib, hashlib, json, shutil, subprocess

root = Path('/Users/shawn/CantorAI')
repo = root / 'Garnet'
w = Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-gpu')
source = root / 'out/research/worldsense-primary-gpu/source'
d = repo / 'results/20261016/worldsense-m5-primary-gpu'
assert d == repo / ('results/20261016/worldsense-m5-' + w.name.removeprefix('worldsense-'))
assert d.is_dir() and (d / 'README.md').exists(), 'Run generic archive with reviewed report first'
assert json.loads((w / 'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert (w / 'decision.json').exists(), 'Explicit terminal decision required'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

manifest = json.loads((w / 'starting-source-manifest.json').read_text())
base = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
patch, changed = [], []
for rel, digest in manifest.items():
    p = source / rel
    assert sha(p) == digest, rel
    old = subprocess.run(['git', '-C', str(repo), 'show', base + ':' + rel], capture_output=True)
    if old.returncode == 0 and old.stdout == p.read_bytes():
        continue
    changed.append(rel)
    t = d / 'source' / rel
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, t)
    patch.extend(difflib.unified_diff(
        old.stdout.decode().splitlines(True) if old.returncode == 0 else [],
        p.read_text().splitlines(True),
        fromfile='a/' + rel if old.returncode == 0 else '/dev/null', tofile='b/' + rel))
(d / 'source-overlay.patch').write_text(''.join(patch))
subprocess.run(['git', '-C', str(repo), 'apply', '--check', str(d / 'source-overlay.patch')], check=True)

for name in ['fixture-1-8', 'fixture-2-64', 'speed', 'confirmation']:
    for p in (w / name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.json', '.log', '.py', '.csv', '.txt']:
            t = d / name / p.relative_to(w / name)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
for p in w.iterdir():
    if p.is_file() and p.suffix in ['.cpp', '.bin']:
        shutil.copy2(p, d / p.name)
for mode in ['off', 'on', 'fault']:
    p = root / ('out/build/garnet-metal-worldsense-primary-gpu-' + mode) / 'primary_projection_words.bin'
    t = d / ('native-' + mode + '-projection-words.bin')
    shutil.copy2(p, t)

(d / 'source-reconstruction.json').write_text(json.dumps({
    'base_commit': base,
    'base_choice': 'Selected after experiment for compatible reconstruction; not build-time checkout identity',
    'changed_files': changed, 'overlay_apply_check': True,
    'full_closure_sha256_file': 'starting-source-manifest.json',
    'fault_source': 'Same frozen source; test-only GARNET_METAL_PRIMARY_TEST_FAULTS macro in separate fault build',
    'fixtures': 'Original absolute paths retained. Regenerate with make_resident_fixtures.py when relocating.',
    'checkpoint_selection': '../worldsense-m5-primary-blocks/matrix-selection.json',
}, indent=2))

calls, routes, expected = [], [], []
for cohort in [d, d / 'speed', d / 'confirmation']:
    for p in sorted(cohort.glob('*.json')):
        if not p.name.startswith(('control', 'candidate')):
            continue
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or 'items' not in data or 'search_plans' not in data:
            continue
        for category, items in [('frame', data['items']), ('query', data['search_plans'])]:
            for index, item in enumerate(items):
                responses = item['result'].get('model_response', {}) if category == 'frame' else {'query_plan': item['result'].get('benchmark_model_response', {})}
                for phase, c in responses.items():
                    if not isinstance(c, dict) or 'model_id' not in c:
                        continue
                    identity = dict(artifact=str(p.relative_to(d)), artifact_sha256=sha(p), worker=data.get('label', p.stem), category=category, sample_index=index, phase=phase, model_id=c['model_id'])
                    expected.append(identity)
                    t = c.get('timings', {})
                    total, first, count = (c.get(k) for k in ['total_seconds', 'ttft_seconds', 'generated_tokens'])
                    rate = (count-1)/(total-first) if all(isinstance(v, (int, float)) for v in [total, first, count]) and total > first and count > 1 else ''
                    calls.append(dict(identity, warmup=item.get('warmup', ''), repetition=item.get('repetition', ''), status=c.get('status', ''), input_tokens=t.get('input_tokens', ''), generated_tokens=count, ttft_seconds=first, total_seconds=total, derived_decode_tokens_per_second=rate, native_call_seconds=item.get('native_call_seconds', ''), image_sha256=item.get('image_sha256', ''), skill_sha256=item.get('skill_sha256', ''), input_prompt_sha256=c.get('input_prompt_sha256', ''), observation_id=item.get('observation_id', ''), skill_ref=item.get('skill_ref', ''), query=item.get('query', ''), eligible_for_speed=cohort != d))
                    row = dict(identity, backend=c.get('backend', ''), primary_capability='primary_bf16_blocks' in c.get('backend', ''))
                    row.update({'primary_storage_after_' + k: v for k,v in t.get('primary_storage_after', {}).items()})
                    row.update({'primary_encoded_delta_' + k: v for k,v in t.get('primary_encoded_delta', {}).items()})
                    row.update({'memory_after_decode_' + k: v for k,v in t.get('memory_after_decode', {}).items()})
                    routes.append(row)
for name, rows in [('calls.csv', calls), ('execution-routes.csv', routes)]:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (d / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
(d / 'export-scope.json').write_text(json.dumps({
    'raw_calls': len(expected), 'calls_csv_rows': len(calls), 'route_csv_rows': len(routes),
    'scope': 'All first/warm returned model calls exactly once. Resource-root timings ineligible for speed. Calls remain grouped by worker/pair; CSV is not an independent result.',
    'decode_metric': '(generated_tokens-1)/(total-TTFT); includes host/stop/post-loop overhead and possibly extra EOS work, not pure GPU decode.',
    'storage_counters': 'Native owner accounting, not driver physical liveness. Encoded deltas count dispatch encoding, not completed GPU operations. Snapshots are not peaks.',
}, indent=2))
for p in d.rglob('__pycache__'):
    shutil.rmtree(p)
(d / 'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)): sha(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name != 'file-manifest.json'}, indent=2))
print(json.dumps({'archive': str(d), 'overlay_files': len(changed), 'calls': len(calls), 'sha_files': len(json.loads((d/'file-manifest.json').read_text()))}))
