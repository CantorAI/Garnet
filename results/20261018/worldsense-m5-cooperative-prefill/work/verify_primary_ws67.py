"""Post-terminal audit, independent recomputation; no new inference or exclusions."""
from pathlib import Path
import ast, datetime, hashlib, json, statistics

b = Path('/Users/shawn/Documents/Codex/2026-10-15/i')
w = b / 'work/worldsense-cooperative-prefill'
p = w / 'primary-speed'
r = Path('/Users/shawn/CantorAI')
s = r / 'out/research/worldsense-cooperative-prefill/source'
load = lambda f: json.loads(f.read_text())

def sha(f):
    h = hashlib.sha256()
    with f.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

assert load(w / 'production-speed/independent-screen-audit.json')['speed_passed']
for f in [w / 'driver-identities-before-dispatch.json',
          w / 'production-speed/driver-identities-before-dispatch.json',
          p / 'driver-identities-before-dispatch.json',
          p / 'prospective-stage-driver-note.json',
          w / 'outer-method-verifier-identities-before-resource.json']:
    assert all(sha(Path(k)) == v for k, v in load(f)['files'].items()), str(f)
assert all(sha(s / k) == v for k, v in load(p / 'starting-source-manifest.json').items())
dep = load(w / 'build-host-dependencies-before-dispatch.json')
assert len(dep['files']) == 1881
assert all(str(Path(k).resolve()) == v for k, v in dep['aliases'].items())
assert all(Path(k).stat().st_size == v['bytes'] and sha(Path(k)) == v['sha256']
           for k, v in dep['files'].items())
assert load(w / 'dependencies-after-primary-speed.json')['all_aliases_and_file_bytes_unchanged']
assets = load(p / 'learned-assets-before-dispatch.json')
assert len(assets) == 19
assert all(Path(k).stat().st_size == v['bytes'] and sha(Path(k)) == v['sha256']
           for k, v in assets.items())
assert sha(Path('/usr/bin/pmset')) == load(p / 'observation-tool-identity.json')['pmset_sha256']
assert all(sha(Path(k)) == v for k, v in load(w / 'learned-identities-before.json')['external'].items())
fn = next(n for n in ast.parse((p / 'analyze.py').read_text()).body
          if isinstance(n, ast.FunctionDef) and n.name == 'native_route')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'original native route', 'exec'), ns)
gold = load(r / 'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json')
refs = {(item['observation_id'], phase): (item, call)
        for item in gold['items'] if item['policy'] == 'candidate'
        for phase, call in item['result']['model_response'].items()
        if isinstance(call, dict) and call.get('status') == 'ok'}
analysis = load(p / 'analysis.json')
manifest = {m['label']: m for m in load(p / 'cohort-manifest.json')}
environment = load(p / 'environment.json')
metrics, wall, metal, footprint, phase_data = {}, {}, {}, {}, {}
count = 0
stable = lambda result: {k: v for k, v in result.items()
                          if k not in ['model_response', 'model_response_json']}
for label in ['control-1', 'candidate-1']:
    d = load(p / (label + '.json'))
    assert d['valid'] and d['assignments_restored'] and not d['search_plans']
    assert d['request'] == load(p / 'cases.json')
    assert [(i['repetition'], i['warmup'], i['policy']) for i in d['items']] == [
        (i, i == 0, 'default') for i in range(5)]
    assert manifest[label]['library_sha256'] == environment['libraries'][
        'on' if label == 'candidate-1' else 'off']
    rows = []
    for item in d['items']:
        calls = [(phase, c) for phase, c in item['result']['model_response'].items()
                 if isinstance(c, dict) and c.get('status') == 'ok']
        assert len(calls) == 2
        for phase, c in calls:
            ri, ref = refs[item['observation_id'], phase]
            assert stable(item['result']) == stable(ri['result'])
            assert all(c[k] == ref[k] for k in ['generated_token_ids', 'text',
                'input_prompt_sha256', 'serving_configuration'])
            assert c['timings']['input_tokens'] == ref['timings']['input_tokens']
            assert item['image_sha256'] == ri['image_sha256']
            assert item['skill_sha256'] == ri['skill_sha256']
            assert ns['native_route'](c, label == 'candidate-1', True)
            assert Path(c['timings']['serving_source']).resolve() == s / 'runtime/metal_serving.py'
            rows.append((item['warmup'], c)); count += 1
    metrics[label] = {}; phase_data[label] = {}
    for model in ['Qwen3-1.7B', 'Qwen3.5-4B']:
        calls = [c for warmup, c in rows if not warmup and c['model_id'] == model]
        assert len(calls) == 4
        metrics[label][model] = dict(samples=4,
            ttft=statistics.median(c['ttft_seconds'] for c in calls),
            total=statistics.median(c['total_seconds'] for c in calls),
            decode_tps=statistics.median((c['generated_tokens']-1) /
                (c['total_seconds']-c['ttft_seconds']) for c in calls))
        phase_data[label][model] = [dict(ttft=c['ttft_seconds'], total=c['total_seconds'],
            prefill=c['timings']['prefill_seconds'], decode=c['timings']['decode_seconds'],
            decode_interval=c['total_seconds']-c['ttft_seconds']) for c in calls]
    wall[label] = statistics.median(i['native_call_seconds'] for i in d['items'] if not i['warmup'])
    metal[label] = max(c['timings']['memory_after_decode']['metal_allocated_bytes'] for _, c in rows)
    footprint[label] = max(c['timings']['memory_after_decode']['task_physical_footprint_bytes'] for _, c in rows)
comparisons = [dict(model=model,
    ttft_change=metrics['candidate-1'][model]['ttft']/metrics['control-1'][model]['ttft']-1,
    total_change=metrics['candidate-1'][model]['total']/metrics['control-1'][model]['total']-1,
    decode_gain=metrics['candidate-1'][model]['decode_tps']/metrics['control-1'][model]['decode_tps']-1)
    for model in sorted(metrics['control-1'])]
wall_gain = 1-wall['candidate-1']/wall['control-1']
vlm_gain = 1-metrics['candidate-1']['Qwen3.5-4B']['ttft']/metrics['control-1']['Qwen3.5-4B']['ttft']
memory_ok = metal['candidate-1']/metal['control-1']-1 <= .02 and footprint['candidate-1']/footprint['control-1']-1 <= .02
passed = memory_ok and vlm_gain >= .05 and wall_gain >= -.02 and all(
    c['ttft_change'] <= .02 and c['total_change'] <= .02 and c['decode_gain'] >= -.02 for c in comparisons)
assert count == 20 and metrics == analysis['metrics'] and comparisons == analysis['comparisons']
assert wall == analysis['native_skill_wall_p50'] and wall_gain == analysis['native_skill_wall_gain']
assert metal == analysis['sampled_max_post_request_metal_bytes'] and footprint == analysis['sampled_max_footprint_bytes']
assert memory_ok == analysis['matched_memory_preserved'] and vlm_gain == analysis['observed_VLM_TTFT_gain']
assert passed == analysis['passed'] == False and analysis['status'] == 'rejected'
assert load(w / 'restoration-verification.json')['benchmark_valid_request_http404']
result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    session=58249, exit_code=0, decision='valid-reject', promotion=False,
    twenty_calls_exact_original_outputs_and_native_routes=True,
    source_driver_runtime_library_model_and_1881_dependencies_unchanged=True,
    metrics=metrics, comparisons=comparisons, whole_skill_wall=wall,
    wall_gain=wall_gain, vlm_ttft_gain=vlm_gain,
    metal_snapshots=metal, footprint_snapshots=footprint, memory_preserved=memory_ok,
    next_stages='Stopped: no RAW20, mirrored48 or canonical integration for this rejected candidate.',
    limits='Fixed-frame real WorldSense/Galaxy calls; no live decode/event persistence/search coverage. First retained; four warm samples/model/worker. Sampled memory, not peaks. Entire profile contrast, no isolated cooperative kernel or causal claim.')
(p / 'independent-stage-audit.json').write_text(json.dumps(result, indent=2)+'\n')
(p / 'posthoc-phase-diagnostics.json').write_text(json.dumps(dict(
    scope='Descriptive posthoc only, no exclusions, new inference or causal conclusion. Separate timing components overlap and are nonadditive.',
    workers=phase_data), indent=2)+'\n')
print(json.dumps(result, indent=2))
