"""Independent full-output, source, shader and contiguous timestamp-record audit."""
import hashlib,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for e in proof['files']:
    p=root/e['path'];assert p.resolve().is_relative_to(root) and p.stat().st_size==e['bytes'] and digest(p)==e['sha256']
folder=root/'logs/counter-trace-file-profiles-20261015';assert json.loads((folder/'exit.json').read_text())['exit_code']==0
jobs=folder/'jobs';meta=json.loads((jobs/'metadata.json').read_text());assert meta['library_sha256']==proof['library_sha256']
for name,sha in meta['source_sha256'].items():assert digest(root/'source/test2026/metal'/name)==sha
ids=json.loads((jobs/'source-identities.json').read_text())
for name,sha in ids['diagnostic_source_sha256'].items():assert digest(root/'counter-trace-file-source'/name)==sha
original=root/'source/src/backends/metal/metal_context.mm';assert digest(original)==ids['original_metal_context_sha256']
pattern=r'R"([A-Za-z_]+)\((.*?)\)\1"'
assert re.findall(pattern,original.read_text(),re.S)==re.findall(pattern,(root/'counter-trace-file-source/src/backends/metal/metal_context.mm').read_text(),re.S)
assert '100% tests passed' in (jobs/'ctest.log').read_text()
stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==28 and all(s['exit_code']==0 and s['interruption'] is None for s in stages)
for s in stages:assert json.loads((jobs/(s['name']+'-monitor.json')).read_text())
profiles=json.loads((jobs/'profiles.json').read_text())['profiles'];assert len(profiles)==24
assert {(p['model'],p['batch'],p['tracing'],p['repeat']) for p in profiles}=={(m,b,t,n) for m in proof['assets'] for b in (1,2) for t in (False,True) for n in (0,1)}
count=0;sample_count=0;command_count=0
for p in profiles:
    model=p['model'];label=model+'-b'+str(p['batch']);name=label+'-'+('traced' if p['tracing'] else 'quiet')+'-'+str(p['repeat'])
    reference=json.loads((jobs/(label+'-expected.json')).read_text());case=json.loads((jobs/(label+'-case.json')).read_text());assert case['requests']==3
    assert len(p['records'])==p['requests']==3
    raw=(jobs/(name+'.log')).read_text();prefix='matched_frame_result ' if model=='Qwen3.5-0.8B' else 'matched_result '
    actual=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith(prefix)];assert actual==p['records']
    for r in actual:
        assert all(r[k]==reference[k] for k in ('token_ids','prompt_token_ids','finished')) and r['case']==case
        assert r['global_half_alpha_enabled'] and not r['prefill_stream_enabled'] and not r['dirty_prefix_reset_enabled']
        if model=='Qwen3.5-0.8B':assert r['vision_reencoded'] and r['patch_sha256']==case['image_metadata']['patch_sha256']
    trace=jobs/name/'gpu-counters.jsonl'
    if not p['tracing']:assert not trace.exists() and p['counter_samples']==p['counter_commands']==0
    else:
        lines=trace.read_text().splitlines();samples=[];commands=[]
        for line in lines:
            prefix,data=line.split(' ',1);value=json.loads(data);assert value['diagnostic'] and not value['serialized']
            if prefix=='metal_counter_profile':samples.append(value)
            else:assert prefix=='metal_counter_command';commands.append(value)
        assert len(samples)==p['counter_samples'] and len(commands)==p['counter_commands'] and samples and commands
        by={}
        for e in samples:
            assert 0<e['start_raw']<=e['end_raw'];by.setdefault((e['context'],e['command']),[]).append(e)
        assert len({(c['context'],c['command']) for c in commands})==len(commands)
        for c in commands:
            assert c['partial_submissions']==0 and 0<=c['samples']<=1024
            assert [e['index'] for e in by.get((c['context'],c['command']),[])]==list(range(c['samples']))
        sample_count+=len(samples);command_count+=len(commands)
    count+=3
result=dict(verified_files=len(proof['files']),full_output_requests=count,counter_samples=sample_count,counter_commands=command_count,
    raw_shader_payloads_unchanged=True,all_counter_records_contiguous=True,scope='Compute-pass diagnostic only; no fresh comparator or speed claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
