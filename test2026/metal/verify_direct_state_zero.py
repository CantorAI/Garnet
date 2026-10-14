"""Verify retained reset evidence and summarize actual full-model measurements."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence'
files=json.loads((e/'files.json').read_text())
for name,item in files.items():
    raw=(root/name).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
manifest=json.loads((e/'source-manifest.json').read_text())
for name,item in manifest.items():
    assert hashlib.sha256((e/'source'/name).read_bytes()).hexdigest()==item['sha256']
    if item['before_sha256']:
        assert hashlib.sha256((e/'original'/name).read_bytes()).hexdigest()==item['before_sha256']
checks=e/'logs/direct-state-zero-checks-20261013'
assert json.loads((checks/'exit.json').read_text())['exit_code']==0
statuses=json.loads((checks/'jobs/statuses.json').read_text())
assert len(statuses)==19 and all(s['exit_code']==0 and not s['interruption'] for s in statuses)
validation_stages=len(statuses)
ctest=(checks/'jobs/ctest.log').read_text()
assert '100% tests passed' in ctest and 'metal_state_zero_reference' in ctest
assert 'state_zero checks 36 PASS' in (checks/'jobs/CTest-LastTest.log').read_text()
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    log=(checks/'jobs'/(model+'-exact-tails.log')).read_text()
    assert log.count('pretrained dense PASS')==8
    assert log.count('dense reference full_attention')==24
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assert (checks/'jobs'/(model+'-private-kv.log')).read_text().count('private_kv_reference PASS')==8
extended=(checks/'jobs/qwen35-extended-prefill.log').read_text()
assert extended.count('pretrained chunk PASS')==12 and extended.count('chunk reference ')==36
metadata=json.loads((checks/'jobs/metadata.json').read_text())
for name,expected in metadata['source_sha256'].items():
    assert hashlib.sha256((e/'check-source'/name).read_bytes()).hexdigest()==expected,name
assert 'METAL_DIRECT_STATE_ZERO' in metadata['options']
assert metadata['control_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
results=[];outputs=0
for job in sorted((e/'logs').glob('direct-state-zero-inference*')):
    assert json.loads((job/'exit.json').read_text())['exit_code']==0
    data=json.loads((job/'jobs/summary.json').read_text());assert len(data['profiles'])==6 and len(data['records'])==72
    library=json.loads((job/'jobs/metadata.json').read_text())['library_sha256']
    assert library['candidate']==metadata['candidate_sha256']
    assert library['control']=='f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088'
    for profile in data['profiles']:
        records=[r for r in data['records'] if r['model_name']==profile['model'] and r['case']['batch']==profile['batch']]
        assert len(records)==12
        for r in records:
            assert all(r[k]==records[0][k] for k in ('token_ids','prompt_token_ids','finished','case','prefill_chunk_plan'))
            assert r['attention_prefix_enabled']
            assert r['direct_state_zero_enabled']==(r['build_policy']=='candidate')
            assert r['reset_seconds']>=0
            if profile['model']=='Qwen3.5-0.8B':assert r['vision_reencoded'] and r['patch_sha256']==r['case']['image_metadata']['patch_sha256']
        warm={p:[r for r in records if r['build_policy']==p and r['request']>0] for p in ('control','candidate')}
        for policy,rows in warm.items():
            assert len(rows)==4
            assert abs(profile['means'][policy]['ttft_ms']-statistics.mean(r['ttft_seconds'] for r in rows)*1000)<1e-9
            assert abs(profile['means'][policy]['decode_tps']-statistics.mean(r['decode_wall_steps_per_second'] for r in rows))<1e-9
        means=profile['means']
        assert abs(profile['ttft_fraction']-means['candidate']['ttft_ms']/means['control']['ttft_ms'])<1e-12
        assert abs(profile['decode_ratio']-means['candidate']['decode_tps']/means['control']['decode_tps'])<1e-12
        results.append(dict(run=job.name,model=profile['model'],batch=profile['batch'],
            ttft_fraction=profile['ttft_fraction'],decode_ratio=profile['decode_ratio'],
            means=profile['means'],reset_ms={p:statistics.mean(r['reset_seconds'] for r in rows)*1000 for p,rows in warm.items()},
            first_request_ttft_ms={p:statistics.mean(r['ttft_seconds'] for r in records if r['build_policy']==p and r['request']==0)*1000 for p in warm}))
        outputs+=len(records)
    statuses=json.loads((job/'jobs/statuses.json').read_text())
    assert len(statuses)==24 and all(s['exit_code']==0 and 'interruption' not in s for s in statuses)
    for status in statuses:
        folder=job/'jobs'/status['name']
        assert json.loads((folder/'exit.json').read_text())['exit_code']==0
        monitors=json.loads((folder/'monitor.json').read_text())
        assert monitors and all(m['guard_passed'] for m in monitors)
assert results,'No complete inference evidence'
result=dict(scope='Native candidate/control; not a fresh vLLM comparison',
    files=len(files),validation_stages=validation_stages,complete_matched_outputs=outputs,profiles=results)
(root/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(files=len(files),validation_stages=validation_stages,complete_matched_outputs=outputs,profiles=len(results))))
