"""Audit actual dispatch, state references, outputs and worker RSS evidence."""
import hashlib,json,re,sys,math,statistics
from pathlib import Path
base=Path(sys.argv[1]).resolve();manifest=json.loads((base/'manifest.json').read_text());logs=base/'logs'
for e in manifest['files']:
    p=base/e['path'];assert p.is_relative_to(base)
    assert p.stat().st_size==e['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==e['sha256']
assert all(v['exit_code']==0 for v in manifest['jobs'].values())
assert manifest['control_sha256']=='cc41fde1f3905fcfb02fed6a50ab0f9d68933e94358352b92db4e44cb5f43dfe'
folder=logs/'compact-joined-checks-20261013/jobs';stages=json.loads((folder/'statuses.json').read_text());assert len(stages)==18 and all(s['exit_code']==0 and s['interruption'] is None for s in stages)
ctest=(folder/'CTest-LastTest.log').read_text();assert 'joined_prefill PASS rows=44 inner=1024 columns=3072' in ctest and 'joined_prefill PASS rows=26 inner=896 columns=4864' in ctest
counter_checks=0
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    text=(folder/(model+'-exact-tails.log')).read_text();counter=re.findall(r'joined_prefill_dispatch PASS (\d+) (\d+) (\d+)',text)
    assert len(counter)==24
    for batch,chunk,count in counter:
        expected=int((model=='Qwen3-0.6B' and int(batch)==2 and int(chunk)==22) or (model=='Qwen2.5-Coder-0.5B-Instruct' and int(batch)==1 and int(chunk)==26))
        assert int(count)==expected
    counter_checks+=len(counter)
    ratios=[float(v) for v in re.findall(r'(?<!_)worst_ratio ([\d.e+-]+)',text)];assert len(ratios)==24 and all(v<=1 for v in ratios)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assert (folder/(model+'-private-kv.log')).read_text().count('private_kv_reference PASS')==8
profiles=[];requests=0
for name,count in [('compact-joined-native-20261013',6),('compact-joined-native-confirm-20261013',2)]:
    if name not in manifest['jobs']:continue
    summary=json.loads((logs/name/'jobs/summary.json').read_text());assert len(summary['profiles'])==count
    for profile in summary['profiles']:
        where=logs/name/'jobs'/Path(profile['evidence']).name;report=json.loads((where/'summary.json').read_text());records=report['records'];assert len(records)==12
        reference=records[0]
        expected=json.loads((where/(profile['model']+'-b'+str(profile['batch'])+'-expected.json')).read_text())
        assert all(reference[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished'))
        for record in records:
            assert all(record[k]==reference[k] for k in ('token_ids','prompt_token_ids','finished','case'))
            expected=(28 if profile['model']=='Qwen3-0.6B' and profile['batch']==2 else 24 if profile['model']=='Qwen2.5-Coder-0.5B-Instruct' and profile['batch']==1 else 0)
            assert record['prefill_profile'].get('joined_prefill_pairs',0)==expected
            plan=([24] if profile['batch']==1 else [22]) if profile['model']=='Qwen3-0.6B' else [26] if profile['model']=='Qwen2.5-Coder-0.5B-Instruct' else [80] if profile['batch']==1 else [32,32,16]
            assert record['prefill_chunk_plan']==plan and record['peak_individual_rss_bytes']>0
            if profile['model']=='Qwen3.5-0.8B':assert record['vision_reencoded'] and record['patch_sha256']==record['case']['image_metadata']['patch_sha256']
        for policy in ('control','candidate'):
            assert len([r for r in records if r['build_policy']==policy and r['request']>0])==4
            for repeat in (0,1):
                worker=where/(profile['model']+'-b'+str(profile['batch'])+'-'+policy+'-'+str(repeat))
                assert json.loads((worker/'exit.json').read_text())['exit_code']==0 and json.loads((worker/'monitor.json').read_text())
                raw=(worker/'output.log').read_text()
                rss=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',raw,re.M)
                selected=[r for r in records if r['build_policy']==policy and r['repeat']==repeat]
                assert len(rss)==1 and len(selected)==3 and all(r['peak_individual_rss_bytes']==int(rss[0]) for r in selected)
            warm=[r for r in records if r['build_policy']==policy and r['request']>0]
            means=profile['means'][policy]
            for key,field,scale in [('ttft_ms','ttft_seconds',1000),('decode_tps','decode_wall_steps_per_second',1),('initialization_s','initialization_seconds',1),('peak_individual_rss_bytes','peak_individual_rss_bytes',1)]:
                assert math.isclose(means[key],statistics.mean(r[field] for r in warm)*scale,rel_tol=1e-12)
        assert math.isclose(profile['decode_ratio'],profile['means']['candidate']['decode_tps']/profile['means']['control']['decode_tps'],rel_tol=1e-12)
        assert math.isclose(profile['ttft_fraction'],profile['means']['candidate']['ttft_ms']/profile['means']['control']['ttft_ms'],rel_tol=1e-12)
        requests+=12;profiles.append(profile)
assert requests in (72,96)
result=dict(verified_files=len(manifest['files']),validation_stages=18,targeted_reference_and_dispatch_checks=counter_checks,private_cache_profiles=24,complete_inference_requests=requests,candidate_sha256=manifest['candidate_sha256'],control_sha256=manifest['control_sha256'],full_goal_met=False,profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
