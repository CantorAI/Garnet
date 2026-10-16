"""Audit incomplete native cohort: exact outputs/routes, stale legacy-counter guard; no timing claim."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='ndarray-native-20261015';assert proof['jobs'][trial]['exit_code']==1
jobs=root/'logs'/trial/'jobs';audit=json.loads((jobs/'checks-verification.json').read_text())
assert audit['archive_sha256']==proof['checks_archive_sha256']=='f60abf2cfd3a3b1a335e096a291b5343118772a50c9ee8d5f68c37db7fd0e388'
identities=json.loads((jobs/'source-identities.json').read_text());assert identities==audit['source_manifest']
for name,sha in identities.items():assert digest(root/'source'/name)==digest(jobs/'source-snapshot'/name)==sha
external=json.loads((jobs/'external-tool-identities.json').read_text())['files']
for name,sha in external.items():assert digest(root/'external-tools'/name)==digest(jobs/'tool-source-snapshot'/name)==sha
saved=sys.argv
try:
    sys.argv=[saved[0],str(base),str(base/'checks-reverification.json')]
    runpy.run_path(str(root/'external-tools/verify_ndarray_native_checks.py'),run_name='__main__')
finally:sys.argv=saved
recheck=json.loads((base/'checks-reverification.json').read_text())
for name,value in recheck.items():
    if name=='verified_files':assert value>=audit[name]
    else:assert value==audit[name],name
summary=json.loads((jobs/'summary.json').read_text());assert len(summary['profiles'])==4
outer=json.loads((jobs/'statuses.json').read_text());assert len(outer)==5
assert all(s['exit_code']==0 for s in outer[:4]) and outer[-1]['exit_code']==1
assert outer[-1]['name']=='Qwen3.5-0.8B-b1-attempt0'
records=0;workers=0;nd_images=0
for status in outer:
    folder=jobs/status['name'];meta=json.loads((folder/'metadata.json').read_text())
    assert meta['library_sha256']=={'control':proof['library_sha256']['build-coalesced-bf16-body'],'candidate':proof['library_sha256']['build-ndarray-prefill-shape80-guardrepair']}
    assert meta['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
    expected_file=next(folder.glob('*-expected.json'));label=expected_file.name.removesuffix('-expected.json');expected=json.loads(expected_file.read_text())
    model=label.rsplit('-b',1)[0];batch=int(label.rsplit('-b',1)[1]);header=json.loads((root/'checkpoint-headers'/(model+'.json')).read_text())
    eligible=[name for name,item in header.items() if name.startswith(('model.layers.','model.language_model.layers.')) and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2 and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]
    assert meta['ndarray_prefill']['tensors']==eligible
    stages=json.loads((folder/'statuses.json').read_text());assert len(stages)==(4 if status['exit_code']==0 else 2)
    for stage in stages:
        assert stage['exit_code']==0
        path=folder/stage['name'];assert json.loads((path/'monitor.json').read_text())
        rows=[json.loads(line.split(' ',1)[1]) for line in (path/'output.log').read_text().splitlines() if line.startswith(('matched_result ','matched_frame_result '))]
        assert len(rows)==3 and {v['request'] for v in rows}=={0,1,2}
        candidate='-candidate-' in stage['name']
        case=json.loads((folder/(label+('-candidate-case.json' if candidate else '-control-case.json'))).read_text())
        for row in rows:
            assert row['case']==case and all(row[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished'))
            expected_nd=len(eligible)*sum(batch*c==80 for c in row['prefill_chunk_plan']) if candidate else 0
            assert row['ndarray_prefill_available']==candidate and row['ndarray_prefill_counts']==[expected_nd]+[0]*row['decode_steps_after_first']
            assert row['prefill_profile'].get('ndarray_prefill_dispatches',0)==expected_nd
            alpha=row['prefill_profile'].get('global_half_alpha_dispatches',0)
            if expected_nd:assert expected_nd==186 and alpha==0;nd_images+=1
            else:assert alpha>0
            assert row['coalesced_block_enabled'] and row['screened_head_enabled'] and row['global_half_alpha_enabled']
            if model=='Qwen3.5-0.8B':assert row['vision_reencoded'] and row['patch_sha256']==case['image_metadata']['patch_sha256']
            records+=1
        workers+=1
assert records==54 and workers==18 and nd_images==3
log=(jobs/'Qwen3.5-0.8B-b1-attempt0.log').read_text()
assert "assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0" in log and 'AssertionError' in log
result=dict(verified_files=len(proof['files']),full_output_records=records,successful_runtime_workers=workers,complete_ABBA_profiles=4,
    incomplete_vision_profile_workers=2,actual_NDArray_image_requests=nd_images,NDArray_dispatches_per_image=186,
    failure='Stale benchmark assumption: full NDArray route correctly has zero legacy global-half-alpha calls.',
    scope='Incomplete benchmark; all54 captured complete outputs match and actual NDArray routes verified. No admitted full-cohort timing, gain, vLLM ratio or target pass.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
