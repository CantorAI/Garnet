"""Audit original full gates plus actual wide prefill, unchanged outputs and bounds."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_interleaved_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-interleaved-screen-index','build-wide-batch-prefill'),('build-small-screen-seed','build-interleaved-screen-index'),('interleaved-head-checks-20261015','wide-batch-prefill-checks-20261015'),('interleaved-head-native-20261015','wide-batch-prefill-native-20261015'),('interleaved-head-original-source-20261015','wide-batch-prefill-original-source-20261015'),('d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876','837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471'),('len(stages)==27','len(stages)==28'),("dependencies['pid']==33362","dependencies['pid']==69725"),("dependencies['completed_stages']==27","dependencies['completed_stages']==28")]:source=source.replace(old,new)
source=source.replace("names={'vision-full-grid'","names={'wide-batch-trained-reference','vision-full-grid'")
source=source.replace("assert row['interleaved_screen_enabled']==(policy=='candidate')","assert row['interleaved_screen_enabled']")
source=source.replace("row['interleaved_screen_counts']==[1 if policy=='candidate' else 0]*len(counts)","row['interleaved_screen_counts']==[1]*len(counts)")
needle="        for row in rows:";assert source.count(needle)==1
source=source.replace(needle,"        original_case=dict(case);case=json.loads((folder/(label+'-'+policy+'-case.json')).read_text())\n        assert {k:v for k,v in case.items() if k!='chunk'}=={k:v for k,v in original_case.items() if k!='chunk'}\n"+needle)
needle="            assert row['screened_head_enabled'] and row['global_half_alpha_enabled']";assert source.count(needle)==1
source=source.replace(needle,needle+"\n            wide=model=='Qwen3.5-0.8B' and batch==2 and policy=='candidate'\n            assert (row['prefill_profile'].get('wide_batch_prefill_dispatches',0)>0)==wide\n            if model=='Qwen3.5-0.8B':assert row['prefill_chunk_plan']==(([80] if wide else [32,32,16]) if batch==2 else [80])")
needle='Path(sys.argv[2]).write_bytes';assert source.count(needle)==1
source=source.replace(needle,'''assert 'Apple M4' in proof['hardware']['machdep.cpu.brand_string'] and int(proof['hardware']['hw.memsize'])==16*2**30
assert 'METAL_QWEN35_WIDE_BATCH_PREFILL' in json.loads((checks/'metadata.json').read_text())['options']
assert 'metal_wide_batch_prefill_reference' in ctext
identities=json.loads((trial/'source-identities.json').read_text())
for name,sha in identities.items():assert digest(trial/'source-snapshot'/name)==sha and digest(root/'source'/name)==sha
stage=next(v for v in json.loads((checks/'statuses.json').read_text()) if v['name']=='wide-batch-trained-reference')
text=(checks/stage['accepted_cases'][0]['evidence']/'wide-batch-trained-reference.log').read_text()
ratios=[float(v) for v in re.findall(r'worst_ratio ([0-9.eE+-]+)',text)]
import math
assert len(ratios)==12 and all(math.isfinite(v) and 0<=v<=1 for v in ratios) and text.count('pretrained chunk PASS ')==4
fixtures=list((checks/'wide-reference/fixtures').glob('*/fixture.json'));assert len(fixtures)==4;coverage=set()
from pathlib import PurePosixPath
for path in fixtures:
    f=json.loads(path.read_text());coverage.add((f['kind'],f['layer']))
    assert f['batch']==2 and f['chunk']==80 and f['capacity']==256 and len(f['steps'])==3
    assert f['config']==json.loads((root/'models/Qwen3.5-0.8B/config.json').read_text())['text_config']
    for step in f['steps']:
        for name,shape in zip(step['inputs'],f['shapes']):
            rel=str(PurePosixPath(name).relative_to('/Users/zsdx/garnet-metal-investigation'));assert proof['omitted'][rel]['bytes']==math.prod(shape)*4
        rel=str(PurePosixPath(step['expected']).relative_to('/Users/zsdx/garnet-metal-investigation'));assert proof['omitted'][rel]['bytes']==step['count']*4
assert coverage=={('linear_attention',0),('linear_attention',22),('full_attention',3),('full_attention',23)}
for p in profiles:
    directory=native/Path(p['evidence']).name;meta=json.loads((directory/'metadata.json').read_text());model=p['model']
    for name in ('model.safetensors','tokenizer.json'):
        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]
result.update(wide_reference_profiles=4,wide_carried_steps=12,worst_wide_original_bound_ratio=max(ratios),source_closure_files=len(identities))
'''+needle)
exec(compile(source,__file__,'exec'))
