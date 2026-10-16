"""Regenerate private decode norm-RoPE integration and all34 original-bound native records."""
from pathlib import Path
import hashlib,json,math,runpy,shutil,sys,tempfile
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='paired-norm-rope-native-checks-20261016';assert proof['jobs'][trial]['exit_code']==0;checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text());preparation=json.loads((root/'preparation.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-paired-norm-rope-decode']
assert preparation['baseline_library_sha256']==proof['library_sha256']['build-vector-vision-width4']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert meta['control_sha256']==proof['library_sha256']['build-resident-kv-inplace']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
for name,sha in proof['library_sha256'].items():assert proof['omitted'][name+'/libgarnet.dylib']['sha256']==sha
for name,sha in preparation['source_sha256'].items():
 p=root/'source'/name
 if p.is_file():assert digest(p)==sha
 else:assert proof['omitted']['source/'+name]['sha256']==sha
sys.path.insert(0,str(root/'source/test2026/metal'));factory=runpy.run_path(str(root/'source/test2026/metal/make_paired_norm_rope_native.py'));operations=factory['operations'](preparation['selected_models']);assert set(preparation['changes'])==set(operations) and len(operations)==12
for name,change in preparation['changes'].items():
 original=root/'original-source'/name;assert digest(original)==change['before']==preparation['baseline_source_sha256'][name]
 assert (root/'source'/name).read_bytes()==operations[name](original.read_text()).encode() and digest(root/'source'/name)==change['after']
for name,sha in preparation['added_sources'].items():assert digest(root/'source'/name)==sha
manifest=preparation['source_sha256']
assert preparation['selected_models']==sorted(v['model'] for v in preparation['selection']['confirmed_workloads'])==['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B'] and all(v['kind']=='decode' for v in preparation['selection']['confirmed_workloads'])
assert preparation['native_admission']==dict(unmet_decode_profiles=[dict(model='Qwen3-0.6B',batch=1),dict(model='Qwen2.5-Coder-0.5B-Instruct',batch=1)],minimum_decode_gain=1.01,at_least_one_same_unmet_profile_in_both=True,each_profile_decode_fraction_min=.98,each_profile_ttft_fraction_max=1.02,all_six_profiles_required=True,complete_cohorts=2,all_original_correctness_required=True)
external=json.loads((checks/'external-driver-identities.json').read_text());assert set(external)=={'run_paired_norm_rope_native_checks.py','run_vector_vision_native_checks.py','run_ndarray_native_checks.py','run_ndarray_native_checks_repaired.py','launch_paired_norm_rope_native_checks.py'}
for name,sha in external.items():assert digest(checks/'external-driver-snapshot'/name)==sha
with tempfile.TemporaryDirectory(prefix='garnet-paired-rope-audit-') as temporary:
 tools=Path(temporary)/'tools';tools.mkdir()
 for p in (root/'source/test2026/metal').glob('*.py'):shutil.copyfile(p,tools/p.name)
 for name in external:shutil.copyfile(checks/'external-driver-snapshot'/name,tools/name)
 saved=sys.argv
 try:
  sys.argv=['generate','/unused',str(Path(temporary)/'generated'),'--generate-only'];runpy.run_path(str(tools/'run_paired_norm_rope_native_checks.py'),run_name='__main__')
 finally:sys.argv=saved
 assert (Path(temporary)/'generated.paired_rope.complete34.executed.py').read_bytes()==(checks.parent/'jobs.paired_rope.complete34.executed.py').read_bytes()
assert 'METAL_PAIRED_NORM_ROPE' in meta['options'] and 'METAL_COALESCED_BF16_BODY' in meta['options'] and 'METAL_MLP_PREFILL_GRAPH' not in meta['options']
# Reuse every original independent validation gate; only add stages and exact provenance paths.
original=(root/'source/test2026/metal/verify_vector_vision_native_checks.py').read_text();fragment=original[original.index("template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'"):original.rindex("exec(compile(source,__file__,'exec'))")]
exec(compile(fragment,'original_full30_auditor','exec'))
replace('assert len(stages)==30','assert len(stages)==34')
replace("assert stages[29]['name']=='wide-batch-trained-reference'","assert stages[33]['name']=='wide-batch-trained-reference'")
replace("names|{'vector-vision-primitive-compile','vector-vision-primitive'}","names|{'vector-vision-primitive-compile','vector-vision-primitive','paired-rope-compile','paired-rope-primitive','paired-rope-text-trained-reference','paired-rope-coder-trained-reference'}")
replace("if stage['name']!='vector-vision-primitive-compile':assert log.stat().st_size>0","if stage['name'] not in ('vector-vision-primitive-compile','paired-rope-compile'):assert log.stat().st_size>0")
replace('validation_stages=30,ctests=7,','validation_stages=34,ctests=7,')
replace('result.update(regenerated_changes=8,external_driver_files=len(external),regenerated_guard_and_full30_runner=True)','result.update(regenerated_changes=12,external_driver_files=len(external),regenerated_guard_and_full34_runner=True)')
needle='Path(sys.argv[2]).write_bytes'
extra='''primitive=next(v for v in stages if v['name']=='paired-rope-primitive')['accepted_cases'][0]
assert (checks/primitive['evidence']/'paired-rope-primitive.log').read_text().strip()=='Paired norm-RoPE60/original-bits/pending/foreign/alias/abort-recovery/fallback PASS checked_values=3710720'
compiled=next(v for v in stages if v['name']=='paired-rope-compile')['accepted_cases'][0]['command']
assert primitive['command']==[compiled[compiled.index('-o')+1]] and str(PurePosixPath(compiled[-1]).relative_to(preparation['target'])) in proof['omitted']
worst=0;serving=0;fallback=0
for model in ('text','coder'):
    trained=next(v for v in stages if v['name']=='paired-rope-'+model+'-trained-reference')['accepted_cases'][0]
    text=(checks/trained['evidence']/('paired-rope-'+model+'-trained-reference.log')).read_text()
    routes=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('paired_norm_rope_trained_route ')]
    assert len(routes)==24 and sum(r['expected'] for r in routes)==18 and all(r['expected']==r['session']==r['graph']==int(r['batch']*r['chunk'] in (1,2)) for r in routes)
    serving+=sum(r['expected'] for r in routes);fallback+=sum(not r['expected'] for r in routes)
    lines=[line for line in text.splitlines() if line.startswith('dense reference full_attention ')];assert len(lines)==24
    for line in lines:
        fields=line.split()
        for label in ('hidden_l2_worst_ratio','value_l2_worst_ratio','worst_ratio'):
            value=float(fields[fields.index(label)+1]);assert math.isfinite(value) and 0<=value<=1;worst=max(worst,value)
    assert sum(line.startswith('pretrained dense PASS ') for line in text.splitlines())==8
    fixtures=list((checks/('trained-paired-rope-'+model)).glob('*/fixture.json'));assert len(fixtures)==8
    for path in fixtures:
        fixture=json.loads(path.read_text());assert len(fixture['steps'])==len(fixture['invalid_inputs'])==3
        for step in fixture['steps']:
            assert len(step['inputs'])==6
            for name in step['inputs']+[step['expected']]:
                relative=PurePosixPath(name).relative_to(preparation['target']).as_posix();assert proof['omitted'][relative]['bytes']>0
result.update(paired_rope_primitive_cases=60,paired_rope_primitive_output_bits=3710720,paired_rope_trained_profiles=16,paired_rope_trained_steps=48,paired_rope_serving_route_checks=serving,paired_rope_fallback_route_checks=fallback,paired_rope_trained_original_bound_worst_ratio=worst)
'''
replace(needle,extra+needle);exec(compile(source,__file__,'exec'))
