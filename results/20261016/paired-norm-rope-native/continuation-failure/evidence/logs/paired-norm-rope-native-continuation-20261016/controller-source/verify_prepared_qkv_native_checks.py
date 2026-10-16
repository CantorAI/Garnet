"""Regenerate exact private QKV integration and all33 original-bound native validation records."""
from pathlib import Path
import hashlib,json,math,runpy,shutil,sys,tempfile
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='prepared-qkv-native-checks-20261016';assert proof['jobs'][trial]['exit_code']==0;checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text());preparation=json.loads((root/'preparation.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-prepared-qkv-text']
assert preparation['baseline_library_sha256']==proof['library_sha256']['build-vector-vision-width4']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert meta['control_sha256']==proof['library_sha256']['build-resident-kv-inplace']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
for name,sha in proof['library_sha256'].items():assert proof['omitted'][name+'/libgarnet.dylib']['sha256']==sha
for name,sha in preparation['source_sha256'].items():
 p=root/'source'/name
 if p.is_file():assert digest(p)==sha
 else:assert proof['omitted']['source/'+name]['sha256']==sha
sys.path.insert(0,str(root/'source/test2026/metal'));factory=runpy.run_path(str(root/'source/test2026/metal/make_prepared_qkv_native.py'));operations=factory['OPERATIONS'];assert set(preparation['changes'])==set(operations) and len(operations)==12
for name,change in preparation['changes'].items():
 original=root/'original-source'/name;assert digest(original)==change['before']==preparation['baseline_source_sha256'][name]
 assert (root/'source'/name).read_bytes()==operations[name](original.read_text()).encode() and digest(root/'source'/name)==change['after']
for name,sha in preparation['added_sources'].items():assert digest(root/'source'/name)==sha
manifest=preparation['source_sha256']
assert preparation['selection']['confirmed_models']==['Qwen3-0.6B'] and preparation['native_admission']==dict(selected_models=['Qwen3-0.6B'],warm_text_ttft_fraction_max=.97,each_batch=True,complete_cohorts=2,decode_fraction_min=.98,all_six_profiles_required=True,all_original_correctness_required=True)
external=json.loads((checks/'external-driver-identities.json').read_text());assert set(external)=={'run_prepared_qkv_native_checks.py','run_vector_vision_native_checks.py','run_ndarray_native_checks.py','run_ndarray_native_checks_repaired.py','launch_prepared_qkv_native_checks.py'}
for name,sha in external.items():assert digest(checks/'external-driver-snapshot'/name)==sha
with tempfile.TemporaryDirectory(prefix='garnet-qkv-audit-') as temporary:
 tools=Path(temporary)/'tools';tools.mkdir()
 for p in (root/'source/test2026/metal').glob('*.py'):shutil.copyfile(p,tools/p.name)
 for name in external:shutil.copyfile(checks/'external-driver-snapshot'/name,tools/name)
 saved=sys.argv
 try:
  sys.argv=['generate','/unused',str(Path(temporary)/'generated'),'--generate-only'];runpy.run_path(str(tools/'run_prepared_qkv_native_checks.py'),run_name='__main__')
 finally:sys.argv=saved
 assert (Path(temporary)/'generated.prepared_qkv.complete33.executed.py').read_bytes()==(checks.parent/'jobs.prepared_qkv.complete33.executed.py').read_bytes()
assert 'METAL_PREPARED_QKV_TEXT' in meta['options'] and 'METAL_COALESCED_BF16_BODY' in meta['options'] and 'METAL_MLP_PREFILL_GRAPH' not in meta['options']
# Reuse every original independent validation gate; only add stages and exact provenance paths.
original=(root/'source/test2026/metal/verify_vector_vision_native_checks.py').read_text();fragment=original[original.index("template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'"):original.rindex("exec(compile(source,__file__,'exec'))")]
exec(compile(fragment,'original_full30_auditor','exec'))
replace('assert len(stages)==30','assert len(stages)==33')
replace("assert stages[29]['name']=='wide-batch-trained-reference'","assert stages[32]['name']=='wide-batch-trained-reference'")
replace("names|{'vector-vision-primitive-compile','vector-vision-primitive'}","names|{'vector-vision-primitive-compile','vector-vision-primitive','prepared-qkv-compile','prepared-qkv-primitive','prepared-qkv-trained-reference'}")
replace("if stage['name']!='vector-vision-primitive-compile':assert log.stat().st_size>0","if stage['name'] not in ('vector-vision-primitive-compile','prepared-qkv-compile'):assert log.stat().st_size>0")
replace('validation_stages=30,ctests=7,','validation_stages=33,ctests=7,')
replace('result.update(regenerated_changes=8,external_driver_files=len(external),regenerated_guard_and_full30_runner=True)','result.update(regenerated_changes=12,external_driver_files=len(external),regenerated_guard_and_full33_runner=True)')
needle='Path(sys.argv[2]).write_bytes'
extra='''qkv=next(v for v in stages if v['name']=='prepared-qkv-primitive')['accepted_cases'][0]
assert (checks/qkv['evidence']/'prepared-qkv-primitive.log').read_text().strip()=='Prepared QKV heterogeneous24/fp32-key/pending/ownership/alias/abort-recovery/fallback/cache64-reject2 PASS weight_bytes=336'
compiled=next(v for v in stages if v['name']=='prepared-qkv-compile')['accepted_cases'][0]['command']
assert qkv['command']==[compiled[compiled.index('-o')+1]] and str(PurePosixPath(compiled[-1]).relative_to(preparation['target'])) in proof['omitted']
trained=next(v for v in stages if v['name']=='prepared-qkv-trained-reference')['accepted_cases'][0]
text=(checks/trained['evidence']/'prepared-qkv-trained-reference.log').read_text()
routes=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('prepared_qkv_trained_route ')]
assert len(routes)==24 and sum(r['expected'] for r in routes)==12 and all(r['expected']==r['session']==r['graph']==int(r['batch']*r['chunk'] in (24,44)) for r in routes)
lines=[line for line in text.splitlines() if line.startswith('dense reference full_attention ')];assert len(lines)==24
worst=0
for line in lines:
    fields=line.split()
    for label in ('hidden_l2_worst_ratio','value_l2_worst_ratio','worst_ratio'):
        value=float(fields[fields.index(label)+1]);assert math.isfinite(value) and 0<=value<=1;worst=max(worst,value)
assert sum(line.startswith('pretrained dense PASS ') for line in text.splitlines())==8
fixtures=list((checks/'trained-qkv-reference').glob('*/fixture.json'));assert len(fixtures)==8
for path in fixtures:
    fixture=json.loads(path.read_text());assert len(fixture['steps'])==len(fixture['invalid_inputs'])==3
    for step in fixture['steps']:
        assert len(step['inputs'])==6
        for name in step['inputs']+[step['expected']]:
            relative=PurePosixPath(name).relative_to(preparation['target']).as_posix();assert proof['omitted'][relative]['bytes']>0
result.update(prepared_qkv_primitive_cases=24,prepared_qkv_trained_profiles=8,prepared_qkv_trained_steps=24,prepared_qkv_serving_route_checks=12,prepared_qkv_fallback_route_checks=12,prepared_qkv_trained_original_bound_worst_ratio=worst)
'''
replace(needle,extra+needle);exec(compile(source,__file__,'exec'))
