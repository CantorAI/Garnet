"""Regenerate width64 native integration/full32 driver and preserve every original numerical gate."""
from pathlib import Path
import hashlib,json,math,re,runpy,shutil,sys,tempfile
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256'],name
trial='delta-value-tile64-native-checks-20261016';assert proof['jobs'][trial]['exit_code']==0;checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text());preparation=json.loads((root/'preparation.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-delta-value-tile64-prefill']
assert preparation['baseline_library_sha256']==proof['library_sha256']['build-vector-vision-width4']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert meta['control_sha256']==proof['library_sha256']['build-resident-kv-inplace']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
for name,sha in proof['library_sha256'].items():assert proof['omitted'][name+'/libgarnet.dylib']['sha256']==sha
manifest=preparation['source_sha256']
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
sys.path.insert(0,str(root/'source/test2026/metal'));factory=runpy.run_path(str(root/'source/test2026/metal/make_delta_value_tiles_native.py'));operations=factory['OPERATIONS'];assert set(operations)==set(preparation['changes']) and len(operations)==12
for name,change in preparation['changes'].items():
 original=root/'original-source'/name;assert digest(original)==change['before']==preparation['baseline_source_sha256'][name]
 assert (root/'source'/name).read_bytes()==operations[name](original.read_text()).encode() and digest(root/'source'/name)==change['after']
assert len(preparation['added_sources'])==4
for name,sha in preparation['added_sources'].items():assert digest(root/'source'/name)==sha
component=runpy.run_path(str(root/'source/test2026/metal/make_delta_value_tiles.py'));reference=root/'selected-component/original-recurrence.metal';assert digest(reference)==preparation['original_shader_sha256'] and reference.read_bytes()==component['shader']((root/'original-source/src/backends/metal/metal_context.mm').read_text()).encode()
assert preparation['selection']['confirmed_workloads']==[dict(width=64,kind='prefill')] and preparation['selection']['source_assets_and_design_unchanged'] and preparation['selection']['complete_numeric_profiles']==416 and preparation['selection']['timing_samples']==12800
assert preparation['native_policy']==dict(batch=[1,2],chunk=80,heads=16,key_dim=128,value_dim=128,tile=64,threads_per_group=128,decode_unchanged=True,other_shapes_retain_original=True,original_arithmetic=True,original_buffer_writes=True,extra_weight_storage=False)
assert preparation['native_admission']==dict(warm_vl_ttft_fraction_max=.99,each_batch=True,complete_cohorts=2,each_profile_decode_fraction_min=.98,each_profile_ttft_fraction_max=1.02,all_six_profiles_required=True,all_original_correctness_required=True)
external=json.loads((checks/'external-driver-identities.json').read_text());assert set(external)=={'run_delta_value_tiles_native_checks.py','run_vector_vision_native_checks.py','run_ndarray_native_checks.py','run_ndarray_native_checks_repaired.py','launch_delta_value_tiles_native_checks.py'}
for name,sha in external.items():assert digest(checks/'external-driver-snapshot'/name)==sha
with tempfile.TemporaryDirectory(prefix='garnet-delta-value64-audit-') as temporary:
 tools=Path(temporary)/'tools';tools.mkdir()
 for p in (root/'source/test2026/metal').glob('*.py'):shutil.copyfile(p,tools/p.name)
 for name in external:shutil.copyfile(checks/'external-driver-snapshot'/name,tools/name)
 saved=sys.argv
 try:
  sys.argv=['generate','/unused',str(Path(temporary)/'generated'),'--generate-only'];runpy.run_path(str(tools/'run_delta_value_tiles_native_checks.py'),run_name='__main__')
 finally:sys.argv=saved
 assert (Path(temporary)/'generated.delta_value64.complete32.executed.py').read_bytes()==(checks.parent/'jobs.delta_value64.complete32.executed.py').read_bytes()
assert 'METAL_DELTA_VALUE_TILE64' in meta['options'] and 'METAL_COALESCED_BF16_BODY' in meta['options'] and 'METAL_MLP_PREFILL_GRAPH' not in meta['options'] and 'METAL_PAIRED_NORM_ROPE' not in meta['options']
assert proof['audit_dependency_sha256']==preparation['audit_dependency_sha256']=={'verify_vector_vision_native_checks.py':'0e57a8ad01c71e549abce786886cc2050283c85f0fc487e816faef0656d013be','verify_mlp_prefill_native_checks.py':'2eb33f027c120dba530aed406ddb60796bfee9484b2e83c10f8a0cc8ab19d2fa'}
for name,sha in proof['audit_dependency_sha256'].items():assert digest(root/'external-validation-tools'/name)==sha
original=(root/'external-validation-tools/verify_vector_vision_native_checks.py').read_text();fragment=original[original.index("template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'"):original.rindex("exec(compile(source,__file__,'exec'))")]
fragment=fragment.replace("template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'","template=root/'external-validation-tools/verify_mlp_prefill_native_checks.py'");exec(compile(fragment,'original_full30_auditor','exec'))
replace('assert len(stages)==30','assert len(stages)==32');replace("assert stages[29]['name']=='wide-batch-trained-reference'","assert stages[31]['name']=='wide-batch-trained-reference'")
replace("names|{'vector-vision-primitive-compile','vector-vision-primitive'}","names|{'vector-vision-primitive-compile','vector-vision-primitive','delta-value64-compile','delta-value64-primitive'}")
replace("if stage['name']!='vector-vision-primitive-compile':assert log.stat().st_size>0","if stage['name'] not in ('vector-vision-primitive-compile','delta-value64-compile'):assert log.stat().st_size>0")
replace('validation_stages=30,ctests=7,','validation_stages=32,ctests=7,')
replace('result.update(regenerated_changes=8,external_driver_files=len(external),regenerated_guard_and_full30_runner=True)','result.update(regenerated_changes=12,external_driver_files=len(external),regenerated_guard_and_full32_runner=True)')
extra='''primitive=next(v for v in stages if v['name']=='delta-value64-primitive')['accepted_cases'][0]
summary=json.loads((checks/primitive['evidence']/'delta-value64-primitive.log').read_text());expected_words=sum(3*b*h*v*(c+128) for b in (1,2,3) for c in (1,16,80) for h in (2,16) for v in (96,128,256))
assert summary==dict(kind='complete',checks=162,selected=6,fallback=156,original_fp32_words_checked=expected_words,carried_steps=3,pending_abort_recovery_alias_uninitialized_foreign=True,model_performance_measured=False)
compiled=next(v for v in stages if v['name']=='delta-value64-compile')['accepted_cases'][0]['command'];assert primitive['command']==[compiled[compiled.index('-o')+1],preparation['target']+'/selected-component/original-recurrence.metal']
assert str(PurePosixPath(primitive['command'][0]).relative_to(preparation['target'])) in proof['omitted']
route_steps=0;serving=0;fallback=0;worst=0
for stage_name,total,selected_count in [('qwen35-extended-prefill',36,6),('wide-batch-trained-reference',12,6)]:
    record=next(v for v in stages if v['name']==stage_name)['accepted_cases'][0];text=(checks/record['evidence']/(stage_name+'.log')).read_text()
    routes=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('delta_value_tile64_trained_route ')]
    assert len(routes)==total and sum(r['expected'] for r in routes)==selected_count
    assert all(r['expected']==r['session']==r['graph']==int(r['kind']=='linear_attention' and r['batch'] in (1,2) and r['chunk']==80) for r in routes)
    ratios=[float(v) for v in re.findall(r'worst_ratio ([0-9.eE+-]+)',text)];assert len(ratios)==total and all(math.isfinite(v) and 0<=v<=1 for v in ratios)
    route_steps+=len(routes);serving+=sum(r['expected'] for r in routes);fallback+=sum(not r['expected'] for r in routes);worst=max(worst,max(ratios))
result.update(delta_value_tile64_primitive_cases=162,delta_value_tile64_selected_cases=6,delta_value_tile64_fallback_cases=156,delta_value_tile64_original_words_checked=expected_words,delta_value_tile64_trained_steps=route_steps,delta_value_tile64_trained_selected_routes=serving,delta_value_tile64_trained_fallback_routes=fallback,delta_value_tile64_trained_original_bound_worst_ratio=worst)
'''
replace('Path(sys.argv[2]).write_bytes',extra+'Path(sys.argv[2]).write_bytes');exec(compile(source,__file__,'exec'))
