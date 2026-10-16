"""Independent FP64 reconstruction of API gate successes and retained failures."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert len(proof['jobs'])==1;trial,status=next(iter(proof['jobs'].items()));out=root/'logs'/trial/'jobs'
meta=json.loads((out/'metadata.json').read_text());snap=out/'source-snapshot'
for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
for name,sha in meta['sdk_headers'].items():assert digest(out/name)==sha
assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
stages=json.loads((out/'statuses.json').read_text());compile_stage=next(v for v in stages if v['name']=='compile')
assert compile_stage['exit_code']==0
binary=Path(compile_stage['command'][compile_stage['command'].index('-o')+1]).name
assert proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']==meta['binary_sha256']
for stage in stages:
    monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
if 'generated_cpp_sha256' in meta:
    import runpy
    factory=runpy.run_path(str(snap/'test2026/metal/make_ndarray_alpha_probe.py'))
    expected=factory['generate']((snap/'test2026/metal/ndarray_compat_gpu.mm').read_text()).encode()
    assert (out/'ndarray_alpha_gpu.mm').read_bytes()==expected and digest(out/'ndarray_alpha_gpu.mm')==meta['generated_cpp_sha256']
summary=json.loads((out/'summary.json').read_text());observed=[];checks=0
for mode in summary['modes']:
    label=mode['mode'];storage=label.split('-')[0]
    stage=next(v for v in stages if v['name']==label);assert stage['exit_code']==mode['exit_code']
    values=[json.loads(v) for v in (out/(label+'.log')).read_text().splitlines() if v.startswith('{')]
    fixtures={v['shape']:v for v in values if v['kind']=='fixture'};quality=[v for v in values if v['kind']=='check']
    ratios=[]
    for f in fixtures.values():
        m,k,n=f['rows'],f['inner'],f['columns'];assert (m,k,n) in ((3,8,16),(16,1024,64),(80,64,32),(160,64,32))
        if 'source_count' in f:
            assert f['variant']==label.split('-')[1]
            assert (f['scale'],f['source_count'])==((0,2) if f['variant']=='unit' else (8,3))
            assert f['alpha_requested']==f['alpha_reported']==math.ldexp(1.,-f['scale']) and f['beta_reported']==0
        raw=np.asarray(f['bf16'],dtype='<u2').reshape(n,k);weights=(raw.astype('<u4')<<16).view('<f4')
        scaled=np.ldexp(weights,f['scale']);half=scaled.astype('<f2').view('<u2')
        assert np.array_equal(half,np.asarray(f['half'],dtype='<u2').reshape(n,k))
        assert np.array_equal(half.T,np.asarray(f['transposed_half'],dtype='<u2').reshape(k,n))
        assert np.array_equal(scaled.T.view('<u4'),np.asarray(f['wide'],dtype='<f4').reshape(k,n).view('<u4'))
        assert np.array_equal(np.ldexp(half.view('<f2').astype('<f4'),-f['scale']).view('<u4'),weights.view('<u4'))
    for q in quality:
        f=fixtures[q['shape']];m,k,n=f['rows'],f['inner'],f['columns'];assert q['pattern'] in range(4)
        raw=np.asarray(f['bf16'],dtype='<u2').reshape(n,k);w=(raw.astype('<u4')<<16).view('<f4').astype(np.float64)
        a=np.asarray(q['inputs'],dtype='<f4').reshape(m,k).astype(np.float64)
        c=np.asarray(q['control'],dtype='<f4').reshape(m,n).astype(np.float64);d=np.asarray(q['candidate'],dtype='<f4').reshape(m,n).astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(c).all() and np.isfinite(d).all()
        reference=a@w.T;absolute=np.abs(a)@np.abs(w).T;terms=math.ceil(k/32)+32;eps=2**-23;bound=terms*eps/(1-terms*eps)*absolute+1e-7
        native=float(np.max(np.abs(c-reference)/bound));candidate=float(np.max(np.abs(d-reference)/bound));comparison=float(np.max(np.abs(d-c)/(5e-6+5e-5*np.abs(c))))
        for key,number in (('native_fp64_worst',native),('fp64_worst',candidate),('comparison_worst',comparison)):
            assert math.isclose(q[key],number,rel_tol=1e-7,abs_tol=1e-10),(label,key,q[key],number)
        assert q['passed']==(native<=1 and candidate<=1 and comparison<=1)
        assert q['left_fp32'] and q['output_fp32'] and q['right_fp16']==(storage=='mixed') and q['right_fp32']==(storage=='f32')
        assert q['guard_preserved'] and q['immutable_inputs']
        diagnostic=float(np.max(np.abs(np.ldexp(d,-f['scale'])-c)/(5e-6+5e-5*np.abs(c))))
        ratios.append(dict(shape=q['shape'],pattern=q['pattern'],native_fp64_worst=native,candidate_fp64_worst=candidate,
            comparison_worst=comparison,passed=q['passed'],diagnostic_inverse_scale_comparison=diagnostic))
    compatible=stage['exit_code']==0
    assert mode['compatible']==compatible and mode['complete_checks']==len(quality)
    if compatible:assert len(fixtures)==4 and len(quality)==16 and all(q['passed'] for q in quality) and values[-1]==dict(kind='complete',checks=16)
    else:assert not any(v['kind']=='complete' for v in values)
    checks+=len(quality);observed.append(dict(mode=label,exit_code=stage['exit_code'],compatible=compatible,numerical_records=len(quality),checks=ratios))
assert summary['all_compatible']==all(v['compatible'] for v in observed)
assert (status['exit_code']==0)==summary['all_compatible']
result=dict(verified_files=len(proof['files']),numerical_records=checks,modes=observed,all_compatible=summary['all_compatible'],
    scope='Synthetic API compatibility only. Failed gates remain failed; inverse-scale comparisons are diagnostics, never admissions. Native bounds unchanged; statistic reconciliation tolerance is not an output tolerance. No model or speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='modes'}))
