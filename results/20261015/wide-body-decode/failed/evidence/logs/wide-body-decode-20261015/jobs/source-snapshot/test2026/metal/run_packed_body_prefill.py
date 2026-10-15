"""Original physical prefill shapes: selected global-alpha MPS versus packed FP32 tiles."""
from pathlib import Path
import ast
wrapper=Path(__file__).with_name('run_streamed_direct_mps_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_direct_mps_body import context as generate_context,harness as generate_harness','from make_packed_body_prefill import context as generate_context,header as generate_header')
source=source.replace("build=root/'build-screened-head'","build=root/'build-interleaved-bf16-body'")
needle="assert digest(build/'libgarnet.dylib')=='2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23'";assert source.count(needle)==1
source=source.replace(needle,"assert digest(build/'libgarnet.dylib')==json.loads((root/'logs/interleaved-body-native-checks-20261015/jobs/metadata.json').read_text())['candidate_sha256']")
source=source.replace("paths=[Path(__file__),","paths=[Path(__file__),s/'make_packed_body_prefill.py',s/'make_prefill_tile_shader.py',s/'packed_body_prefill_gpu.mm',s/'run_streamed_direct_mps_body.py',root/'source/src/backends/metal/interleaved_bf16_body_shader.h',root/'source/src/backends/metal/screened_head_index.h',root/'source/src/backends/metal/screened_head_shader.h',root/'source/src/backends/metal/screened_head_native_impl.h',")
source=source.replace("cpp.write_bytes((s/'streamed_direct_mps_body_gpu.mm').read_bytes())","cpp.write_bytes((s/'packed_body_prefill_gpu.mm').read_bytes())\nheader_cpp=out/'metal_context.h';header_cpp.write_bytes(generate_header((root/'source/src/backends/metal/metal_context.h').read_text()).encode())\nmetadata['private_header_sha256']=digest(header_cpp)")
source=source.replace("out/'streamed_direct_mps_body_gpu","out/'packed_body_prefill_gpu").replace("out/'metal_context_direct.mm'","out/'metal_context_packed_prefill.mm'")
source=source.replace("run(model+'-gpu',[str(binary),str(assets)])","run(model+'-gpu',[str(binary),str(assets),model])")
source=source.replace('original-sized inputs; raw BF16 control and forced mixed-storage MPS GEMM in the same isolated context copy','Frozen physical prefill rows24/44,26/52,80/160; FP32 activation and accumulation, original BF16 values in four-output storage; current global-alpha MPS control. Every projection set uses one submission; no complete-model or vLLM ratio')
needle='defines=[v for v in shlex.split(match.group(1)) if v.startswith(\'-DGARNET_METAL_\')]';assert source.count(needle)==1
source=source.replace(needle,needle+"\nassert '-DGARNET_METAL_INTERLEAVED_BF16_BODY=1' in defines")
start=source.index("    checks=[v for v in values");source=source[:start]+'''    checks=[v for v in values if v['kind']=='check'];assert len(checks)==24
    all_checks.extend(dict(model=model,**v) for v in checks)
    for batch in (1,2):
        rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
        for policy in (1,2,3):
            relevant=[v for v in checks if (v['batch'],v['policy'])==(batch,policy)];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
            assert all(v['rows']==rows and v['projections']==len(info['cases']) for v in relevant)
            windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
            if not all(v['passed'] for v in relevant):assert not windows;continue
            assert len(windows)==8 and all(v['rows']==rows for v in windows)
            for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
            assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
            means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,batch=batch,physical_rows=rows,policy=policy,checks=relevant,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,full_goal_met=False),indent=2)+'\\n').encode());print('Completed packed prefill component',model,flush=True)
'''
doc=ast.parse(source).body[0];assert isinstance(doc,ast.Expr) and isinstance(doc.value,ast.Constant) and isinstance(doc.value.value,str)
source=repr(__doc__)+'\n'+''.join(source.splitlines(keepends=True)[doc.end_lineno:])
exec(compile(source,__file__,'exec'))
