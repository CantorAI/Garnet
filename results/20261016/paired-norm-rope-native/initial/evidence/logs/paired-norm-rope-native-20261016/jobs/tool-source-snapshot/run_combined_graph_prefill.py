"""Single native submission versus combined placeholder/constant FP32 weight graphs."""
from pathlib import Path
source=Path(__file__).with_name('run_mpsgraph_prefill.py').read_text()
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body').replace('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015')
source=source.replace("s/'mpsgraph_prefill_gpu.mm'","s/'combined_graph_prefill_gpu.mm'")
source=source.replace('paths=[Path(__file__),',"paths=[Path(__file__),s/'run_mpsgraph_prefill.py',")
source=source.replace("binary=out/'mpsgraph_prefill_gpu'","binary=out/'combined_graph_prefill_gpu'")
source=source.replace('Independent repeatedly resident projections; allocations/uploads/first graph compile outside warm windows. Native and graph use separate queues sequentially and synchronize each product.','Complete independent resident first-layer projection sets, all matrices included; native single command submission versus one synchronous multi-result graph. Policy1 has FP32 weight placeholders, policy2 has original FP32 weight constants. Initialization/uploads/first graph compile outside warm windows.')
source=source.replace('Private MPSGraph prefill, wall latency only. Original FP32 activation, matmul and output; original BF16 weights either expanded to FP32 or losslessly scaled-half then cast back before FP32 matmul.','Combined MPSGraph prefill, wall latency only. Original FP32 activation, weight values, matmul and output. No input deduplication, graph-visible model prefixes, complete-layer dependency graph or whole-model result.')
start=source.index("    skip=[v for v in values");source=source[:start]+'''    checks=[v for v in values if v['kind']=='check'];assert len(checks)==16
    all_checks.extend(dict(model=model,**v) for v in checks);first.extend(dict(model=model,**v) for v in values if v['kind']=='first_graph_run')
    for batch in (1,2):
        rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
        for policy in (1,2):
            relevant=[v for v in checks if (v['batch'],v['policy'])==(batch,policy)]
            assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4)) and all(v['rows']==rows and v['projections']==len(info['cases']) for v in relevant)
            windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
            if not all(v['passed'] for v in relevant):assert not windows;continue
            assert len(windows)==8
            for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
            assert all(len(v['wall_ms'])==8 and all(math.isfinite(x) and x>0 for x in v['wall_ms']) for v in windows)
            means={name:statistics.mean(x for v in windows if v['candidate']==candidate for x in v['wall_ms']) for name,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,batch=batch,rows=rows,policy=policy,checks=relevant,wall_ms=means,wall_fraction=means['candidate']/means['control']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,first_graph_runs=first,full_goal_met=False),indent=2)+'\\n').encode());print('Completed combined graph prefill component',model,flush=True)
'''
exec(compile(source,__file__,'exec'))
