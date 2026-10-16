"""Preserve exact combined-graph math; identify failing projections without timing."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,"probe=out/'combined_graph_error_diagnostic_gpu.mm'\nrun('harness',[sys.executable,str(s/'make_combined_graph_error_diagnostic.py'),str(probe)])\n"+needle+"s/'make_combined_graph_error_diagnostic.py',s/'run_combined_graph_prefill.py',")
source=source.replace("str(s/'combined_graph_prefill_gpu.mm')","str(probe)")
source=source.replace("binary=out/'combined_graph_prefill_gpu'","binary=out/'combined_graph_error_diagnostic_gpu'")
needle="(out/'metadata.json').write_bytes";at=source.index(needle)
source=source[:at]+"metadata.update(scope='Per-projection numerical diagnosis of unchanged combined graph math, all original matrices/patterns. No performance measurements.',limitations='No tolerance changes, input changes, graph routing changes or timed windows. First graph calls remain setup observations only.',generated_cpp_sha256=digest(probe))\n"+source[at:]
source=source.replace('profiles=[];all_checks=[];skipped=[];first=[]','profiles=[];all_checks=[];skipped=[];first=[];per_projection=[]')
start=source.index("    checks=[v for v in values")
source=source[:start]+'''    assert all(v['kind'] in ('projection_check','check','first_graph_run') for v in values)
    checks=[dict(model=model,**v) for v in values if v['kind']=='check'];assert len(checks)==16
    details=[dict(model=model,**v) for v in values if v['kind']=='projection_check'];assert len(details)==len(info['cases'])*16
    all_checks.extend(checks);per_projection.extend(details)
    first.extend(dict(model=model,**v) for v in values if v['kind']=='first_graph_run')
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],checks=all_checks,per_projection=per_projection,first_graph_runs=first,full_goal_met=False),indent=2)+'\\n').encode())
    print('Completed combined graph numerical diagnosis',model,flush=True)
'''
exec(compile(source,__file__,'exec'))
