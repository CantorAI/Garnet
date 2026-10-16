"""Fixed four-endpoint workload screen after complete 32-stage chained calibration."""
from pathlib import Path
import sys
template=Path(__file__).with_name('analyze_paired_norm_rope_learned.py').read_text()
needle='result=dict(profiles=profiles,';assert template.count(needle)==1
template=template.replace(needle,'''workloads=[]
for model in sorted({v['model'] for v in profiles}):
 for kind in ('decode','prefill'):
  selected=[v for v in profiles if v['model']==model and ('decode' if v['rows']<=2 else 'prefill')==kind];assert len(selected)==4
  for v in selected:v['profile_screen_pass']=v['null_within_five_percent'] and all(v['fractions']['comparison'][f]<=.95 for f in ('wall_ms','gpu_ms'))
  workloads.append(dict(model=model,kind=kind,endpoint_profiles=4,component_screen_pass=all(v['profile_screen_pass'] for v in selected)))
result=dict(carried_stages=32,workloads=workloads,unchanged_confirmation_admitted=any(v['component_screen_pass'] for v in workloads),profiles=profiles,''')
template=template.replace('Descriptive component results only. GPU kernel/CPU encode savings and single-submit wall times are distinct. Own-nulls are not gain normalization. Native admission requires separate repeated complete-model evidence.','Distinct carried-chain component: require >=5% GPU AND wall gain at all first/last-layer x B1/B2 endpoints of a workload, with each own-null within 5%. Preserve all samples; no null normalization. One complete unchanged confirmation only on a passing complete workload. No native/model gain.')
template=template.replace("stable_null_profiles=sum(p['null_within_five_percent'] for p in profiles),", "stable_null_profiles=sum(p['null_within_five_percent'] for p in profiles),workloads=workloads,unchanged_confirmation_admitted=result['unchanged_confirmation_admitted'],")
exec(compile(template,__file__,'exec'))
