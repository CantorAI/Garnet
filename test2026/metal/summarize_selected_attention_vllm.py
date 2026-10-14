"""Record remaining targets and separate worker initialization/first-use data."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();job=root/'evidence/logs/selected-attention-vllm-20261013/jobs'
summary=json.loads((job/'summary.json').read_text());rows=[]
for profile in summary['profiles']:
    result=json.loads((job/Path(profile['evidence']).name/'summary.json').read_text());means=result['means'];g=means['garnet'];v=means['vllm-metal']
    first={backend:statistics.mean(r['ttft_seconds'] for r in result['records'] if r['backend']==backend and r['request']==0)*1000 for backend in means}
    rows.append(dict(model=profile['model'],batch=profile['batch'],garnet_warm_tps_per_row=g['decode_tps'],vllm_warm_tps_per_row=v['decode_tps'],garnet_warm_tps_aggregate=g['decode_tps']*profile['batch'],vllm_warm_tps_aggregate=v['decode_tps']*profile['batch'],decode_ratio=result['decode_ratio'],garnet_warm_ttft_ms=g['ttft_ms'],vllm_warm_ttft_ms=v['ttft_ms'],ttft_fraction=result['ttft_fraction'],decode_target_tps_per_row=2*v['decode_tps'],ttft_target_ms=.5*v['ttft_ms'],additional_decode_factor_required=max(1,2*v['decode_tps']/g['decode_tps']),additional_ttft_reduction_percent_required=max(0,100*(1-.5*v['ttft_ms']/g['ttft_ms'])),initialization_seconds={b:means[b]['initialization_s'] for b in means},first_request_ttft_ms=first,decode_target_met=result['decode_target_met'],ttft_target_met=result['ttft_target_met']))
output=dict(scope='Additional matched-FP32 selected-library study; BF16 versus lossless FP32 weight-storage widths differ; no stock BF16 claim',initialization_and_first_use='Separate observations; no sum presented as an observed cold end-to-end TTFT; OS/driver caches not flushed',profiles=rows,all_targets_met=all(r['decode_target_met'] and r['ttft_target_met'] for r in rows))
(root/'remaining-budgets.json').write_bytes((json.dumps(output,indent=2)+'\n').encode())
for r in rows:print(json.dumps({k:r[k] for k in ('model','batch','garnet_warm_tps_per_row','vllm_warm_tps_per_row','decode_ratio','garnet_warm_ttft_ms','vllm_warm_ttft_ms','ttft_fraction','additional_decode_factor_required','additional_ttft_reduction_percent_required')}))
