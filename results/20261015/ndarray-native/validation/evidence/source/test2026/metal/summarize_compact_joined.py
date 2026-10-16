"""Keep first-use TTFT, startup and worker RSS separate from warm results."""
import json,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();results=[]
for name in ('compact-joined-native-20261013','compact-joined-native-confirm-20261013'):
    folder=base/'logs'/name/'jobs'
    if not folder.exists():continue
    summary=json.loads((folder/'summary.json').read_text())
    for profile in summary['profiles']:
        records=json.loads((folder/Path(profile['evidence']).name/'summary.json').read_text())['records'];values={}
        for policy in ('control','candidate'):
            first=[v for v in records if v['build_policy']==policy and v['request']==0];assert len(first)==2
            values[policy]=dict(first_use_requests=2,first_use_ttft_ms=statistics.mean(v['ttft_seconds'] for v in first)*1000,initialization_s=statistics.mean(v['initialization_seconds'] for v in first),peak_individual_rss_bytes=statistics.mean(v['peak_individual_rss_bytes'] for v in first),warm_ttft_ms=profile['means'][policy]['ttft_ms'])
        results.append(dict(run=name,model=profile['model'],batch=profile['batch'],values=values,first_use_ttft_fraction=values['candidate']['first_use_ttft_ms']/values['control']['first_use_ttft_ms'],rss_delta_mib=(values['candidate']['peak_individual_rss_bytes']-values['control']['peak_individual_rss_bytes'])/2**20))
output=dict(scope=__doc__,cold_hardware_established=False,initialization_and_ttft_clocks_separate=True,unique_gpu_memory_established=False,profiles=results)
Path(sys.argv[2]).write_bytes((json.dumps(output,indent=2)+'\n').encode())
for p in results:print(p['run'],p['model'],p['batch'],p['values']['control']['first_use_ttft_ms'],p['values']['candidate']['first_use_ttft_ms'],p['rss_delta_mib'])
