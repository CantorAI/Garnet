"""Completed2B B1 experiments on a common equal32 output subset, no projections."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
stages=[('', 'Original'),('-fused-attention','Attention'),('-fused-attention-chunk16','Chunk16'),
    ('-verified-mtp-d3','MTP depth3'),('-verified-mtp-d3-indexed8192','MTP3 + indexed'),
    ('-verified-mtp-d7-indexed8192','MTP7 + indexed'),
    ('-verified-mtp-d3-indexed8192-delta-replay','MTP3 + replay'),
    ('-verified-mtp-d7-indexed8192-delta-replay','MTP7 + replay'),
    ('-hybrid-p32-d4-mtp3-k8192-replay','Shared prefill32/decode4'),
    ('-hybrid-p64-d4-mtp3-k8192-replay','Shared P64 / D4'),
    ('-hybrid-p128-d4-mtp3-k8192-replay','Shared prefill128/decode4'),
    ('-packed-p64-d4-mtp3-k8192-replay','Packed P64 / D4'),
    ('-hybrid-p64-d4-mtp3-k8192-replay-resident','Shared + resident IO'),
    ('-hybrid-p64-d4-mtp3-k8192-replay-resident-serial','Resident IO + one context'),
    ('-hybrid-p64-d4-mtp3-k8192-replay-resident-serial-register128','Register delta128'),
    ('-packed-p64-d4-mtp3-k8192-replay-flat','Packed flat concat'),
    ('-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat','Packed + register + IO'),
    ('-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0','One target context + aux0')]
def read(file):return json.loads((root/file).read_text())
def key(row):return (row['trial'],row['workload_group'])
data=[]
for suffix,label in stages:
    path=root/f'matched512-garnet-2b-b1{suffix}.json'
    if not path.exists():continue
    rows=read(path.name)
    if len(rows)!=28:continue
    data.append((suffix,label,{key(r):r for r in rows}))
competitors={e:{key(r):r for r in read(f'matched512-{e}-2b-b1'+('-partitioned' if e=='lmstudio' else '')+'.json')} for e in ('lmstudio','vllm')}
mtp_path=root/'matched512-vllm-2b-b1-mtp3-cache128.json'
if mtp_path.exists():
    mtp_rows=read(mtp_path.name)
    if len(mtp_rows)==28:competitors['vllm_mtp']={key(r):r for r in mtp_rows}
keys=[k for k,r in data[0][2].items() if not r['warmup']
    and all(rows[k]['sampled_tokens_including_stop']==[32] for _,_,rows in data)
    and all(len(item['token_ids'])==32 for engine,rows in competitors.items() if engine.startswith('vllm') for item in rows[k]['records'])
    and all(item['result']['stats']['total_output_tokens']==32 for item in competitors['lmstudio'][k]['records'])]
assert keys
for k in keys:
    hashes=[rows[k]['input_sha256'] for _,_,rows in data]+[rows[k]['input_sha256'] for rows in competitors.values()]
    assert all(h==hashes[0] for h in hashes)
values=[statistics.median(rows[k]['wall_seconds']*1000 for k in keys) for _,_,rows in data]
labels=[label for _,label,_ in data];colors=['#a8acb8' if not suffix else '#7655d2' if 'mtp' not in suffix else '#dd9347' for suffix,_,_ in data]
for engine in competitors:
    values.append(statistics.median(competitors[engine][k]['wall_seconds']*1000 for k in keys))
    labels.append({'lmstudio':'LM Studio (MTP)','vllm':'vLLM (speculation off)','vllm_mtp':'vLLM MTP3'}[engine]);colors.append({'lmstudio':'#3687ca','vllm':'#159b83','vllm_mtp':'#0d665c'}[engine])
fig,(lat,mem)=plt.subplots(1,2,figsize=(17,8),facecolor='#f7f9fc',gridspec_kw={'width_ratios':[1.7,1]})
bars=lat.barh(labels,values,color=colors,height=.62);lat.set_ylim(len(values)-.5,-.5);lat.set_xlim(0,max(values)*1.2)
for bar,value in zip(bars,values):lat.text(value+6,bar.get_y()+bar.get_height()/2,f'{value:.1f} ms',va='center')
target=min(values[-len(competitors):])/2;lat.axvline(target,color='#d94a4a',linestyle='--',label=f'2× target: {target:.1f} ms (unmet)');lat.legend(loc='lower right',frameon=False)
lat.set_xlabel('Median group completion latency · lower is better');lat.set_title(f'Completed experiments · {len(keys)} common equal32 groups',loc='left',weight='bold')
owned=[]
for _,_,rows in data:
    points={rows[k]['owned_memory']['owned_model_bytes'] for k in keys};assert len(points)==1;owned.append(points.pop()/1e9)
bars=mem.barh(range(len(data)),owned,color=colors[:len(data)],height=.62)
for bar,value in zip(bars,owned):mem.text(value+.04,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center',fontsize=10)
mem.set_yticks(range(len(values)),['']*len(values));mem.tick_params(axis='y',length=0);mem.set_ylim(len(values)-.5,-.5)
mem.set_xlim(0,max(owned)*1.2);mem.set_xlabel('Retained owned GB (decimal)');mem.set_title('Garnet owned allocations',loc='left',weight='bold')
for ax in (lat,mem):
    ax.set_facecolor('#f7f9fc');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Qwen3.5-2B · batch1 · CUDA development',x=.03,ha='left',fontsize=20,weight='bold')
fig.text(.03,.91,'RTX4080 GPU0 · sequential runs · same unique JPEGs · 85prompt / 32output tokens · 512context slots',fontsize=11)
fig.text(.03,.075,'LM Studio uses MTP (runtime verified); vLLM results include speculation OFF and MTP3 when completed. Proposal settings differ.',fontsize=10)
fig.text(.03,.035,'Experimental options remain default OFF; independent quality gaps remain. Ownership is not process peak VRAM. No2× acceptance.',fontsize=10,color='#a52b2b')
fig.subplots_adjust(left=.18,right=.97,top=.82,bottom=.20,wspace=.18)
for ext in ('png','svg'):fig.savefig(root/f'progress-speculative-2b-b1.{ext}',dpi=160,facecolor=fig.get_facecolor())
print(root/'progress-speculative-2b-b1.png')
