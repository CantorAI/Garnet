"""Common-subset latency history and separately sampled Windows memory."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
base='packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat'
stages=[('Original\ncombined',''),('One context\naux0','-targetserial-aux0'),
        ('Deferred\ncommit','-targetserial-aux0-deferred'),
        ('Native prefix\ncommit','-targetserial-aux0-deferred-nativecommit'),
        ('Shared\noutputs','-targetserial-aux0-deferred-nativecommit-arena')]
whole_suffix='-targetserial-aux0-deferred-nativecommit-arena-wholeprogram'
whole_memory='memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-wholeprogram-summary.json'
whole_ready=(root/f'matched512-garnet-4b-b8-{base}{whole_suffix}.json').exists() and (root/whole_memory).exists()
if whole_ready:stages.append(('No RDC\nexperimental',whole_suffix))
sources={label:f'matched512-garnet-4b-b8-{base}{suffix}.json' for label,suffix in stages}
sources.update({'vLLM serial':'matched512-vllm-4b-b8.json',
    'vLLM MTP3':'matched512-vllm-4b-b8-mtp3-cache128-convbfloat16.json',
    'LM Studio':'matched512-lmstudio-4b-b8-partitioned.json'})
data={label:{(r['trial'],r['workload_group']):r for r in json.loads((root/name).read_text())} for label,name in sources.items()}
assert all(len(rows)==28 for rows in data.values())
def count(label,r):
    if label.startswith('vLLM'):return [len(q['token_ids']) for q in r['records']]
    if label=='LM Studio':return [q['result']['stats']['total_output_tokens'] for q in r['records']]
    return r['sampled_tokens_including_stop']
original=data[stages[0][0]]
for key,r in original.items():assert all(rows[key]['input_sha256']==r['input_sha256'] for rows in data.values())
keys=[k for k,r in original.items() if not r['warmup'] and all(count(label,rows[k])==[32]*8 for label,rows in data.items())]
assert keys
medians={label:statistics.median(rows[k]['wall_seconds'] for k in keys)*1000 for label,rows in data.items()}
memory=[('Old short\n1 context','memory-combined-4b-b8-512-summary.json'),
        ('One context\naux0','memory-combined-4b-b8-512-targetserial-aux0-summary.json'),
        ('Deferred\ncommit','memory-combined-4b-b8-512-targetserial-aux0-deferred-summary.json'),
        ('Native prefix\ncommit','memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-summary.json'),
        ('Shared\noutputs','memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-summary.json'),
        ('LM Studio','memory-lmstudio-4b-b8-512-partitioned-summary.json')]
if whole_ready:memory.insert(-1,('No RDC\nexperimental',whole_memory))
samples=[];all_hashes=[]
for label,name in memory:
    summary=json.loads((root/name).read_text());events=json.loads((root/name.replace('-summary.json','-session.json')).read_text())
    hashes=[e['result']['input_sha256'] for e in events if 'result' in e] if label=='LM Studio' else [e['input_sha256'] for e in events if 'input_sha256' in e]
    assert len(hashes)==6;all_hashes.append(hashes)
    steady=summary['phases']['steady_after']['bytes']
    session_name=name.replace('-summary.json','-session.json')
    samples.append({'label':label,'source':name,'source_sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),
        'session_source':session_name,'session_sha256':hashlib.sha256((root/session_name).read_bytes()).hexdigest(),
        'dedicated':steady['DedicatedUsage']['median']/1e9,'shared':steady['SharedUsage']['median']/1e9})
assert all(h==all_hashes[0] for h in all_hashes)
fig,axes=plt.subplots(1,2,figsize=(14,6),facecolor='#f7f9fc')
ax=axes[0];x=np.arange(len(stages));values=[medians[label] for label,_ in stages]
bars=ax.bar(x,values,color=['#b9c0c9','#6b9fbd','#458dad','#247e9e','#d58b37','#9d7ab0'][:len(stages)],width=.64)
for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+20,f'{v:.0f}',ha='center',fontsize=10)
for label,color,style in [('LM Studio','#3687ca','--'),('vLLM MTP3','#91a9a2',':'),('vLLM serial','#159b83','--')]:
    ax.axhline(medians[label],color=color,linestyle=style,linewidth=1.4,label=f'{label}: {medians[label]:.0f} ms')
threshold=min(medians[label] for label in ('LM Studio','vLLM MTP3','vLLM serial'))/2
ax.axhline(threshold,color='#c73f3f',linestyle=':',label=f'2× target: {threshold:.0f} ms')
ax.set_xticks(x,[label for label,_ in stages],fontsize=9);ax.set_ylabel('Median group completion ms')
ax.set_title(f'Latency · same {len(keys)} equal32 groups',loc='left',weight='bold');ax.set_ylim(0,max(values)*1.16)
ax.legend(frameon=False,fontsize=8,loc='upper right')
ax=axes[1];x=np.arange(len(samples));ded=[s['dedicated'] for s in samples];shared=[s['shared'] for s in samples]
memory_colors=['#b9c0c9','#6b9fbd','#458dad','#247e9e','#d58b37']+(['#9d7ab0'] if whole_ready else [])+['#3687ca']
ax.bar(x,ded,color=memory_colors,width=.63,label='Dedicated')
ax.bar(x,shared,bottom=ded,color='#e0b6bb',width=.63,label='Shared')
for i,(d,s) in enumerate(zip(ded,shared)):ax.text(i,d+s+.2,f'{d:.2f}',ha='center',fontsize=9)
ax.set_xticks(x,[s['label'] for s in samples],fontsize=8);ax.set_ylim(0,18.5)
ax.set_title('Warm Windows process GPU counters',loc='left',weight='bold');ax.set_ylabel('GB decimal · sampled dedicated + shared')
ax.legend(frameon=False,fontsize=8,loc='upper right')
for ax in axes:
    ax.set_facecolor('#f7f9fc');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Qwen3.5-4B · batch 8 · completed memory methods',x=.04,ha='left',fontsize=18,weight='bold')
fig.text(.04,.89,'RTX4080 GPU0 · engines run sequentially · BF16 source · 512 tokens/slot · fresh real-image work',fontsize=10)
fig.text(.04,.075,'Memory: separate six-group diagnostics, identical JPEG hashes. Old short run had one context; old timing reached two. No-RDC result needs repeat.',fontsize=8.2)
fig.text(.04,.04,'Counters are not residency or lifetime peaks. vLLM WDDM worker usage unavailable. Proposal/arithmetic settings differ. Independent quality open; 2× unmet.',fontsize=8.5,color='#a52b2b')
fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.21,wspace=.24)
for ext in ('png','svg'):fig.savefig(root/f'progress-4b8-latency-memory.{ext}',dpi=160,facecolor=fig.get_facecolor())
manifest={'scope':'Common-subset latency; separate matched six-group sampled memory. No inferred missing values.',
    'latency_sources':{label:{'file':name,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for label,name in sources.items()},
    'common_group_keys':keys,'median_ms':medians,'memory':samples,'goal_met':False}
(root/'progress-4b8-latency-memory-sources.json').write_text(json.dumps(manifest,indent=2))
print(root/'progress-4b8-latency-memory.png')
