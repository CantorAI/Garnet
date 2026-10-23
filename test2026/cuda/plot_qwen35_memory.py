"""Common WDDM process-counter scope; do not mix in allocator or VM counters."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
sources={(engine,size):json.loads((root/f'memory-{engine}-{size}-b1-summary.json').read_text())
    for engine in ('garnet','lmstudio') for size in ('2b','4b')}
assert len({r['adapter']['uuid'] for r in sources.values()})==1
for record in sources.values():
    assert not record['ambiguous_nonzero_duplicate_samples']
    for phase in ('steady_before','steady_after'):assert record['phases'][phase]['samples']>=2
fig,ax=plt.subplots(figsize=(11,6),facecolor='#f7f9fc');ax.set_facecolor('#f7f9fc')
x=np.arange(2);width=.29
for engine,offset,color,label in [('garnet',-width/2,'#7c4dff','Garnet fused'),('lmstudio',width/2,'#3687ca','LM Studio / llama.cpp')]:
    values=[sources[engine,size]['phases']['steady_before']['bytes']['DedicatedUsage']['median']/1e9 for size in ('2b','4b')]
    bars=ax.bar(x+offset,values,width,color=color,label=label)
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.18,f'{v:.2f} GB',ha='center',weight='bold',fontsize=12)
ax.set_xticks(x,['Qwen3.5-2B','Qwen3.5-4B']);ax.set_ylim(0,11.6)
ax.set_ylabel('Warm dedicated usage (decimal GB)',labelpad=10)
ax.yaxis.grid(alpha=.15);ax.set_axisbelow(True);ax.tick_params(length=0)
for s in ax.spines.values():s.set_visible(False)
ax.legend(frameon=False,loc='upper left')
fig.suptitle('GPU memory: isolated model processes',x=.055,ha='left',y=.96,fontsize=21,weight='bold')
fig.text(.055,.865,'GPU0 verified by PCI ID + CUDA LUID · WDDM process counters · warm idle before inference',fontsize=11,color='#536174')
fig.text(.055,.15,'Shared usage, recorded separately: Garnet 0.082 GB; LM Studio 0.090 GB, for both model sizes.',fontsize=11)
fig.text(.055,.10,'Warm values also stable after inference. Sampled observations, not true peaks. LM loading was outside this capture.',fontsize=10,color='#536174')
fig.text(.055,.055,'Batch1; Garnet context384, LM512. vLLM uses a separate worker-allocator scope; Windows VM counters are ambiguous.',fontsize=10,color='#536174')
fig.subplots_adjust(left=.12,right=.97,top=.79,bottom=.24)
for ext in ('png','svg'):fig.savefig(root/f'progress-memory.{ext}',dpi=170,facecolor=fig.get_facecolor())
print(root/'progress-memory.png')
