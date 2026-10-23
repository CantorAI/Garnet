"""Retained ownership across completed matched512 profiles, not process peak."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
fig,ax=plt.subplots(figsize=(10,6),facecolor='#f7f9fc');ax.set_facecolor('#f7f9fc')
for size,color in [('2b','#3687ca'),('4b','#7c4dff')]:
    data=json.loads((root/f'matched512-{size}-allocation-growth.json').read_text())
    profiles=data['profiles'];x=[p['batch'] for p in profiles]
    y=[p['bytes']['owned_model_bytes']/1e9 for p in profiles]
    ax.plot(x,y,'o-',color=color,label=f'Qwen3.5-{size.upper()}',linewidth=2)
    for b,value in zip(x,y):ax.annotate(f'{value:.2f} GB',(b,value),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
ax.set_xticks([1,2,4,8]);ax.set_ylim(0,15);ax.grid(alpha=.15)
ax.set_xlabel('Batch profile');ax.set_ylabel('Retained owned bytes / 1e9 (GB)');ax.legend(loc='upper left')
for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Qwen3.5 CUDA: retained model ownership',x=.08,ha='left',fontsize=19,weight='bold')
fig.text(.08,.89,'RTX 4080 GPU0 · 512 context · chunk4 · BF16 weights / FP32 state',fontsize=11)
fig.text(.08,.075,'Includes tracked engine allocations, request state, packed output, shared head and vision.',fontsize=10)
fig.text(.08,.035,'Excludes some transient/driver/library allocations. This is not process peak VRAM or physical residency.',fontsize=10,color='#536174')
fig.subplots_adjust(left=.1,right=.96,top=.84,bottom=.18)
for ext in ('png','svg'):fig.savefig(root/f'progress-matched512-owned-memory.{ext}',dpi=160,facecolor=fig.get_facecolor())
print(root/'progress-matched512-owned-memory.png')
