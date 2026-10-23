"""Selected experimental attention measurements; correctness remains open."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
fig,axes=plt.subplots(2,2,figsize=(13,9),facecolor='#f7f9fc')
for row,(size,batch) in enumerate([('2b',1),('4b',8)]):
    suffix='-fused-attention-chunk16' if size=='2b' else '-fused-attention'
    path=root/f'matched512-{size}-b{batch}{suffix}-candidate.json'
    if not path.exists():
        for ax in axes[row]:ax.axis('off');ax.text(.5,.5,'Pending',ha='center')
        continue
    data=json.loads(path.read_text());times=data['median_group_ms']
    ax=axes[row,0];values=[times[e] for e in ('control','candidate','lmstudio','vllm')]
    candidate_label='Attention + chunk16' if size=='2b' else 'Attention candidate'
    bars=ax.barh(['Prior Garnet',candidate_label,'LM Studio','vLLM'],values,
        color=['#c1abf5','#7c4dff','#3687ca','#159b83'],height=.55)
    ax.invert_yaxis();ax.set_xlim(0,max(values)*1.25)
    for bar,value in zip(bars,values):ax.text(value+max(values)*.02,bar.get_y()+bar.get_height()/2,f'{value:,.0f} ms',va='center')
    target=min(times['lmstudio'],times['vllm'])/2;ax.axvline(target,color='#df4c4c',linestyle='--')
    ax.set_title(f'{size.upper()} batch {batch} · {data["equal_32_four_way_groups"]} matched groups\nLatency: {(1-data["candidate_control_latency_fraction"])*100:.1f}% lower than prior Garnet',loc='left',weight='bold',fontsize=12)
    ax.set_xlabel('Median group completion (ms) · lower is better')
    ax=axes[row,1];owned=data['retained_owned_bytes']
    assert all(len(owned[e])==1 for e in ('control','candidate'))
    values=[owned[e][0]/1e9 for e in ('control','candidate')]
    bars=ax.bar(['Prior Garnet',candidate_label],values,color=['#c1abf5','#7c4dff'],width=.5)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.15,f'{value:.3f} GB',ha='center',weight='bold')
    ax.set_ylim(0,max(values)*1.24);ax.set_ylabel('Retained owned bytes / 1e9 (GB)')
    ax.set_title(f'{size.upper()} batch {batch} · retained ownership\n{values[0]-values[1]:.3f} GB lower; not process peak VRAM',loc='left',weight='bold',fontsize=12)
for ax in axes.flat:
    ax.set_facecolor('#f7f9fc');ax.grid(axis='x' if ax in axes[:,0] else 'y',alpha=.15);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Experimental CUDA changes: faster, smaller, still below target',x=.04,ha='left',y=.98,fontsize=20,weight='bold')
fig.text(.04,.925,'RTX 4080 GPU0 · sequential runs · independent 512-token slots · 32 output tokens per row in selected groups',fontsize=11)
fig.text(.04,.105,'Default OFF. Strict batch8 numerical check and independent teacher-forced choices remain unresolved; no quality acceptance.',fontsize=10,color='#a52b2b')
fig.text(.04,.065,'2B uses chunk16; 4B uses chunk4. Same images/prompt counts; saved competitor cohorts reused. Arithmetic/API timing differ.',fontsize=10)
fig.text(.04,.025,'Dashed line: 2× fastest-baseline goal (unmet). Ownership excludes some transient/driver/library memory; it is not peak VRAM.',fontsize=10,color='#536174')
fig.subplots_adjust(left=.16,right=.96,top=.82,bottom=.2,wspace=.35,hspace=.65)
for ext in ('png','svg'):fig.savefig(root/f'progress-fused-attention.{ext}',dpi=160,facecolor=fig.get_facecolor())
print(root/'progress-fused-attention.png')
