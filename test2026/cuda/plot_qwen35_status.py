"""Completed measurements only; keep historical timing and owned memory distinct."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
two=json.loads((root/'fused-chunk4-graph-comparison.json').read_text())
four=json.loads((root/'pretrained-4b-image-fused-chunk4.json').read_text())
p=two['paired_wall_ms']
fig,(ax,mem)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[2.4,1]},facecolor='#f7f9fc')
plt.rcParams.update({'font.size':12})
for a in (ax,mem):
    a.set_facecolor('#f7f9fc');a.set_axisbelow(True)
    for spine in a.spines.values():spine.set_visible(False)
    a.tick_params(length=0)
labels=['Garnet original','Garnet chunk-4','Garnet fused','LM Studio / llama.cpp','vLLM']
values=[p['serial'],p['chunk4_graph'],p['fused_chunk4_graph'],p['lmstudio'],p['vllm']]
bars=ax.barh(labels,values,color=['#a4adba','#b5a0eb','#7c4dff','#3687ca','#159b83'],height=.55)
ax.invert_yaxis();ax.set_xlim(0,990);ax.xaxis.grid(alpha=.15)
for b,v in zip(bars,values):ax.text(v+12,b.get_y()+b.get_height()/2,f'{v:.0f} ms',va='center',weight='bold')
target=min(p['vllm'],p['lmstudio'])/2
ax.axvline(target,color='#df4c4c',linestyle='--',linewidth=2)
ax.set_title('2B latency: last completed comparison',loc='left',weight='bold',fontsize=14,pad=20)
ax.set_xlabel('Median full-request latency (ms) · lower is better',labelpad=14)
ax.text(.98,.99,f'2x goal: {target:.0f} ms',transform=ax.transAxes,ha='right',va='top',color='#c13737',weight='bold')
sizes=[two['owned_model_bytes']/1e9,four[-1]['owned_memory']['owned_model_bytes']/1e9]
bars=mem.bar(['2B','4B'],sizes,color=['#7c4dff','#e89b40'],width=.55)
mem.set_ylim(0,11.5);mem.yaxis.grid(alpha=.15);mem.set_ylabel('Retained owned model memory (decimal GB)',labelpad=10)
mem.set_title('Memory diagnostics',loc='left',weight='bold',fontsize=14,pad=20)
for b,v in zip(bars,sizes):mem.text(b.get_x()+b.get_width()/2,v+.2,f'{v:.2f} GB',ha='center',weight='bold')
fig.suptitle('Garnet CUDA: latest completed evidence',x=.04,ha='left',fontsize=22,weight='bold',y=.96)
fig.text(.04,.885,'2B: 12 matched cases, 32 sampled tokens, sequential engine runs. 4B competitive cohort is in progress.',color='#536174',fontsize=12)
fig.text(.04,.14,'Timing and memory above precede the explicit cuBLAS reduction-policy fix; current-binary revalidation is pending.',fontsize=11,weight='bold')
fig.text(.04,.095,'Memory counts retained tensors + tracked TensorRT runtime; excludes external/transient/graph overhead. Not process peak VRAM.',fontsize=10,color='#536174')
fig.text(.04,.055,'4B memory is a two-request smoke diagnostic. Precision/API/context limitations and broader quality validation remain open.',fontsize=10,color='#536174')
fig.subplots_adjust(left=.17,right=.97,top=.79,bottom=.25,wspace=.35)
for ext in ('png','svg'):fig.savefig(root/f'progress-status.{ext}',dpi=160,facecolor=fig.get_facecolor())
print(root/'progress-status.png')
