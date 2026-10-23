"""Plot complete audited matched-capacity cohorts; missing batches stay pending."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
fig,axes=plt.subplots(2,4,figsize=(18,9),facecolor='#f7f9fc')
for ax,(size,batch) in zip(axes.flat,[(s,b) for s in ('2b','4b') for b in (1,2,4,8)]):
    path=root/f'matched512-{size}-b{batch}-partitioned-summary.json'
    if not path.exists():
        ax.axis('off');ax.text(.5,.5,f'Batch {batch}: pending',ha='center');continue
    data=json.loads(path.read_text())
    assert all(data[k] for k in ('all_three_engines_present','all_input_hashes_equal','garnet_vllm_prompt_ids_equal','lmstudio_independent_capacity_verified'))
    values=[data['equal_work_median_group_completion_ms'][e] for e in ('garnet','lmstudio','vllm')]
    ax.set_facecolor('#f7f9fc')
    bars=ax.barh(['Garnet','LM Studio','vLLM'],values,color=['#7c4dff','#3687ca','#159b83'],height=.5)
    ax.invert_yaxis();ax.set_xlim(0,max(values)*1.3);ax.xaxis.grid(alpha=.15);ax.set_axisbelow(True)
    for bar,value in zip(bars,values):ax.text(value+10,bar.get_y()+bar.get_height()/2,f'{value:.0f} ms',va='center',weight='bold')
    target=min(values[1:])/2;ax.axvline(target,color='#df4c4c',linestyle='--')
    ax.set_title(f'{size.upper()} · batch {batch}\n{data["equal_32_sample_groups"]} equal-output groups',loc='left',weight='bold',pad=14,fontsize=11)
    ax.text(.98,.96,f'2× goal: {target:.0f} ms',transform=ax.transAxes,ha='right',va='top',color='#c13737',fontsize=9)
    ax.set_xlabel('Group completion (ms), lower is better',fontsize=9)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Qwen3.5 CUDA: completed matched-work baseline',x=.04,ha='left',y=.98,fontsize=22,weight='bold')
fig.text(.04,.925,'RTX 4080 GPU0 · sequential engines · independent 512-token slots · 32 sampled tokens per row in selected groups',fontsize=12)
fig.text(.04,.105,'Same JPEG hashes; Garnet/vLLM prompt IDs match. LM Studio uses llama.cpp; raw prompt IDs unavailable.',fontsize=11)
fig.text(.04,.065,'Intermediate arithmetic and API timing differ. Full prompt work verified; all-cache-off settings and broad quality remain unverified.',fontsize=10,color='#536174')
fig.text(.04,.025,'Dashed line: half the fastest baseline latency. The 2× goal remains unmet. Group latency is separate from aggregate throughput.',fontsize=11,color='#536174')
fig.subplots_adjust(left=.085,right=.98,top=.82,bottom=.21,wspace=.55,hspace=.65)
for ext in ('png','svg'):fig.savefig(root/f'progress-matched512.{ext}',dpi=160,facecolor=fig.get_facecolor())
print(root/'progress-matched512.png')
