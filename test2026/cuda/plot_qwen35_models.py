"""Per-model comparisons with historical/current precision policy explicit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
two=json.loads((root/'fused-chunk4-graph-comparison.json').read_text())['paired_wall_ms']
four=json.loads((root/'initial-4b-batch1-comparison.json').read_text())['equal_sampled_token_work_subset']
fig,axes=plt.subplots(1,2,figsize=(13,6.2),facecolor='#f7f9fc')
items=[('2B: historical checkpoint','12 matched cases; precision-policy revalidation pending',
    [two['fused_chunk4_graph'],two['lmstudio'],two['vllm']]),
    ('4B: current strict-reduction build','20 matched cases; 32 sampled tokens each',
    [four[k]['median_wall_ms'] for k in ['garnet','lmstudio','vllm']])]
current_two=root/'initial-2b-strict-batch1-comparison.json'
if current_two.exists():
    current=json.loads(current_two.read_text())['equal_sampled_token_work_subset']
    count=current['garnet']['requests'];tokens=current['garnet']['sampled_token_counts']
    items[0]=('2B: current strict-reduction build',f'{count} matched cases; sampled token counts: {tokens}',
        [current[k]['median_wall_ms'] for k in ['garnet','lmstudio','vllm']])
for ax,(title,subtitle,values) in zip(axes,items):
    ax.set_facecolor('#f7f9fc')
    bars=ax.barh(['Garnet fused','LM Studio / llama.cpp','vLLM'],values,color=['#7c4dff','#3687ca','#159b83'],height=.5)
    ax.invert_yaxis();ax.set_xlim(0,1500);ax.xaxis.grid(alpha=.15);ax.set_axisbelow(True)
    for b,v in zip(bars,values):ax.text(v+18,b.get_y()+b.get_height()/2,f'{v:,.0f} ms',va='center',weight='bold')
    target=min(values[1:])/2
    ax.axvline(target,color='#df4c4c',linestyle='--',linewidth=2)
    ax.set_title(title,loc='left',fontsize=14,weight='bold',pad=35)
    ax.text(0,1.035,subtitle,transform=ax.transAxes,fontsize=9,color='#536174')
    ax.text(.98,.50,f'2x target: {target:.0f} ms',transform=ax.transAxes,ha='right',va='center',color='#c13737',fontsize=10)
    ax.set_xlabel('Median full-request latency (ms) · lower is better',labelpad=12,fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
fig.suptitle('Qwen3.5 CUDA comparisons so far',x=.035,ha='left',fontsize=22,weight='bold',y=.96)
fig.text(.035,.86,'Batch 1 · common real-image variants · engines run sequentially · 2x goal remains unmet',fontsize=12)
fig.text(.035,.12,'BF16 source weights; intermediate arithmetic / API timing differ. LM Studio context512 vs384; actual prompt IDs unavailable.',fontsize=10,color='#536174')
fig.text(.035,.075,'Separate memory captures now available. These are latency results; broad quality and comparative batch performance remain pending.',fontsize=10,color='#536174')
fig.subplots_adjust(left=.15,right=.97,top=.68,bottom=.25,wspace=.60)
for ext in ['png','svg']:fig.savefig(root/f'progress-models.{ext}',dpi=160,facecolor=fig.get_facecolor())
print(root/'progress-models.png')
