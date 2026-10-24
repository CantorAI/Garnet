from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
out=Path(__file__).parent;source=out/'native-controls-abba-verified.json';v=json.loads(source.read_text());assert v['status']=='four_leg_controls_abba_evidence_verified';reference=out/'native-mtp3-shared-joined-verified.json';refs=json.loads(reference.read_text())['runs'];legs=['a1','b1','b2','a2'];colors=['#8797a7','#2279bd','#2279bd','#8797a7']
fig,axes=plt.subplots(2,3,figsize=(14,8.5));fig.patch.set_facecolor('#f7f9fc');records={}
metrics=[('decode_median_tokens_per_second','Decode · higher better','tok/s'),('ttft_median_ms','First token · lower better','ms'),('output_tokens_per_second','Cohort throughput · higher better','tok/s')]
for row,case in enumerate(('text','image')):
 for col,(metric,label,unit) in enumerate(metrics):
  ax=axes[row,col];data=v['table'][case+'-c1'][metric];values=[data['per_leg'][k] for k in legs];ordinary=[refs[k]['cohorts'][case+'-c1']['metrics'][metric] for k in ('vllm','sglang')];target=min(ordinary)/2 if col==1 else max(ordinary)*2
  records[case+'-'+metric]={'per_leg':dict(zip(legs,values)),'ordinary_reference_values':ordinary,'target':target,'on_off_ratio_of_means':data['on_off_ratio_of_means']}
  bars=ax.bar(range(4),values,color=colors,width=.65);ax.axhline(target,color='#c54a4a',ls='--',lw=1.25);ax.set_ylim(0,max(max(values)*1.2,target*1.18));ax.set_xticks(range(4),['A1\nOFF','B1\nON','B2\nON','A2\nOFF'],fontsize=9);ax.set_ylabel(unit);ax.set_title(case.title()+' · '+label,loc='left',fontsize=10,pad=10)
  for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+ax.get_ylim()[1]*.016,f'{value:.1f}',ha='center',va='bottom',fontsize=9)
  ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
fig.suptitle('RTX 5090 · controlled Garnet buffer test · 2× unmet',x=.055,ha='left',fontsize=18,fontweight='bold',y=.97)
fig.text(.055,.918,'Fresh OFF / ON / ON / OFF runs · identical native binary, hashed application sources, requests and token outputs verified',fontsize=10,color='#49556a')
fig.legend(handles=[Patch(color=colors[0],label='Reusable buffers OFF'),Patch(color=colors[1],label='Reusable buffers ON'),Line2D([0],[0],color='#c54a4a',ls='--',label='2× ordinary reference; half TTFT')],loc='upper left',bbox_to_anchor=(.049,.908),ncol=3,frameon=False,fontsize=9)
fig.text(.055,.102,'Observed mean of two decode medians: +6.7% text / +10.4% image. Text throughput is worse on average; all observed pauses are retained.',fontsize=9,color='#49556a')
fig.text(.055,.074,'Batch 1 · mixed NVFP4/FP8 + MTP3 · T128 prefill / T4 decode · 8 requests per cohort · 128 outputs · 41/104 input tokens · capacity 4096.',fontsize=9,color='#49556a')
fig.text(.055,.046,'Two runs per mode; performance varies. No default promotion. Before/after telemetry only. True GPU concurrency and tuned references remain pending.',fontsize=9,color='#49556a')
fig.subplots_adjust(left=.06,right=.98,top=.80,bottom=.19,hspace=.42,wspace=.30)
for ext in ('png','svg'):fig.savefig(out/('Garnet-Controls-ABBA.'+ext),dpi=170,facecolor=fig.get_facecolor())
(out/'Garnet-Controls-ABBA-data.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_source_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'data':records,'scope':v['scope']},indent=2))
