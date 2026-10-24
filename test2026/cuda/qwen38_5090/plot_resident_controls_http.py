from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
source=root/'native-resident-controls-http-verified.json';new=json.loads(source.read_text(encoding='utf8'));assert new['status']=='resident_controls_http_evidence_verified'
oldsource=root/'native-mtp3-shared-joined-verified.json';old=json.loads(oldsource.read_text(encoding='utf8'))
d={'runs':dict(old['runs'],resident=new['run']),'archive':{'sha256':new['archive_sha256'],'files':new['files_verified']}}
series=[('joined','Garnet\nprevious pilot','#7d8b9a'),('resident','Garnet\nreusable buffers','#2279bd'),('vllm','vLLM\nordinary','#efad43'),('sglang','SGLang\nordinary','#64a790')]
fig,axes=plt.subplots(2,2,figsize=(12,8));fig.patch.set_facecolor('#f7f9fc')
data={'verified_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'prior_input_sha256':hashlib.sha256(oldsource.read_bytes()).hexdigest(),'archive':d['archive'],'series':{},'scope':'One B1 pilot per mode, 8 measured requests per cohort, 128 outputs, 41/104 prompt tokens, capacity4096. Ordinary references; tuning/MTP, broader precision/quality, long context/concurrency and repetitions remain pending. No 2x qualification.'}
for col,case in enumerate(('text','image')):
 key=case+'-c1'
 for row,(metric,title,unit) in enumerate([('decode_median_tokens_per_second','Decode speed — higher is better','tok/s'),('ttft_median_ms','Time to first token — lower is better','ms')]):
  ax=axes[row,col];values=[d['runs'][label]['cohorts'][key]['metrics'][metric] for label,_,_ in series]
  data['series'][case+'-'+metric]=dict(zip([label for label,_,_ in series],values))
  ax.set_facecolor('white');bars=ax.bar(range(4),values,color=[color for _,_,color in series],width=.62)
  baseline=max(values[2:]) if row==0 else min(values[2:]);target=baseline*2 if row==0 else baseline/2
  ax.axhline(target,color='#c54a4a',ls='--',lw=1.3)
  ax.text(3.4,target+2,('2× ordinary reference' if row==0 else 'Half fastest ordinary TTFT'),ha='right',va='bottom',fontsize=8,color='#ad3b3b')
  ax.set_ylim(0,max(max(values)*1.18,target*1.22));ax.set_xticks(range(4),[label for _,label,_ in series],fontsize=8)
  ax.set_ylabel(unit);ax.set_title(case.title()+' · '+title,loc='left',fontsize=11,pad=12)
  for bar,value in zip(bars,values):
   near=abs(value-target)<ax.get_ylim()[1]*.07
   ax.text(bar.get_x()+bar.get_width()/2,value-ax.get_ylim()[1]*.025 if near else value+ax.get_ylim()[1]*.018,f'{value:.1f}',ha='center',va='top' if near else 'bottom',fontsize=9,color='white' if near else 'black')
  ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
fig.suptitle('Reusable-buffer pilot vs references · 2× unmet',x=.06,ha='left',fontsize=17,fontweight='bold',y=.97)
fig.text(.06,.918,'RTX 5090 32GB · batch 1 · both Garnet modes: prefill T128 / decode T4, MTP3 / draft T4',fontsize=10,color='#49556a')
fig.text(.06,.09,'Matched request bytes/IPv4/counts verified. New pilot: 14 model controls + 20 exact target/MTP pairs + 20 retained identities + SSE gates.',fontsize=9,color='#49556a')
fig.text(.06,.065,'Short prompts: text 41 / image 104 tokens; 128 output tokens. Capacity 4096 does not mean a 4K-prompt benchmark.',fontsize=9,color='#49556a')
fig.text(.06,.04,'Separate pilots/binaries; no causal gain or error bars. Ordinary references. Tuning/MTP, broad quality, repeats and native concurrency pending.',fontsize=9,color='#49556a')
fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.20,hspace=.57,wspace=.23)
for ext in ('png','svg'):fig.savefig(root/('Garnet-Resident-Controls-vs-References-B1.'+ext),dpi=170,facecolor=fig.get_facecolor())
(root/'Garnet-Resident-Controls-vs-References-B1-data.json').write_text(json.dumps(data,indent=2))
