from pathlib import Path
import hashlib,json,sys,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent;small_path=root/'native-4b-http-verified.json';large_path=root/'native-mtp3-shared-joined-verified.json';small=json.loads(small_path.read_text(encoding='utf8'));large=json.loads(large_path.read_text(encoding='utf8'));assert small['status']=='native4b_http_evidence_verified'
assert small['prompt_ids_match_27b']==16
archive=root/'native-mtp3-shared-joined-evidence.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()==large['archive']['archive_sha256']
with zipfile.ZipFile(archive) as z:
 names=[n for n in z.namelist() if '/benchmark/' in n and n.endswith('-manifest.json')];assert len(names)==1;large_raw=json.loads(z.read(names[0]))['cohorts']
runs=[large['runs']['joined'],small['run']];colors=['#2279bd','#8b66c3'];labels=['27B mixed NVFP4/FP8 + MTP3 · T128/T4','4B BF16 greedy · T16/T16'];cases=['text','image']
fig,axes=plt.subplots(1,3,figsize=(12,5));fig.patch.set_facecolor('#f7f9fc');data={}
for ax,metric,title,unit,scale in zip(axes,['decode_median_tokens_per_second','ttft_median_ms','sampled_peak_adapter_used_bytes'],['Decode speed · higher better','Time to first token · lower better','Serving-cohort sampled VRAM'],['tok/s','ms','GiB'],[1,1,1/(1<<30)]):
 for i,run in enumerate(runs):
  if metric=='sampled_peak_adapter_used_bytes':
   source=large_raw if i==0 else small['raw_cohorts'];values=[next(v for v in source if v['case']==case)['sampled_peak_adapter_used_bytes'] for case in cases]
  else:values=[run['cohorts'][case+'-c1']['metrics'][metric] for case in cases]
  values=[v*scale for v in values];data.setdefault(metric,{})[str(i)]=values;xs=[x+(i-.5)*.34 for x in range(2)];bars=ax.bar(xs,values,width=.32,color=colors[i],label=labels[i])
  for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.1f}',ha='center',va='bottom',fontsize=10)
 ax.set_xticks([0,1],['Text','Image']);ax.set_ylabel(unit);ax.set_title(title,fontsize=11,loc='left');ax.set_ylim(0,ax.get_ylim()[1]*1.14);ax.spines[['top','right']].set_visible(False);ax.set_facecolor('white');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('RTX 5090 · Garnet 27B and 4B verified batch-1 pilots',x=.07,ha='left',fontsize=17,fontweight='bold',y=.97)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.06,.89),frameon=False,ncol=1,fontsize=10)
fig.text(.07,.16,'Same request content/images and prompt IDs verified; 41/104 input tokens, 128 outputs, context capacity 4096.',fontsize=9,color='#49556a')
fig.text(.07,.115,'Separate pilots with different architectures, precision, speculation and binaries; no causal gain or broad quality parity.',fontsize=9,color='#49556a')
fig.text(.07,.07,'VRAM is adapter-wide sampled usage. 4B references/repeats and true native batching remain pending; 27B 2× goal unmet.',fontsize=9,color='#49556a')
fig.subplots_adjust(left=.07,right=.98,top=.70,bottom=.28,wspace=.30)
for ext in ('png','svg'):fig.savefig(root/('Garnet-27B-vs-4B-B1.'+ext),dpi=170,facecolor=fig.get_facecolor())
(root/'Garnet-27B-vs-4B-B1-data.json').write_text(json.dumps({'verified_inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (small_path,large_path)},'series':labels,'metrics':data,'two_x_established':False},indent=2),encoding='utf8')
