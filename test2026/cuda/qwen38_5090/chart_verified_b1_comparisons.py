"""Reproduce historical B1 wins and model/precision charts from hash-bound evidence."""
from pathlib import Path
import json,hashlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Input directory contains retained verified JSONs; artifacts are written beside them.
# Usage: python chart_verified_b1_comparisons.py <evidence-directory>
out=Path(sys.argv[1]) if len(sys.argv)==2 else None
if out is None or not out.is_dir():
 raise SystemExit('Usage: chart_verified_b1_comparisons.py <existing evidence-directory>')
def load(n):return json.loads((out/n).read_text(encoding='utf-8-sig'))
d=load('Garnet-Joined-MLP-vs-References-B1-data.json')
assert hashlib.sha256((out/'native-mtp3-shared-joined-verified.json').read_bytes()).hexdigest()==d['verified_input_sha256']
f=load('Garnet-27B-vs-4B-B1-data.json')
for n,h in f['verified_inputs'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h
colors=['#287ab5','#e6a43b','#58a08b']
fig,axes=plt.subplots(2,2,figsize=(12,9));fig.set_facecolor('#f7f9fc')
for col,case in enumerate(['text','image']):
 for row,(metric,unit) in enumerate([('decode_median_tokens_per_second','tokens / second ↑'),('ttft_median_ms','milliseconds ↓')]):
  ax=axes[row,col];s=d['series'][case+'-'+metric];v=[s[k] for k in ['joined','vllm','sglang']]
  bars=ax.bar(range(3),v,color=colors,width=.6)
  for bar,x in zip(bars,v):ax.text(bar.get_x()+bar.get_width()/2,x+max(v)*.035,f'{x:.2f}',ha='center',fontweight='bold')
  win=v[0]/max(v[1:])-1 if row==0 else min(v[1:])/v[0]
  title=f'{case.title()} decode: Garnet +{win:.1%}' if row==0 else f'{case.title()} TTFT: Garnet {win:.2f}× faster'
  ax.set_title(title,loc='left',fontsize=13,pad=13);ax.set_xticks(range(3),['Garnet MTP3','vLLM ordinary','SGLang ordinary']);ax.set_ylabel(unit);ax.set_ylim(0,max(v)*1.25)
  ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Where Garnet measured ahead · batch 1',x=.06,y=.97,ha='left',fontsize=20,fontweight='bold')
fig.text(.06,.918,'Qwen3.8-27B VLM · RTX 5090 32GB · wins relative to the faster ordinary reference',fontsize=11,color='#465269')
fig.text(.06,.12,'Garnet: joined MLP, mixed NVFP4/FP8 + MTP3, T128 prefill / T4 decode. Separate historical B1 profile.',fontsize=10,color='#465269')
fig.text(.06,.085,'Matched short requests: 41 text / 104 image prompt tokens, 128 outputs, 8 requests/cohort. Capacity 4096.',fontsize=10,color='#465269')
fig.text(.06,.05,'Single pilots; ordinary references, no error bars. Broader quality, tuned references and actual 4K prompts pending.',fontsize=10,color='#465269')
fig.subplots_adjust(left=.08,right=.97,top=.85,bottom=.23,hspace=.44,wspace=.27)
for ext in ['png','svg']:fig.savefig(out/('Garnet-B1-Verified-Wins.'+ext),dpi=160,facecolor=fig.get_facecolor())
# Model and quantization tradeoffs, same native B1 workload; not model quality parity.
i=load('native-int4-http-verified.json');four=load('native-4b-http-verified.json');joined=load('native-mtp3-shared-joined-verified.json')
fig,axes=plt.subplots(2,2,figsize=(12,9));fig.set_facecolor('#f7f9fc')
for col,case in enumerate(['text','image']):
 for row,(metric,unit) in enumerate([('decode_median_tokens_per_second','tokens / second ↑'),('sampled_peak_adapter_used_bytes','GiB · adapter-wide sampled peak')]):
  ax=axes[row,col]
  if row==0:v=[joined['runs']['joined']['cohorts'][case+'-c1']['metrics'][metric],next(x[metric] for x in i['cohorts'] if x['case']==case),four['run']['cohorts'][case+'-c1']['metrics'][metric]]
  else:v=[f['metrics'][metric]['0'][col],next(x[metric] for x in i['cohorts'] if x['case']==case)/2**30,f['metrics'][metric]['1'][col]]
  bars=ax.bar(range(3),v,color=['#287ab5','#9672b2','#8996aa'],width=.6)
  for bar,x in zip(bars,v):ax.text(bar.get_x()+bar.get_width()/2,x+max(v)*.035,f'{x:.2f}',ha='center',fontweight='bold')
  ax.set_title(case.title()+' · '+('decode speed' if row==0 else 'GPU memory'),loc='left',fontsize=13);ax.set_xticks(range(3),['27B mixed\nNVFP4/FP8 + MTP3','27B INT4\nW4A16 greedy','4B BF16\ngreedy']);ax.set_ylabel(unit);ax.set_ylim(0,max(v)*1.23);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Native Garnet · model and quantization tradeoffs',x=.06,y=.97,ha='left',fontsize=19,fontweight='bold')
fig.text(.06,.918,'RTX 5090 32GB · batch 1 · measured short-request pilots',fontsize=11,color='#465269')
fig.text(.06,.12,'4B is a different model and precision; these bars do not establish equivalent output quality.',fontsize=10,color='#465269')
fig.text(.06,.085,'27B profiles also differ in MTP and execution settings. Adapter-wide sampled memory is not process-only allocation.',fontsize=10,color='#465269')
fig.text(.06,.05,'41 text / 104 image prompt tokens, 128 outputs; capacity 4096, not actual 4K prompts. Single pilots; no error bars.',fontsize=10,color='#465269')
fig.subplots_adjust(left=.08,right=.97,top=.85,bottom=.23,hspace=.5,wspace=.27)
for ext in ['png','svg']:fig.savefig(out/('Garnet-Quantization-Model-Comparison.'+ext),dpi=160,facecolor=fig.get_facecolor())
