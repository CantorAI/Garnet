"""Completed combined profiles only; no extrapolated batches or memory claims."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
suffix='-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat'
fig,axes=plt.subplots(2,2,figsize=(13,8),facecolor='#f7f9fc')
colors={'candidate':'#d58b37','lmstudio':'#3687ca','vllm':'#159b83'}
sources=[]
for column,size in enumerate(('2b','4b')):
    cohorts=[]
    for batch in (1,2,4,8):
        path=root/f'matched512-{size}-b{batch}{suffix}-candidate.json'
        mtp_path=root/f'matched512-{size}-b{batch}{suffix}-mtp3-cache128-candidate.json'
        if mtp_path.exists():path=mtp_path
        for extra in ('-targetserial-aux0','-targetserial-aux0-mtp3-cache128',
                      '-targetserial-aux0-deferred','-targetserial-aux0-deferred-mtp3-cache128',
                      '-targetserial-aux0-deferred-nativecommit','-targetserial-aux0-deferred-nativecommit-mtp3-cache128',
                      '-targetserial-aux0-deferred-nativecommit-arena','-targetserial-aux0-deferred-nativecommit-arena-mtp3-cache128'):
            newer=root/f'matched512-{size}-b{batch}{suffix}{extra}-candidate.json'
            if newer.exists():path=newer
        if path.exists():
            cohorts.append((batch,json.loads(path.read_text())))
            sources.append({'model':size,'batch':batch,'summary':path.name,
                'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    ax=axes[0,column];mem=axes[1,column];x=np.arange(len(cohorts));width=.23
    for index,(engine,label) in enumerate((('candidate','Garnet D4 profiles'),('lmstudio','LM Studio'),('vllm','vLLM'))):
        vals=[d['median_group_ms'][engine] for _,d in cohorts]
        bars=ax.bar(x+(index-1)*width,vals,width,color=colors[engine],label=label)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+7,f'{v:.0f}',ha='center',fontsize=8)
    targets=[min(d['median_group_ms']['lmstudio'],d['median_group_ms']['vllm'])/2 for _,d in cohorts]
    ax.scatter(x,targets,marker='_',s=250,color='#c73f3f',label='2× threshold',zorder=5)
    ax.set_title(f'Qwen3.5-{size.upper()} · group latency',loc='left',weight='bold');ax.set_ylabel('Median ms · lower is better')
    ax.set_xticks(x,[f'Batch {b}\n{d["equal_32_four_way_groups"]} equal32 groups' for b,d in cohorts],fontsize=9);ax.set_ylim(0,max([d['median_group_ms'][e] for _,d in cohorts for e in colors])*1.2)
    lo=[min(d['retained_owned_bytes']['candidate'])/1e9 for _,d in cohorts]
    hi=[max(d['retained_owned_bytes']['candidate'])/1e9 for _,d in cohorts]
    bars=mem.bar(x,hi,color=colors['candidate'],width=.5)
    for bar,l,h in zip(bars,lo,hi):mem.text(bar.get_x()+bar.get_width()/2,h+.1,f'{h:.3f}' if l==h else f'{l:.3f}–{h:.3f}',ha='center',fontsize=9)
    mem.set_xticks(x,[f'Batch {b}' for b,_ in cohorts]);mem.set_ylim(0,max(hi)*1.2)
    mem.set_title(f'GPU allocations kept by Garnet · {size.upper()}',loc='left',weight='bold');mem.set_ylabel('GB decimal · includes MTP')
for ax in axes.flat:
    ax.set_facecolor('#f7f9fc');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
axes[0,0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('Qwen3.5 CUDA · batch profiles measured so far',x=.04,ha='left',fontsize=18,weight='bold')
fig.text(.04,.91,'RTX4080 GPU0 · sequential engines · identical JPEGs · 85 prompt / 32 output tokens · independent 512-token slots',fontsize=10)
fig.text(.04,.065,'D4 references; NoRDC/D8 trials remain experimental. Options/subsets differ. Kept GPU allocations exclude some temporary/driver memory.',fontsize=8.5)
fig.text(.04,.03,'Displayed vLLM: 2B B1 MTP3, others serial; 4B B8 MTP3 also tested. LM2B MTP observed / LM4B unresolved. Quality open; 2× unmet.',fontsize=8.5,color='#a52b2b')
fig.subplots_adjust(left=.07,right=.97,top=.84,bottom=.14,hspace=.43,wspace=.20)
for ext in ('png','svg'):fig.savefig(root/f'progress-combined-profiles.{ext}',dpi=160,facecolor=fig.get_facecolor())
(root/'progress-combined-profiles-sources.json').write_text(json.dumps({'sources':sources,
    'selection':'D4 profile history through native prefix/shared outputs; excludes unrepeated NoRDC, rejected D8 and pending binding reuse.',
    'quality_accepted':False,'goal_2x_achieved':False},indent=2))
print(root/'progress-combined-profiles.png')
