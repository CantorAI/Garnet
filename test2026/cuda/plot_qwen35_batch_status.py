"""Recorded batch evidence, including failed profiles and open parity gaps."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
fig,axes=plt.subplots(1,2,figsize=(13,6.4),facecolor='#f7f9fc')
for ax,size in zip(axes,('2b','4b')):
    data=json.loads((root/f'batch-validation-{size}-summary.json').read_text())
    ax.set_facecolor('#f7f9fc')
    for p in data['profiles']:
        b=p['batch'];v=p['owned_model_bytes'][0]/1e9
        ax.bar(str(b),v,color='#7c4dff',width=.5)
        ax.text(str(b),v+.2,f'{v:.2f} GB',ha='center',weight='bold')
        ax.text(str(b),.8,f"{p['reference_maximum_set_matches']}/{p['reference_decisions']}\nmax-set",ha='center',color='white',fontsize=10)
    if size=='4b':
        log=(root/'batch-validation-4b-b8.log').read_text(errors='replace')
        assert 'buildSerializedNetwork failed' in log
        ax.bar('8',0,color='#dd5555')
        ax.text('8',3,'BUILD FAILED\nworkspace limit\nno result',ha='center',color='#bc3434',weight='bold')
    ax.set_ylim(0,12.5);ax.set_xlabel('Batch size');ax.set_title(f'Qwen3.5-{size.upper()}',loc='left',weight='bold')
    ax.yaxis.grid(alpha=.15);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
axes[0].set_ylabel('Retained owned model memory (decimal GB)')
fig.suptitle('Batch validation: current evidence',x=.055,ha='left',y=.96,fontsize=22,weight='bold')
fig.text(.055,.865,'Completed profiles pass runtime/reset/slot checks. Decoder reference acceptance remains open.',fontsize=11)
fig.text(.055,.15,'2B: 1,279/1,280 exact IDs; all choices in reference max set (one tie). Batch2 mixed-image check pending.',fontsize=10)
fig.text(.055,.10,'4B: batch2 188/192 and batch4 377/384 max-set matches; non-tied differences remain unresolved.',fontsize=10)
fig.text(.055,.055,'Ownership excludes driver overhead. Diagnostic runs provide no comparative batch timings. Projection candidate not yet GPU-tested.',fontsize=10,color='#536174')
fig.subplots_adjust(left=.09,right=.97,top=.77,bottom=.25,wspace=.25)
for ext in ('png','svg'):fig.savefig(root/f'progress-batch-status.{ext}',dpi=170,facecolor=fig.get_facecolor())
print(root/'progress-batch-status.png')
