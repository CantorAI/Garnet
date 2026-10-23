"""Plot all-group repeated studies; comparator subsets are intentionally separate."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--joined-decode',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
study='joined-decode' if a.joined_decode else 'promptpack-reusehead'
letters='ab' if a.joined_decode else 'abc'
output='progress-joined-study' if a.joined_decode else 'progress-packed-study'
studies=[]
for size,batch in (('2b',1),('4b',8)):
    path=root/f'packed-study-{size}-b{batch}-{study}-summary.json'
    if path.exists():studies.append((size,batch,path,json.loads(path.read_text())))
if not studies:raise SystemExit('No completed paired study')
fig,axes=plt.subplots(1,len(studies),figsize=(max(10,6.5*len(studies)),5.2),squeeze=False,facecolor='#f7f9fc')
colors=('#8293aa','#4688c4','#159b83');labels=('A: prior D4','B: + packed inputs','C: + head reuse')
if a.joined_decode:labels=('A: packed + head reuse','B: + joined decode')
sources=[]
for ax,(size,batch,path,data) in zip(axes[0],studies):
    values=[data['aggregate_group_median_ms'][k] for k in letters]
    bars=ax.bar(range(len(letters)),values,color=colors[:len(letters)],width=.58)
    for index,(key,bar,v) in enumerate(zip(letters,bars,values)):
        ax.text(index,v+max(values)*.035,f'{v:.1f} ms',ha='center',weight='bold',fontsize=11)
        for repeat,offset in (('1',-.10),('2',.10)):
            ax.scatter(index+offset,data['per_run'][key+repeat]['median_group_ms'],s=26,color='#263247',zorder=3)
    ax.set_title(f'Qwen3.5-{size.upper()} · batch {batch}',loc='left',weight='bold',fontsize=13)
    ax.set_xticks(range(len(letters)),labels,fontsize=9);ax.set_ylabel('Full-request group median · ms (lower is better)')
    ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_facecolor('#f7f9fc')
    for spine in ax.spines.values():spine.set_visible(False)
    comparison='a_to_b' if a.joined_decode else 'a_to_c'
    reduction=data['comparisons'][comparison]['reduction_fraction']*100
    change=f'{abs(reduction):.1f}% '+('lower' if reduction>=0 else 'higher')+' latency'
    ax.text(.02,.90,f'A → {letters[-1].upper()}: {change}',transform=ax.transAxes,fontsize=10)
    sources.append({'model':size,'batch':batch,'summary':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
title='Joined decode projections' if a.joined_decode else 'Prompt packing + verified-head reuse'
order='four fresh processes in A1 B1 B2 A2 order' if a.joined_decode else 'six fresh processes in A1 B1 C1 C2 B2 A2 order'
fig.suptitle(title+' · repeated real-image requests',x=.04,ha='left',weight='bold',fontsize=15)
fig.text(.04,.89,'Same build · '+order+' · strictly sequential',fontsize=10)
fig.text(.04,.085,'Bars: median of per-group repeat medians, all 20 warm groups. Dots: each run’s median; not confidence intervals.',fontsize=8.5)
fig.text(.04,.04,'All 28 input/output/stop signatures must match. Competitors not rerun here; independent quality and 2× goal remain open.',fontsize=8.5,color='#a52b2b')
fig.subplots_adjust(left=.07,right=.97,bottom=.20,top=.79,wspace=.24)
for ext in ('png','svg'):fig.savefig(root/f'{output}.{ext}',dpi=160,facecolor=fig.get_facecolor())
(root/f'{output}-sources.json').write_text(json.dumps({'sources':sources,'goal_2x_achieved':False,'independent_quality_accepted':False},indent=2))
print(root/f'{output}.png')
