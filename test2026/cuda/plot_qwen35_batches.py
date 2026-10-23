"""Plot recorded batch validation; diagnostic ownership is not WDDM VRAM."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[3] / 'work/qwen35-cuda'
data = json.loads((root / 'batch-validation-2b-summary.json').read_text())
profiles = data['profiles']
fig, ax = plt.subplots(figsize=(11, 6), facecolor='#f7f9fc')
ax.set_facecolor('#f7f9fc')
values = [p['owned_model_bytes'][0] / 1e9 for p in profiles]
bars = ax.bar([str(p['batch']) for p in profiles], values, color='#7c4dff', width=.5)
for bar, value, p in zip(bars, values, profiles):
    ax.text(bar.get_x() + bar.get_width()/2, value+.12, f'{value:.2f} GB', ha='center', weight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, 1.0,
            f"{p['reference_top1_matches']}/{p['reference_decisions']} exact IDs\n"
            f"{p['reference_maximum_set_matches']}/{p['reference_decisions']} in max set",
            ha='center', color='white', fontsize=11)
ax.set_ylim(0, 6.8)
ax.set_xlabel('Batch size')
ax.set_ylabel('Retained owned model memory (decimal GB)')
ax.yaxis.grid(alpha=.15)
ax.set_axisbelow(True)
for spine in ax.spines.values(): spine.set_visible(False)
fig.suptitle('Qwen3.5-2B: batch validation so far', x=.055, ha='left', y=.96, fontsize=21, weight='bold')
fig.text(.055, .86, 'Reset/replay and slot accounting passed at batches 2, 4, 8; retained ownership stable.', fontsize=11)
fig.text(.055, .15, 'One exact-ID difference is a reference tie; all 1,280 selected tokens belong to the reference maximum set.', fontsize=10)
fig.text(.055, .10, 'Decoder reference uses native image embeddings. Mixed-image checks: batches4/8; batch2 pending. 4B pending.', fontsize=10)
fig.text(.055, .055, 'Diagnostic traces invalidate timing. Ownership excludes driver overhead; this is not the process-VRAM comparison.', fontsize=10, color='#536174')
fig.subplots_adjust(left=.12, right=.97, top=.78, bottom=.25)
for ext in ('png', 'svg'): fig.savefig(root / f'progress-batches.{ext}', dpi=170, facecolor=fig.get_facecolor())
print(root / 'progress-batches.png')
