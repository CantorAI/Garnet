"""Plot the matched-work benchmark checkpoint; no benchmark execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[3] / 'work/qwen35-cuda'
latest = root / 'fused-chunk4-graph-comparison.json'
data = json.loads((latest if latest.exists() else root / 'chunk4-graph-comparison.json').read_text())
p = data['paired_wall_ms']
labels = ['Garnet original', 'Garnet latest*', 'LM Studio / llama.cpp', 'vLLM']
values = [p['serial'], p['chunk4_graph'], p['lmstudio'], p['vllm']]
colors = ['#a4adba', '#7c4dff', '#3687ca', '#159b83']
if 'fused_chunk4_graph' in p:
    labels = ['Garnet original', 'Garnet chunk-4', 'Garnet fused latest*', 'LM Studio / llama.cpp', 'vLLM']
    values = [p['serial'], p['chunk4_graph'], p['fused_chunk4_graph'], p['lmstudio'], p['vllm']]
    colors = ['#a4adba', '#b5a0eb', '#7c4dff', '#3687ca', '#159b83']
target = min(p['lmstudio'], p['vllm']) / 2
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
fig, ax = plt.subplots(figsize=(12, 6.8), facecolor='#f7f9fc')
ax.set_facecolor('#f7f9fc')
bars = ax.barh(labels, values, color=colors, height=.55)
ax.invert_yaxis()
for bar, value in zip(bars, values):
    ax.text(value + 12, bar.get_y() + bar.get_height()/2, f'{value:.0f} ms', va='center', weight='bold')
ax.axvline(target, color='#df4c4c', linestyle='--', linewidth=2)
ax.text(.98, .98, f'Dashed line = 2x target: {target:.0f} ms', transform=ax.transAxes, ha='right', va='top', color='#be3333', weight='bold', fontsize=10)
ax.set_xlim(0, 940)
ax.set_xlabel('Median full-request latency (ms) — lower is better', labelpad=12)
ax.set_title('Qwen3.5-2B CUDA: progress so far', loc='left', fontsize=21, weight='bold', pad=40)
ax.text(0, 1.045, f"Batch 1 · {data['paired_equal_32token_cases']} matched cases · 32 sampled tokens · sequential engine runs", transform=ax.transAxes, color='#536174')
ax.xaxis.grid(True, alpha=.15)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis='both', length=0)
improvement = 100 * (1 - p.get('fused_chunk4_graph', p['chunk4_graph']) / p['serial'])
fig.text(.04, .085, f'*Experimental profiles. {improvement:.1f}% less latency than original; target not yet reached.', fontsize=11)
fig.text(.04, .045, 'BF16 source weights; intermediate precision and API timing differ. LM context rounds to 512 vs 384.', fontsize=10, color='#536174')
fig.subplots_adjust(left=.235, right=.97, top=.79, bottom=.22)
for ext in ['png', 'svg']:
    fig.savefig(root / f'progress-latency.{ext}', dpi=170, facecolor=fig.get_facecolor())
print(root / 'progress-latency.png')
