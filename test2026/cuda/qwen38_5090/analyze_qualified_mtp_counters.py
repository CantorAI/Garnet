from pathlib import Path
import hashlib,json,zipfile,sys
out=Path(sys.argv[1]) if len(sys.argv)==2 else Path(__file__).parent
archive=out/'native-mtp3-shared-joined-evidence.zip';rows=[]
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='19bafb0f2cf8f518ad199677ad509c66e903abbecd94eb122b498048a437081f'
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  if '/benchmark/' not in name or not name.endswith('-events.jsonl') or not any('-s3-'+case+'-c1-r' in name for case in ('text','image')):continue
  events=[json.loads(line)['data'] for line in z.read(name).decode().splitlines()];final=next(json.loads(v) for v in reversed(events) if v!='[DONE]');metrics=final['study']['speculative_metrics']
  case='image' if '-s3-image-' in name else 'text';generated=final['usage']['completion_tokens'];assert generated==128 and metrics['target_prefix_recomputes']==0 and metrics['target_head_reuses']==0
  assert metrics['target_forward_calls']==metrics['target_verification_calls'] and metrics['target_head_evaluations']==metrics['target_verification_calls']+metrics['target_prefix_replays']
  rows.append({'case':case,'artifact':name,'generated':generated,**{k:metrics[k] for k in ('mtp_prefill_calls','mtp_draft_calls','mtp_correction_calls','mtp_head_calls','target_verification_calls','target_prefix_replays','proposed_draft_tokens','accepted_draft_tokens','target_forward_calls','target_head_evaluations')}})
assert len(rows)==16 and all(sum(r['case']==case for r in rows)==8 for case in ('text','image'))
summary=[]
for case in ('text','image'):
 selected=[r for r in rows if r['case']==case];totals={key:sum(r[key] for r in selected) for key in selected[0] if key not in ('case','artifact')}
 summary.append({'case':case,'requests':8,'totals':totals,'accepted_fraction':totals['accepted_draft_tokens']/totals['proposed_draft_tokens'],'post_initial_tokens_per_full_target_forward':(totals['generated']-8)/totals['target_forward_calls']})
result={'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'rows':rows,'summary':summary,'scope':'Retained qualified pilot counters only, not a new timing experiment. Prefix replay uses retained projections, not another full target forward; do not convert counts into stage durations or speedup. Hot-stage profiling remains needed.'}
(out/'qualified-mtp-counter-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
