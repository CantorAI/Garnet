from pathlib import Path
import json,zipfile,hashlib,collections,statistics,sys
p=Path(sys.argv[1]);destination=Path(sys.argv[2]);rows=[]
with zipfile.ZipFile(p) as z:
 e=json.loads(z.read('evidence-manifest.json'))
 for name in z.namelist():
  if not name.startswith('results/shared-joined-mlp-http-diagnostic/s3-') or not name.endswith('-mtp-response.json'):continue
  raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==e['files'][name]
  d=json.loads(raw);m=d['study']['speculative_metrics'];ids=d['choices'][0]['response_token_ids'];assert len(ids)==128
  counts=[it['processed_target_counts'][0] for it in m['iterations']]
  assert sum(counts)==127 and len(counts)==m['target_verification_calls']
  assert m['iterations'][-1]['sampled_so_far'][0]-len(counts)==m['accepted_draft_tokens']
  rows.append({'response':name,'case':'image' if '-image-' in name else 'text','verification_calls':len(counts),'partial_replays':m['target_prefix_replays'],'proposed':m['proposed_draft_tokens'],'accepted':m['accepted_draft_tokens'],'tokens_advanced_per_verification':127/len(counts),'advance_histogram':dict(collections.Counter(counts))})
assert len(rows)==16
summary={}
for case in ('text','image'):
 a=[r for r in rows if r['case']==case];summary[case]={'requests':len(a),'draft_acceptance_fraction':sum(r['accepted'] for r in a)/sum(r['proposed'] for r in a),'verification_calls_median':statistics.median(r['verification_calls'] for r in a),'partial_replays_median':statistics.median(r['partial_replays'] for r in a),'advanced_per_verification_weighted':127*len(a)/sum(r['verification_calls'] for r in a)}
result={'source_archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows,'summary':summary,'scope':'Saved joined MTP3 diagnostic counters only; arithmetic checked against all16 complete128-token outputs. Prefix replay reuses retained projections and does not rerun target GEMMs; calls must not be counted as equivalent full target passes. No timing/causal or counterfactual speedup claim.'}
destination.write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
