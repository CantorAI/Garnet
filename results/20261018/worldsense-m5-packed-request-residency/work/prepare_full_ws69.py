"""Prepare WS69 guards before execution; every original adaptation has an inverse."""
from pathlib import Path
import ast, hashlib, json, shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i')
w=b/'work/worldsense-packed-request-residency'
old=w.parent/'worldsense-request-weight-residency'
proof={}
names=['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']
for name in names:
 original=(old/name).read_text()
 text=original.replace('worldsense-request-weight-residency','worldsense-packed-request-residency')
 ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(name,x,text.count(x))
  text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if name in ['build_corpus.py','build_direct_corpus.py','request_session_guard.py','request_full_session_guard.py']:
  change('garnet-metal-worldsense-packed-request-residency-on','garnet-metal-worldsense-packed-request-residency-compat56')
 if name in ['request_session_guard.py','request_full_session_guard.py']:
  # Preserve the original raw-policy proof in COMPAT56 independently of the new packed policy.
  out='request-session-guard' if name=='request_session_guard.py' else 'request-full-session-guard'
  change(out+'.json',out+'-compat56.json')
  if name=='request_session_guard.py':change('request-session-guard-progress.json','request-session-guard-progress-compat56.json')
 if name=='resident_controller.py':change("['off','primary','on']","['off','primary','on','compat56']")
 if name=='resident_guard.py':change("endswith('-on')","endswith('-compat56')")
 if name=='api_state_guard.py':
  change("['off','primary','on'];expected=0 if mode=='off'else 1;vector_expected=mode=='on'", "['off','primary','on','compat56'];expected=0 if mode=='off'else 1;vector_expected=mode=='compat56';packed_expected=mode=='on'")
  change(" else:assert 'request_weight_residency'not in timings", " else:assert 'request_weight_residency'not in timings\n assert ('packed_request_weight_residency'in call['backend'])==packed_expected\n if packed_expected:\n  q=timings['packed_request_weight_residency'];eligible=name=='Qwen3.5-4B'\n  assert timings['packed_request_eligible']==eligible and q['enabled'] and q['limit_bytes']==8589934592\n  assert not q['scope_open'] and not q['attached'] and not q['requested']\n  assert q['active_bytes']==q['global_used_bytes']==q['owner_count']==q['set_allocation_count']==0\n  assert 'memory_after_packed_request_cleanup'in timings\n  if eligible:\n   assert q['calls']==q['admitted']==q['ends']>0 and q['fallback']==0 and q['reason']==1\n   assert q['last_count']==q['attempted_count']==402 and q['last_bytes']==q['attempted_bytes']>=6402163712\n   assert q['coverage_cache_hits']>0 and q['coverage_checks']>0 and q['coverage_fallbacks']==0 and 33<q['last_coverage_graphs']<=256\n  else:\n   assert q['calls']==q['ends']==q['admitted']==q['fallback']==q['coverage_checks']==q['coverage_cache_hits']==0\n else:assert 'packed_request_weight_residency'not in timings")
 if name.endswith('.py'):ast.parse(text)
 inverse=text
 for op in reversed(ops):
  assert inverse.count(op['after'])==op['count'],name
  inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-packed-request-residency','worldsense-request-weight-residency')==original,name
 assert not(w/name).exists(),name
 (w/name).write_text(text)
 proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('WS69 original learned guards prepared: exact inverses, COMPAT56 five readers and original raw guards, four API modes; no learned execution.')
