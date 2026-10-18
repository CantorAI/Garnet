"""Real197-master model global-budget/close/reset correctness; no speed claim."""
import sys,json
from pathlib import Path
s='/Users/shawn/CantorAI/out/research/worldsense-owned-storage-local/source';sys.path.insert(0,s);sys.path.insert(0,s+'/runtime');sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-owned-storage-local-compat56')
import garnet,metal_serving
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-owned-storage-local');name='Qwen3-1.7B';p=dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True);ready=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,s+'/xModel/qwen/metal_dense','/Users/shawn/CantorAI/out/cache/'+name,json.dumps(p),name));assert ready['ready'];model=metal_serving._models[name];graphs=[x['graph']for x in model.decoder.layers];head=model.decoder.head;a=garnet.decoder_session();b=garnet.decoder_session();records=[]
def stats(session):return json.loads(session.request_residency_stats_json)
try:
 a.build(graphs,head);b.build(graphs,head);assert a.begin_request_residency(graphs);sa=stats(a);assert sa['active_bytes']==3440902144 and sa['last_count']==197
 assert not b.begin_request_residency(graphs);sb=stats(b);assert sb['reason']==5 and sb['active_bytes']==0 and sb['global_used_bytes']==sa['active_bytes'];records.append(dict(case='real_global_budget_contention',a=sa,b=sb));b.end_request_residency();a.close();assert b.begin_request_residency(graphs);sb=stats(b);assert sb['active_bytes']==3440902144;records.append(dict(case='closed_session_releases_budget',b=sb));b.reset();assert stats(b)['active_bytes']==stats(b)['global_used_bytes']==0;b.close()
 a.build(graphs,head);assert a.begin_request_residency(graphs);model.decoder.close();rejected=False
 try:a.begin_request_residency(graphs)
 except Exception:rejected=True
 assert rejected;a.close();(w/'request-full-session-guard-compat56.json').write_text(json.dumps({'real197masters_global4GiB_contention_close_readmission_reset':True,'graph_model_close_nested_scope_rejected_and_session_close':True,'records':records,'scope':'Actual raw text model native allocation admission/cleanup. No physical-residency or speed claim.'},indent=2));print('Real197master global4GiB budget contention/close/readmission/reset PASS',flush=True)
finally:a.close();b.close();model.decoder.close()
