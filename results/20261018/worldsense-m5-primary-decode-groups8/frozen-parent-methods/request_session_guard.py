"""Actual trained dense layer request coverage/lifecycle correctness; no timing claim."""
import sys,json,math
source='/Users/shawn/CantorAI/out/research/worldsense-minimal-full-prefill/source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-minimal-full-prefill-compat56';sys.path.insert(0,source);sys.path.insert(0,build)
import garnet,tensor
from xModel.qwen.metal_dense.layers import T,DenseLayer
from pathlib import Path
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-minimal-full-prefill');model='/Users/shawn/CantorAI/out/models/Qwen3-1.7B';c=json.loads(open(model+'/config.json').read());c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);checkpoint=model+'/model.safetensors.index.json';cap=8;cache=[1,c['num_key_value_heads'],cap,c['head_dim']];shapes=[[1,c['hidden_size']],cache,cache,[1,3],[1],[1]]
symbols=[tensor.input('request_dense_'+str(i),shape=s,dtype=tensor.float32)for i,s in enumerate(shapes)]
@T.fusion(name='request_scope_dense_layer',boundary='required')
def forward(hidden,key,value,position,slot,active):return DenseLayer(hidden,key,value,position,slot,active,c,0,cap)
def state(session):return [x.tolist()for x in session.snapshot(0)]
def packed(session):return session.hidden().tolist()+state(session)[0]+state(session)[1]
def stats(session):return json.loads(session.request_residency_stats_json)
g=garnet.compiled_graph();other=garnet.compiled_graph();shared=garnet.compiled_graph();session=garnet.decoder_session();records=[]
position=tensor.tensor([0.,0.,0.],shape=[1,3],dtype=tensor.float32);slot=tensor.tensor([0.],shape=[1],dtype=tensor.float32);active=tensor.tensor([1.],shape=[1],dtype=tensor.float32);hidden=tensor.tensor([float(i%23-11)*.01 for i in range(c['hidden_size'])],shape=shapes[0],dtype=tensor.float32)
def record(name):records.append(dict(case=name,stats=stats(session)));(w/'request-session-guard-progress-compat56.json').write_text(json.dumps(records,indent=2))
def baseline(graph):return graph.run([hidden,*session.snapshot(0),position,slot,active]).tolist()
try:
 g.build_bf16(forward(*symbols),symbols,checkpoint);other.build_bf16(forward(*symbols),symbols,checkpoint);shared.build_shared(forward(*symbols),symbols,checkpoint,g);session.build([g],None)
 want=baseline(g);assert all(math.isfinite(v)for v in want);assert session.begin_request_residency([g]);saved=state(session);rejected=False
 try:session.begin_request_residency([g])
 except Exception:rejected=True
 assert rejected and state(session)==saved and stats(session)['active_bytes']>0
 session.run([g],hidden,position,slot,active,-1);assert packed(session)==want and stats(session)['active_bytes']>0 and stats(session)['coverage_cache_hits']>0;session.end_request_residency();assert stats(session)['active_bytes']==stats(session)['global_used_bytes']==0;record('admitted_cached_run_and_nested_guard')
 saved=state(session);before=hidden.tolist();assert session.begin_request_residency([g]);rejected=False
 try:session.run([g],hidden,position,slot,tensor.tensor([2.],shape=[1],dtype=tensor.float32),-1)
 except Exception:rejected=True
 assert rejected and saved==state(session)and before==hidden.tolist()and stats(session)['active_bytes']==stats(session)['global_used_bytes']==0;session.end_request_residency();record('malformed_active_cancels_without_private_or_host_mutation')
 session.reset();want=baseline(other);assert session.begin_request_residency([g]);session.run([other],hidden,position,slot,active,-1);assert packed(session)==want and stats(session)['reason']==8 and stats(session)['coverage_fallbacks']==1 and stats(session)['active_bytes']==stats(session)['global_used_bytes']==0;session.end_request_residency();record('actual_alternate_uncovered_graph_falls_back_exactly')
 session.reset();want=baseline(other);assert session.begin_request_residency([other]);session.run([other],hidden,position,slot,active,-1);assert packed(session)==want and stats(session)['active_bytes']>0;session.end_request_residency();record('actual_supplied_alternate_graph_admitted')
 session.reset();want=baseline(shared);assert session.begin_request_residency([g]);session.run([shared],hidden,position,slot,active,-1);assert packed(session)==want and stats(session)['coverage_checks']>0 and stats(session)['coverage_fallbacks']==0 and stats(session)['active_bytes']>0;session.end_request_residency();record('new_shared_graph_coverage_validated')
 session.reset();assert session.begin_request_residency([g]);g.close();g.build_bf16(forward(*symbols),symbols,checkpoint);want=baseline(g);session.run([g],hidden,position,slot,active,-1);assert packed(session)==want and stats(session)['reason']==8 and stats(session)['active_bytes']==0;session.end_request_residency();record('generation_change_revalidates_and_falls_back')
 assert session.begin_request_residency([g]);session.reset();assert stats(session)['active_bytes']==stats(session)['global_used_bytes']==0;record('reset_releases_scope')
 assert session.begin_request_residency([g]);g.close();saved=state(session);rejected=False
 try:session.run([g],hidden,position,slot,active,-1)
 except Exception:rejected=True
 assert rejected and saved==state(session)and stats(session)['active_bytes']==stats(session)['global_used_bytes']==0;record('closed_supplied_graph_cancels')
 (w/'request-session-guard-compat56.json').write_text(json.dumps({'all_cases_pass':True,'records':records,'scope':'Trained six-input dense layer, real native APIs/weights/private state. No serving speed claim.'},indent=2));print('Trained request scope/coverage/generation/nesting/reset/error/private-host parity PASS',flush=True)
finally:session.close();g.close();other.close();shared.close()
