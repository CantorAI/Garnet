"""Actual trained 4B linear layer under a packed lease, exact explicit/private parity."""
import sys,json,math
from pathlib import Path
s='/Users/shawn/CantorAI/out/research/worldsense-packed-request-residency/source';sys.path.insert(0,s);sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-packed-request-residency-on')
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer
w=Path(__file__).resolve().parent;root='/Users/shawn/CantorAI/out/models/Qwen3.5-4B';c=json.loads(open(root+'/config.json').read())['text_config'];checkpoint=root+'/model.safetensors.index.json'
channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
shapes=[[1,c['hidden_size']],[1,channels,c['linear_conv_kernel_dim']],[1,c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']],[1]]
symbols=[tensor.input('packed_request_linear_'+str(i),shape=v,dtype=tensor.float32)for i,v in enumerate(shapes)]
@T.fusion(name='packed_request_actual_4b_linear',boundary='required')
def forward(hidden,conv,state,active):return Qwen35LinearDecoderLayer(hidden,conv,state,active,c,0)
g=garnet.compiled_graph();other=garnet.compiled_graph();shared=garnet.compiled_graph();a=garnet.decoder_session();records=[]
hidden=tensor.tensor([float(i%23-11)*.01 for i in range(c['hidden_size'])],shape=shapes[0],dtype=tensor.float32)
position=tensor.tensor([0.,0.,0.],shape=[1,3],dtype=tensor.float32);slot=tensor.tensor([0.],shape=[1],dtype=tensor.float32);active=tensor.tensor([1.],shape=[1],dtype=tensor.float32)
def state():return [v.tolist()for v in a.snapshot(0)]
def packed():return a.hidden().tolist()+state()[0]+state()[1]
def stats():return json.loads(a.packed_request_residency_stats_json)
def clean():
 q=stats();assert not q['scope_open'] and not q['attached'] and not q['requested']
 assert q['active_bytes']==q['global_used_bytes']==q['set_allocation_count']==q['owner_count']==0
 return q
def record(case):records.append(dict(case=case,stats=stats()));(w/'packed-layer-session-progress.json').write_text(json.dumps(records,indent=2))
def baseline(graph):
 control=[hidden,*a.snapshot(0),active];before=[v.tolist()for v in control];wanted=graph.run(control).tolist()
 assert all(math.isfinite(v)for v in wanted)and before==[v.tolist()for v in control]
 return wanted
try:
 g.build_bf16(forward(*symbols),symbols,checkpoint);other.build_bf16(forward(*symbols),symbols,checkpoint);shared.build_shared(forward(*symbols),symbols,checkpoint,g);assert g.wide_primary_eligible
 a.build([g],None);wanted=baseline(g);assert a.begin_packed_request_residency([g]);saved=state();before=stats();rejected=False
 try:a.begin_packed_request_residency([g])
 except Exception:rejected=True
 assert rejected and saved==state()and stats()==before
 a.run([g],hidden,position,slot,active,-1);assert packed()==wanted and stats()['active_bytes']>0 and stats()['coverage_cache_hits']>0 and stats()['coverage_fallbacks']==0
 a.end_packed_request_residency();clean();record('cached_active_run_nested_guard_and_exact_reference')
 for invalid_graphs,invalid_active in [([g],tensor.tensor([2.],shape=[1],dtype=tensor.float32)),([1],active)]:
  a.reset();saved=state();before=hidden.tolist();assert a.begin_packed_request_residency([g]);rejected=False
  try:a.run(invalid_graphs,hidden,position,slot,invalid_active,-1)
  except Exception:rejected=True
  assert rejected and saved==state()and hidden.tolist()==before;clean();a.end_packed_request_residency();clean();record('invalid_active_or_graph_type_cancels_without_mutation')
 a.reset();wanted=baseline(other);assert a.begin_packed_request_residency([g]);a.run([other],hidden,position,slot,active,-1)
 assert packed()==wanted and stats()['reason']==8 and stats()['coverage_fallbacks']==1;clean();record('uncovered_actual_alternate_fallback_exact')
 a.reset();wanted=baseline(other);assert a.begin_packed_request_residency([other]);a.run([other],hidden,position,slot,active,-1);assert packed()==wanted and stats()['active_bytes']>0 and stats()['coverage_fallbacks']==0
 a.end_packed_request_residency();clean();record('supplied_actual_alternate_admitted')
 a.reset();wanted=baseline(shared);assert a.begin_packed_request_residency([g]);a.run([shared],hidden,position,slot,active,-1)
 assert packed()==wanted and stats()['coverage_checks']>0 and stats()['coverage_fallbacks']==0 and stats()['active_bytes']>0
 a.end_packed_request_residency();clean();record('new_shared_graph_covered_with_exact_original_owners')
 a.reset();assert a.begin_packed_request_residency([g]);g.close();g.build_bf16(forward(*symbols),symbols,checkpoint);wanted=baseline(g);a.run([g],hidden,position,slot,active,-1)
 assert packed()==wanted and stats()['reason']==8 and stats()['coverage_fallbacks']==1;clean();record('same_object_generation_revalidated_uncovered_exact')
 assert a.begin_packed_request_residency([g]);a.reset();clean();record('reset_releases_scope')
 assert a.begin_packed_request_residency([g]);g.close();saved=state();rejected=False
 try:a.run([g],hidden,position,slot,active,-1)
 except Exception:rejected=True
 assert rejected and saved==state();clean();record('closed_graph_rejected_cleanly')
 # The separately retained shared graph must survive closure of the source graph.
 a.build([shared],None);wanted=baseline(shared);assert a.begin_packed_request_residency([shared]);a.run([shared],hidden,position,slot,active,-1)
 assert packed()==wanted and stats()['active_bytes']>0 and stats()['coverage_fallbacks']==0;a.end_packed_request_residency();clean();record('shared_survives_source_close_and_recovery')
 (w/'packed-layer-session-guard.json').write_text(json.dumps(dict(all_cases_pass=True,records=records,scope='Actual learned four-input linear graph exact finite explicit/private/host state comparison; no timing claim.'),indent=2)+'\n')
 print('Actual4B trained packed request cached/shared/alternate/generation/type/invalid-active/reset/closed/error recovery exact PASS',flush=True)
finally:a.close();g.close();other.close();shared.close()
