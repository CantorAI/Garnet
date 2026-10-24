"""Four actual layers: per-layer versus embedded packed program, full state snapshots.

Diagnostic transfers are bounded to1M F32 words each; no timing benchmark.
"""
import hashlib,json,math,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram
checkpoint,output=sys.argv[1:3]
os.makedirs(output,exist_ok=True)
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)['text_config']
config['layer_types']=config['layer_types'][:4];config['num_hidden_layers']=4
assert config['layer_types']==['linear_attention']*3+['full_attention']
serial=None;program=None;report={'status':'running','capacity':4096,'batch':1,'chunk':16,'layers':4,'chunks':[],
 'scope':'Actual four-layer cross-program hidden/all-state regression; not independent CPU math, full-model quality or speed'}
def save():
 with open(os.path.join(output,'manifest.json'),'w') as f:json.dump(report,f,indent=2)

def dump(parts,path):
 digest=hashlib.sha256();total=0
 with open(path,'wb') as f:
  for value,width in parts:
   flat=io.storage_view(value,[1,width])
   for start in range(0,width,1048576):
    block=io.slice(flat,start,[1,min(1048576,width-start)])
    raw=io.finite_f32_bytes(block);f.write(raw);digest.update(raw);total+=len(raw)
 return {'file':os.path.basename(path),'bytes':total,'sha256':digest.hexdigest()}
try:
 serial=CUDADecoder(checkpoint,config,1,4096,16,True,True);serial.enable_cuda_graph(True);io=serial.io
 program=CUDAPackedDecoderProgram(checkpoint,config,io,1,4096,16,flat_concat=True,embedded_projection_weights=True)
 assert not program.weights and not program.weight_names
 state=io.zeros([1,program.state_width]);hidden=io.zeros([1,16,config['hidden_size']]);active=io.zeros([1,16]);positions=io.zeros([1,16,3]);slots=io.zeros([1,16])
 report['layout']=program.layout;report['hidden_width']=program.hidden_width;report['state_width']=program.state_width;save()
 cursor=0
 for block_index,count in enumerate([16,2]):
  values=[math.sin((cursor+t)*.17+i*.03)*.3 if t<count else 0. for t in range(16) for i in range(config['hidden_size'])]
  io.update(hidden,values);rows=io.zeros([16,config['hidden_size']]);io.update(rows,values)
  pos=[[3+cursor+t,5+cursor+t,7+cursor+t] for t in range(count)]
  serial.chunk_hidden([io.row(rows,t) for t in range(count)],[[1] for t in range(count)],[pos],False)
  ref_parts=[(serial.hidden,program.hidden_width)]
  for layer,item in zip(serial.layers,program.layout):ref_parts.extend([(layer['state1'],item['state1_width']),(layer['state2'],item['state2_width'])])
  row={'start':cursor,'active_tokens':count,'reference':dump(ref_parts,os.path.join(output,'chunk'+str(block_index)+'-serial.f32')),'runs':[]}
  report['chunks'].append(row);save()
  io.update(active,[1.]*count+[0.]*(16-count));io.update(positions,[v for p in pos for v in p]+[0.]*(3*(16-count)))
  io.update(slots,list(range(cursor,cursor+count))+[0.]*(16-count))
  for replay,enabled in enumerate([False,True,True,True,True]):
   program.graph.enable_cuda_graph(enabled);packed=program.run([hidden,state,active,positions,slots])
   snapshot=dump([(packed,program.output_width)],os.path.join(output,'chunk'+str(block_index)+'-packed'+str(replay)+'.f32'))
   snapshot['cuda_graph_enabled']=enabled;row['runs'].append(snapshot);save()
  io.slice_into(packed,program.hidden_width,state);cursor+=count
  print('PACKED_EMBEDDED_CHUNK_SNAPSHOTS_WRITTEN',block_index,count,flush=True)
 report['program']=program.memory_info();stats=report['program']['graph'];save()
 assert stats['native_nvfp4_projection_weights']==12 and stats['native_fp8_projection_weights']==13 and stats['external_bf16_weight_inputs']==0,stats
 assert stats['graph_replays']>=2 and stats['graph_capture_failures']==0,stats
 report['status']='snapshots_completed';save();print('PACKED_EMBEDDED_SNAPSHOTS_COMPLETE_CPU_COMPARISON_REQUIRED',flush=True)
finally:
 if program is not None:program.close()
 if serial is not None:serial.close()
