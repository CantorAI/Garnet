"""Actual vision output and graph replay qualification; no timing/speed claim."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.run_logged import source_identity
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous vision qualification')
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_text(encoding='utf8'))
report={'status':'starting','source':source_identity(),'cases':[],'scope':'One existing compiled vision engine; exact direct/staged outputs, changing image content, shape guard and actual graph counters. No model TTFT, HTTP or speed claim.'}
path.write_text(json.dumps(report,indent=2));io=garnet.cuda_tensor_io();vision=None
try:
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 patches=[];expected=[]
 for i,color in enumerate(['red','green','blue']*2):
  oracle=root/'results/native-mtp3-http'/(color+'-request.json');body=json.loads(oracle.read_text(encoding='utf8'))
  item=next(c for c in body['messages'][0]['content'] if c['type']=='image_url');data=base64.b64decode(item['image_url']['url'].split(',',1)[1],validate=True)
  image=root/'tmp'/'vision-resident-qualification.jpg';image.write_bytes(data)
  prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
  patches.append(prepared[0]) # Retain allocations so direct bindings cannot accidentally recycle.
  raw=bytes(io.finite_f32_bytes(vision.encode_patches(patches[-1])));expected.append(raw)
  (out/('direct-'+str(i)+'.f32')).write_bytes(raw)
  report['cases'].append({'index':i,'color':color,'request_sha256':hashlib.sha256(oracle.read_bytes()).hexdigest(),'image_sha256':hashlib.sha256(data).hexdigest(),'direct_sha256':hashlib.sha256(raw).hexdigest()})
 report['direct_graph']=json.loads(vision.graph.stats_json());path.write_text(json.dumps(report,indent=2))
 assert len({c['direct_sha256'] for c in report['cases'][:3]})==3,'Changing image colors produced identical vision outputs'
 vision.enable_resident_input(io);report['resident_input_bytes']=vision.resident_input_bytes
 for i,patch in enumerate(patches):
  raw=bytes(io.finite_f32_bytes(vision.encode_patches(patch)));(out/('resident-'+str(i)+'.f32')).write_bytes(raw)
  report['cases'][i].update(resident_sha256=hashlib.sha256(raw).hexdigest(),exact_match=raw==expected[i]);path.write_text(json.dumps(report,indent=2))
  assert raw==expected[i],'Resident vision output differs; raw outputs preserved'
 after=json.loads(vision.graph.stats_json());report['resident_graph']=after;path.write_text(json.dumps(report,indent=2))
 assert after['graph_replays']-report['direct_graph']['graph_replays']>=4,'No actual stable-input graph replay'
 assert after['graph_capture_failures']==report['direct_graph']['graph_capture_failures']
 before=after.copy();bad=io.zeros([1,1,1536]);rejected=False
 try:vision.encode_patches(bad)
 except Exception as error:rejected=True;report['wrong_shape_error']=repr(error)
 assert rejected,'Invalid patch shape was accepted'
 report['after_wrong_shape_graph']=json.loads(vision.graph.stats_json())
 assert report['after_wrong_shape_graph']['graph_replays']==before['graph_replays']
 report['status']='vision_component_qualified';path.write_text(json.dumps(report,indent=2));print('VISION_RESIDENT_EXACT_OUTPUTS_AND_ACTUAL_REPLAY_PASS',flush=True)
except Exception as error:
 report['status']='failed';report['error']=repr(error);path.write_text(json.dumps(report,indent=2));raise
finally:
 if vision is not None:vision.close()
 io.close()
