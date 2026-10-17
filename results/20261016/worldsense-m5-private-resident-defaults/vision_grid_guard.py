"""Native still-image embedding parity and shared-vision graph lifecycle, correctness only."""
import sys,json,pathlib,hashlib,struct,math
mode=sys.argv[1];w=pathlib.Path(__file__).resolve().parent;source='/Users/shawn/CantorAI/out/research/worldsense-private-resident-defaults/source';sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-defaults-'+mode);sys.path.insert(0,source);sys.path.insert(0,source+'/runtime')
import garnet,metal_serving
name='Qwen3.5-4B';profile=dict(maxInputTokens=384,maxOutputTokens=128,prefillChunk=64,linearChunk=True,imagePixels=65536,shareVisionWeights=True);result=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,source+'/xModel/qwen35/vl_4b','/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name));assert result['ready'];m=metal_serving._models[name];records=[]
try:
 for image_name in ['car.jpg','traffic.jpg','bunny.jpg','car.jpg']:
  image=w.parent/'skill-expansion'/image_name;prepared=garnet.prepare_image(str(image),m.image_options);gh,gw=prepared[1],prepared[2]
  if m.grid!=[gh,gw]:
   previous=m.vision;replacement=m.model_factory.create_vision(m.checkpoint,m.config,gh,gw,previous);m.vision,m.grid=replacement,[gh,gw]
   if previous is not None:previous.close();assert previous.graph.graph_workspace_allocated_bytes==0
  print('before explicit plan',image_name,m.vision.graph.vision_materialized_weight_slots,m.vision.graph.vision_pooled_weight_slots,flush=True)
  plan=json.loads(m.vision.graph.workspace_plan_json())
  print('after explicit plan',image_name,m.vision.graph.vision_pooled_weight_slots,flush=True)
  assert m.vision.graph.vision_materialized_weight_slots==(96 if mode=='on' else 0)
  assert m.vision.graph.vision_pooled_weight_slots==(96 if mode=='on' else 0)
  assert m.vision.graph.vision_raw_weight_payload_bytes==(603979776 if mode=='on' else 0)
  embedding=m.vision.encode_patches(prepared[0]);values=embedding.tolist();assert all(math.isfinite(v)for v in values);h=hashlib.sha256()
  for i in range(0,len(values),4096):b=values[i:i+4096];h.update(struct.pack('<'+'f'*len(b),*b))
  assert (m.vision.graph.graph_workspace_buffer_count>0)==(mode=='on');assert (m.vision.graph.graph_workspace_bound_slots>m.vision.graph.graph_workspace_buffer_count)==(mode=='on')
  records.append(dict(image=image_name,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),grid=[gh,gw],embedding_values=len(values),embedding_sha256=h.hexdigest()));(w/('vision-grid-progress-'+mode+'.json')).write_text(json.dumps(records,indent=2))
 assert len({tuple(r['grid'])for r in records})>=2 and records[0]==records[-1]
 if mode=='on':assert records==json.loads((w/'vision-grid-off.json').read_text())
 (w/('vision-grid-'+mode+'.json')).write_text(json.dumps(records,indent=2));print('Native vision embeddings/grid changes/shared owner close/car repeat exact PASS '+mode,flush=True)
finally:
 if m.vision is not None:m.vision.close()
 m.decoder.close();del metal_serving._models[name]
