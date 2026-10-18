"""Native still-image embedding parity and shared-vision graph lifecycle, correctness only."""
import sys,json,pathlib,hashlib,struct,math
mode=sys.argv[1];w=pathlib.Path(__file__).resolve().parent;source='/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source';sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-'+mode);sys.path.insert(0,source);sys.path.insert(0,source+'/runtime')
import garnet,metal_serving
name='Qwen3.5-4B';profile=dict(maxInputTokens=384,maxOutputTokens=128,prefillChunk=64,linearChunk=True,imagePixels=65536,shareVisionWeights=True);result=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,source+'/xModel/qwen35/vl_4b','/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name));assert result['ready'];m=metal_serving._models[name];records=[]
try:
 for image_name in ['car.jpg','traffic.jpg','bunny.jpg','car.jpg']:
  image=w.parent/'skill-expansion'/image_name;prepared=garnet.prepare_image(str(image),m.image_options);gh,gw=prepared[1],prepared[2]
  if m.grid!=[gh,gw]:
   previous=m.vision;replacement=m.model_factory.create_vision(m.checkpoint,m.config,gh,gw,previous);m.vision,m.grid=replacement,[gh,gw]
   if previous is not None:previous.close()
  embedding=m.vision.encode_patches(prepared[0]);values=embedding.tolist();assert all(math.isfinite(v)for v in values);h=hashlib.sha256()
  for i in range(0,len(values),4096):b=values[i:i+4096];h.update(struct.pack('<'+'f'*len(b),*b))
  records.append(dict(image=image_name,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),grid=[gh,gw],embedding_values=len(values),embedding_sha256=h.hexdigest()));(w/('vision-grid-progress-'+mode+'.json')).write_text(json.dumps(records,indent=2))
 assert len({tuple(r['grid'])for r in records})>=2 and records[0]==records[-1]
 assert records==json.loads((w.parent/'worldsense-request-residency/vision-grid-off.json').read_text())
 if mode=='on':assert records==json.loads((w/'vision-grid-off.json').read_text())
 (w/('vision-grid-'+mode+'.json')).write_text(json.dumps(records,indent=2));print('Native vision embeddings/grid changes/shared owner close/car repeat exact PASS '+mode,flush=True)
finally:
 if m.vision is not None:m.vision.close()
 m.decoder.close();del metal_serving._models[name]
