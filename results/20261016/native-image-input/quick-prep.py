import sys,json,pathlib,time,statistics,subprocess,hashlib
sys.path.insert(0,sys.argv[1])
import garnet
root=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/native-image');options=json.dumps({'patch_size':16,'merge_size':2,'temporal_patch_size':2,'pixels':65536,'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]});records=[]
for name in ['bunny','car','traffic']:
 image=str(root.parent/'skill-expansion'/(name+'.jpg'));old=[];new=[]
 for i in range(5):
  methods=['old','native'] if i%2==0 else ['native','old']
  for method in methods:
   start=time.monotonic()
   if method=='old':
    subprocess.run(['/Users/shawn/CantorAI/out/deps/garnet-metal-frontend/bin/python','/Users/shawn/CantorAI/Garnet/runtime/metal_prepare_image.py','/Users/shawn/CantorAI/out/models/Qwen3.5-4B',image,str(root/'control-frame'),'65536'],check=True)
    old.append(time.monotonic()-start)
   else:
    result=garnet.prepare_image(image,options);new.append(time.monotonic()-start)
 reference=root.parent/'4b-vision'/(name+'.f32');assert (root/'control-frame.f32').read_bytes()==reference.read_bytes()
 records.append({'image':name,'old_helper_seconds':old,'native_seconds':new,'old_median':statistics.median(old),'native_median':statistics.median(new)})
(root/'quick-prep.json').write_text(json.dumps(records,indent=2));print('PREP_METRICS',json.dumps(records),flush=True)
