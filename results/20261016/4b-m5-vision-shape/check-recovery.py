import sys,json,pathlib,hashlib,struct
build,source,out=sys.argv[1:4];sys.path.insert(0,build)
import garnet
sys.path.insert(0,source)
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
root=pathlib.Path('/Users/shawn/CantorAI/out/models/Qwen3.5-4B');config=json.loads((root/'config.json').read_text())['vision_config'];checkpoint=str(root/'model.safetensors.index.json');base=pathlib.Path(out)/'bunny';gh,gw=json.loads(base.with_suffix('.json').read_text())['grid'];old=MetalImageEncoder(checkpoint,config,gh,gw)
a=old.encode_file(str(base.with_suffix('.f32'))).tolist();bad=dict(config);bad['depth']=25
try:MetalImageEncoder(checkpoint,bad,gh,gw,shared_encoder=old)
except Exception:pass
else:raise AssertionError('Missing vision layer was accepted')
b=old.encode_file(str(base.with_suffix('.f32'))).tolist();assert a==b,'Failed replacement damaged old graph'
# Changed patch bytes must be freshly encoded, not retrieved as an image result.
changed=pathlib.Path(out)/'changed.f32';data=bytearray(base.with_suffix('.f32').read_bytes())
for i in range(128):struct.pack_into('<f',data,4*i,struct.unpack_from('<f',data,4*i)[0]+1)
changed.write_bytes(data);c=old.encode_file(str(changed)).tolist();assert c!=a,'Changed image patches reused a previous output';old.close()
pathlib.Path(out,'recovery-checks.json').write_text(json.dumps({'failed_rebuild_preserves_old_encoder':True,'changed_patches_produce_changed_embeddings':True,'repeat_original_embeddings_equal':True},indent=2));print('RECOVERY_CHECKS_PASS',flush=True)
