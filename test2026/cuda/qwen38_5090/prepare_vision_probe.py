"""CPU Transformers vision oracle only; no full-model load or timing benchmark."""
import argparse,hashlib,importlib.metadata,json,types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mixed-projections',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    cp=a.checkpoint;config=json.loads((cp/'config.json').read_text())
    index=json.loads((cp/'model.safetensors.index.json').read_text())['weight_map']
    names=[n for n in index if n.startswith('model.visual.')]
    if not names:raise ValueError('No actual vision weights in checkpoint index')
    cfg=Qwen3_5VisionConfig(**config['vision_config']);cfg._attn_implementation='eager'
    model=Qwen3_5VisionModel(cfg).to(dtype=torch.float32,device='cpu').eval()
    weights={}
    for filename in sorted({index[n] for n in names}):
        with safe_open(cp/filename,framework='pt',device='cpu') as f:
            for name in names:
                if index[name]==filename:weights[name.removeprefix('model.visual.')]=f.get_tensor(name).float()
    model.load_state_dict(weights,strict=True);del weights
    mixed_count=0
    if a.mixed_projections:
        # Match the declared native arithmetic, while retaining Transformers'
        # independent attention, position, normalization and activation code.
        def linear(module,value):
            projected=torch.nn.functional.linear(value.bfloat16(),module.weight.bfloat16()).float()
            return projected+module.bias.float() if module.bias is not None else projected
        def patch(module,value):
            if list(value.shape[1:])!=[3,2,16,16]:raise ValueError('Unexpected patch Conv3D geometry')
            projected=torch.nn.functional.linear(value.reshape(-1,1536).bfloat16(),module.weight.reshape(-1,1536).bfloat16()).float()
            if module.bias is not None:projected=projected+module.bias.float()
            return projected.reshape(-1,module.out_channels,1,1,1)
        for module in model.modules():
            if isinstance(module,torch.nn.Linear):module.forward=types.MethodType(linear,module);mixed_count+=1
            elif isinstance(module,torch.nn.Conv3d):module.forward=types.MethodType(patch,module);mixed_count+=1
        if mixed_count!=111:raise ValueError('Unexpected native projection count: '+str(mixed_count))
    y,x=np.mgrid[:256,:256];rgb=np.stack([(x*7+y*3)%256,(x^y)%256,(x*2+y*5)%256],axis=-1).astype(np.uint8)
    image_path=a.output/'vision-pattern.jpg';Image.fromarray(rgb).save(image_path,quality=97,subsampling=0)
    processor=AutoImageProcessor.from_pretrained(str(cp),local_files_only=True)
    inputs=processor(images=Image.open(image_path).convert('RGB'),return_tensors='pt')
    patches=inputs['pixel_values'].float();grid=inputs['image_grid_thw']
    if grid.tolist()!=[[1,16,16]] or list(patches.shape)!=[256,1536]:raise ValueError('Unexpected pinned processor shape')
    with torch.inference_mode():output=model(patches,grid_thw=grid).pooler_output.float()
    if list(output.shape)!=[64,5120] or not torch.isfinite(output).all():raise ValueError('Invalid vision oracle')
    patches.numpy().astype('<f4').tofile(a.output/'patches.f32')
    output.numpy().astype('<f4').tofile(a.output/'expected.f32')
    import inspect
    source=Path(inspect.getsourcefile(Qwen3_5VisionModel))
    manifest={'grid':grid.tolist()[0],'patch_shape':[1,256,1536],'output_shape':[1,64,5120],
        'vision_weights':len(names),'reference_dtype':('BF16 input/weight/output projections, F32 bias/residual/norm/attention; eager CPU Transformers' if a.mixed_projections else 'CPU float32 on decoded original BF16 weights; eager attention'),
        'mixed_projection_count':mixed_count,
        'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors','pillow']},
        'model_source_sha256':digest(source),'config_sha256':digest(cp/'config.json'),
        'index_sha256':digest(cp/'model.safetensors.index.json'),
        'files':{n:digest(a.output/n) for n in ['patches.f32','expected.f32','vision-pattern.jpg']},
        'acceptance':{'relative_l2_max':.02,'max_absolute_error_max':.5},
        'scope':'One independent full27-layer CPU vision fixture; native BF16 projection rounding differs; no timing/quality generalization'}
    (a.output/'oracle.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
