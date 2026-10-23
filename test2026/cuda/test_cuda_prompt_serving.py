"""Exercise production prompt preparation with a deliberately reused vision buffer."""
import json,os,struct,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
io=garnet.cuda_tensor_io();h=7;checks=[]

class Tokenizer:
    def encode(self,text):return [1+ord(c)%100 for c in text]

class FakeIO:
    def __getattr__(self,key):return getattr(io,key)
    def embedding(self,ids):return io.from_host([float(t*10+j) for t in ids for j in range(h)],[len(ids),h])
    def prepare_image(self,image,pixels):return [int(image),2,4]

class ReusedVision:
    def __init__(self):self.storage=io.zeros([1,2,h])
    def encode_patches(self,image):
        io.update(self.storage,[float(image*1000+j) for j in range(2*h)])
        return self.storage

try:
    for batch in (1,2,4,8):
        for packed in (False,True):
            model=Model.__new__(Model);model.batch=batch;model.pack_prompt=packed
            model.decoder=SimpleNamespace(io=FakeIO());model.tokenizer=Tokenizer();model.pixels=65536
            model.config={'image_token_id':1000,'vision_config':{'patch_size':16}};model.max_input=384
            model.vision=ReusedVision();model.grid=[2,4]
            prompts=['short' if row%2 else 'longer prompt' for row in range(batch)]
            images=[str(row+1) if row%3!=2 else None for row in range(batch)]
            rows=[model._prepare(p,image) for p,image in zip(prompts,images)]
            if packed:output=list(io.pack_prompt([r[0] for r in rows],[r[6] for r in rows],[r[7] for r in rows]))
            else:
                zero=io.zeros([1,h])
                output=[io.stack_rows([r[0][t] if t<len(r[0]) else zero for r in rows]) for t in range(max(len(r[2]) for r in rows))]
            visual_cursor=[0]*batch
            for t,value in enumerate(output):
                expected=[]
                for row,r in enumerate(rows):
                    if t>=len(r[2]):expected.extend([0.]*h)
                    elif r[2][t]==1000:
                        expected.extend([float((row+1)*1000+visual_cursor[row]*h+j) for j in range(h)])
                        visual_cursor[row]+=1
                    else:expected.extend([float(r[2][t]*10+j) for j in range(h)])
                assert io.finite_f32_bytes(value)==struct.pack('<'+str(len(expected))+'f',*expected),(batch,packed,t)
            assert visual_cursor==[2 if image else 0 for image in images]
            checks.append({'batch':batch,'pack_prompt':packed,'exact':True})
    Path(sys.argv[1]).write_text(json.dumps({'all_passed':True,'cases':checks,
        'scope':'Production Model._prepare and native packing with reused fake vision output and deterministic fake embeddings; independent expected image IDs/text/padding. No pretrained quality or performance claim.'},indent=2))
    print('CUDA_PROMPT_SERVING_PASS',len(checks),flush=True)
finally:io.close()
