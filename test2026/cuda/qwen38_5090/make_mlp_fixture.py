"""Synthetic identity NVFP4 MLP for native capture/refit/graph correctness."""
import argparse,json,struct
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--bf16',action='store_true');a=p.parse_args()
    width=256;entries={}
    for name in ['gate_proj','up_proj','down_proj']:
        prefix='model.language_model.layers.0.mlp.'+name
        if a.bf16:
            raw=bytearray(width*width*2)
            for row in range(width):raw[(row*width+row)*2:(row*width+row)*2+2]=b'\x80\x3f'
            entries[prefix+'.weight']=('BF16',[width,width],bytes(raw))
            continue
        weight=bytearray(width*width//2)
        for row in range(width): weight[row*(width//2)+row//2]=2<<(4*(row%2))
        entries[prefix+'.weight']=('U8',[width,width//2],bytes(weight))
        entries[prefix+'.weight_scale']=('F8_E4M3',[width,width//16],bytes([56])*(width*width//16))
        entries[prefix+'.weight_scale_2']=('F32',[],struct.pack('<f',1.))
        entries[prefix+'.input_scale']=('F32',[],struct.pack('<f',1./448.))
    header={};data=b''
    for name,(dtype,shape,raw) in entries.items():
        header[name]={'dtype':dtype,'shape':shape,'data_offsets':[len(data),len(data)+len(raw)]};data+=raw
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+data)
    config={} if a.bf16 else {'quantization_config':{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION'}}
    (a.output/'config.json').write_text(json.dumps(config))
