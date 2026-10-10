"""Read only the pinned safetensors header; no full weight download."""
import json,struct,sys,urllib.request
from pathlib import Path
REVISION="2fc06364715b967f1860aea9cf38778875588b17"
URL=f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{REVISION}/model.safetensors-00001-of-00001.safetensors"
def read_exact(stream,size):
    data=bytearray()
    while len(data)<size:
        chunk=stream.read(size-len(data))
        if not chunk:raise ValueError("incomplete checkpoint header")
        data.extend(chunk)
    return bytes(data)
try:
    with urllib.request.urlopen(URL,timeout=30) as response:
        length=struct.unpack('<Q',read_exact(response,8))[0]
        if length>16*1024*1024:raise ValueError("unexpected header length")
        header=json.loads(read_exact(response,length))
    destination=Path(sys.argv[1]);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(header,indent=2))
    prefix='model.language_model.layers.0.linear_attn'
    expected={'.in_proj_qkv.weight':[6144,1024],'.conv1d.weight':[6144,1,4],'.norm.weight':[128],'.out_proj.weight':[1024,2048]}
    for suffix,shape in expected.items():
        entry=header[prefix+suffix]
        if entry['shape']!=shape:raise ValueError("checkpoint dimensions differ from config")
        print(prefix+suffix,entry['dtype'],entry['shape'])
    print('pinned revision',REVISION,'header bytes',length,'tensor entries',len(header))
except Exception as error:
    # Redirect URLs can contain signed query strings; never log HTTP exceptions.
    print('checkpoint header inspection failed:',type(error).__name__,file=sys.stderr)
    raise SystemExit(1)
