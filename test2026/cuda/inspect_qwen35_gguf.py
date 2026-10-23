"""Verify converted BF16 GGUF storage and retain conversion provenance.

Run outside measured inference. Large tensor inspection reads metadata; hashes
stream full files. This does not prove semantic equivalence of the converter.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('workspace');parser.add_argument('--model',choices=['2B','4B'],default='4B')
args=parser.parse_args();workspace=Path(args.workspace).resolve()
llama=workspace/'ThirdPartySDK/llama.cpp';sys.path.insert(0,str(llama/'gguf-py'))
from gguf import GGUFReader
root=workspace/'work/qwen35-cuda';name='Qwen3.5-'+args.model
records=[]
promotions={'v.patch_embd.weight':3145728,'v.patch_embd.weight.1':3145728,'v.position_embd.weight':9437184}
for path in [root/(name+'-BF16.gguf'),root/('mmproj-'+name+'-BF16.gguf')]:
    reader=GGUFReader(str(path));types={};exceptions=[]
    for tensor in reader.tensors:
        kind=tensor.tensor_type.name;size=int(tensor.n_bytes)
        types[kind]=types.get(kind,0)+size
        if size>=1048576 and kind!='BF16':
            assert path.name.startswith('mmproj-') and kind=='F32' and promotions.get(tensor.name)==size,(tensor.name,kind,size)
            exceptions.append({'name':tensor.name,'bytes':size,'storage':'F32',
                'reason':'Pinned converter explicitly promotes patch convolution and position table; BF16 values are exactly representable in F32.'})
    if path.name.startswith('mmproj-'):assert {t['name'] for t in exceptions}==set(promotions)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    records.append({'file':str(path),'bytes':path.stat().st_size,'sha256':digest,
        'tensor_count':len(reader.tensors),'tensor_bytes_by_type':types,'large_tensor_f32_promotions':exceptions})
converter=llama/'convert_hf_to_gguf.py'
provenance=root/'models'/name/'download-provenance.json'
result={'checkpoint_provenance':json.loads(provenance.read_text()) if provenance.exists() else None,
    'converter_commit':subprocess.check_output(['git','-C',str(llama),'rev-parse','HEAD'],text=True).strip(),
    'converter_sha256':hashlib.sha256(converter.read_bytes()).hexdigest(),
    'converter_support_sha256':{str(p.relative_to(llama)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [llama/'conversion/base.py',llama/'conversion/qwen.py',llama/'conversion/qwen3vl.py']},
    'commands_script':'convert-'+args.model.lower()+'.cmd','outtype':'bf16',
    'large_text_tensors_all_bf16':True,'vision_f32_promotions_explicit':True,'outputs':records,
    'limitation':'Storage and file identity only; no proof of output equivalence or runtime arithmetic.'}
output=root/('conversion-'+args.model.lower()+'-provenance.json')
output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
