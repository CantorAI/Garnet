"""Expose installed wheel CUDA libraries at the layout expected by JIT builds."""
import argparse,hashlib,json,subprocess
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['vllm','sglang']);a=p.parse_args()
    root=Path('/opt/cantorai5090')/a.engine
    cuda=root/'lib/python3.12/site-packages/nvidia/cu13'
    library=cuda/'lib/libcudart.so.13'
    if not library.is_file() or not (cuda/'bin/nvcc').is_file(): raise ValueError('Expected existing CUDA wheel SDK')
    links=[(cuda/'lib64','lib'),(cuda/'lib/libcudart.so','libcudart.so.13')]
    for path,target in links:
        if path.exists() or path.is_symlink():
            if path.resolve()!=(path.parent/target).resolve(): raise ValueError('Refusing to replace existing CUDA SDK path')
        else: path.symlink_to(target,target_is_directory=(target=='lib'))
    result={'engine':a.engine,'links':[{'path':str(path),'target':target} for path,target in links],
        'cudart_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'nvcc_version':subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True),
        'scope':'SDK linker discovery only; model/JIT execution still requires validation'}
    output=Path('/mnt/d/CantorAI/5090-study-20261023/results/references')/a.engine/'cuda-layout.json'
    if output.exists(): raise ValueError('Preserve existing setup provenance before rerunning')
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
