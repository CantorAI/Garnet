"""Pin an official checkpoint revision and retain download provenance."""
import json,sys
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
model,destination=sys.argv[1:3]
assert model in ('Qwen/Qwen3.5-2B','Qwen/Qwen3.5-4B')
info=HfApi().model_info(model)
root=Path(destination);root.mkdir(parents=True,exist_ok=True)
metadata={'repo_id':model,'revision':info.sha,'method':'huggingface_hub snapshot_download, BF16 original weights'}
(root/'download-provenance.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata),flush=True)
snapshot_download(model,revision=info.sha,local_dir=str(root),
    allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'],max_workers=2)
print('QWEN35_CHECKPOINT_DOWNLOAD_COMPLETE',model,info.sha,flush=True)
