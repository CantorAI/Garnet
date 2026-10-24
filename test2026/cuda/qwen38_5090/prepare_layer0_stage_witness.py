"""Extract immutable real layer0 witnesses; run only outside a timed GPU cohort."""
from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');archive=r/'results/native-prefix12-internals-evidence.zip';expected='32586aea53c0fd77b1d96935f700abe0061bd24f2cc1915400454929aac10564';assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
out=r/'results/layer0-stage-witness';assert not out.exists();selected={}
with zipfile.ZipFile(archive) as z:
 report=json.loads(z.read('results/native-prefix12-internals/report.json'))
 for n in z.namelist():
  if 'prefix-1-batch-' in n and '-repeat-0-chunk-0-' in n:
   name=Path(n).name;assert name not in selected;selected[name]=z.read(n)
 assert len(selected)==6
 assert hashlib.sha256((r/'models/Qwen3.8-27B-NVFP4/config.json').read_bytes()).hexdigest()==report['config_sha256']
 out.mkdir()
 for n,b in selected.items():(out/n).write_bytes(b)
 (out/'manifest.json').write_text(json.dumps({'archive_sha256':expected,'input_prompt_ids':report['input_prompt_ids'],'config_sha256':report['config_sha256'],'files':{n:hashlib.sha256(b).hexdigest() for n,b in selected.items()},'scope':'Historical real16token layer0 hidden and full convolution/recurrent state; current ordinary/staged witness equality must be checked separately.'},indent=2))
print('SIX_REAL_LAYER0_WITNESSES_EXTRACTED')
