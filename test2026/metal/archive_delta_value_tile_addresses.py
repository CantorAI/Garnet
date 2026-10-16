"""Freeze terminal local CPU indexing evidence, source snapshots and audit tools."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
folder=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists();report=json.loads((folder/'report.json').read_text());assert not report['gpu_executed'] and len(report['profiles'])==12 and report['all_state_and_output_words_owned_once']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,sha in report['source_sha256'].items():assert digest(folder/'source-snapshot'/Path(name).name)==sha
files={'evidence/'+p.relative_to(folder).as_posix():p for p in folder.rglob('*') if p.is_file()}
for name in ('archive_delta_value_tile_addresses.py','verify_delta_value_tile_addresses.py'):
 p=Path(__file__).with_name(name);files['evidence/tools/'+name]=p
proof=dict(files={},exit_code=0,scope='Local CPU addressing/literal-expression proof only; no GPU/numerical or model speed admission.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),exit_code=0,gpu_executed=False)))
