import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['decode-input-tiles-20261013','decode-input-tiles-complete-heads-20261013','decode-input-tiles-runtime-heads-20261013']
with tarfile.open(root/'decode-input-tile-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name;assert json.loads((folder/'exit.json').read_text())['exit_code']==0
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix!='.bin':archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['decode_tile_gpu.mm','make_decode_tile_shader.py','prepare_decode_tile_probe.py','run_decode_tile_probe.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='current-probe-source/'+name)
 archive.add(root/'source/src/backends/metal/metal_context.mm',arcname='current-probe-source/metal_context.mm')
 archive.add(root/'archive_decode_tiles.py',arcname='current-probe-source/archive_decode_tiles.py')
 # Recreate the two historical preparation versions and verify their recorded
 # digests before saving; current sources must not masquerade as prior ones.
 current=(root/'source/test2026/metal/prepare_decode_tile_probe.py').read_text()
 line="    config=json.loads((checkpoint.parent/'config.json').read_text())\n    head='model.language_model.embed_tokens.weight' if 'model.language_model.embed_tokens.weight' in header else 'model.embed_tokens.weight' if config.get('tie_word_embeddings',False) else 'lm_head.weight'\n"
 second=current.replace(line,"    head='model.language_model.embed_tokens.weight' if 'model.language_model.embed_tokens.weight' in header else 'lm_head.weight' if 'lm_head.weight' in header else 'model.embed_tokens.weight'\n").replace("assert cases and any(case['tensor']==head for case in cases), 'Runtime head was not collected as BF16'","assert cases")
 first=second.replace("    head='model.language_model.embed_tokens.weight' if 'model.language_model.embed_tokens.weight' in header else 'lm_head.weight' if 'lm_head.weight' in header else 'model.embed_tokens.weight'\n",'').replace('and tensor!=head:continue',"and tensor not in ('lm_head.weight','model.language_model.embed_tokens.weight'):continue")
 for name,source in zip(names[:2],(first,second)):
  recorded=json.loads((root/'logs'/name/'jobs/metadata.json').read_text())['source_sha256']['prepare_decode_tile_probe.py']
  assert hashlib.sha256(source.encode()).hexdigest()==recorded,name
  path=root/(name+'-prepare-source.py');path.write_bytes(source.encode());archive.add(path,arcname=name+'/historical-prepare-source.py')
print('Archived complete decode input-tile evidence')
