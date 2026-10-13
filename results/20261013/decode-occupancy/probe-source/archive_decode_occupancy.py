import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['decode-occupancy-20261013','decode-occupancy-confirm-20261013']
with tarfile.open(root/'decode-occupancy-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix!='.bin':archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['make_decode_occupancy_shader.py','decode_occupancy_gpu.mm','prepare_decode_tile_probe.py','run_decode_occupancy_probe.py','run_decode_tile_probe.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='probe-source/'+name)
 first=root/'logs'/names[0]/'jobs/metadata.json'
 assert hashlib.sha256((root/'occupancy_control_context.mm').read_bytes()).hexdigest()==json.loads(first.read_text())['source_sha256']['metal_context.mm']
 archive.add(root/'occupancy_control_context.mm',arcname='probe-source/first-control-metal_context.mm')
 current=root/'source/src/backends/metal/metal_context.mm'
 second=json.loads((root/'logs'/names[1]/'jobs/metadata.json').read_text())
 assert hashlib.sha256(current.read_bytes()).hexdigest()==second['source_sha256']['metal_context.mm']
 archive.add(current,arcname='probe-source/confirm-metal_context.mm')
 # The first probe mistakenly recorded the input-tile runner digest, not its
 # own. Preserve that script and reconstruct the actual initial runner too.
 runner=(root/'source/test2026/metal/run_decode_occupancy_probe.py').read_text()
 initial=runner.replace("scripts/'run_decode_occupancy_probe.py'","scripts/'run_decode_tile_probe.py'")
 path=root/'initial_decode_occupancy_probe.py';path.write_bytes(initial.encode())
 archive.add(path,arcname='probe-source/initial-runner.py')
 archive.add(root/'archive_decode_occupancy.py',arcname='probe-source/archive_decode_occupancy.py')
print('Archived initial and repeated decode work-distribution probes')
