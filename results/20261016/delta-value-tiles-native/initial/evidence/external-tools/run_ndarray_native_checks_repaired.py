"""Fresh30 validation; exclude only the exact primitive in its verified owned process group."""
from pathlib import Path
import hashlib,json,sys
p=Path(__file__).with_name('run_ndarray_native_checks.py');ns={'__file__':str(p)}
saved=sys.argv;sys.argv=[saved[0],saved[1],saved[2],'--generate-only']
exec(compile(p.read_text(),str(p),'exec'),ns);sys.argv=saved;source=ns['source']
source=source.replace('build-ndarray-prefill-shape80','build-ndarray-prefill-shape80-guardrepair')
old="if owned_group is not None and fields and fields[-1].lower().endswith('/xlang3'):";assert source.count(old)==1
source=source.replace(old,"if owned_group is not None and fields and (fields[-1].lower().endswith('/xlang3') or fields[-1]==str(owned_ndarray_binary)):")
old='if group==owned_group:owned+=1;continue';assert source.count(old)==1
source=source.replace(old,"if group==owned_group:\n                if fields[-1].lower().endswith('/xlang3'):owned+=1\n                continue")
needle='raw_snapshot=snapshot;owned_group=None';assert source.count(needle)==1
source=source.replace(needle,needle+"\nowned_ndarray_binary=Path(sys.argv[2]).resolve()/'ndarray_prefill_cache_gpu'")
needle="assert not build.exists(),'Keep existing build artifacts; use a fresh candidate directory'";assert source.count(needle)==1
source=source.replace(needle,"""external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('launch_ndarray_native_checks_repaired.py')]
external_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in external}
for p in external:
    dest=output/'external-driver-snapshot'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
(output/'external-driver-identities.json').write_text(json.dumps(external_hashes,indent=2)+'\\n')
"""+needle)
source+='\nfor name,sha in external_hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha\n'
destination=Path(sys.argv[2]).with_suffix('.complete30.repaired.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
