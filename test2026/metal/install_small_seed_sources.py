"""Install only after both same-process components are terminal and audited."""
from pathlib import Path
source=Path(__file__).with_name('install_serial_vision_sources.py').read_text()
for old,new in [('serial-vision-staging','small-seed-staging'),('serial-vision-context-20261015','screened-seed-abba-20261015'),('serial-vision-context-confirm-20261015','screened-seed-abba-confirm-20261015'),('serial-vision-original-source-20261015','small-seed-original-source-20261015')]:source=source.replace(old,new)
source=source.replace('backup.mkdir(exist_ok=False)','backup.mkdir(exist_ok=True)')
needle='target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())';assert source.count(needle)==1
source=source.replace(needle,'target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists() or target.read_bytes()==p.read_bytes();target.write_bytes(p.read_bytes())')
source=source.replace('pinned serial vision files','pinned small-seed files')
exec(compile(source,__file__,'exec'))
