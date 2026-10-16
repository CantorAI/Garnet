"""One trained norm at a time; same packed fusion and exact original controls."""
from pathlib import Path
import sys
assert len(sys.argv)==5 and int(sys.argv[3]) in range(6) and sys.argv[4] in ('regular','extra')
source=Path(__file__).with_name('run_prefill_residual_norm_probe.py').read_text()
if sys.argv[4]=='extra':source=source.replace("str(shader),str(probe)])","str(shader),str(probe),'--extra-chunks'])")
needle='destination=Path(sys.argv[2]).with_suffix'
assert source.count(needle)==1
source=source.replace(needle,"source=source.replace(\"assetfile=out/'norm-assets.json'\",\"cases=[cases[int(sys.argv[3])]];assetfile=out/'norm-assets.json'\")\n"+needle)
exec(compile(source,str(Path(__file__)),'exec'))
