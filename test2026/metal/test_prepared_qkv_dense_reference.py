"""Original trained dense bounds, selected 24/44 rows and fallbacks, retained random fixtures."""
from pathlib import Path
import contextlib,sys
output_root=Path(sys.argv[6]).resolve();output_root.mkdir(parents=True,exist_ok=False);sys.argv=sys.argv[:6]
source=Path(__file__).with_name('test_exact_dense_tail_reference.py').read_text()
assert source.count('for chunk in (22,26):')==1
source=source.replace('for chunk in (22,26):','for chunk in (24,22):')
old="with tempfile.TemporaryDirectory(prefix='dense-metal-reference-') as temporary:";assert source.count(old)==1
source=source.replace(old,"with contextlib.nullcontext(str(output_root/('layer'+str(layer)+'-b'+str(batch)+'-chunk'+str(chunk)))) as temporary:")
old='folder=Path(temporary)';assert source.count(old)==1;source=source.replace(old,old+';folder.mkdir(parents=True,exist_ok=False)')
exec(compile(source,__file__,'exec'))
