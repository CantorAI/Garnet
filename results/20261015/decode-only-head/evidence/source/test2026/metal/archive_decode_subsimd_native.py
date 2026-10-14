"""Archive terminal native layout checks/inference separately from components."""
from pathlib import Path
import sys
source=Path(__file__).with_name('archive_direct_state_zero.py').read_text()
source=source.replace('direct-state-zero','decode-subsimd')
source=source.replace("(r/'logs').glob('decode-subsimd-*')", "(p for p in (r/'logs').glob('decode-subsimd-*') if p.name.startswith(('decode-subsimd-checks','decode-subsimd-inference','decode-subsimd-original-source')))")
source=source.replace("'archive_direct_state_zero.py'", "'archive_decode_subsimd_native.py','run_direct_state_zero_checks.py','compare_direct_state_zero.py','run_decode_subsimd_checks.py','compare_decode_subsimd.py'")
source=source.replace("r/'work/install_direct_state_zero.py'", "r/'work/install_decode_subsimd.py'")
needle="files['evidence/installer.py']=r/'work/install_decode_subsimd.py'"
source=source.replace(needle,needle+"\nfiles['evidence/installer-base.py']=r/'work/install_direct_state_zero.py'")
exec(compile(source,str(Path(sys.argv[2]).with_suffix('.executed.py')),'exec'))
