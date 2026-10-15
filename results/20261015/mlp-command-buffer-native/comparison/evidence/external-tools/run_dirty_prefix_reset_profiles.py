"""All six guarded native profiles; discard interrupted profiles completely."""
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2])
subprocess.run([sys.executable,str(Path(__file__).with_name('run_guarded_prefix_reset_session_tests.py')),str(root),str(out.with_name(out.name+'-session-gates'))],check=True)
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text().replace('compare_vision_global_alpha.py','compare_dirty_prefix_reset.py')
exec(compile(source,str(Path(__file__)),'exec'))
