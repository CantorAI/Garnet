"""All six original complete profiles, quiet guard and sequential native ABBA."""
from pathlib import Path
import sys

p=Path(__file__).with_name('run_vision_global_alpha_profiles.py')
source=p.read_text()
needle="str(s/'compare_vision_global_alpha.py')";assert source.count(needle)==1
source=source.replace(needle,"str(Path(__file__).with_name('compare_command_buffer_native.py'))")
needle="confirm='--confirm' in sys.argv[3:]";assert source.count(needle)==1
source=source.replace(needle,'confirm=False')
source=repr(__doc__)+source[source.index('\n'):]
Path(sys.argv[2]).with_suffix('.runner.executed.py').write_bytes(source.encode())
exec(compile(source,__file__,'exec'))
