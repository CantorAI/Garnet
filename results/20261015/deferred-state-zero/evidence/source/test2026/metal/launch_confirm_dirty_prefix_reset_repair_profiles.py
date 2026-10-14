"""Repeat all six repaired-reset profiles only after the initial run exits 0."""
from pathlib import Path
import json,sys,hashlib
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/dirty-prefix-reset-repair-native-20261014/exit.json').read_text())['exit_code']==0
assert hashlib.sha256((root/'build-dirty-prefix-reset-repaired/libgarnet.dylib').read_bytes()).hexdigest()=='b5614830b706e3ae8bf8ca3333f492b7b79e47e1bca289041267d0fb706efaee'
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_dirty_prefix_reset_repair_profiles.py')
exec(compile(source,__file__,'exec'))
