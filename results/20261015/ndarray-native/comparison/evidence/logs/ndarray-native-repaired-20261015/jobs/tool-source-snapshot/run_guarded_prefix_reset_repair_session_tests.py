"""Run the original six recovery gates against the newly repaired binary."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_prefix_reset_session_tests.py').read_text().replace('build-dirty-prefix-reset','build-dirty-prefix-reset-repaired')
exec(compile(source,__file__,'exec'))
