"""Fresh complete validation after fixing only a C++ test declaration."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_single_coder_head_checks.py').read_text().replace('build-single-coder-head','build-single-coder-head-repaired')
exec(compile(source,str(Path(__file__)),'exec'))
