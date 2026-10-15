"""Continue the archived27-plus-fresh-wide sequence without altering frozen helpers."""
from pathlib import Path
p=Path(__file__).with_name('finalize_mlp_profiles_and_launch_eight.py')
source=p.read_text().replace('mlp-prefill-native-finalize-20261015','mlp-prefill-native-finalize-completed-20261015').replace('upstream_pid=82980','upstream_pid=99015')
exec(compile(source,__file__,'exec'))
