"""Derive clocks from the audited command-buffer cohort, without new inference."""
from pathlib import Path
import json,sys

base=Path(sys.argv[1]).resolve()
archive=json.loads((base/'archive-verification.json').read_text())['archive_sha256']
source=Path(__file__).with_name('analyze_mlp_native_phases.py').read_text()
for old,new in (
    ('logs/mlp-prefill-native-20261015/native','logs/mlp-command-buffer-native-retry-20261015/native'),
    ('9d494b2795485f03def18f463673b28da754fb6b5b99cce8db6c0ac96f7c5242',archive),
    ('Synchronous candidate encode includes graph execution.','Command-buffer encoding has no explicit per-graph CPU completion wait.'),
    ('Derived clock descriptions of the rejected synchronous MLP cohort and its retained control.',
     'Derived clock descriptions of the command-buffer MLP cohort and its retained control.')):
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
