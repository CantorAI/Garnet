"""Install only after the archived all-head full runs are terminal."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('fused-head-argmax-native-repair-20261015','fused-head-argmax-confirm-native-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
assert hashlib.sha256((root/'build-fused-head-argmax/libgarnet.dylib').read_bytes()).hexdigest()=='8cfcfb392d19fad68695ad7d80c8c3b5989f01ca956fa50370c7b2f960230177'
source=Path(__file__).with_name('install_residual_norm_sources.py').read_text().replace('residual-norm-staging','decode-only-head-staging').replace('residual-norm-original-source-20261014','decode-only-head-original-source-20261015').replace('residual-norm','decode-only-head')
exec(compile(source,__file__,'exec'))
