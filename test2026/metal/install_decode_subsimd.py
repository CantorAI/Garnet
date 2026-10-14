"""Reuse the verified isolated-root installer with the new explicit manifest."""
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
source=(root/'work/install_direct_state_zero.py').read_text()
source=source.replace('direct-state-zero','decode-subsimd').replace('assert len(manifest)==10','assert len(manifest)==8')
exec(compile(source,str(root/'work/install-decode-subsimd.executed.py'),'exec'))
