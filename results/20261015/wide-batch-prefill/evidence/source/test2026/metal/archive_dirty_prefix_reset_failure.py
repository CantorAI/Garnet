"""Preserve the completed gates and rejected first profile before a source repair."""
from pathlib import Path
import sys
source=Path(__file__).with_name('archive_dirty_prefix_reset_native.py').read_text()
source=source.replace("assert status['exit_code']==0;jobs[name]=status", "assert status['exit_code']==(0 if name=='dirty-prefix-reset-checks-20261014' else 1);jobs[name]=status")
exec(compile(source,str(Path(__file__)),'exec'))
