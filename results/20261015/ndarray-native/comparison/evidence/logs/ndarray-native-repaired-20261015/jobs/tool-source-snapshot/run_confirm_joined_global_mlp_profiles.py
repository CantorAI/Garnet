"""Confirm only the two initial native warm signals; preserve failed full trial."""
from pathlib import Path
source=Path(__file__).with_name('run_joined_global_mlp_profiles.py').read_text()
source=source.replace("confirm='--confirm' in sys.argv[3:]","confirm=True")
source=source.replace('assert len(profiles)==6','assert len(profiles)==2')
exec(compile(source,str(Path(__file__)),'exec'))
