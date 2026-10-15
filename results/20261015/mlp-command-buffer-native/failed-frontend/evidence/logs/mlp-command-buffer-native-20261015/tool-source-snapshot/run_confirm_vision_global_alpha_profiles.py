"""Fresh confirmation of both actual image profiles; retain full-six evidence."""
from pathlib import Path
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text()
source=source.replace("confirm='--confirm' in sys.argv[3:]","confirm=True")
source=source.replace("[('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1)]","[('Qwen3.5-0.8B',1),('Qwen3.5-0.8B',2)]").replace('assert len(profiles)==6','assert len(profiles)==2')
exec(compile(source,str(Path(__file__)),'exec'))
