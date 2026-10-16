"""Run one isolated CPU frontend trial after the native repeat is terminal."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/body-shape-confirm-native-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_striped_frame_probe.py')
source=source.replace("try:code=subprocess.call([sys.executable,", "try:code=subprocess.call([str(root/'comparators/vllm/venv/bin/python'),")
exec(compile(source,__file__,'exec'))
