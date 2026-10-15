"""Wait only on the exact owned vision inference handle before CPU timing."""
from pathlib import Path
source=Path(__file__).with_name('queue_packed_prefill_norm_profiles.py').read_text()
source=source.replace('packed-prefill-norm-checks-20261014','vision-global-alpha-native-20261014').replace('68972','16264')
source=source.replace('launch_quiet_packed_prefill_norm_checks.py','queue_vision_global_alpha_profiles.py').replace("str(prior),'checks','--worker'","str(prior),'--worker'")
source=source.replace('run_packed_prefill_norm_profiles.py','run_compact_frame_probe.py').replace('All validation stages passed; starting six sequential complete-output profiles','Owned inference trial completed; starting independent CPU-only preparation checks')
source=source.replace('code=subprocess.call([sys.executable,',"code=subprocess.call([str(root/'comparators/vllm/venv/bin/python'),")
exec(compile(source,str(Path(__file__)),'exec'))
