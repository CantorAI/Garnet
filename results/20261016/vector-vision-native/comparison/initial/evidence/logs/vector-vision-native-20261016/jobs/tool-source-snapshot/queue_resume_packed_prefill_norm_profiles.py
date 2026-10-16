"""Queue six complete-output benchmarks only after resumed full validation."""
from pathlib import Path
source=Path(__file__).with_name('queue_packed_prefill_norm_profiles.py').read_text()
source=source.replace('packed-prefill-norm-checks-20261014','packed-prefill-norm-resume-checks-20261014').replace('68972','75785')
source=source.replace('launch_quiet_packed_prefill_norm_checks.py','launch_resume_packed_prefill_norm_checks.py').replace("str(prior),'checks','--worker']","str(prior),'--worker']")
exec(compile(source,str(Path(__file__)),'exec'))
