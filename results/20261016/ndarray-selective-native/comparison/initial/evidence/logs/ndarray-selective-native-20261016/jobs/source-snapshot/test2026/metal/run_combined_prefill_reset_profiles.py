"""All six original sequential profiles with combined feature counters checked."""
from pathlib import Path
source=Path(__file__).with_name('run_prefill_stream_profiles.py').read_text().replace('compare_prefill_stream.py','compare_combined_prefill_reset.py')
exec(compile(source,__file__,'exec'))
