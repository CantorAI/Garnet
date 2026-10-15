"""Keep observed matched metrics, cold/warm separation and every unmet target."""
from pathlib import Path
source=Path(__file__).with_name('summarize_global_alpha_vllm.py').read_text().replace('global-alpha-vllm-20261014','screened-head-vllm-20261015')
exec(compile(source,__file__,'exec'))
