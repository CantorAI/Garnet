"""Recompute observed native phases from the completed screened-head cohort."""
from pathlib import Path
source=Path(__file__).with_name('summarize_selected_attention_phases.py').read_text().replace('selected-attention-vllm-20261013','screened-head-vllm-20261015')
exec(compile(source,__file__,'exec'))
