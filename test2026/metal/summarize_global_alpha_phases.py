"""Recompute existing global-alpha native phase telemetry without inference."""
from pathlib import Path
source=Path(__file__).with_name('summarize_selected_attention_phases.py').read_text()
source=source.replace('selected-attention-vllm-20261013','global-alpha-vllm-20261014')
exec(compile(source,str(Path(__file__)),'exec'))
