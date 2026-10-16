"""Compute all original matched targets from two independently audited cohorts."""
from pathlib import Path

source=Path(__file__).with_name('summarize_wide_batch_vllm_trials.py').read_text()
source=source.replace('wide-batch-vllm','coalesced-body-vllm')
exec(compile(source,__file__,'exec'))
