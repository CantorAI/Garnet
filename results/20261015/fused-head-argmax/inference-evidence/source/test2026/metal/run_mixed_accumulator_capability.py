"""Exclusive guarded mixed matrix accumulator precision capability probe."""
from pathlib import Path
source=Path(__file__).with_name('run_metal_counter_capabilities.py').read_text().replace('metal_counter_capabilities_probe.mm','mixed_accumulator_capability_probe.mm').replace('metal_counter_capabilities_probe','mixed_accumulator_capability_probe')
exec(compile(source,__file__,'exec'))
