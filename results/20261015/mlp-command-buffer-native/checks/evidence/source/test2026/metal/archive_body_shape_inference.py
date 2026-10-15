"""Archive terminal original gates, body route counters and native trials."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('archive_deferred_state_zero_inference.py').read_text();exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source']
for old,new in (('deferred-state-zero','body-shape'),('deferred_state_zero','body_shape'),('deferred_zero','body_shape')):source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
