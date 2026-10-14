"""Independent full-output and bounded-reset audit including copy regression gates."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('verify_dirty_prefix_reset_native.py').read_text()
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('dirty-prefix-reset-checks-20261014','dirty-prefix-reset-repair-checks-20261014').replace('dirty-prefix-reset-native-20261014','dirty-prefix-reset-repair-native-20261014').replace('dirty-prefix-reset-staging','dirty-prefix-reset-repair-staging').replace('dirty-prefix-reset-original-source-20261014','dirty-prefix-reset-repair-original-source-20261014')
source=source.replace('assert len(stages)==19','assert len(stages)==22').replace('numerical_stages=19','numerical_stages=22')
source=source.replace("meta=json.loads((checks/'metadata.json').read_text())",'''copies=[s for s in stages if s['name'].endswith('-prefix-copy')];assert len(copies)==3
for s in copies:
 case=s['accepted_cases'][0];text=(checks/case['evidence']/(s['name']+'.log')).read_text()
 values=[json.loads(line.split(' ',2)[2]) for line in text.splitlines() if line.startswith('prefix_reset_prefill PASS ')];assert len(values)==2 and {v['batch'] for v in values}=={1,2}
 assert all(all(v[k] for k in ('prefill_copy_admitted','complete_zero_verified','inactive_high_slot_ignored','other_session_unchanged')) for v in values)
meta=json.loads((checks/'metadata.json').read_text())''')
exec(compile(source,__file__,'exec'))
