"""Archive successful profiles after the non-inference validation-path repair."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('archive_prefill_stream_native.py').read_text();exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('prefill-stream-native-20261015','prefill-stream-repair-native-20261015')
source=source.replace("'archive_prefill_stream_native')","'archive_prefill_stream_native','archive_prefill_stream_repaired_native','launch_repaired_prefill_stream_profiles')")
source=source.replace("for name in ('prefill-stream-checks-20261015','prefill-stream-repair-native-20261015'):","names=['prefill-stream-checks-20261015','prefill-stream-repair-native-20261015']\nif (root/'logs/prefill-stream-confirm-native-20261015').exists():names.append('prefill-stream-confirm-native-20261015')\nfor name in names:")
source=source.replace("for name in set(manifest)|", "if len(names)>2:\n for helper in ('launch_confirm_prefill_stream_profiles.py','launch_confirm_global_half_alpha_profiles.py'):add(root/'source/test2026/metal'/helper)\nfor name in set(manifest)|")
exec(compile(source,__file__,'exec'))
