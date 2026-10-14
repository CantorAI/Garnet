"""Full output/counter audit against original complete streamed-prefill validation."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('verify_prefill_stream_native.py').read_text();exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace('prefill-stream-native-20261015','prefill-stream-repair-native-20261015')
if '--confirm' in __import__('sys').argv[3:]:source=source.replace('prefill-stream-repair-native-20261015','prefill-stream-confirm-native-20261015')
exec(compile(source,__file__,'exec'))
