"""Use the corrected, independently audited archive; retain all probe gates."""
from pathlib import Path

p=Path(__file__).with_name('launch_coalesced_eight_body.py')
source=p.read_text()
for old,new in (
    ('mlp-prefill-native-verification.json','mlp-prefill-native-complete-source-verification.json'),
    ('mlp-prefill-native-20261015.tar.gz','mlp-prefill-native-complete-source-20261015.tar.gz')):
    assert source.count(old)==1
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
