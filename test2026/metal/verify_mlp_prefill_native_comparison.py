"""Preserve numeric gates; distinguish original-check and combined archive sizes."""
from pathlib import Path

p=Path(__file__).with_name('verify_mlp_prefill_native_profiles.py')
source=p.read_text()
needle='for key,value in checks_audit.items():assert audit[key]==value,key'
assert source.count(needle)==1
source=source.replace(needle,'''for key,value in checks_audit.items():
    if key=='verified_files':
        # The combined archive additionally contains the complete native
        # comparison and its source snapshots. Every file is checked above.
        assert value>=audit[key]
    else:assert audit[key]==value,key''')
needle='result=dict(verified_files='
assert source.count(needle)==1
source=source.replace(needle,"result=dict(original_checks_archive_verified_files=audit['verified_files'],verified_files=")
needle="        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]"
assert source.count(needle)==1
source=source.replace(needle,"""        asset='models/'+model+'/'+name
        # Tokenizer JSON bytes are archived; checkpoint tensors are hash-only.
        if 'evidence/'+asset in proof['files']:
            assert digest(root/asset)==meta['assets'][model][name]
        else:assert proof['omitted'][asset]['sha256']==meta['assets'][model][name]""")
needle="assert proof['library_sha256']['build-coalesced-bf16-body']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'"
assert source.count(needle)==1
source=source.replace(needle,"""repair=proof['archival_repair']
assert repair['original_archive_sha256']=='4c54a571ab781c9635943839c7fb341db27bd18632d1b7d52f329467fb63f852'
original=root/'archival-repair/original-manifest.json'
assert digest(original)==repair['original_manifest_sha256']=='d705a65c1fe4526b24f361b0b66537ac50d93b64ac9d1d10d546d0a828b8b2e0'
original_proof=json.loads(original.read_text())
for name,info in original_proof['files'].items():assert proof['files'][name]==info
for key in ('jobs','library_sha256','checks_archive_sha256','hardware','scope','full_goal_met'):
    assert proof[key]==original_proof[key]
added=repair['added_source_names'];assert len(added)==len(set(added))==13
for name in added:
    assert name.endswith('.metal')
    for prefix in ('source/','logs/mlp-prefill-native-20261015/source-snapshot/'):
        key=prefix+name
        assert proof['files']['evidence/'+key]==original_proof['omitted'][key]
assert {k:v for k,v in original_proof['omitted'].items() if k in proof['omitted']}==proof['omitted']
assert len(original_proof['omitted'])-len(proof['omitted'])==26
"""+needle)
exec(compile(source,__file__,'exec'))
