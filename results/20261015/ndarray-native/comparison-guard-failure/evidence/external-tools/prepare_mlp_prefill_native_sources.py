"""Freeze the optional native MLP graph prototype and its original-source identities."""
from pathlib import Path
import hashlib,io,json,tarfile

root=Path(__file__).resolve().parents[2];stage=root/'.work/mlp-prefill-native-staging';stage.mkdir(exist_ok=False)
original=json.loads((root/'.work/mlp-prefill-native-original.json').read_text())
extra=['src/backends/metal/mlp_prefill_graph_impl.h','test2026/metal/mlp_prefill_graph_test.cpp','test2026/metal/run_guarded_mlp_prefill_native_checks.py','test2026/metal/launch_mlp_prefill_native_checks.py','test2026/metal/prepare_mlp_prefill_native_sources.py','test2026/metal/install_mlp_prefill_native_sources.py']
audit=json.loads((root/'results/20261015/stage-rounded-mlp-graph/verification.json').read_text())
assert audit['numeric_checks']==48
all_checks=[c for p in audit['profiles']+audit['rejected_profiles'] for c in p['checks']]
assert len(all_checks)==48 and all(c['finite'] and c['comparison_worst']<=1 for c in all_checks)
gates=dict(scope='Experimental integration only: original graph/native comparisons pass all stress cases. Some native AND graph reference stress gates fail; none are erased or waived as trained-model evidence. Full original trained validation is required.',native_comparison_checks=48,reference_rejected_profiles=len(audit['rejected_profiles']),full_goal_met=False)
(stage/'component-gates.json').write_bytes((json.dumps(gates,indent=2)+'\n').encode())
manifest={}
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in list(original)+extra:
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'native prototype files; original-source backup required before install.')
