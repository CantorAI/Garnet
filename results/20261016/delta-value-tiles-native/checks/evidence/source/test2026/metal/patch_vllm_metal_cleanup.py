"""Fix only vLLM 0.30.0 shutdown for the MLX platform's Torch CPU device.

Arguments: isolated comparator root, evidence folder. Forward/scheduler/model
execution is unchanged. Atomically replace the private source to detach any
package-cache hardlink. Refuse other source versions.
"""
import difflib,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();evidence=Path(sys.argv[2]);evidence.mkdir(parents=True,exist_ok=True)
path=root/'venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py'
assert path.resolve().is_relative_to(root)
original_hash='d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6'
before=path.read_bytes();assert hashlib.sha256(before).hexdigest()==original_hash
old='    if not current_platform.is_cpu():\n        from vllm.triton_utils import HAS_TRITON\n'
new='    if not current_platform.is_cpu() and current_platform.device_type != "cpu":\n        from vllm.triton_utils import HAS_TRITON\n'
source=before.decode();assert source.count(old)==1
after=source.replace(old,new).encode()
(root/'downloads/parallel_state-unpatched.py').write_bytes(before)
temporary=path.with_name(path.name+'.garnet-cleanup-patch');temporary.write_bytes(after)
temporary.chmod(path.stat().st_mode);temporary.replace(path)
patch=''.join(difflib.unified_diff(source.splitlines(True),after.decode().splitlines(True),
    fromfile='vllm/distributed/parallel_state.py (official v0.30.0)',tofile='vllm/distributed/parallel_state.py (Metal teardown compatibility)'))
(evidence/'cleanup_patch.diff').write_text(patch)
manifest=dict(path=str(path),before_sha256=original_hash,after_sha256=hashlib.sha256(after).hexdigest(),
    scope='Only distributed resource teardown; skip Torch accelerator cleanup for an out-of-tree platform with Torch CPU device type. MLX forward/model/cache execution unchanged.')
(evidence/'cleanup_patch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
