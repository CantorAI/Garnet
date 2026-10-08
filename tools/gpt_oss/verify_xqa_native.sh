#!/usr/bin/env bash
# Private native prerequisite only. Never run beside another inference job.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
binary=${GARNET_XQA_PARITY_BINARY:-$build/bin/garnet_gpt_oss_xqa_attention_parity}
[[ -x $binary && -x $python ]] || exit 2
command -v compute-sanitizer >/dev/null
command -v nsys >/dev/null
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
[[ -z $(git -C "$repo" status --porcelain) ]] || exit 1
directory=$1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
phase=metadata
terminal() {
    local status=$?
    trap - EXIT
    "$python" - "$directory" "$phase" "$status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'terminal.json'
assert not p.exists()
p.write_text(json.dumps(dict(terminal=True,phase=sys.argv[2],actual_exit=int(sys.argv[3]),
 scope='Private native sampled FP64/full metadata/changed capture/read-only KV/memcheck/dispatch only; no compiled/model/performance proof'),indent=2)+'\n')
PY
    exit "$status"
}
trap terminal EXIT
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$binary" "$repo/test2026/gpt_oss/xqa_attention_parity.cu" \
    "$repo/plugins/gpt_oss/cuda/gpt_oss_xqa_attention.cu" \
    "$repo/plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.cu" \
    "$repo/tools/gpt_oss/prepare_xqa_sources.py" >"$directory/native-source.sha256"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" \
    "$build/bin/garnet_gpt_oss_kernel_parity" >>"$directory/native-source.sha256"
"$python" - "$repo" "$build" "$binary" "$directory" <<'PY'
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo,build,binary,out=map(Path,sys.argv[1:])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_names=['test2026/gpt_oss/xqa_attention_parity.cu','plugins/gpt_oss/cuda/gpt_oss_xqa_attention.cu',
 'plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.cu','plugins/gpt_oss/include/gpt_oss_xqa_attention.h',
 'plugins/gpt_oss/include/gpt_oss_xqa_layout.h','plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.h',
 'plugins/gpt_oss/include/gpt_oss_kernels.h','plugins/gpt_oss/CMakeLists.txt','tools/gpt_oss/prepare_xqa_sources.py']
for name in source_names:
 p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,p)
vendor=repo/'plugins/gpt_oss/third_party/flashinfer_xqa'
shutil.copytree(vendor,out/'vendor',symlinks=True)
key=hashlib.sha256((sha(repo/'tools/gpt_oss/prepare_xqa_sources.py')+':'+sha(vendor/'SHA256SUMS')).encode()).hexdigest()
# The saved target project imports Garnet through a wrapper subdirectory;
# standalone and root builds have different binary-directory prefixes.
candidates=list(build.glob('**/xqa-'+key))
assert len(candidates)==1 and candidates[0].is_dir() and not candidates[0].is_symlink(), 'Ambiguous generated build closure'
generated=candidates[0]
subprocess.run([sys.executable,str(repo/'tools/gpt_oss/prepare_xqa_sources.py'),str(vendor),str(generated),'--verify-existing'],check=True)
shutil.copytree(generated,out/'generated',symlinks=True)
(out/'native-payload').mkdir();shutil.copy2(binary,out/'native-payload/xqa-parity')
inference={}
for name in ('libgarnet.so','libgarnet_gpt_oss.so','garnet_gpt_oss_kernel_parity'):
 shutil.copy2(build/'bin'/name,out/'native-payload'/name);inference[name]=sha(build/'bin'/name)
(out/'gate-metadata.json').write_text(json.dumps(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo),text=True).strip(),
 private_binary_sha256=sha(binary),inference_binaries=inference,source_sha256={n:sha(repo/n) for n in source_names},
 generated_sha256={p.name:sha(p) for p in generated.iterdir()}),indent=2)+'\n')
PY
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
compute-sanitizer --version >"$directory/sanitizer-version.txt" 2>&1
nsys --version >"$directory/nsys-version.txt" 2>&1
phase=ordinary-native
"$binary" >"$directory/native.log" 2>&1
grep -Fq 'XQA_NATIVE_PARITY_ACTUAL_PASS devices=2 cases=36 ' "$directory/native.log"
phase=unrestricted-native-memcheck
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all --launch-timeout 0 \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/native-mem.xml" \
    "$binary" >"$directory/native-mem.log" 2>&1
grep -Fq 'XQA_NATIVE_PARITY_ACTUAL_PASS devices=2 cases=36 ' "$directory/native-mem.log"
if grep -Eq 'No attachable process found|compute-sanitizer timed-out|========= Error:' "$directory/native-mem.log"; then
    echo 'Incomplete native sanitizer coverage' >&2; exit 1
fi
grep -Fq 'ERROR SUMMARY:' "$directory/native-mem.log"
"$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
    "$directory/native-mem.xml" "$directory/native-mem.audit.json"
"$python" - "$directory/native-mem.audit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['passed'] and r['records']==0
assert not r['excluded_known_initialization'] and not r['unexpected']
PY
phase=native-dispatch
nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
    --output="$directory/native-trace" "$binary" >"$directory/native-trace.log" 2>&1
grep -Fq 'XQA_NATIVE_PARITY_ACTUAL_PASS devices=2 cases=36 ' "$directory/native-trace.log"
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/native-trace.nsys-rep" \
    >"$directory/native-kernels.csv" 2>"$directory/native-kernels.log"
grep -Fq 'GarnetGptOssXqaKernel' "$directory/native-kernels.csv"
grep -Fq 'gptOssXqaPrepare' "$directory/native-kernels.csv"
grep -Fq 'gptOssXqaFinalize' "$directory/native-kernels.csv"
sha256sum --check "$directory/native-source.sha256"
phase=complete-native-only
echo 'XQA native prerequisite complete; no TensorRT/pretrained/throughput acceptance'
