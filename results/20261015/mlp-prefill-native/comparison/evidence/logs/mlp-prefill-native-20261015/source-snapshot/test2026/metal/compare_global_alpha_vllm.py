"""Matched additional-FP32 native/vLLM comparison using the recorded library policy."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_retained_prefill_vllm.py').read_text()
source=source.replace('import ast,hashlib,json,os,signal,statistics,subprocess,sys,time','import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil')
needle="build=r/'build-resident-kv-inplace'"
assert source.count(needle)==1
source=source.replace(needle,'''selection_path=r/'work/global-alpha-next-comparator-selection.json'
selection=json.loads(selection_path.read_text())
policy=selection['models'][model][str(batch)];selected=selection['builds'][policy]
build=(r/selected['directory']).resolve()
assert build.is_relative_to(r.resolve()),'Build outside investigation root'
(output/'runtime-selection.json').write_bytes(selection_path.read_bytes())''')
needle="assert metadata['native_library_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'"
assert source.count(needle)==1
source=source.replace(needle,"assert metadata['native_library_sha256']==selected['sha256']\nmetadata.update(native_build_policy=policy,attention_prefix_enabled=selected['attention_prefix_enabled'],selection_sha256=digest(selection_path))")
source=source.replace("        if backend=='garnet':\n            expected_plan=","        if backend=='garnet':\n            assert value['attention_prefix_enabled']==selected['attention_prefix_enabled']\n            expected_plan=")
source=source.replace('def guard(label):\n    strict_guard()', '''def guard(label):
    space=shutil.disk_usage(r)
    (output/(label+'-disk.json')).write_text(json.dumps(dict(total_bytes=space.total,used_bytes=space.used,free_bytes=space.free))+'\\n')
    if space.free<2*2**30:raise RuntimeError('Investigation disk free space below 2 GiB')
    strict_guard()''')
source=source.replace("if fields and fields[-1].lower().endswith(('/split_half_prefill_probe','/lossless_block_gpu_probe','/lossless_gpu_probe','/prefill_tile_gpu','/parallel_delta_gpu','/decode_accumulators_gpu','/decode_accumulators_head_gpu','/decode_unroll_gpu','/decode_single_output_gpu','/decode_wide_outputs_gpu')):","if fields and fields[-1].lower().endswith(('_gpu','_probe')):")
source=source.replace("            else:assert len(set(value['encoder_cache_identities']))==batch","            else:assert len(set(value['encoder_cache_identities']))==batch")
source=source.replace("    if backend=='garnet':native_log=folder/'output.log'","    if vl and backend=='vllm-metal':assert len({identity for value in values for identity in value['encoder_cache_identities']})==batch*3,'Encoder identity reused across requests'\n    if backend=='garnet':native_log=folder/'output.log'")
source=source.replace("assert value['attention_prefix_enabled']==selected['attention_prefix_enabled']","assert value['attention_prefix_enabled']==selected['attention_prefix_enabled']\n            assert value['global_half_alpha_enabled'] and value['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n            assert not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
