"""Original full component audit, with both-kernel timing structure checked."""
from pathlib import Path
source=Path(__file__).with_name('verify_decode_subsimd.py').read_text().replace("glob('decode-subsimd-probe*')","glob('decode-splitk-*')").replace('bf16_decode_s16_t64','bf16_decode_sk2').replace('bf16_decode_s16_t128','bf16_decode_sk4')
needle="    job=folder/'jobs';metadata=json.loads((job/'metadata.json').read_text())"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
    harness=(job/'decode_splitk_gpu.mm').read_text()
    assert 'columns<=256?"bf16_decode_small":"bf16_decode"' in harness
    assert '[reduce dispatchThreadgroups:' in harness and '[reduce endEncoding];}' in harness
    assert '5e-6+5e-5*std::abs' in harness and 'std::ceil(double(inner)/32)+32' in harness
''')
needle="print(json.dumps(dict(files=len(files),omitted_matrices=len(proof),checks=count,scope='Matrix components only; no full-model claim')))"
assert source.count(needle)==1
source=source.replace(needle,"result=dict(files=len(files),omitted_matrices=len(proof),checks=count,scope='Both-kernel matrix components only; no full-model claim',full_goal_met=False)\n(r/'verification.json').write_bytes((json.dumps(result,indent=2)+'\\n').encode())\nprint(json.dumps(result))")
exec(compile(source,__file__,'exec'))
