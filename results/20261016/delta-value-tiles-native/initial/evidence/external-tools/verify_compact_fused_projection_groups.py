"""Regenerate compact-aware shader/harness and audit original bits/bounds/all ABBA samples."""
from pathlib import Path
import hashlib,json,runpy,sys
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());assert len(proof['jobs'])==1
trial=next(iter(proof['jobs']));out=root/'evidence/logs'/trial/'jobs';tools=out/'source-snapshot/test2026/metal'
generator=runpy.run_path(str(tools/'make_compact_fused_projection_groups.py'))['generate']
source=(tools/'verify_concurrent_bf16_groups.py').read_text()
start=source.index("    shader=('#include <metal_stdlib>");end=source.index('    stages=json.loads',start)
source=source[:start]+'''    original=(out/'source-snapshot/test2026/metal/concurrent_bf16_group_gpu.mm').read_text()
    shader,harness=generator(native,(out/'source-snapshot/src/backends/metal/coalesced_body_block_shader.h').read_text(),original)
    assert (out/'compact-fused-projections.metal').read_bytes()==shader.encode() and digest(out/'compact-fused-projections.metal')==meta['native_shader_sha256']
    assert (out/'compact_fused_projection_group_gpu.mm').read_bytes()==harness.encode() and digest(out/'compact_fused_projection_group_gpu.mm')==meta['generated_harness_sha256']
    assert 'Projection or padding changed bits' in harness and 'Original FP64 envelope exceeded' in harness
    assert 'Compact original-bit roundtrip failed' in harness and 'Fallback interleave original-bit roundtrip failed' in harness
    assert proof['omitted_binaries']['logs/'+trial+'/jobs/compact_fused_projection_group_gpu']['sha256']==meta['binary_sha256']
    assert meta['timing_design']==dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,all_samples_retained=True)
'''+source[end:]
source=source.replace("c['validated_executions']>=252","c['validated_executions']>=652").replace("warm[0]['elapsed_ms']>=250","warm[0]['elapsed_ms']>=2000")
source=source.replace('len(windows)==8','len(windows)==40').replace('for rnd in (0,1):','for rnd in range(10):').replace('len(v[k])==12','len(v[k])==5').replace('timed+=96','timed+=200')
needle="        records=[json.loads(line)";assert source.count(needle)==1
source=source.replace(needle,"""        for group in assets['groups']:
            for case in group['cases']:
                expected=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
                assert case['float_key']==expected
                omitted=proof['omitted_binaries']['logs/'+trial+'/jobs/'+model+'-assets/'+case['name']+'.bin']
                assert omitted['sha256']==case['sha256'] and omitted['bytes']==case['inner']*case['columns']*2
"""+needle)
needle="        checks=[v for v in records if v['kind']=='check'];";assert source.count(needle)==1
source=source.replace(needle,"""        packing=[v for v in records if v['kind']=='packing']
        assert len(packing)==2*sum(len(g['cases']) for g in assets['groups'])
        for row in packing:
            group=next(g for g in assets['groups'] if g['name']==row['group']);case=next(c for c in group['cases'] if c['name']==row['matrix'])
            n,k=case['columns'],case['inner'];eligible=not case['float_key'] and 256<n<=8192 and k%128==0 and k>=(1048576+n-1)//n
            assert row['rows'] in (1,2) and row['load_pack_allocate_ms']>=0 and row['blocks']==(((n+3)//4)*(k//128) if eligible else 0)
            assert 0<=row['compact_blocks']<=row['blocks'] and row['selected']==bool(row['compact_blocks'])
            assert row['weight_buffer_bytes']==(((n+3)//4)*4*k*2 if row['selected'] else n*k*2)
            assert row['payload_buffer_bytes']==(row['blocks']*768 if row['selected'] else 4)
            assert row['metadata_buffer_bytes']==(row['blocks']*4 if row['selected'] else 4)
"""+needle)
source=source.replace('Independent projection component only, original shader bits; concurrent API request does not prove hardware overlap; no full inference or vLLM gain','Compact-aware grouped64 component only, original accumulation/reduction and lossless BF16 words; no full inference or vLLM gain')
exec(compile(source,__file__,'exec'))
