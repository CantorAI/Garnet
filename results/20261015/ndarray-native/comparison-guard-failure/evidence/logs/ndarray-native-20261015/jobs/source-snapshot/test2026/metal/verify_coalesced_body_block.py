"""Audit coalesced bit reconstruction, independent block coverage and strict timing gates."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_wide_body_decode.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('make_wide_body_decode','make_coalesced_body_block'),('metal_context_wide_body','metal_context_coalesced_block'),('wide_body_decode_gpu','coalesced_body_block_gpu')]:source=source.replace(old,new)
source=source.replace("('check','samples','controls')","('check','samples','controls','packing')")
needle="for trial,status in proof['jobs'].items():";assert source.count(needle)==1
source=source.replace(needle,"admission_path=root/'evidence/logs/coalesced-body-block-admission-20261015'\nadmission=json.loads((admission_path/'report.json').read_text());assert digest(admission_path/'source.py')==admission['source_sha256']\n"+needle)
start=source.index('        eligible=sum(');end=source.index('        for batch in (1,2):',start)
source=source[:start]+'''        cpu=next(v for v in admission['models'] if v['model']==model)
        assert cpu['checkpoint_sha256']==assets['checkpoint_sha256']
        records={c['name']:c for c in cpu['cases']}
        packing=[v for v in values if v['kind']=='packing'];assert len(packing)==len(assets['cases'])
        eligible=0
        for case in assets['cases']:
            actual_packing=next(v for v in packing if v['matrix']==case['name'])
            if case['name'] not in records:
                assert actual_packing==dict(kind='packing',matrix=case['name'],blocks=0,compact_blocks=0,selected=False)
                continue
            c=records[case['name']];assert c['raw_sha256']==case['sha256']['raw.bin'] and c['cpu_original_bits_exact']
            assert c['rows']==case['rows'] and c['inner']==case['inner'] and c['blocks']==case['rows']//4*(case['inner']//128)
            assert c['compact_fraction']==c['compact_blocks']/c['blocks'] and 0<c['compact_blocks']<=c['blocks']
            assert actual_packing==dict(kind='packing',matrix=case['name'],blocks=c['blocks'],compact_blocks=c['compact_blocks'],selected=True)
            eligible+=1
        assert eligible>0
'''+source[end:]
source=source.replace('Private eight-output BF16 decode versus current four-output layout. Exact native output bits, original FP64 bound, tail/fallback/prefill/ownership/abort tests.64/128/256-thread variants. No full-model or vLLM speed claim.','Private coalesced exact12-bit BF16 blocks with uniform native fallback. Independent CPU block coverage and runtime counts, exact native output bits and original FP64 envelope; tail, ownership and recovery controls. Component only; no model TPS/TTFT or vLLM claim.')
exec(compile(source,__file__,'exec'))
