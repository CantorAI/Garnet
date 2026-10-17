from pathlib import Path
import re,json,hashlib
old=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-decode512/source');new=old.parent.parent/'worldsense-dense-decode1024/source';w=Path(__file__).resolve().parent
for rel in ['src/backends/metal/body_shape_shader.h','src/backends/metal/dense_decode256_shader.h','src/backends/metal/dense_decode512_shader.h']:assert (old/rel).read_bytes()==(new/rel).read_bytes(),rel
s=(new/'src/backends/metal/dense_decode1024_shader.h').read_text().replace('DenseDecode1024','DenseDecode512').replace('DENSE1024','DENSE512').replace('dense1024','dense512').replace('ulong(group)*128+simd*4','ulong(group)*64+simd*4');assert s==(old/'src/backends/metal/dense_decode512_shader.h').read_text()
a=(old/'src/backends/metal/metal_context.mm').read_text();b=(new/'src/backends/metal/metal_context.mm').read_text();pattern=r'R"([A-Za-z0-9_]*)\((.*?)\)\1"';aa=re.findall(pattern,a,re.S);bb=re.findall(pattern,b,re.S);assert aa==bb
cuda=[p for p in old.rglob('*') if p.is_file() and ('cuda' in str(p.relative_to(old)).lower() or p.suffix=='.cu')];assert all(p.read_bytes()==(new/p.relative_to(old)).read_bytes() for p in cuda)
assert 'dense_decode1024_encoded_total' in (new/'src/backends/metal/decoder_session.h').read_text()
r={'original_inline_MSL_blocks_unchanged':len(aa),'original_body_256_512_headers_unchanged':True,'new_shader_only_name_and_group_stride_differ':True,'cuda_files_unchanged':len(cuda),'new_source_sha256':hashlib.sha256((new/'src/backends/metal/dense_decode1024_shader.h').read_bytes()).hexdigest(),'scope':'Source math and isolation audit; full native output, model states, recovery and actual route remain required.'};(w/'source-scope-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
