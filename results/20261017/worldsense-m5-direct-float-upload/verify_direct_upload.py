from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-direct-float-upload/source');old=w.parent/'worldsense-duo-prefill';before=Path('/Users/shawn/CantorAI/out/research/worldsense-duo-prefill/source')
seed=json.loads((w/'ws49-seed-source-manifest.json').read_text());assert hashlib.sha256((before/'src/backends/metal/compiled_graph.cpp').read_bytes()).hexdigest()==seed['src/backends/metal/compiled_graph.cpp']
original=(before/'src/backends/metal/compiled_graph.cpp').read_text();current=(s/'src/backends/metal/compiled_graph.cpp').read_text();a=original.index('    std::vector<float> data(rows*columns);');b=original.index('    if(modified!=',a);original_block=original[a:b];body=original_block[:original_block.index('    const auto id=')]
assert current.count(body)==1,'Original F32/BF16 vector fallback body changed'
x=current.index('    auto materialize=[&]() -> MetalMatrix {');y=current.index('    if(modified!=',x);assert current[:x]+original_block+current[y:]==original,'Metadata/shared/checkpoint validation outside materialization changed'
program=json.loads((w/'actual-program-proof.json').read_text());prior=json.loads((old/'actual-program-proof.json').read_text());assert program['programs']==prior['programs'],'Actual shader programs differ from49'
for mode in ['on','fault']:
 p=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-direct-float-upload-'+mode)/'Testing/Temporary/LastTest.log';assert 'Direct float upload all65536patterns/unaligned/padded/alias/source-lifetime/invalid/openbatch/Abort PASS'in p.read_text()
report={'entire_loader_recovered_by_replacing_only_materialize_block':True,'original_F32_fallback_body_byte_exact':True,'actual_four_build_programs_byte_exact_WS49':True,'direct_native_all65536_patterns_and_bounds_lifetime_pass':True,'scope':'Source/actual program/native correctness; no memory/performance gain claim.'}
if(w/'direct-upload-corpus.json').exists():
 selection=json.loads((w/'direct-upload-selection.json').read_text());result=json.loads((w/'direct-upload-corpus.json').read_text());expected={(x['model'],x['tensor']):x for x in selection['weights']};actual={(x['model'],x['tensor']):x for x in result['weights']};assert len(expected)==len(actual)==546 and expected.keys()==actual.keys()
 for key,x in expected.items():
  y=actual[key];assert all(x[k]==y[k]for k in ['shape','rows','columns']);assert y['FP32_words_checked']==x['rows']*x['columns'] and len(y['BF16_sha256'])==64 and y['storage']=='matrix_fp32'
 assert result['totals']=={'weights':546,'FP32_words_checked':334609152};report['all546_learned_vision_norm_depthwise_patch_candidates_FP32_words_exact']=True;report['direct_learned_words']=334609152
(w/'independent-direct-upload-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
