"""Prepare future gates without changing the live learned proof closure."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-raw-wide-vocab';proof={}
for stem in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed','verify_screen','admission','audit_routes']:
 original=(d/(stem+'_ws74.py')).read_text();t=original.replace('worldsense-minimal-selective-defaults','worldsense-raw-wide-vocab').replace('ws74.py','ws75.py').replace('WS74','WS75').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('len(m)==1721','len(m)==1723').replace('source_files=1721','source_files=1723')
 if stem=='admission':
  before='results={}';after="""wide_vocab=['-DGARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=ON'];wide_vocab_require='Raw wide vocabulary requires primary weights and wide checkpoint storage'
cases += [('wide-vocab-primary',wide_vocab+['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF'],wide_vocab_require),('wide-vocab-wide',wide_vocab+['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF'],wide_vocab_require)]
results={}""";assert t.count(before)==1;t=t.replace(before,after).replace('len(results)==18','len(results)==20').replace('three_selective_dependencies_rejected=True','three_selective_dependencies_rejected=True,two_raw_wide_vocab_dependencies_rejected=True').replace('original7+8core+3selective configuration','original7+8core+3selective+2rawwidevocab configuration')
 if stem=='audit_routes':
  before="'primary_decode_groups2','primary_decode_groups8']";assert t.count(before)==2;t=t.replace(before,"'primary_decode_groups2','primary_decode_groups8','primary_raw_wide_vocab']")
  before="   if c['model_id']=='Qwen3-1.7B'and mode!='off':";after="""   if enabled:
    for key in ['head_encoded','vector_head_encoded']:
     bad=copy.deepcopy(c);bad['timings']['primary_storage_before'][key]+=1;bad['timings']['primary_storage_after'][key]+=1;reject(bad)
    if c['model_id']=='Qwen3.5-4B':bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['active_weights']=201;reject(bad)
   if c['model_id']=='Qwen3-1.7B'and mode!='off':""";assert t.count(before)==1;t=t.replace(before,after)
  before="    bad=copy.deepcopy(both);bad['timings']['primary_storage_after']['active_weights']=398;reject(bad)";after=before+"\n    if enabled:bad=copy.deepcopy(both);bad['timings']['primary_storage_after']['active_weights']=201;reject(bad)";assert t.count(before)==1;t=t.replace(before,after)
  t=t.replace('corezero/selective graphpolicy and legacy compatibility','corezero/selective graphpolicy/200rawhead-vs201packedhead and legacy compatibility')
 ast.parse(t);p=d/(stem+'_ws75.py');assert not p.exists();p.write_text(t);proof[stem]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(t.encode()).hexdigest(),scope='Same original numerical/golden thresholds.60native/1723source admission IDs; strict new200/headzero/rawcap plus two new missing-dependency exclusions and malformed copied counters. No dependent stage executed.')
(w/'pipeline-helper-adaptations.json').write_text(json.dumps(proof,indent=2)+'\n');print('WS75 eight future pipeline helpers prepared; live learned closure untouched, no dependent dispatch.')
