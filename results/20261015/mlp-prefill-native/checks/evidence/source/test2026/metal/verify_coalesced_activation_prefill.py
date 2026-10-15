"""Independent reconstruction/bounds/ABBA audit against selected MPS prefill."""
from pathlib import Path
source=Path(__file__).with_name('verify_joined_activation_decode_probe.py').read_text()
start=source.index("assert proof['jobs']['joined-activation-decode-probe")
end=source.index('meta=json.loads',start)
source=source[:start]+"assert proof['jobs']['coalesced-activation-prefill-20261015']['exit_code']==0\njobs=root/'evidence/logs/coalesced-activation-prefill-20261015/jobs'\n"+source[end:]
source=source.replace('joined-activation-decode-repair-probe-20261014','coalesced-activation-prefill-20261015').replace('joined_activation_decode_gpu','coalesced_activation_prefill_gpu')
source=source.replace('assert len(stages)==7','assert len(stages)==10').replace("assert len(checks)==2*(len(assets['cases'])-len(excluded))","assert len(checks)==4*(len(assets['cases'])-len(excluded))")
source=source.replace('assert len(rows)==64 and timed==6144',"assert len(rows)==4*sum(len(a['cases']) for a in meta['assets'].values())-4*len(exclusions) and timed==96*len(rows)")
source=source.replace('for batch in (1,2):','for batch in (8,16,32,80):')
source=source.replace('matrix_batch_checks=len(rows)','matrix_prefill_row_checks=len(rows)').replace("scope='Component only; exact representations and bounded products; no full inference or vLLM gain'","scope='Component only: synthetic FP32 inputs and actual first-layer matrices, rows 8/16/32/80; selected global-alpha MPS control; no inference or vLLM gain'")
needle='rows=[];exclusions=[];validated=0;timed=0';assert source.count(needle)==1
source=source.replace(needle,'''harness=(jobs/'source-snapshot/test2026/metal/coalesced_activation_prefill_gpu.mm').read_text()
assert 'alpha:std::ldexp(1.0,-commonScale) beta:0' in harness
assert '[controlProduct encodeToCommandBuffer:cb leftMatrix:controlInput rightMatrix:wMatrix resultMatrix:controlOutput]' in harness
assert 'for(uint64_t rows:{8,16,32,80})' in harness
'''+needle)
exec(compile(source,__file__,'exec'))
