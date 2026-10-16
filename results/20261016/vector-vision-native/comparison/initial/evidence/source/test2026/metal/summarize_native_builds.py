"""Validate the close native-build ABBA comparison, without a vLLM claim."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);groups=[];records=[]
assert json.loads((root/'metadata.json').read_text())['comparison_kind'].startswith('native ABBA')
for batch in (1,2):
    for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        samples={'garnet':[],'garnet-control':[]};expected=None
        for repeat in range(2):
            for backend in samples:
                folder=root/(model+'-b'+str(batch)+'-'+backend+'-'+str(repeat))
                assert json.loads((folder/'exit.json').read_text())['exit_code']==0
                text=(folder/'output.log').read_text()
                assert 'Segfault encountered' not in text
                values=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('matched_result ')]
                assert len(values)==3
                for value in values:
                    assert value['clock']=='time.monotonic' and all(value['finished'])
                    assert value['ttft_seconds']>0 and value['decode_window_seconds']>0
                    assert len(value['token_ids'])==batch
                    assert value['decode_steps_after_first']==max(map(len,value['token_ids']))-1
                    identity=(value['prompt_token_ids'],value['token_ids'])
                    if expected is None:expected=identity
                    assert identity==expected
                    value['folder']=folder.name;records.append(value)
                    if value['request']>0:samples[backend].append(value)
        def mean(values):
            return dict(ttft_seconds=statistics.mean(v['ttft_seconds'] for v in values),
                decode_steps_per_second=statistics.mean(v['decode_wall_steps_per_second'] for v in values))
        candidate=mean(samples['garnet']);control=mean(samples['garnet-control'])
        group=dict(model=model,batch=batch,candidate=candidate,control=control,
            ttft_fraction=candidate['ttft_seconds']/control['ttft_seconds'],
            decode_ratio=candidate['decode_steps_per_second']/control['decode_steps_per_second'],
            exact_outputs=True,requests=12)
        groups.append(group);print(json.dumps(group),flush=True)
(root/'native_summary.json').write_bytes((json.dumps(dict(scope='native ABBA, not a vLLM comparison',groups=groups,records=records),indent=2)+'\n').encode())
