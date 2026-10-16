"""Validate native ABBA real-frame comparison; no vLLM/target advantage claim."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);groups=[];records=[]
assert json.loads((root/'metadata.json').read_text())['comparison_kind'].startswith('native ABBA')
for batch in (1,2):
    samples={'garnet':[],'garnet-control':[]};identity=None;all_values=[]
    for repeat in range(2):
        for backend in samples:
            folder=root/('Qwen3.5-0.8B-b'+str(batch)+'-'+backend+'-'+str(repeat))
            assert json.loads((folder/'exit.json').read_text())['exit_code']==0
            log=(folder/'output.log').read_text();assert 'Segfault encountered' not in log
            values=[json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('matched_frame_result ')]
            assert len(values)==3
            for value in values:
                assert value['runtime']=='garnet' and value['vision_reencoded'] and all(value['finished'])
                assert value['clock']=='time.monotonic' and value['case']['batch']==batch and value['case']['max_tokens']==32
                assert value['case']['capacity']==256 and len(value['prompt_token_ids'])==79
                assert value['prompt_token_ids'].count(248056)==54
                assert value['decode_steps_after_first']==max(map(len,value['token_ids']))-1
                key=(value['prompt_token_ids'],value['token_ids'])
                if identity is None:identity=key
                assert key==identity
                value['folder']=folder.name;records.append(value);all_values.append(value)
                if value['request']>0:samples[backend].append(value)
    assert len({v['rgb_sha256'] for v in all_values})==1 and len({v['patch_sha256'] for v in all_values})==1
    def mean(values):
        return {key:statistics.mean(v[key] for v in values) for key in ('ttft_seconds','decode_wall_steps_per_second','total_seconds','common_preprocessing_seconds')}
    candidate=mean(samples['garnet']);control=mean(samples['garnet-control'])
    group=dict(batch=batch,candidate=candidate,control=control,ttft_fraction=candidate['ttft_seconds']/control['ttft_seconds'],
        decode_ratio=candidate['decode_wall_steps_per_second']/control['decode_wall_steps_per_second'],
        exact_outputs=True,all_requests_finished=True,same_rgb_and_patch_bytes=True,requests=len(all_values))
    groups.append(group);print(json.dumps(group),flush=True)
(root/'native_frame_summary.json').write_bytes((json.dumps(dict(scope='Native ABBA real frames; not a vLLM speed comparison',groups=groups,records=records),indent=2)+'\n').encode())
