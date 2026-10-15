"""Validate image-frame evidence and report matched-input speed, including limits."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);groups=[];records=[]
for batch in (1,2):
    samples={'garnet':[],'vllm-metal':[]};all_values=[];processes=[]
    for repeat in range(2):
        for backend in samples:
            folder=root/('Qwen3.5-0.8B-b'+str(batch)+'-'+backend+'-'+str(repeat))
            assert json.loads((folder/'exit.json').read_text())['exit_code']==0
            log=(folder/'output.log').read_text()
            assert 'Segfault encountered' not in log
            values=[json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('matched_frame_result ')]
            assert len(values)==3
            for value in values:
                assert value['clock']=='time.monotonic' and value['case']['max_tokens']==32
                assert value['case']['capacity']==256 and value['case']['batch']==batch
                assert len(value['prompt_token_ids'])==79 and value['prompt_token_ids'].count(248056)==54
                assert value['ttft_seconds']>0 and value['decode_window_seconds']>0
                assert value['decode_steps_after_first']==max(map(len,value['token_ids']))-1
                if backend=='garnet':assert value['vision_reencoded']
                else:
                    assert not value['prefix_caching'] and not value['speculative_decoding']
                    assert len(set(value['encoder_cache_identities']))==batch
                value['folder']=folder.name;records.append(value);all_values.append(value)
                if value['request']>0:samples[backend].append(value)
    assert len({v['rgb_sha256'] for v in all_values})==1
    assert len({v['patch_sha256'] for v in all_values})==1
    assert all(v['prompt_token_ids']==all_values[0]['prompt_token_ids'] for v in all_values)
    identities=[identity for v in all_values if v['runtime']=='vllm-metal' for identity in v['encoder_cache_identities']]
    assert len(set(identities))==len(identities)
    exact=all(v['token_ids']==all_values[0]['token_ids'] for v in all_values)
    finished=all(all(v['finished']) for v in all_values)
    def mean(values):
        return {key:statistics.mean(v[key] for v in values) for key in
            ('ttft_seconds','decode_wall_steps_per_second','total_seconds','common_preprocessing_seconds')}
    native=mean(samples['garnet']);comparator=mean(samples['vllm-metal'])
    ratio=native['decode_wall_steps_per_second']/comparator['decode_wall_steps_per_second']
    fraction=native['ttft_seconds']/comparator['ttft_seconds']
    group=dict(batch=batch,garnet=native,vllm=comparator,decode_ratio=ratio,ttft_fraction=fraction,
        same_rgb_pixels=True,same_prompt_ids=True,fresh_encoder_identities=True,exact_outputs=exact,
        all_requests_finished=finished,decode_target_met=exact and finished and ratio>=2,
        ttft_target_met=exact and finished and fraction<=.5)
    groups.append(group);print(json.dumps(group),flush=True)
(root/'frame_summary.json').write_bytes((json.dumps(dict(groups=groups,records=records,
    precision_scope='same BF16 checkpoint; native FP32 vision/activations/KV versus vLLM BF16 model/vision/KV',
    timing_scope='common CPU image preparation and persistent worker protocol; first/done markers timestamped before diagnostics'),indent=2)+'\n').encode())
