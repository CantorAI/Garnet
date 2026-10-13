"""Validate saved matched requests and report the 2x/half-TTFT target honestly."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);records=[];groups=[]
for batch in (1,2):
    for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        samples={'garnet':[],'vllm-metal':[]};all_case=[]
        for repeat in range(2):
            for backend in samples:
                folder=root/(name+'-b'+str(batch)+'-'+backend+'-'+str(repeat))
                status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0,status
                values=[];processes=[];prompt=None
                for line in (folder/'output.log').read_text().splitlines():
                    marker,_,payload=line.partition(' ')
                    if marker=='matched_prompt':prompt=json.loads(payload)
                    elif marker=='matched_result':values.append(json.loads(payload))
                    elif marker=='matched_process':processes.append(json.loads(payload))
                assert len(values)==3 and processes[-1]['exit_code']==0,folder
                for value in values:
                    assert value['case']['batch']==batch and value['case']['max_tokens']==64
                    assert value['clock']=='time.monotonic' and value['ttft_seconds']>0
                    assert all(t>0 for t in value['decode_step_seconds'])
                    assert value['decode_window_seconds']>0
                    ids=[list(row) for row in value['token_ids']]
                    if backend=='vllm-metal':
                        assert not value['prefix_caching'] and not value['speculative_decoding']
                        # vLLM may omit the terminal stop ID from RequestOutput.
                        # Reconstruct it only from its recorded finish/stop reason.
                        with open(folder.parent/(name+'-b'+str(batch)+'-garnet-'+str(repeat))/'output.log') as stream:
                            native_prompt=next(json.loads(line.split(' ',1)[1]) for line in stream if line.startswith('matched_prompt '))
                        for row,done,reason in zip(ids,value['finished'],value['stop_reasons']):
                            if done:
                                stop=reason if isinstance(reason,int) else native_prompt['stop_token_ids'][0]
                                if not row or row[-1] not in native_prompt['stop_token_ids']:row.append(stop)
                    value.update(normalized_token_ids=ids,folder=folder.name,process=processes[-1])
                    value['step_accounting_valid']=value['decode_steps_after_first']==max(map(len,ids))-1
                    all_case.append(value);records.append(value)
                    if value['request']>0:samples[backend].append(value)
        expected=all_case[0]
        matched=all(v['normalized_token_ids']==expected['normalized_token_ids'] and v['finished']==expected['finished'] for v in all_case)
        same_prompt=all(v['prompt_token_ids']==expected['prompt_token_ids'] for v in all_case)
        accounting=all(v['step_accounting_valid'] for v in all_case)
        def mean(values):
            return dict(ttft_seconds=statistics.mean(v['ttft_seconds'] for v in values),
                decode_steps_per_second=statistics.mean(v['decode_wall_steps_per_second'] for v in values),
                total_seconds=statistics.mean(v['total_seconds'] for v in values),
                peak_process_rss_gib=statistics.mean(v['process']['peak_rss_bytes']/2**30 for v in values))
        garnet=mean(samples['garnet']);vllm=mean(samples['vllm-metal'])
        ratio=garnet['decode_steps_per_second']/vllm['decode_steps_per_second'];fraction=garnet['ttft_seconds']/vllm['ttft_seconds']
        valid=matched and same_prompt and accounting and all(expected['finished'])
        group=dict(model=name,batch=batch,garnet=garnet,vllm=vllm,decode_ratio=ratio,ttft_fraction=fraction,
            exact_outputs=matched,exact_prompt_ids=same_prompt,one_prediction_per_step=accounting,
            output_tokens_including_stop=list(map(len,expected['normalized_token_ids'])),requests=12,
            decode_target_met=valid and ratio>=2,ttft_target_met=valid and fraction<=0.5)
        groups.append(group);print(json.dumps(group),flush=True)
result=dict(groups=groups,records=records,all_targets_met=all(v['decode_target_met'] and v['ttft_target_met'] for v in groups),
    warm_requests_per_configuration=4,precision_scope='identical BF16 checkpoint weights; Garnet FP32 activations/KV versus vLLM BF16 model/KV',
    timing_scope='direct native library versus direct vLLM engine; initialization excluded from warm timing; decode uses the complete wall interval between first and final prediction, including time between engine calls')
(root/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print('All 2x/half-TTFT targets met:',result['all_targets_met'])
