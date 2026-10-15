"""Independent source, raw worker output, dtype, screening and metric audit."""
from pathlib import Path
source=Path(__file__).with_name('verify_global_alpha_vllm.py').read_text().replace("glob('global-alpha-vllm-20261014*')","glob('screened-head-vllm-20261015*')")
needle="outputs=0;audits=0;profiles=[]";assert source.count(needle)==1
source=source.replace(needle,needle+"\nfor name,sha in selection['source_manifest'].items():assert hashlib.sha256((e/'source'/name).read_bytes()).hexdigest()==sha")
needle="        for value in values:";assert source.count(needle)==1
source=source.replace(needle,'''        assert metadata['threaded_decoder_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
        for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((e/'source/test2026/metal'/name).read_bytes()).hexdigest()==sha
        for backend in ('garnet','vllm-metal'):
            for repeat in (0,1):
                worker=folder/(backend+'-'+str(repeat));lines=(worker/'output.log').read_text().splitlines()
                prefix='matched_frame_result ' if model=='Qwen3.5-0.8B' else 'matched_result '
                raw=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(prefix)];assert len(raw)==3
                processes=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('matched_process ')];assert processes and processes[-1]['exit_code']==0
                stored=[v for v in values if v['backend']==backend and v['repeat']==repeat];assert len(stored)==3
                for observed,record in zip(raw,stored):
                    assert all(record[k]==v for k,v in observed.items()) and record['process']==processes[-1]
                monitor=json.loads((worker/'monitor.json').read_text());assert monitor and all(v['guard_passed'] for v in monitor)
'''+needle)
needle="                assert value['attention_prefix_enabled']==chosen['attention_prefix_enabled']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                assert value['screened_head_enabled'] and value['screened_head_counts']==[1]*(value['decode_steps_after_first']+1)
                assert not value['fused_head_argmax_enabled'] and not value['body_shape_enabled']
                assert not value['prefill_stream_enabled'] and not value['dirty_prefix_reset_enabled']
                assert not value['deferred_state_zero_enabled'] and not value['reused_zero_source_enabled']
''')
needle="                assert value['patch_sha256']==value['case']['image_metadata']['patch_sha256']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                assert value['frontend_policy']=='combined' and value['frontend_route']=='threaded-striped'
                assert all(value['rgb_sha256']==v['rgb_sha256'] for v in values)
''')
source=source.replace('verification=dict(verified_files=',"assert outputs==72 and audits==3 and len(profiles)==6\nverification=dict(verified_files=")
exec(compile(source,__file__,'exec'))
