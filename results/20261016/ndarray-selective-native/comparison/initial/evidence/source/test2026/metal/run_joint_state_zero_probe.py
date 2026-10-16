"""Exclusive real-config reset plus dependent-read component; no inference claim."""
from pathlib import Path
base=Path(__file__).with_name('run_decode_subsimd_probe.py').read_text()
exec(compile(base[:base.index('native=r/')],__file__,'exec'))
profiles=[];configs={}
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    path=r/'models'/model/'config.json';raw=json.loads(path.read_text());c=raw.get('text_config',raw);configs[model]=dict(sha256=digest(path),config=raw)
    for batch in (1,2):
        columns=[]
        for layer in range(c['num_hidden_layers']):
            kind=c.get('layer_types',['full_attention']*c['num_hidden_layers'])[layer]
            if kind=='linear_attention':
                channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim'];columns.extend([channels*c['linear_conv_kernel_dim'],c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']])
            else:
                width=c['num_key_value_heads']*256*c.get('head_dim',c['hidden_size']//c['num_attention_heads']);columns.extend([width,width])
        profiles.append(dict(name=model+'-b'+str(batch),batch=batch,columns=columns,logical_bytes=batch*sum(columns)*4))
profiles.append(dict(name='padded',batch=3,columns=[1,3,17,65],logical_bytes=3*(1+3+17+65)*4))
profile_path=output/'profiles.tsv';profile_path.write_text('\n'.join(p['name']+' '+str(p['batch'])+' '+str(len(p['columns']))+' '+' '.join(map(str,p['columns'])) for p in profiles)+'\n')
paths=[Path(__file__),s/'joint_state_zero_gpu.mm',s/'run_decode_subsimd_probe.py',s/'compare_vllm_metal.py']
metadata=dict(scope='Synthetic state reset plus full dependent GPU read, one CB for joint candidate; no full-model inference or vLLM claim',policies=['temporary_zero_vector_and_copy_then_gpu_read','cpu_memset_then_gpu_read','gpu_blit_fill_and_read_same_command_buffer'],conditioning='GPU writes/wait before every sample excluded; CPU validation outside timing',configs=configs,profiles=profiles,source_sha256={p.name:digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
(output/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode());snapshot_dir=output/'source-snapshot';snapshot_dir.mkdir()
for p in paths:(snapshot_dir/p.name).write_bytes(p.read_bytes())
binary=output/'joint_state_zero_gpu';run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'joint_state_zero_gpu.mm'),'-o',str(binary)])
run('gpu',[str(binary),str(profile_path)])
records=[json.loads(line) for line in (output/'gpu.log').read_text().splitlines() if line.startswith('{')]
checks=[v for v in records if v['kind']=='check'];assert len(checks)==63 and all(v['positive_zero_cpu_gpu'] and v['padding_preserved'] for v in checks)
summary=[]
for profile in profiles:
    windows=[v for v in records if v['kind']=='timing' and v['profile']==profile['name']];assert len(windows)==12
    means={str(policy):{metric:statistics.mean(x for v in windows if v['policy']==policy for x in v[metric]) for metric in ('wall_ms','gpu_ms')} for policy in range(3)}
    summary.append(dict(profile=profile['name'],logical_bytes=profile['logical_bytes'],means=means,joint_over_legacy_wall=means['2']['wall_ms']/means['0']['wall_ms'],joint_over_direct_cpu_wall=means['2']['wall_ms']/means['1']['wall_ms']))
(output/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=summary,full_goal_met=False),indent=2)+'\n').encode());guard();print(json.dumps(summary),flush=True)
