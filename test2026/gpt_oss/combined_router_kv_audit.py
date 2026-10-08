"""CPU mock-artifact audit tests; never GPU/model/performance evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'tools/gpt_oss'))
from audit_combined_router_kv import audit

def save(path, value): path.write_text(json.dumps(value))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);(root/'fixture').mkdir();(root/'native-payload').mkdir()
    config=dict(num_hidden_layers=2,head_dim=64,num_attention_heads=16,num_key_value_heads=2,
        sliding_window=17,intermediate_size=64,vocab_size=4)
    save(root/'fixture/config.json',config)
    save(root/'fixture/request.json',dict(input_ids=list(range(65)),max_new_tokens=3))
    cpu=[[.1,.2,-.3,.4],[.2,-.4,.6,.3],[.5,.3,-.2,.6]]
    save(root/'fixture/expected.json',dict(generation_logits=cpu,generated=[3,2,3]))
    (root/'fixture/model.safetensors').write_bytes(b'CPU MOCK ONLY, no native weights')
    binaries={}
    for name in ('libgarnet.so','libgarnet_gpt_oss.so'):
        (root/'native-payload'/name).write_bytes(b'CPU MOCK ONLY '+name.encode());binaries[name]=sha(root/'native-payload'/name)
    common={'GARNET_TRT_SYNC_ALLOCATOR':'1','GARNET_GPT_OSS_MARLIN_PREPACKED':'1',
        'GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL':'0','GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE':'1',
        'GARNET_GPT_OSS_PREFILL_FLASHINFER':'1','GARNET_GPT_OSS_DECODE_FLASHINFER':'1'}
    metadata=dict(directory=str(root),source_commit='a'*40,native_binaries=binaries,
        hardware_csv='CPU MOCK hardware, NOT GPU',common_environment=common,
        fixture_sha256={n:sha(root/'fixture'/n) for n in ('model.safetensors','config.json','request.json','expected.json')})
    save(root/'gate-metadata.json',metadata)
    save(root/'terminal.json',dict(terminal=True,actual_exit=0,phase='complete-raw-combined'))
    for axis in ('expert','intermediate'):
        b,chunk=(128,16) if axis=='expert' else (512,8);pages=((65+chunk+15)//16)
        for router in (0,1):
            for hybrid in (0,1):
                env=dict(common,GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=str(int(axis=='expert')),
                    GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=str(int(axis=='intermediate')),
                    GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=str(router),GARNET_GPT_OSS_HYBRID_KV=str(hybrid))
                plan=dict(config=config,mode='gpt-oss-tensor-parallel-tp2',schema=4 if hybrid else 3,
                    batch=b,capacity=pages*16,max_tokens=chunk,kv_pages=b*pages,local_kv_heads=1,
                    marlin_prepacked=True,expert_weight_shards=axis=='expert',moe_intermediate_shards=axis=='intermediate',
                    estimated_per_gpu_bytes=100,stages=[dict(rank=i,device_id=i,start=0,end=2,budget_bytes=1000) for i in (0,1)])
                ring=min(pages,(17+chunk-1+15)//16)
                size=b*16*64*4*(pages+ring) if hybrid else 2*b*pages*16*64*4
                table=b*pages*4 if hybrid else 0
                if hybrid:plan['kv_layout']=dict(schema=1,mode='gpt-oss-alternating-hybrid-kv-v1',page_size=16,dtype='bfloat16',
                    max_prefill_tokens=chunk,sliding_window=17,window_layers=[0],global_layers=[1],physical_layer_rule='logical_layer//2',
                    logical_pages_per_request=pages,window_pages_per_request=ring,global_shape=[1,b*pages,16,1,64],
                    window_shape=[1,b*ring,16,1,64],window_table_shape=[b,pages],shared_kv_bytes=size,shared_auxiliary_bytes=table,
                    extra_tensor_arguments=['window_keys','window_values','window_table'])
                r=dict(source_commit=metadata['source_commit'],native_binaries=binaries,hardware_csv=metadata['hardware_csv'],
                    fixture_sha256=metadata['fixture_sha256'],kernel_environment=env,cache_root=str(root)+f'/{axis}-router{router}-hybrid{hybrid}-cache',
                    measurement='Synthetic CPU-prefix KV diagnostic, not performance',passed=True,failure=None,profiled_diagnostic=False,
                    batch=b,chunk=chunk,input_tokens=65,padded_tail_tokens=(-65)%chunk,complete_generations=2,plan=plan,
                    shared_kv_bytes=size,shared_auxiliary_bytes=table,
                    steps=[dict(generation=i//3,step=i%3,cpu_logits=cpu[i%3],tp_logits=cpu[i%3]*b,
                        maximum_absolute_error=0,within_existing_tolerance=True) for i in range(6)])
                for phase in ('cold','warm'):save(root/f'{axis}-router{router}-hybrid{hybrid}-{phase}.json',r)
    baseline=json.loads((root/'intermediate-router1-hybrid1-warm.json').read_text())
    save(root/'compiled-combined-mem.json',baseline)
    trace=deepcopy(baseline);trace['profiled_diagnostic']=True;save(root/'combined-trace.json',trace)
    xml='''<ComputeSanitizerOutput><record><kind>Api</kind><what><api>cudaFuncGetAttributes</api><result>209</result></what>
<hostStack><saveLocation>error</saveLocation><frame><func>program</func><module>app</module></frame>
<frame><func>cudaFuncGetAttributes</func><module>/lib/libnccl.so.2</module></frame>
<frame><func>ncclInitKernelsForDevice(int)</func><module>/lib/libnccl.so.2</module></frame></hostStack></record></ComputeSanitizerOutput>'''
    (root/'compiled-combined-mem.xml').write_text(xml)
    (root/'compiled-combined-mem.log').write_text('ERROR SUMMARY: 1 errors\n')
    (root/'combined-kernels.csv').write_text('Name,Instances\nSinkAttention,8\nrouteScoresTensorCore,8\n')
    positive=audit(root,root/'positive.json')
    assert len(positive['compiled_files'])==18 and positive['max_cpu_abs']==0
    hazards=0
    def reject_json(name, mutate):
        global hazards
        path=root/name; original=path.read_bytes();value=json.loads(original);mutate(value);save(path,value)
        output=root/f'negative-{hazards}.json'
        try:
            try:audit(root,output)
            except ValueError:pass
            else:raise AssertionError('Corrupt raw artifact accepted: '+name)
            assert not output.exists()
            hazards+=1
        finally:path.write_bytes(original)
    name='intermediate-router1-hybrid1-warm.json'
    for mutation in (
        lambda r:r.update(source_commit='b'*40),lambda r:r.update(hardware_csv='other hardware'),
        lambda r:r.update(passed=False),lambda r:r.update(failure='failed native app'),
        lambda r:r.update(measurement='performance'),lambda r:r.update(batch=511),
        lambda r:r.update(complete_generations=1),lambda r:r.update(padded_tail_tokens=0),
        lambda r:r.update(shared_kv_bytes=r['shared_kv_bytes']+4),lambda r:r.update(shared_auxiliary_bytes=0),
        lambda r:r.update(cache_root='/other/cache'),lambda r:r['kernel_environment'].update(GARNET_TRT_SYNC_ALLOCATOR='0'),
        lambda r:r['plan']['kv_layout'].update(dtype='float32'),lambda r:r['plan'].update(expert_weight_shards=True),
        lambda r:r['plan']['stages'][0].update(budget_bytes=1),lambda r:r['steps'].pop(),
        lambda r:r['steps'][0]['cpu_logits'].__setitem__(0,.101),
        lambda r:r['steps'][0]['tp_logits'].__setitem__(0,float('nan')),
        lambda r:r['steps'][0]['tp_logits'].pop()):reject_json(name,mutation)
    def drift(r):
        for i,step in enumerate(r['steps']):
            step['tp_logits'][0]+=.001
            step['maximum_absolute_error']=max(abs(a-b) for a,b in zip(step['tp_logits'],cpu[i%3]*512))
    reject_json(name,drift) # passes numerical bound and replay equality, fails cross-mode exactness
    reject_json('terminal.json',lambda r:r.update(actual_exit=1))
    reject_json('terminal.json',lambda r:r.update(terminal=False))
    reject_json('gate-metadata.json',lambda r:r['common_environment'].update(GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL='1'))
    reject_json('fixture/expected.json',lambda r:r['generation_logits'][0].__setitem__(0,.101))
    for mutate in (lambda e:e.find('record/kind').__setattr__('text','Memory'),
                   lambda e:e.find('record/what/result').__setattr__('text','719'),
                   lambda e:e.findall('record/hostStack/frame')[2].find('module').__setattr__('text','libunrelated.so')):
        e=ET.fromstring(xml);mutate(e);(root/'compiled-combined-mem.xml').write_bytes(ET.tostring(e))
        try:audit(root,root/'bad-xml.json')
        except ValueError:hazards+=1
        else:raise AssertionError('Memory/unrelated API accepted')
        (root/'compiled-combined-mem.xml').write_text(xml)
    (root/'combined-kernels.csv').write_text('Name,Instances\nSinkAttention,8\nrouteScoresTensorCore,4\n')
    try:audit(root,root/'bad-dispatch.json')
    except ValueError:hazards+=1
    else:raise AssertionError('Missing decode TC dispatch accepted')
    before=(root/'positive.json').read_bytes()
    try:audit(root,root/'positive.json')
    except ValueError:hazards+=1
    else:raise AssertionError('Audit overwritten')
    assert (root/'positive.json').read_bytes()==before
    print('CPU MOCK combined raw audit:',hazards,'scope/identity/matrix/byte/API/memory/dispatch/overwrite hazards rejected; no GPU proof')
