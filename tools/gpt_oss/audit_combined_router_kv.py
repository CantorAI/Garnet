"""Independent archived OPT49/50 combined synthetic gate; no performance claim.

No Garnet, pipeline, KV-layout, admission or sanitizer-gate imports. Requires
the terminal shell status and original fixture vectors/payloads, not only a
result file's self-reported reference or PASS flags. DIRECTORY NEW_REPORT.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_matrices(root):
    root = Path(root)
    metadata = read(root/'gate-metadata.json')
    require(re.fullmatch(r'[0-9a-f]{40}', metadata['source_commit']), 'invalid source identity')
    require(set(metadata['native_binaries']) == {'libgarnet.so','libgarnet_gpt_oss.so'}, 'native identity names')
    for name, digest in metadata['native_binaries'].items():
        require(sha(root/'native-payload'/name) == digest, 'native payload changed')
    fixture = root/'fixture'
    hashes = {name: sha(fixture/name) for name in ('model.safetensors','config.json','request.json','expected.json')}
    require(hashes == metadata['fixture_sha256'], 'original CPU fixture changed')
    config, request, expected = (read(fixture/name) for name in ('config.json','request.json','expected.json'))
    require(len(request['input_ids']) == 65 and request['max_new_tokens'] == 3, 'full synthetic input/output contract')
    require(config['num_hidden_layers'] == 2 and config['head_dim'] == 64 and
        config['num_attention_heads'] == 16 and config['num_key_value_heads'] == 2 and
        config['sliding_window'] == 17 and config['intermediate_size'] == 64, 'fixture dimensions')
    require(len(expected['generation_logits']) == 3 and len(expected['generated']) == 3, 'CPU prefix count')
    common = metadata['common_environment']
    for key, value in {'GARNET_TRT_SYNC_ALLOCATOR':'1',
        'GARNET_GPT_OSS_MARLIN_PREPACKED':'1','GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL':'0',
        'GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE':'1','GARNET_GPT_OSS_PREFILL_FLASHINFER':'1',
        'GARNET_GPT_OSS_DECODE_FLASHINFER':'1'}.items():
        require(common.get(key) == value, 'common kernel policy mismatch: '+key)
    checked, reference, maximum = [], {}, 0

    def one(path, axis, router, hybrid, diagnostic=False):
        nonlocal maximum
        r = read(path)
        batch, chunk = (128,16) if axis == 'expert' else (512,8)
        wanted_env = dict(common, GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=str(int(axis=='expert')),
            GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=str(int(axis=='intermediate')),
            GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=str(router), GARNET_GPT_OSS_HYBRID_KV=str(hybrid))
        require(r['source_commit'] == metadata['source_commit'] and r['native_binaries'] == metadata['native_binaries'] and
            r['hardware_csv'] == metadata['hardware_csv'] and r['fixture_sha256'] == hashes and
            r['kernel_environment'] == wanted_env, 'raw source/native/hardware/fixture/kernel provenance')
        require(r['cache_root'] == metadata['directory']+f'/{axis}-router{router}-hybrid{hybrid}-cache', 'cache identity')
        require(r['measurement']=='Synthetic CPU-prefix KV diagnostic, not performance' and
            r['passed'] is True and r['failure'] is None and r['profiled_diagnostic'] is diagnostic,
            'completed diagnostic scope')
        require((r['batch'],r['chunk'],r['input_tokens'],r['padded_tail_tokens'],r['complete_generations']) ==
            (batch,chunk,65,(-65)%chunk,2), 'full batch/input/tail/replay shape')
        plan = r['plan']; capacity = ((65+chunk+15)//16)*16; pages = capacity//16
        require(plan['config']==config and plan['mode']=='gpt-oss-tensor-parallel-tp2' and
            plan['batch']==batch and plan['capacity']==capacity and plan['max_tokens']==chunk and
            plan['kv_pages']==batch*pages and plan['local_kv_heads']==1 and plan['marlin_prepacked'] is True and
            plan['expert_weight_shards'] is (axis=='expert') and plan['moe_intermediate_shards'] is (axis=='intermediate'),
            'plan/weight-axis shape')
        require(len(plan['stages'])==2 and [s['rank'] for s in plan['stages']]==[0,1] and
            len({s['device_id'] for s in plan['stages']})==2 and
            all(s['start']==0 and s['end']==2 and 2*plan['estimated_per_gpu_bytes']<=s['budget_bytes']
                for s in plan['stages']), 'complete resident TP2 stages/budget')
        if hybrid:
            require(plan['schema']==4, 'hybrid plan schema')
            # Independent token-distance capacity calculation, no layout helper.
            rings = min(pages, (17+chunk-1+15)//16)
            bytes_ = (config['num_hidden_layers']//2)*batch*16*64*4*(pages+rings)
            table_bytes = batch*pages*4
            layout = plan['kv_layout']
            wanted_layout=dict(schema=1,mode='gpt-oss-alternating-hybrid-kv-v1',page_size=16,dtype='bfloat16',
                max_prefill_tokens=chunk,sliding_window=17,window_layers=[0],global_layers=[1],
                physical_layer_rule='logical_layer//2',logical_pages_per_request=pages,window_pages_per_request=rings,
                global_shape=[1,batch*pages,16,1,64],window_shape=[1,batch*rings,16,1,64],
                window_table_shape=[batch,pages],shared_kv_bytes=bytes_,shared_auxiliary_bytes=table_bytes,
                extra_tensor_arguments=['window_keys','window_values','window_table'])
            require(layout==wanted_layout,
                'independent bank/layout byte accounting')
        else:
            require(plan['schema']==3 and 'kv_layout' not in plan, 'original plan schema')
            bytes_ = config['num_hidden_layers']*batch*pages*16*64*4; table_bytes=0
        require(r['shared_kv_bytes']==bytes_ and r['shared_auxiliary_bytes']==table_bytes,
            'independent runtime KV/auxiliary byte accounting')
        require(len(r['steps'])==6, 'all six prefix matrices')
        matrices=[]; worst=0
        for i, step in enumerate(r['steps']):
            cpu = expected['generation_logits'][i%3]
            require(len(cpu)==config['vocab_size'] and all(math.isfinite(x) for x in cpu), 'CPU reference extent/finite')
            require((step['generation'],step['step'])==(i//3,i%3) and step['cpu_logits']==cpu, 'original CPU prefix reference')
            actual = step['tp_logits']; wanted = cpu*batch
            require(len(actual)==len(wanted) and all(math.isfinite(x) for x in actual), 'actual full matrix extent/finite')
            errors=[abs(a-b) for a,b in zip(actual,wanted)]
            require(step['within_existing_tolerance'] is True and step['maximum_absolute_error']==max(errors) and
                all(e<=.025*(1+abs(b)) for e,b in zip(errors,wanted)), 'unchanged compiled numerical bound')
            matrices.append(actual); worst=max(worst,max(errors))
        require(matrices[:3]==matrices[3:], 'exact reused-request matrices')
        maximum=max(maximum,worst)
        return matrices, dict(path=path.name,sha256=sha(path),batch=batch,chunk=chunk,max_cpu_abs=worst,
            kv_bytes=bytes_,auxiliary_bytes=table_bytes)

    for axis in ('expert','intermediate'):
        matrices=None
        for router in (0,1):
            for hybrid in (0,1):
                for phase in ('cold','warm'):
                    values, record = one(root/f'{axis}-router{router}-hybrid{hybrid}-{phase}.json',axis,router,hybrid)
                    if matrices is None: matrices=values
                    require(values==matrices, 'exact cold/warm/router/bank cross-mode matrices')
                    checked.append(record)
        reference[axis]=matrices
    for name, diagnostic in (('compiled-combined-mem',False),('combined-trace',True)):
        values, record = one(root/(name+'.json'),'intermediate',1,1,diagnostic)
        require(values==reference['intermediate'], 'exact memory/trace matrices against original')
        checked.append(record)
    return dict(metadata=metadata,compiled_files=checked,max_cpu_abs=maximum,
        exact_cold_warm_router_banks_repeated_requests=True)


def exact_initialization(record):
    require(record.findtext('kind')=='Api', 'all memory records fatal')
    api, code = record.findtext('what/api'), record.findtext('what/result')
    frames=record.findall('hostStack/frame')
    require(record.findtext('hostStack/saveLocation')=='error' and len(frames)>=3 and
        frames[1].findtext('func')==api, 'complete API origin')
    modules=[Path(f.findtext('module','')).name for f in frames]
    caller=frames[2].findtext('func','')
    if (api,code) in (('cudaFuncGetAttributes','209'),('cudaGetLastError','209')):
        require(modules[1:3]==['libnccl.so.2']*2 and re.fullmatch(r'ncclInitKernelsForDevice(?:\(.*\))?',caller), 'exact kernel initialization')
    elif (api,code)==('cudaGetLastError','704'):
        if modules[1:3]==['libnccl.so.2']*2:
            require(any(re.fullmatch(re.escape(n)+r'(?:\(.*\))?',caller) for n in
                ('p2pRecvConnect','p2pMap','p2pMap [clone .isra.7]')), 'exact NCCL peer initialization')
        else:
            require(modules[1:3]==['libcudart.so.13','libgarnet.so'] and
                re.fullmatch(r'Garnet::GarnetAPI::TensorToDevice(?:\(.*\))?',caller), 'exact Garnet already-enabled peer clearing')
    else:
        raise ValueError('unrelated API record fatal')
    return f'{api}/{code}/{caller}'


def audit(root, output):
    root, output = Path(root), Path(output)
    require(not output.exists(), 'refuse audit overwrite')
    terminal=read(root/'terminal.json')
    require(terminal['terminal'] is True and terminal['actual_exit']==0 and
        terminal['phase']=='complete-raw-combined', 'actual terminal application status')
    result=audit_matrices(root)
    xml=root/'compiled-combined-mem.xml'; tree=ET.fromstring(xml.read_bytes())
    require(tree.tag=='ComputeSanitizerOutput','complete unrestricted XML schema')
    records=tree.findall('record'); exclusions=Counter(exact_initialization(r) for r in records)
    log=(root/'compiled-combined-mem.log').read_text()
    require(re.search(r'ERROR SUMMARY:\s*'+str(len(records))+r' errors',log), 'complete tool summary/count')
    require(not any(x in log for x in ('No attachable process found','compute-sanitizer timed-out',
        '========= Error:','Target application returned an error')), 'tool/attachment/application failure')
    result['sanitizer']=dict(records=len(records),exact_initialization=dict(exclusions),unexpected=0,sha256=sha(xml))
    counts=Counter(); header=None
    for row in csv.reader((root/'combined-kernels.csv').read_text().splitlines()):
        if 'Name' in row and 'Instances' in row:header=row;continue
        if header and len(row)==len(header):
            for kernel in ('SinkAttention','routeScoresTensorCore'):
                if kernel in row[header.index('Name')]:counts[kernel]+=int(row[header.index('Instances')])
    require(all(counts[n]>=8 for n in ('SinkAttention','routeScoresTensorCore')), 'both ranks/layers prefill+decode dispatch')
    result['dispatch']=dict(counts)
    result['scope']='Combined high-batch synthetic/native-matched compiled matrices/memory/dispatch; NOT pretrained/capacity/speed'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('COMBINED_ROUTER_KV_RAW_AUDIT_PASS',len(result['compiled_files']),'files',result['max_cpu_abs'],flush=True)
    return result


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Expected COMPLETE_DIRECTORY NEW_REPORT')
    audit(sys.argv[1],sys.argv[2])
