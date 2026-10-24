"""Isolated actual-checkpoint embedding gate, prepared for a fresh native build.

Not a benchmark. Run only after the prior worker closes and build provenance is retained.
Arguments: study-root fixture-directory new-output-directory expected-new-library-sha256
"""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from test2026.cuda.qwen38_5090.run_logged import source_identity
root=Path(sys.argv[1]);fixtures=Path(sys.argv[2]);out=Path(sys.argv[3]);expected_library=sys.argv[4]
sha=lambda b:hashlib.sha256(b).hexdigest()
library_sha=sha((root/'build/bin/garnet.dll').read_bytes())
assert library_sha==expected_library and library_sha!='3c3207b84b0e048f7e69736f3ae927ffb629c40ffe086994765e9a70ac5f58dd'
fixture=json.loads((fixtures/'fixture.json').read_text(encoding='utf-8'))
assert fixture['status']=='checkpoint_embedding_rows_prepared_cpu_only'
vocab,width=fixture['shape'];ids=fixture['ids'];oracle={}
for token,paths in fixture['rows'].items():
    raw=(fixtures/paths['bf16']).read_bytes();f32=(fixtures/paths['f32']).read_bytes()
    assert len(raw)==width*2 and len(f32)==width*4 and sha(raw)==paths['bf16_sha256'] and sha(f32)==paths['f32_sha256']
    assert f32==b''.join(b'\x00\x00'+raw[i:i+2] for i in range(0,len(raw),2))
    oracle[int(token)]=f32
out.mkdir(exist_ok=False);path=out/'report.json'
report={'status':'starting','library_sha256':library_sha,'source':source_identity(),
    'fixture_sha256':sha((fixtures/'fixture.json').read_bytes()),'shape':[vocab,width],'cases':[],
    'scope':'Selected embedding rows, batch/chunk/layout/invalid-input/mode/head lifecycle only. No full-model, sanitizer, memory-saving or speed admission.'}
io=garnet.cuda_tensor_io()
def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
def reject(name,call):
    try:call()
    except Exception as error:report.setdefault('rejections',[]).append({'name':name,'error':repr(error)});save();return
    raise AssertionError('Expected rejection: '+name)
def record(name,value,expected):
    raw=bytes(io.finite_f32_bytes(value));(out/(name+'.f32')).write_bytes(raw)
    report['cases'].append({'name':name,'sha256':sha(raw),'bytes':len(raw),'exact':raw==expected});save()
    assert raw==expected,name+' differs from checkpoint row bytes'
try:
    checkpoint=root/'models/Qwen3.8-27B-NVFP4'
    assert checkpoint.resolve()==Path(fixture['checkpoint']).resolve()
    io.open(str(checkpoint));report['initial_memory']=json.loads(io.memory_json())
    assert report['initial_memory']['model_open'] and report['initial_memory']['embedding_table_bytes']==vocab*width*2
    reject('host_requires_native_head',lambda:io.enable_host_embedding(True))
    io.enable_nvfp4_head(True)
    predictions=[]
    for mode in ('device','host'):
        if mode=='host':io.enable_host_embedding(True)
        memory=json.loads(io.memory_json());report[mode+'_memory']=memory
        assert memory['model_open'] and memory['embedding_mode']==('host_mapped_bf16' if mode=='host' else 'device_bf16')
        assert memory['embedding_table_bytes']==(0 if mode=='host' else vocab*width*2)
        assert memory['host_embedding_mapped_tensor_bytes']==(vocab*width*2 if mode=='host' else 0)
        embedded=io.embedding(ids)
        record(mode+'-vector',embedded,b''.join(oracle[token] for token in ids))
        predictions.append(list(io.predict(io.embedding([ids[0]]))))
        # Metadata/projection access must remain open after GPU-table retirement.
        report[mode+'_embedding_metadata']=json.loads(io.weight_info('model.language_model.embed_tokens.weight'))
        assert report[mode+'_embedding_metadata']['shape']==[vocab,width]
        for batch in (1,2,4,8):
            for time in (1,4,16,128):
                dst=io.zeros([batch,time,width])
                counts=(time,) if time==1 else (time,time-1)
                for count in counts:
                    selected=[[ids[(t*batch+b)%len(ids)] for b in range(batch)] for t in range(count)]
                    io.embedding_chunk(selected,dst)
                    flat=io.storage_view(dst,[batch,time*width])
                    for row in range(batch):
                        expected=b''.join(oracle[selected[t][row]] if t<count else bytes(width*4) for t in range(time))
                        record(mode+f'-b{batch}-t{time}-n{count}-r{row}',io.row(flat,row),expected)
                dst=None;flat=None
        reject(mode+'-negative-vector-id',lambda:io.embedding([-1]))
        reject(mode+'-high-vector-id',lambda:io.embedding([vocab]))
        reject(mode+'-empty-vector',lambda:io.embedding([]))
        destination=io.zeros([1,4,width]);io.embedding_chunk([[ids[1]]]*4,destination)
        before=bytes(io.finite_f32_bytes(destination))
        record(mode+'-invalid-before',destination,oracle[ids[1]]*4)
        for name,tokens in [('negative',[[-1]]),('high',[[vocab]]),('wrong-batch',[[0,1]]),('empty',[]),('over-time',[[0]]*5)]:
            reject(mode+'-chunk-'+name,lambda tokens=tokens:io.embedding_chunk(tokens,destination))
            record(mode+'-invalid-after-'+name,destination,before)
        reject(mode+'-shape',lambda:io.embedding_chunk([[0],[1]],io.zeros([1,1,width])))
        destination=None;embedded=None
    assert predictions[0]==predictions[1];report['head_predictions']=predictions
    reject('cannot_disable_native_head_in_host_mode',lambda:io.enable_nvfp4_head(False))
    io.enable_host_embedding(True) # repeated enable is idempotent
    io.enable_host_embedding(False)
    record('restored-device-vector',io.embedding(ids),b''.join(oracle[token] for token in ids))
    report['restored_memory']=json.loads(io.memory_json())
    assert report['restored_memory']['embedding_table_bytes']==vocab*width*2 and report['restored_memory']['host_embedding_mapped_tensor_bytes']==0
    io.enable_nvfp4_head(False);report['native_head_disabled_memory']=json.loads(io.memory_json());io.close()
    reject('closed-vector',lambda:io.embedding([0]))
    reject('closed-host-enable',lambda:io.enable_host_embedding(True))
    report['closed_memory']=json.loads(io.memory_json());assert not report['closed_memory']['model_open']
    io.open(str(checkpoint));record('reopened-default-vector',io.embedding(ids),b''.join(oracle[token] for token in ids))
    report['reopened_memory']=json.loads(io.memory_json())
    assert report['reopened_memory']['embedding_mode']=='device_bf16'
    report['status']='embedding_gate_passed_needs_independent_replay'
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:io.close();save()
