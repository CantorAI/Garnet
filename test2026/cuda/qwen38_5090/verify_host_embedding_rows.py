"""Replay embedding captures against retained checkpoint BF16 rows.

Usage: python verify_host_embedding_rows.py <capture-directory> <fixture-directory> <expected-library-sha256>
This does not verify build provenance, full-model arithmetic, sanitizers or performance.
"""
from pathlib import Path
import hashlib,json,math,struct,sys
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def verify(captures,fixtures,library):
    captures=Path(captures);fixtures=Path(fixtures)
    raw_fixture=(fixtures/'fixture.json').read_bytes();fixture=json.loads(raw_fixture)
    report=json.loads((captures/'report.json').read_bytes())
    assert report['status']=='embedding_gate_passed_needs_independent_replay'
    assert report['library_sha256']==library and len(library)==64
    assert report['fixture_sha256']==sha(raw_fixture)
    assert fixture['status']=='checkpoint_embedding_rows_prepared_cpu_only'
    vocab,width=fixture['shape'];assert report['shape']==[vocab,width]
    ids=fixture['ids'];assert ids[:4]==[0,vocab-1,1,0] and len(ids)==16
    rows={}
    for token in set(ids):
        entry=fixture['rows'][str(token)];raw=(fixtures/entry['bf16']).read_bytes()
        assert len(raw)==width*2 and sha(raw)==entry['bf16_sha256']
        # A separate format-based oracle over numeric BF16 words.
        words=struct.unpack('<'+str(width)+'H',raw)
        f32=struct.pack('<'+str(width)+'I',*(word*65536 for word in words))
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',f32))
        assert f32==(fixtures/entry['f32']).read_bytes() and sha(f32)==entry['f32_sha256']
        rows[token]=f32
    wanted={}
    for mode in ('device','host'):
        wanted[mode+'-vector']=b''.join(rows[token] for token in ids)
        for batch in (1,2,4,8):
            for time in (1,4,16,128):
                for count in ((time,) if time==1 else (time,time-1)):
                    for row in range(batch):
                        wanted[mode+f'-b{batch}-t{time}-n{count}-r{row}']=b''.join(
                            rows[ids[(token*batch+row)%len(ids)]] if token<count else bytes(width*4) for token in range(time))
        for suffix in ['invalid-before']+['invalid-after-'+name for name in ('negative','high','wrong-batch','empty','over-time')]:
            wanted[mode+'-'+suffix]=rows[ids[1]]*4
    vector=b''.join(rows[token] for token in ids)
    wanted['restored-device-vector']=vector;wanted['reopened-default-vector']=vector
    observed={entry['name']:entry for entry in report['cases']}
    assert len(observed)==len(report['cases']) and set(observed)==set(wanted)
    for name,expected in wanted.items():
        entry=observed[name];actual=(captures/(name+'.f32')).read_bytes()
        assert actual==expected and entry['exact'] and entry['bytes']==len(actual) and entry['sha256']==sha(actual),name
    rejections={entry['name']:entry['error'] for entry in report['rejections']}
    expected_rejections={'host_requires_native_head','cannot_disable_native_head_in_host_mode','closed-vector','closed-host-enable'}
    for mode in ('device','host'):
        expected_rejections.update(mode+'-'+name for name in ('negative-vector-id','high-vector-id','empty-vector','shape'))
        expected_rejections.update(mode+'-chunk-'+name for name in ('negative','high','wrong-batch','empty','over-time'))
    assert set(rejections)==expected_rejections and len(report['rejections'])==len(expected_rejections)
    assert all(isinstance(message,str) and message for message in rejections.values())
    size=vocab*width*2
    initial=report['initial_memory'];assert initial['model_open'] and initial['embedding_table_bytes']==size
    assert initial['embedding_mode']=='device_bf16' and initial['host_embedding_mapped_tensor_bytes']==0
    for mode in ('device','host'):
        memory=report[mode+'_memory'];assert memory['model_open']
        assert memory['embedding_table_bytes']==(0 if mode=='host' else size)
        assert memory['host_embedding_mapped_tensor_bytes']==(size if mode=='host' else 0)
        assert memory['embedding_mode']==('host_mapped_bf16' if mode=='host' else 'device_bf16')
        assert memory['head_precision']=='modelopt_nvfp4_w4a4' and memory['untied_head_bytes']==initial['untied_head_bytes']>0
        assert report[mode+'_embedding_metadata']['shape']==[vocab,width]
    assert len(report['head_predictions'])==2 and report['head_predictions'][0]==report['head_predictions'][1]
    assert len(report['head_predictions'][0])==1 and 0<=report['head_predictions'][0][0]<vocab
    for key in ('restored_memory','reopened_memory'):
        memory=report[key];assert memory['model_open'] and memory['embedding_table_bytes']==size
        assert memory['embedding_mode']=='device_bf16' and memory['host_embedding_mapped_tensor_bytes']==0
    disabled=report['native_head_disabled_memory'];assert disabled['head_precision']=='modelopt_nvfp4_w4a16'
    assert disabled['embedding_table_bytes']==size and disabled['model_open']
    closed=report['closed_memory'];assert not closed['model_open']
    assert closed['embedding_table_bytes']==closed['host_embedding_mapped_tensor_bytes']==closed['untied_head_bytes']==0
    return {'status':'host_embedding_row_capture_replay_passed','capture_report_sha256':sha((captures/'report.json').read_bytes()),
        'fixture_sha256':sha(raw_fixture),'library_sha256':library,'raw_arrays_verified':len(wanted),
        'rejections_verified':len(rejections),'selected_checkpoint_rows':len(rows),
        'scope':'Selected raw embedding rows/layout/padded tails and retained lifecycle observations only. Build/source provenance, allocation-failure injection, sanitizers, full-model state and speed require separate evidence.',
        'performance_admitted':False}

if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit(__doc__)
    print(json.dumps(verify(*sys.argv[1:]),indent=2))
