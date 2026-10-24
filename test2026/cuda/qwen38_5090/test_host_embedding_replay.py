"""Synthetic replay admission negatives. These are not real native/GPU captures."""
import json,struct,tempfile,unittest
from pathlib import Path
from host_embedding_fixture import prepare,NAME
from verify_host_embedding_rows import verify,sha

class ReplayTests(unittest.TestCase):
    def fixture(self,base):
        checkpoint=base/'checkpoint';checkpoint.mkdir()
        header=json.dumps({NAME:{'dtype':'BF16','shape':[4,2],'data_offsets':[0,16]}}).encode()
        (checkpoint/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<8H',*[0x3f80,0x4000,0x4040,0x4080,0x40a0,0x40c0,0x40e0,0x4100]))
        fixtures=base/'fixture';f=prepare(checkpoint,fixtures);captures=base/'synthetic-captures';captures.mkdir()
        ids=f['ids'];rows={int(k):(fixtures/v['f32']).read_bytes() for k,v in f['rows'].items()}
        gpu={'model_open':True,'embedding_table_bytes':16,'host_embedding_mapped_tensor_bytes':0,'embedding_mode':'device_bf16','untied_head_bytes':8,'head_precision':'modelopt_nvfp4_w4a4'}
        host=dict(gpu,embedding_table_bytes=0,host_embedding_mapped_tensor_bytes=16,embedding_mode='host_mapped_bf16')
        report={'status':'embedding_gate_passed_needs_independent_replay','library_sha256':'a'*64,'shape':[4,2],
            'fixture_sha256':sha((fixtures/'fixture.json').read_bytes()),'cases':[],'rejections':[],
            'initial_memory':dict(gpu,head_precision='modelopt_nvfp4_w4a16'),'device_memory':gpu,'host_memory':host,
            'device_embedding_metadata':{'shape':[4,2]},'host_embedding_metadata':{'shape':[4,2]},
            'head_predictions':[[1],[1]],'restored_memory':gpu,'reopened_memory':gpu,
            'native_head_disabled_memory':dict(gpu,head_precision='modelopt_nvfp4_w4a16'),
            'closed_memory':dict(gpu,model_open=False,embedding_table_bytes=0,untied_head_bytes=0)}
        def add(name,raw):
            (captures/(name+'.f32')).write_bytes(raw);report['cases'].append({'name':name,'sha256':sha(raw),'bytes':len(raw),'exact':True})
        for mode in ('device','host'):
            add(mode+'-vector',b''.join(rows[t] for t in ids))
            for batch in (1,2,4,8):
                for time in (1,4,16,128):
                    for count in ((time,) if time==1 else (time,time-1)):
                        for row in range(batch):
                            raw=bytearray(time*8)
                            for t in range(count):raw[t*8:(t+1)*8]=rows[ids[(t*batch+row)%16]]
                            add(mode+f'-b{batch}-t{time}-n{count}-r{row}',bytes(raw))
            for suffix in ['invalid-before']+['invalid-after-'+n for n in ('negative','high','wrong-batch','empty','over-time')]:add(mode+'-'+suffix,rows[ids[1]]*4)
            for name in ('negative-vector-id','high-vector-id','empty-vector','shape'):report['rejections'].append({'name':mode+'-'+name,'error':'synthetic expected rejection'})
            for name in ('negative','high','wrong-batch','empty','over-time'):report['rejections'].append({'name':mode+'-chunk-'+name,'error':'synthetic expected rejection'})
        for name in ('restored-device-vector','reopened-default-vector'):add(name,b''.join(rows[t] for t in ids))
        for name in ('host_requires_native_head','cannot_disable_native_head_in_host_mode','closed-vector','closed-host-enable'):report['rejections'].append({'name':name,'error':'synthetic expected rejection'})
        (captures/'report.json').write_text(json.dumps(report));return captures,fixtures,report
    def test_synthetic_complete_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            captures,fixtures,_=self.fixture(Path(temp));result=verify(captures,fixtures,'a'*64)
            self.assertEqual(result['raw_arrays_verified'],226);self.assertFalse(result['performance_admitted'])
    def test_corruption_rejected_even_when_producer_hash_is_updated(self):
        for name in ('host-b8-t128-n127-r7','host-invalid-after-high','reopened-default-vector'):
            with tempfile.TemporaryDirectory() as temp:
                captures,fixtures,report=self.fixture(Path(temp));p=captures/(name+'.f32');raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
                next(e for e in report['cases'] if e['name']==name)['sha256']=sha(raw)
                (captures/'report.json').write_text(json.dumps(report))
                with self.assertRaises(AssertionError):verify(captures,fixtures,'a'*64)
    def test_missing_capture_and_missing_guard_rejected(self):
        for kind in ('capture','guard'):
            with tempfile.TemporaryDirectory() as temp:
                captures,fixtures,report=self.fixture(Path(temp))
                report['cases' if kind=='capture' else 'rejections'].pop()
                (captures/'report.json').write_text(json.dumps(report))
                with self.assertRaises(AssertionError):verify(captures,fixtures,'a'*64)

if __name__=='__main__':unittest.main()
