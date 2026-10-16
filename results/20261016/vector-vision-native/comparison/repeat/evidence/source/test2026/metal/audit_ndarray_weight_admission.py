"""CPU-only full-checkpoint exact-half audit, roundtrips and observed baseline-count reconciliation."""
from pathlib import Path
import hashlib,json,mmap,os,struct,sys,time
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();code=1
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def codec(bits,scale):
    mag=bits&32767;nonzero=mag!=0;normal=(mag>>7)&255
    scale=np.asarray(scale,dtype=np.int16);exponent=normal.astype(np.int16)-127+scale
    valid=(~nonzero)|((normal!=0)&(normal!=255)&(exponent>=-24)&(exponent<=15))
    valid&=(scale>=-64)&(scale<=64)
    discarded=np.clip(-exponent-17,0,24).astype(np.uint32)
    valid&=(~(nonzero&(exponent<-17)))|(((128+(mag&127)).astype(np.uint32)&((np.uint32(1)<<discarded)-1))==0)
    good=bool(valid.all());roundtrip=False
    if good:
        encoded=(bits&32768).astype(np.uint32);norm=nonzero&(exponent>=-14)
        encoded[norm]|=(((exponent[norm]+15).astype(np.uint32)<<10)|((mag[norm]&127).astype(np.uint32)<<3))
        sub=nonzero&~norm;shift=exponent.astype(np.int32)+17
        fraction=((128+(mag&127)).astype(np.uint32)<<np.clip(shift,0,24).astype(np.uint32))>>np.clip(-shift,0,24).astype(np.uint32)
        encoded[sub]|=fraction[sub]
        restored=np.ldexp(encoded.astype('<u2').view('<f2').astype('<f4'),-scale)
        roundtrip=bool(np.array_equal(restored.view('<u4'),bits.astype('<u4')<<16));assert roundtrip
    return dict(admitted=good,bitwise_roundtrip=roundtrip,unrepresentable_values=int(np.count_nonzero(~valid)),unrepresentable_rows=int(np.count_nonzero(np.any(~valid,axis=1))))
try:
    source_paths=[Path(__file__),root/'source/src/backends/metal/exact_half_codec.h',root/'source/src/backends/metal/metal_context.mm']
    identities={p.as_posix():digest(p) for p in source_paths}
    snap=out/'source-snapshot';snap.mkdir()
    for p in source_paths:(snap/p.name).write_bytes(p.read_bytes())
    reference=json.loads((root/'logs/ndarray-prefill-repaired-20261015/jobs/metadata.json').read_text())
    candidate=root/'ndarray-native-investigation-20261015'
    native_audit=json.loads((candidate/'ndarray-native-profiles-verification.json').read_text())
    assert native_audit['full_output_requests']==72 and native_audit['archive_sha256']=='7ed9d48e71a3723cf84e62fc95170bbee52cf614d90ef8fd7c93df338dc92c33'
    models=[]
    (out/'pid.json').write_text(json.dumps(dict(pid=os.getpid())))
    (out/'metadata.json').write_text(json.dumps(dict(source_sha256=identities,command=sys.argv,python=sys.version,numpy=np.__version__,
        baseline_archive_sha256=native_audit['archive_sha256'],assumptions='Retained validated exact-half pipeline/allocation available; aligned body K%8==0 and <=16MiB storage cap. CPU eligibility is reconciled with actual baseline counters, not a per-weight GPU instrument.',
        scope=__doc__,full_goal_met=False),indent=2))
    for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
        checkpoint=root/'models'/model/'model.safetensors';sha=digest(checkpoint);assert sha==reference['assets'][model]['checkpoint_sha256']
        cases=[];exports=[];assets=out/'assets'/model;assets.mkdir(parents=True)
        with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
            length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);data_start=8+length
            (out/(model+'-header.json')).write_bytes(mapping[8:8+length])
            for name,item in header.items():
                if not(name.startswith(('model.layers.','model.language_model.layers.')) and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2):continue
                if name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'):continue
                n,k=item['shape'];assert k%8==0;lo,hi=item['data_offsets'];assert hi-lo==n*k*2
                raw=mapping[data_start+lo:data_start+hi];bits=np.frombuffer(raw,dtype='<u2').reshape(n,k);mag=bits&32767
                assert not np.any(((mag>>7)&255)==255)
                largest=int(mag.max());e=(largest>>7)&255;scale=0 if largest==0 else 142-e
                common=codec(bits,scale);common_ok=(largest==0 or e not in (0,255)) and -64<=scale<=64 and common['admitted']
                rowwise=None
                if not common_ok:
                    row_max=mag.max(axis=1);row_e=(row_max>>7)&255;scales=np.where(row_max==0,0,142-row_e.astype(np.int16)).astype(np.int16).reshape(n,1)
                    rowwise=codec(bits,scales)
                cap=len(raw)<=16*1024*1024
                global_ready=bool(cap and common_ok);exact_ready=bool(cap and (common_ok or rowwise['admitted']))
                c=dict(name='matrix-'+str(len(cases)),tensor=name,rows=n,inner=k,raw_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw),
                    global_half_ready=global_ready,exact_half_ready=exact_ready,common_scale=scale,common=common,rowwise=rowwise,
                    full_f32_cache_bytes=n*k*4,storage_cap_admitted=cap)
                cases.append(c)
                if not global_ready:
                    folder=assets/c['name'];folder.mkdir();(folder/'raw.bin').write_bytes(raw);exports.append(c)
        observed=[];ready=sum(c['global_half_ready'] for c in cases)
        for batch in (1,2):
            label=model+'-b'+str(batch);folder=candidate/'logs/ndarray-native-repaired-20261015/jobs'/(label+'-attempt0')
            path=folder/(label+'-control-0/output.log');prefix='matched_frame_result ' if model=='Qwen3.5-0.8B' else 'matched_result '
            rows=[json.loads(line.split(' ',1)[1]) for line in path.read_text().splitlines() if line.startswith(prefix)];assert len(rows)==3
            for r in rows:
                chunks=sum(2<batch*c<=128 or batch*c==160 for c in r['prefill_chunk_plan']);expected=ready*chunks
                actual=r['prefill_profile'].get('global_half_alpha_dispatches',0);assert actual==expected,(model,batch,actual,expected)
                observed.append(dict(batch=batch,request=r['request'],expected_global_half_calls=expected,actual_global_half_calls=actual))
        report=dict(model=model,checkpoint_sha256=sha,cases=cases,matrices=len(cases),global_half_ready=ready,exact_half_ready=sum(c['exact_half_ready'] for c in cases),
            non_common=[c['tensor'] for c in cases if not c['global_half_ready']],no_exact_half=[c['tensor'] for c in cases if not c['exact_half_ready']],
            all_f32_cache_bytes=sum(c['full_f32_cache_bytes'] for c in cases),non_common_f32_cache_bytes=sum(c['full_f32_cache_bytes'] for c in cases if not c['global_half_ready']),
            no_exact_f32_cache_bytes=sum(c['full_f32_cache_bytes'] for c in cases if not c['exact_half_ready']),baseline_counter_checks=observed)
        (assets/'manifest.json').write_text(json.dumps(dict(model=model,checkpoint_sha256=sha,cases=exports),indent=2))
        models.append(report);(out/'report.json').write_text(json.dumps(dict(models=models,full_goal_met=False),indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('cases','baseline_counter_checks')}),flush=True)
    assert all(digest(Path(p))==sha for p,sha in identities.items())
    code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
