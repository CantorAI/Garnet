"""CPU-only exact BF16 exponent statistics for a possible 11-bit-plus-escape decoder.

No GPU use, inference, packet implementation, speed or memory-RSS claim.
All original first-layer fixtures are retained; decode admission matches the
current large-body geometry. Byte budgets are hypothetical, not GPU traffic.
"""
from pathlib import Path
import hashlib,json,mmap,os,struct,sys,time
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();code=1
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def align16(v):return (v+15)//16*16
try:
    paths=[Path(__file__),root/'source/src/backends/metal/coalesced_body_block_shader.h',root/'source/src/backends/metal/metal_context.mm']
    identities={p.as_posix():digest(p) for p in paths}
    for p in paths:
        q=out/'source-snapshot'/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    (out/'pid.json').write_text(json.dumps(dict(pid=os.getpid())))
    metadata=dict(source_sha256=identities,command=sys.argv,python=sys.version,numpy=np.__version__,scope=__doc__,full_goal_met=False)
    models=[]
    for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
        assets=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets')
        fixtures=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors'
        assert digest(checkpoint)==fixtures['checkpoint_sha256'];cases=[]
        with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
            length=struct.unpack_from('<Q',mapped)[0];header=json.loads(mapped[8:8+length]);base=8+length
            for fixture in fixtures['cases']:
                n,k=fixture['rows'],fixture['inner'];name=fixture['tensor'];item=header[name]
                assert item['dtype']=='BF16' and item['shape']==[n,k]
                lo,hi=item['data_offsets'];raw=(assets/fixture['name']/'raw.bin').read_bytes()
                assert hashlib.sha256(raw).hexdigest()==fixture['sha256']['raw.bin']==hashlib.sha256(mapped[base+lo:base+hi]).hexdigest()
                special=name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight')
                eligible=not special and 256<n<=8192 and k%128==0 and n*k>=1048576
                record=dict(tensor=name,rows=n,inner=k,raw_sha256=fixture['sha256']['raw.bin'],eligible=eligible)
                if not eligible:cases.append(record);continue
                words=np.frombuffer(raw,dtype='<u2').reshape(n,k)
                padded=(n+3)//4*4;packed=np.zeros((padded,k),dtype='<u2');packed[:n]=words
                blocks=packed.reshape(padded//4,4,k//128,128).transpose(0,2,3,1).reshape(-1,512)
                span_hist=np.zeros(256,dtype=np.int64);escape_hist=np.zeros(513,dtype=np.int64)
                compact=roundtrips=ideal=ranked=0
                for at in range(0,len(blocks),1024):
                    original=blocks[at:at+1024];exponents=(original>>7)&255
                    spans=exponents.max(axis=1)-exponents.min(axis=1);span_hist+=np.bincount(spans,minlength=256)
                    use=spans<=15;compact+=int(np.count_nonzero(use));selected=original[use];e=exponents[use]
                    if not len(selected):continue
                    offsets=np.arange(len(selected),dtype=np.int32)[:,None]*256
                    hist=np.bincount((e.astype(np.int32)+offsets).ravel(),minlength=len(selected)*256).reshape(-1,256)
                    sums=np.pad(np.cumsum(hist,axis=1),((0,0),(1,0)))
                    window=sums[:,7:]-sums[:,:-7];best=np.argmax(window,axis=1).astype(np.uint16)[:,None]
                    inside=(e>=best)&(e<best+7);tags=np.where(inside,e-best,7).astype(np.uint16)
                    escapes=np.count_nonzero(~inside,axis=1);escape_hist+=np.bincount(escapes,minlength=513)
                    # Escape table uses the original exponent byte; never rounds,
                    # drops tiny values or substitutes an approximate value.
                    restored_e=np.where(tags==7,e,tags+best)
                    sign_mant=((selected&127)|((selected>>8)&128)).astype(np.uint16)
                    restored=(sign_mant&127)|((sign_mant&128)<<8)|(restored_e<<7)
                    assert np.array_equal(restored,selected);roundtrips+=selected.size
                    # Four-byte per-block metadata in all estimates. Ideal
                    # random escape rank assumes an unimplemented free lookup.
                    # Ranked adds one byte per SIMD lane (32) plus 16-byte
                    # alignment and admits only the lossless smaller packet.
                    ideal+=int(np.minimum(772,708+escapes).sum())
                    ranked+=int(np.minimum(772,align16(736+escapes)+4).sum())
                assert int(span_hist.sum())==len(blocks) and int(escape_hist.sum())==compact and roundtrips==compact*512
                record.update(blocks=len(blocks),compact_blocks=compact,raw_fallback_blocks=len(blocks)-compact,
                    exponent_span_histogram=span_hist.tolist(),escape_count_histogram=escape_hist.tolist(),
                    exact_expression_roundtrip_values=roundtrips,retained_compact_payload_metadata_bytes=compact*772,
                    hypothetical_ideal_packet_bytes=ideal,hypothetical_ranked_aligned_packet_bytes=ranked,
                    logical_raw_fallback_read_bytes=(len(blocks)-compact)*1024,
                    caveat='Estimates omit rank-computation GPU work and pointer/index metadata implementation; neither is GPU traffic or measured speed.')
                cases.append(record)
        assert len(cases)==(15 if model=='Qwen3.5-0.8B' else 7)
        totals={key:sum(c.get(key,0) for c in cases) for key in ('blocks','compact_blocks','raw_fallback_blocks','exact_expression_roundtrip_values',
            'retained_compact_payload_metadata_bytes','hypothetical_ideal_packet_bytes','hypothetical_ranked_aligned_packet_bytes','logical_raw_fallback_read_bytes')}
        totals['eligible_matrices']=sum(c['eligible'] for c in cases)
        current=totals['retained_compact_payload_metadata_bytes']+totals['logical_raw_fallback_read_bytes']
        for name in ('ideal','ranked_aligned'):
            totals[name+'_packet_fraction']=(totals['hypothetical_'+name+'_packet_bytes']+totals['logical_raw_fallback_read_bytes'])/current
        models.append(dict(model=model,checkpoint_sha256=fixtures['checkpoint_sha256'],cases=cases,totals=totals))
        (out/'report.json').write_text(json.dumps(dict(models=models,scope=__doc__,full_goal_met=False),indent=2));print(json.dumps(dict(model=model,**totals)),flush=True)
    assert all(digest(Path(p))==sha for p,sha in identities.items())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
