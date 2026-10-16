"""CPU-only exhaustive BF16 -> scaled FP16 -> BF16 bit-roundtrip survey.

Arguments: investigation root, new evidence folder. Does not run inference,
change any checkpoint, or establish MPS support/performance.
"""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
metadata=dict(scope=__doc__,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,rows_per_window=1024)
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');reports=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 checkpoint=root/'models'/model/'model.safetensors';matrices=[]
 with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);data_start=8+length
  for name,info in header.items():
   if name=='__metadata__' or info['dtype']!='BF16' or len(info['shape'])!=2:continue
   rows,columns=info['shape'];start,end=info['data_offsets'];assert end-start==rows*columns*2
   bad_values=bad_rows=0;first_failures=[];min_scale=1000;max_scale=-1000
   for begin in range(0,rows,1024):
    count=min(1024,rows-begin)
    bits=np.frombuffer(mapping,dtype='<u2',count=count*columns,offset=data_start+start+begin*columns*2).copy().reshape(count,columns)
    raw=bits.astype(np.uint32)<<16;values=raw.view(np.float32)
    maximum=np.max(np.abs(values),axis=1);assert np.all(np.isfinite(maximum))
    _,exponents=np.frexp(maximum);scale=np.where(maximum==0,0,16-exponents).astype(np.int32)
    min_scale=min(min_scale,int(scale.min()));max_scale=max(max_scale,int(scale.max()))
    with np.errstate(under='ignore'):
     encoded=np.ldexp(values,scale[:,None]).astype(np.float16)
     recovered=np.ldexp(encoded.astype(np.float32),-scale[:,None])
    wrong=recovered.view(np.uint32)!=raw;bad_values+=int(wrong.sum());bad_rows+=int(np.any(wrong,axis=1).sum())
    for row,col in np.argwhere(wrong)[:max(0,8-len(first_failures))]:
     first_failures.append(dict(row=begin+int(row),column=int(col),original_bits=int(raw[row,col]),recovered_bits=int(recovered.view(np.uint32)[row,col]),scale=int(scale[row])))
   matrix=dict(name=name,shape=info['shape'],values=rows*columns,failed_values=bad_values,failed_rows=bad_rows,all_bits_exact=bad_values==0,
      row_scale_range=[min_scale,max_scale],first_failures=first_failures,checkpoint_bytes=end-start)
   matrices.append(matrix)
  del bits,raw,values,maximum,encoded,recovered,wrong
 report=dict(model=model,checkpoint_size=checkpoint.stat().st_size,matrices=matrices,
   scope='All rank-two BF16 checkpoint tensors, including embeddings/vision; not only runtime projections',
   values=sum(v['values'] for v in matrices),failed_values=sum(v['failed_values'] for v in matrices),failed_rows=sum(v['failed_rows'] for v in matrices),all_bits_exact=all(v['all_bits_exact'] for v in matrices))
 reports.append(report);(output/(model+'.json')).write_text(json.dumps(report,indent=2)+'\n')
 print('Exact-half CPU survey',model,'matrices',len(matrices),'values',report['values'],'failed_values',report['failed_values'],'failed_rows',report['failed_rows'],flush=True)
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,models=reports),indent=2)+'\n')
