"""CPU-only generated indexing/ownership proof; no GPU arithmetic or speed evidence."""
from pathlib import Path
import datetime,hashlib,json,re,sys,time
from make_delta_value_tiles import arithmetic_block,band,kernel,shader
source_path=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=source_path.read_text();snap=out/'source-snapshot';snap.mkdir()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[Path(__file__).resolve(),Path(__file__).with_name('make_delta_value_tiles.py').resolve(),source_path]
for p in paths:(snap/p.name).write_bytes(p.read_bytes())
generated=out/'delta-value-tiles.metal';generated.write_bytes(shader(source).encode());profiles=[]
block=arithmetic_block(source);original=kernel(source,16)
prefix=original[original.index('            float q='):original.index('            float4 memory=')]
for width in (16,32,64):
 text=kernel(source,width);assert text.count(prefix)==1
 for i in range(width//16):
  # Inverse identifier/address substitution must recover the exact original FP32 body.
  value=band(block,i)
  for word in ('memories','memory','change','result'):value=re.sub(r'\b'+word+'_b'+str(i)+r'\b',word,value)
  value=value.replace('(column+'+str(i*16)+')','column');assert value==block and band(block,i) in text
 # Read the actual generated group/tile expressions, rather than assuming width in the enumeration.
 tile=int(re.search(r'ulong tiles=p.valueDim/(\d+),',text)[1]);column_width=int(re.search(r'column=\(ulong\(id\)%tiles\)\*(\d+)\+simd\*4;',text)[1]);assert tile==column_width==width
 for batch in (1,2):
  for padding in (0,4):
   heads,keydim,valuedim=16,128,128;logical=batch*heads*keydim*valuedim;sstride=heads*keydim*valuedim+padding;nextstride=sstride+padding;ostride=heads*valuedim+padding;visits=bytearray(logical);aligned_vectors=0
   for group in range(batch*heads*(valuedim//tile)):
    head_id=group//(valuedim//tile);head=head_id%heads;row=head_id//heads
    for simd in range(4):
     column=(group%(valuedim//tile))*column_width+simd*4
     for lane in range(32):
      for j in range(4):
       key=lane+32*j
       for band_index in range(width//16):
        first=column+band_index*16;read=row*sstride+head*keydim*valuedim+key*valuedim+first;write=row*nextstride+head*keydim*valuedim+key*valuedim+first
        assert read%4==write%4==0 and first+4<=valuedim and row*sstride<=read and read+4<=row*sstride+heads*keydim*valuedim and row*nextstride<=write and write+4<=row*nextstride+heads*keydim*valuedim;aligned_vectors+=1
        for component in range(4):
         value=first+component;index=((row*heads+head)*keydim+key)*valuedim+value;assert visits[index]==0;visits[index]=1
   assert all(v==1 for v in visits) and aligned_vectors*4==logical
   output_totals={}
   for chunk in (1,16,80):
    count=batch*chunk*heads*valuedim;outputs=bytearray(count)
    for group in range(batch*heads*(valuedim//tile)):
     head_id=group//(valuedim//tile);head=head_id%heads;row=head_id//heads
     for token in range(chunk):
      for simd in range(4):
       column=(group%(valuedim//tile))*column_width+simd*4
       for band_index in range(width//16):
        first=column+band_index*16;address=(row*chunk+token)*ostride+head*valuedim+first;assert address%4==0 and address+4<=(row*chunk+token)*ostride+heads*valuedim
        for component in range(4):
         index=((row*chunk+token)*heads+head)*valuedim+first+component;assert outputs[index]==0;outputs[index]=1
    assert all(v==1 for v in outputs);output_totals[str(chunk)]=count
   profiles.append(dict(width=width,batch=batch,padding_floats=padding,heads=heads,key_dim=keydim,value_dim=valuedim,groups=batch*heads*(valuedim//tile),unique_state_words=logical,aligned_state_vectors=aligned_vectors,output_words_by_chunk=output_totals,source_and_next_padding_untouched=True))
result=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={str(p):digest(p) for p in paths},generated_shader_sha256=digest(generated),original_kernel_sha256=hashlib.sha256(kernel(source,16).encode()).hexdigest(),profiles=profiles,original_qk_norm_prefix_bytes_preserved=True,per_value_original_arithmetic_body_recovered=True,all_state_and_output_words_owned_once=True,gpu_executed=False,numerical_equivalence_proven=False,performance_measured=False,wall_seconds=time.monotonic()-start,scope='CPU generated indexing/ownership and literal arithmetic-body prerequisite only. Wider tiles may change register use/compiler behavior. Actual GPU original-bit, learned state, timing/null/confirmation and native/model tests remain required.',full_goal_met=False)
(out/'report.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(profiles=len(profiles),state_words_checked=sum(p['unique_state_words'] for p in profiles),output_words_checked=sum(sum(p['output_words_by_chunk'].values()) for p in profiles),wall_seconds=result['wall_seconds'],gpu_executed=False,full_goal_met=False)))
