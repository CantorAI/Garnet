"""Bounded actual checkpoint slice and independent double-accumulation oracle."""
import argparse,hashlib,json,math,struct
from pathlib import Path
PREFIX='model.language_model.layers.11.self_attn.o_proj'
def tensor(root,index,name):
 file=index['weight_map'][name]
 if Path(file).name!=file:raise ValueError('Unsafe shard name')
 with (root/file).open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0]
  if not 2<=length<=4*1024*1024:raise ValueError('Unbounded header')
  header=json.loads(f.read(length));entry=header[name]
 return root/file,8+length,entry

def read(path,base,entry,size):
 begin,end=entry['data_offsets']
 if size>end-begin:raise ValueError('Tensor slice exceeds span')
 with path.open('rb') as f:f.seek(base+begin);data=f.read(size)
 if len(data)!=size:raise ValueError('Truncated tensor')
 return data

def bf16(word):return struct.unpack('<f',struct.pack('<I',word<<16))[0]
def round_bf16(value):
 bits=struct.unpack('<I',struct.pack('<f',value))[0]
 return bf16(((bits+0x7fff+((bits>>16)&1))>>16)&0xffff)

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise ValueError('Preserve existing actual fixture')
 index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text());m,n,k=16,128,6144
 path,base,e=tensor(a.checkpoint,index,PREFIX+'.weight_packed')
 if e['dtype']!='I32' or e['shape']!=[5120,k//8]:raise ValueError('Wrong packed projection')
 packed=read(path,base,e,n*k//2)
 sp,sbase,se=tensor(a.checkpoint,index,PREFIX+'.weight_scale')
 if se['dtype']!='BF16' or se['shape']!=[5120,k//128]:raise ValueError('Wrong scale projection')
 scale_bytes=read(sp,sbase,se,n*k//128*2)
 scales=[bf16(v[0]) for v in struct.iter_unpack('<H',scale_bytes)]
 if not all(math.isfinite(v) and v>0 for v in scales):raise ValueError('Invalid scale')
 hp,hbase,he=tensor(a.checkpoint,index,PREFIX+'.weight_shape')
 if he['dtype']!='I64' or struct.unpack('<qq',read(hp,hbase,he,16))!=(5120,k):raise ValueError('Stored logical shape differs')
 values=[((i*5+i//k)%7-3)/2 for i in range(m*k)]
 inputs=b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v))[0]>>16) for v in values)
 expected=[]
 for row in range(m):
  for col in range(n):
   total=0.0
   for j in range(k):
    i=col*k+j;code=((packed[i//2]>>((i%2)*4))&15)-8
    total+=values[row*k+j]*code*scales[col*(k//128)+j//128]
   expected.append(round_bf16(total))
 files={'packed.bin':packed,'scales.bf16':scale_bytes,'input.bf16':inputs,'expected.f32':struct.pack('<'+'f'*len(expected),*expected)}
 for name,data in files.items():(out/name).write_bytes(data)
 report={'checkpoint_revision':'4e2c2d8f592d5641beefa6a75357f43cf3756392','prefix':PREFIX,'logical_projection':[5120,k],'slice_rows':[0,n],'m':m,'n':n,'k':k,'source_shard':str(path),'source_index_sha256':hashlib.sha256((a.checkpoint/'model.safetensors.index.json').read_bytes()).hexdigest(),
  'oracle':'Python double sum of decoded actual offset-binary codes times BF16 group scales and exact dyadic BF16 inputs, rounded to BF16',
  'tolerance_before_gpu':'Each output absolute difference <=max(1e-6,abs(expected)*0.0078125); all finite required',
  'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},'scope':'First128 rows of one actual pinned INT4 projection, not full-layer/full-model quality or performance'}
 (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
