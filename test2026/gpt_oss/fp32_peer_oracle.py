"""Independent integer RN binary32 oracle for every private FP32 transport byte.

No GPU timing claim. The fixture corpus uses finite normal inputs and signed
zeros; cancellation outputs remain normal or zero. This is not arbitrary NaN,
subnormal-FTZ, TensorRT, or pretrained qualification.
"""
import argparse,hashlib,json,math,random,struct
from pathlib import Path

def add_rn(a,b):
    def parts(x):
        sign=-1 if x>>31 else 1;e=(x>>23)&255;m=x&0x7fffff
        if e==255:raise ValueError('Finite oracle only')
        return sign,m|(0x800000 if e else 0),e-150 if e else -149
    sa,ma,ea=parts(a);sb,mb,eb=parts(b);base=min(ea,eb)
    n=sa*(ma<<(ea-base))+sb*(mb<<(eb-base))
    if not n:return 0x80000000 if a==b==0x80000000 else 0
    sign=0x80000000 if n<0 else 0;n=abs(n);high=n.bit_length()-1
    exponent=high+base+127
    shift=(high-23) if exponent>0 else (-149-base)
    if shift>0:
        q,r=divmod(n,1<<shift);half=1<<(shift-1);q+=int(r>half or (r==half and q&1))
    else:q=n<<(-shift)
    if exponent<=0:
        assert q<=0x800000
        return sign|q
    if q==0x1000000:q>>=1;exponent+=1
    if exponent>=255:return sign|0x7f800000
    return sign|(exponent<<23)|(q&0x7fffff)

def input_bits(i,family,offset):
    j=(i+offset)%4096;h=(j*2654435761+family*2246822519)&0xffffffff
    a=(h&0x80000000)|((87+(j//32+family*19)%80)<<23)|(h&0x7fffff)
    if family==0:b=a
    elif family==1:b=a^0x80000000
    elif family==2:b=a+1
    elif family==3:b=(a&0x80000000)|((((a>>23)&255)-24)<<23)|((h>>7)&0x7fffff)
    elif family==4:b=((h&0x80000000)^0x80000000)|((87+(j//16+23)%80)<<23)|((h>>3)&0x7fffff)
    else:return (j&1)<<31,(j&2)<<30
    return a,b

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def selfcheck():
    cases=[(0,0,0),(0x80000000,0x80000000,0x80000000),(0,0x80000000,0),
           (0x3f800000,0xbf800000,0),(0x3f800000,0x33800000,0x3f800000),
           (0x3f800001,0x33800000,0x3f800002),(0xbf800001,0xb3800000,0xbf800002),
           (0x007fffff,1,0x00800000),(1,1,2),(0x7f7fffff,0x7f7fffff,0x7f800000)]
    for a,b,wanted in cases:assert add_rn(a,b)==wanted
    rng=random.Random(75075)
    comparisons=0
    for _ in range(10000):
        a=rng.getrandbits(32);b=rng.getrandbits(32)
        if ((a>>23)&255)==255 or ((b>>23)&255)==255:continue
        comparisons+=1
        x=struct.unpack('<f',struct.pack('<I',a))[0];y=struct.unpack('<f',struct.pack('<I',b))[0]
        try:wanted=struct.unpack('<I',struct.pack('<f',x+y))[0]
        except OverflowError:wanted=(0x80000000 if x+y<0 else 0)|0x7f800000
        assert add_rn(a,b)==wanted,(hex(a),hex(b))
    return dict(boundary_cases=len(cases),random_draws=10000,independent_finite_host_comparisons=comparisons)

def audit(folder,rows,arm,mode):
    count=rows*2880;capacity=(256 if rows<=256 else 512)*2880
    outputs=[];inputs=[];expected_files=set()
    for family in range(6):
        for graph in range(2):
            stem=f'family{family}-'+('graph' if graph else 'eager');offset=131 if graph else 0
            outputs.append((stem,family,offset))
    outputs.extend([('life-reacquire-eager',2,513),('life-reacquire-graph',2,514)])
    if arm=='peer':outputs.extend([('life-reset',4,257),('life-external',3,1021)])
    if mode=='cost':outputs.extend((f'cost-trial{i}',4,521) for i in range(9))
    checked=[]
    for stem,family,offset in outputs:
        pairs=[input_bits(i,family,offset) for i in range(4096)]
        for rank in range(2):
            name=f'{stem}-input{rank}.f32';p=folder/name;expected_files.add(name)
            wanted_inputs=b''.join(struct.pack('<I',x[rank]) for x in pairs)*(capacity//4096)
            assert p.read_bytes()==wanted_inputs
            inputs.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
        unit=b''.join(struct.pack('<I',add_rn(*input_bits(i,family,offset))) for i in range(4096))
        wanted=unit*(count//4096)+unit[:(count%4096)*4]+struct.pack('<I',0xc2f68000)*(capacity-count)
        assert len(wanted)==capacity*4
        for rank in range(2):
            name=f'{stem}-rank{rank}.f32';p=folder/name;expected_files.add(name)
            assert p.read_bytes()==wanted,('FP32 oracle mismatch',name)
            checked.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p),values=capacity,active=count))
    assert {p.name for p in folder.glob('*.f32')}==expected_files
    if mode=='cost':
        cost=json.loads((folder/'cost.json').read_text())
        assert cost['protocol']=='exact-fp32-copy-sum-native-v1' and cost['arm']==arm and cost['rows']==rows and cost['capacity']==capacity
        assert cost['graph_operations']==72 and cost['replays']==3 and cost['warmups']==3 and cost['original_blocks']==32
        assert len(cost['microseconds'])==9 and all(math.isfinite(x) and x>0 for x in cost['microseconds'])
    return dict(inputs=inputs,outputs=checked,active_values_verified=count*len(checked),tail_values_verified=(capacity-count)*len(checked),
                scope='All saved finite normal/signed-zero native corpus values and inactive tails. No timing isolation, memory, arbitrary NaN/FTZ, TensorRT or pretrained proof.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path,nargs='?');p.add_argument('--rows',type=int,choices=(224,512));p.add_argument('--arm',choices=('original','peer'));p.add_argument('--mode',choices=('parity','cost'));p.add_argument('--output',type=Path)
    a=p.parse_args();checks=selfcheck()
    if a.folder is None:print('INTEGER_FP32_ORACLE_CPU_SELF_CHECK',json.dumps(checks))
    else:
        if not all((a.rows,a.arm,a.mode,a.output)):p.error('Complete audit arguments required')
        assert not a.output.exists()
        result=dict(passed=True,selfcheck=checks,**audit(a.folder,a.rows,a.arm,a.mode))
        a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print('FP32_FULL_NATIVE_ORACLE',sha(a.output))
