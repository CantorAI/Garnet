"""Independent integer dot and BF16 RN oracle for the private finite corpus.

Checks EVERY decoded input/output byte, including unchanged-input readbacks.
No arbitrary/pretrained router, softmax/topK, compiled or serving claim.
"""
import argparse,gzip,hashlib,json,math,struct
from pathlib import Path

def bf_bits(v):
    bits=struct.unpack('<I',struct.pack('<f',v))[0]
    assert bits&0x7f800000!=0x7f800000
    return (bits+0x7fff+((bits>>16)&1))&0xffff0000

def xint(q,k,f):return ((q%17)*23+k*11+f*7)%65-32
def wint(e,k,f):return ((e%37)*19+k*7+f*13)%65-32
def biasint(e,f):return (e*5+f*3)%33-16

def contract():
    for value,expected in [(0,0),(1,0x3f800000),(-1,0xbf800000),(1+1/256,0x3f800000),(1+3/256,0x3f820000),(-1-1/256,0xbf800000),(-1-3/256,0xbf820000)]:assert bf_bits(value)==expected
    for h in [33,2880]:assert h*32*32+16*128<1<<24
    print('INTEGER_BF16_RN_CONTRACT_PASS seven_boundaries exact_common_unit_bound')

def audit(folder,output):
    assert not output.exists()
    output.write_text(json.dumps(dict(passed=False,stage='integer-audit-started'))+'\n',encoding='utf-8',newline='\n')
    meta=json.loads((folder/'dataset.json').read_text())
    rows,h,e=(meta[k] for k in ['rows','hidden','experts'])
    assert (rows,h,e) in [(19,33,37),(512,2880,128),(4096,2880,128)]
    assert meta['arms']==[0,4,2,1] and meta['mode'] in ['parity','cost']
    families=1 if meta['mode']=='cost' else 3;assert meta['families']==families
    records={};expected_files={'dataset.json'};score_values=0;input_values=0
    def check_file(name,expected):
        nonlocal score_values,input_values
        assert name not in records;expected_files.add(name)
        data=gzip.decompress((folder/name).read_bytes())
        assert data==expected,(name,'ALL decoded bytes must match independent integer corpus')
        records[name]=dict(compressed_sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest(),decoded_sha256=hashlib.sha256(data).hexdigest(),decoded_bytes=len(data))
        if '-arm' in name:score_values+=len(data)//4
        else:input_values+=len(data)//4
    last_inputs=None;last_expected=None
    for family in range(families):
        # Factory periodicity is verified over every full decoded input byte.
        pattern=b''.join(struct.pack('<f',xint(q,k,family)/128) for q in range(17) for k in range(h))
        x=(pattern*((rows+16)//17))[:rows*h*4]
        pattern=b''.join(struct.pack('<f',wint(ex,k,family)/4096) for ex in range(37) for k in range(h))
        w=(pattern*((e+36)//37))[:e*h*4]
        b=b''.join(struct.pack('<f',biasint(ex,family)/4096) for ex in range(e))
        dot=[[sum(xint(q,k,family)*wint(ex,k,family) for k in range(h)) for ex in range(37)] for q in range(17)]
        pattern=b''.join(struct.pack('<I',bf_bits((dot[q][ex%37]+biasint(ex,family)*128)/524288)) for q in range(17) for ex in range(e))
        expected=(pattern*((rows+16)//17))[:rows*e*4]
        for label,values in [('x',x),('w',w),('b',b)]:
            check_file(f'family{family}-{label}.f32.gz',values)
            check_file(f'family{family}-{label}.after.f32.gz',values)
        for arm in [0,4,2,1]:
            for mode in ['eager','graph']:check_file(f'family{family}-arm{arm}-{mode}.f32.gz',expected)
        last_inputs=(x,w,b);last_expected=expected
    cost=None
    if meta['mode']=='cost':
        expected_files.add('cost.json');cost=json.loads((folder/'cost.json').read_text())
        assert cost['warmups']==5 and cost['trials']==9 and cost['flush_bytes']>=256<<20
        assert [row['arm'] for row in cost['arms']]==[0,4,2,1]
        for row in cost['arms']:
            assert len(row['samples_ms'])==9 and all(math.isfinite(v) and v>0 for v in row['samples_ms'])
            for trial in range(9):check_file(f"cost-arm{row['arm']}-trial{trial}.f32.gz",last_expected)
        for label,values in zip(['x','w','b'],last_inputs):check_file(f'cost-{label}.after.f32.gz',values)
    assert {p.name for p in folder.iterdir()}==expected_files
    result=dict(protocol='private-tensor-router-integer-v1',passed=True,shape=meta,files=records,score_values_verified=score_values,input_values_verified=input_values,cost=cost,
                scope='Finite bounded integer-dot/BF16 RN corpus only. All decoded inputs, readbacks and scalar logits checked. No topK/softmax/pretrained/provider/serving qualification.')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PRIVATE_ROUTER_INTEGER_ALL_BYTES_PASS',hashlib.sha256(output.read_bytes()).hexdigest())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path,nargs='?');parser.add_argument('--output',type=Path)
    args=parser.parse_args();contract()
    if args.folder:
        if args.output is None:parser.error('--output required')
        audit(args.folder,args.output)
