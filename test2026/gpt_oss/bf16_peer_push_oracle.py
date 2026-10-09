"""Independent CPU full-matrix oracle for private owned BF16 peer control.

Preparatory tables are not GPU/NCCL qualification. Cost mode is native-only.
"""
import argparse,hashlib,json,math,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('rows',type=int,choices=[224,512,3584,4096]);p.add_argument('mode',choices=['prepare','parity','cost']);p.add_argument('--blocks',type=int,required=True,choices=[32,64,128]);p.add_argument('output',type=Path);p.add_argument("--transport",required=True,choices=["original-pull-grid1","private-push-grid1"]);a=p.parse_args()
assert not a.output.exists()
def f32(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def sum_round(a,b):
    exact=float(a)+float(b)
    try:n=struct.unpack('<I',struct.pack('<f',exact))[0]
    except OverflowError:n=0xff800000 if exact<0 else 0x7f800000
    return ((n+0x7fff+((n>>16)&1))&0xffff0000)&0xffffffff
def finite(bits):return bits^0x80 if (bits&0x7f80)==0x7f80 else bits
assert sum_round(f32(0x3f800000),f32(0x3b800000))==0x3f800000
assert sum_round(f32(0x3f810000),f32(0x3b800000))==0x3f820000
assert sum_round(f32(0x00010000),f32(0x00010000))==0x00020000
assert sum_round(f32(0x80000000),f32(0x80000000))==0x80000000
assert sum_round(f32(0x7f7f0000),f32(0x7f7f0000))==0x7f800000
assert sum_round(f32(0xff7f0000),f32(0xff7f0000))==0xff800000
tables={}
for family in range(6):
    table=bytearray()
    for i in range(65536):
        x=finite(i)
        y=[x,x^0x8000,finite((x+1)&65535),x&0x8000,(x&0x8000)|1,(x&0x8000)|0x3f80][family]
        table.extend(struct.pack('<I',sum_round(f32(x<<16),f32(y<<16))))
    for tag,offset in [('eager',0),('graph',131)]:
        rotated=bytes(table[offset*4:]+table[:offset*4]);tables[f'family{family}-{tag}']=rotated
cost=bytearray()
for i in range(127*91):cost.extend(struct.pack('<I',sum_round((i%127-63)/64,(i%91-45)/32)))
tables['cost']=bytes(cost)
for name,family,offset in [('life-reset',5,257),('life-reacquire-eager',4,513),('life-reacquire-graph',4,514),('life-external-eager',2,1021),('life-external-graph',2,1022)]:
    period=tables[f'family{family}-eager'];tables[name]=period[offset*4:]+period[:offset*4]
count=a.rows*2880;records=[]
def check(path,period):
    assert path.stat().st_size==count*4
    digest=hashlib.sha256();checked=0;phase=0
    with path.open('rb') as f:
        while True:
            chunk=f.read(1<<20)
            if not chunk:break
            wanted=(period*((phase+len(chunk)+len(period)-1)//len(period)))[phase:phase+len(chunk)]
            assert chunk==wanted,'Whole raw matrix mismatch at chunk byte offset '+str(checked)
            digest.update(chunk);checked+=len(chunk);phase=(phase+len(chunk))%len(period)
    assert checked==count*4
    return dict(path=path.name,bytes=checked,values=count,sha256=digest.hexdigest(),every_byte_exact=True)
if a.mode!='prepare':
    wanted=set()
    for family in range(6):
        for tag in ('eager','graph'):
            stem=f'family{family}-{tag}'
            for rank in (0,1):
                path=a.folder/(stem+f'-rank{rank}.bin');wanted.add(path.name);records.append(check(path,tables[stem]))
    for stem in ['life-reset','life-reacquire-eager','life-reacquire-graph','life-external-eager','life-external-graph']:
        for rank in (0,1):
            path=a.folder/(stem+f'-rank{rank}.bin');wanted.add(path.name);records.append(check(path,tables[stem]))
    assert len(records)==34
    if a.mode=='cost':
        meta=json.loads((a.folder/'cost.json').read_text());assert meta['rows']==a.rows and meta['elements']==count and meta['graph_operations']==72 and meta['replays_per_trial']==3 and meta['warmups']==3
        assert meta['protocol']=='PRIVATE_PULL_PUSH_GRID_COMPARISON' and meta['transport']==a.transport and meta['blocks']==a.blocks and len(meta['microseconds_per_owned_pack_peer_unpack'])==9
        assert all(math.isfinite(x) and x>0 for x in meta['microseconds_per_owned_pack_peer_unpack'])
        for trial in range(9):
            for rank in (0,1):
                path=a.folder/f'cost-trial{trial}-rank{rank}.bin';wanted.add(path.name);records.append(check(path,tables['cost']))
    assert {q.name for q in a.folder.glob('*.bin')}==wanted
result=dict(protocol='independent-owned-bf16-peer-full-bit-matrices-v1',blocks=a.blocks,rows=a.rows,elements=count,mode=a.mode,transport=a.transport,
            every_raw_element_recomputed=a.mode!='prepare',records=records,period_sha256={k:hashlib.sha256(v).hexdigest() for k,v in tables.items()},
            scope='Prepare means CPU-only oracle readiness, NO actual GPU qualification. Other modes require external actual app0/source/binary/sanitizer/idle/dispatch evidence. Native cost is not serving speed.')
a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print('CPU_OWNED_BF16_PEER_ORACLE_'+a.mode.upper(),hashlib.sha256(a.output.read_bytes()).hexdigest(),flush=True)
