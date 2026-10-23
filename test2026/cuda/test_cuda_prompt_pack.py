"""Exact byte oracle for ragged text/visual prompt packing and view lifetime."""
import json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet,tensor
io=garnet.cuda_tensor_io();checks=[]
patterns=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
          0x3f800000,0xbf800000,0x3eaaaaab,0x7f7fffff,0xff7fffff]

def make(count,seed):
    blob=struct.pack('<'+str(count)+'I',*[patterns[(i+seed)%len(patterns)] for i in range(count)])
    return list(struct.unpack('<'+str(count)+'f',blob)),blob

def reject(name,action):
    failed=False
    try:action()
    except BaseException:failed=True
    assert failed,name
    checks.append(name)

try:
    for b in (1,2,4,8):
        for h in (7,16):
            emb=[];visual=[];starts=[];expected_rows=[];lengths=[]
            for row in range(b):
                n=3+row%3;lengths.append(n)
                values,blob=make(n*h,row);emb.append(io.from_host(values,[n,h]))
                start=(-1 if row%3==1 else (0 if row%3==0 else n-2))
                starts.append(start)
                if start>=0:
                    values,vblob=make(2*h,row+4);visual.append(io.from_host(values,[1,2,h]))
                    blob=blob[:start*h*4]+vblob+blob[(start+2)*h*4:]
                else:visual.append(None)
                expected_rows.append(blob)
            source_before=[io.finite_f32_bytes(x) for x in emb]
            visual_before=[io.finite_f32_bytes(x) if x is not None else None for x in visual]
            packed=list(io.pack_prompt(emb,visual,starts));assert len(packed)==max(lengths)
            zero=io.zeros([1,h]);expected=[]
            for t,out in enumerate(packed):
                want=b''.join(expected_rows[row][t*h*4:(t+1)*h*4] if t<lengths[row] else bytes(h*4) for row in range(b))
                expected.append(want);assert io.finite_f32_bytes(out)==want
                legacy=[]
                for row in range(b):
                    if t>=lengths[row]:legacy.append(zero)
                    elif starts[row]>=0 and starts[row]<=t<starts[row]+2:legacy.append(io.visual_token(visual[row],t-starts[row]))
                    else:legacy.append(io.row(emb[row],t))
                assert io.finite_f32_bytes(io.stack_rows(legacy))==want
            assert [io.finite_f32_bytes(x) for x in emb]==source_before
            assert [io.finite_f32_bytes(x) if x is not None else None for x in visual]==visual_before
            # All output views share storage; drop sources and other views.
            survivor=packed[1];want=expected[1];del packed,emb,visual,legacy,out
            assert io.finite_f32_bytes(survivor)==want
            chunk=io.zeros([b,3,h]);io.write_chunk_token(survivor,chunk,1)
            chunk_expected=b''.join(bytes(h*4)+want[row*h*4:(row+1)*h*4]+bytes(h*4) for row in range(b))
            assert io.finite_f32_bytes(chunk)==chunk_expected
            checks.append(f'b{b}_h{h}_ragged_visual_legacy_exact_source_preserved_lifetime_downstream')
    a=io.zeros([3,7]);v=io.zeros([1,2,7])
    alias_values,alias_bytes=make(21,3);alias=io.from_host(alias_values,[3,7])
    alias_visual=io.storage_view(alias,[1,2,7])
    aliased=list(io.pack_prompt([alias,alias],[alias_visual,alias_visual],[1,1]))
    alias_expected=[alias_bytes[:28],alias_bytes[:28],alias_bytes[28:56]]
    for out,want_row in zip(aliased,alias_expected):assert io.finite_f32_bytes(out)==want_row*2
    assert io.finite_f32_bytes(alias)==alias_bytes
    checks.append('shared_read_storage_aliases_deduplicated_source_preserved')
    reject('empty_batch',lambda:io.pack_prompt([],[],[]))
    reject('invalid_batch3',lambda:io.pack_prompt([a]*3,[None]*3,[-1]*3))
    reject('mismatched_lists',lambda:io.pack_prompt([a],[],[-1]))
    reject('embedding_widths_differ',lambda:io.pack_prompt([a,io.zeros([3,8])],[None,None],[-1,-1]))
    reject('noninteger_start',lambda:io.pack_prompt([a],[v],[0.5]))
    reject('missing_visual_with_start',lambda:io.pack_prompt([a],[None],[0]))
    reject('visual_with_negative_start',lambda:io.pack_prompt([a],[v],[-1]))
    reject('visual_out_of_range',lambda:io.pack_prompt([a],[v],[2]))
    reject('width_mismatch',lambda:io.pack_prompt([a],[io.zeros([1,2,8])],[0]))
    reject('embedding_rank',lambda:io.pack_prompt([v],[None],[-1]))
    reject('visual_rank',lambda:io.pack_prompt([a],[a],[0]))
    reject('visual_batch',lambda:io.pack_prompt([a],[io.zeros([2,1,7])],[0]))
    reject('cpu_embedding',lambda:io.pack_prompt([garnet.tensor_to_cpu(a)],[None],[-1]))
    reject('cpu_visual',lambda:io.pack_prompt([a],[garnet.tensor_to_cpu(v)],[0]))
    reject('symbolic_embedding',lambda:io.pack_prompt([tensor.input('unmaterialized_prompt',shape=[3,7],dtype=tensor.float32)],[None],[-1]))
    reject('too_many_tokens',lambda:io.pack_prompt([io.zeros([8193,1])],[None],[-1]))
    reject('too_wide',lambda:io.pack_prompt([io.zeros([1,16385])],[None],[-1]))
    # A small source repeated across rows makes total output exceed the cap.
    big=io.zeros([8192,2048])
    reject('output_over_256MiB',lambda:io.pack_prompt([big]*8,[None]*8,[-1]*8))
    io.close();assert io.finite_f32_bytes(survivor)==want
    checks.append('views_outlive_io_close')
    Path(sys.argv[1]).write_text(json.dumps({'all_passed':True,'checks':checks,
        'scope':'Exact finite FP32 bytes versus host and legacy copy oracles; ragged visual padding, source preservation, invalid inputs and ownership. No performance or pretrained quality claim.'},indent=2))
    print('CUDA_PROMPT_PACK_PASS',len(checks),flush=True)
finally:io.close()
