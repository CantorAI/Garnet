"""Logical contiguous spans must not overwrite neighboring SDK token views."""
import json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
io=garnet.cuda_tensor_io();checks=[]

def blob(values):return struct.pack('<'+str(len(values))+'f',*values)
def check(views,expected):
    for index,(value,want) in enumerate(zip(views,expected)):
        actual=io.finite_f32_bytes(value)
        assert actual==want,('view_index',index,'actual_prefix',actual[:32].hex(),'expected_prefix',want[:32].hex())

try:
    for b in (1,2,4,8):
        for h in (7,16):
            rows=[[float(1000*row+100*t+j) for t in range(3) for j in range(h)] for row in range(b)]
            source=[io.from_host(row,[3,h]) for row in rows]
            views=list(io.pack_prompt(source,[None]*b,[-1]*b))
            expected=[blob([rows[row][t*h+j] for row in range(b) for j in range(h)]) for t in range(3)]
            check(views,expected)
            io.zero(views[1]);expected[1]=bytes(b*h*4);check(views,expected)
            # The first two views have accessible backing spans larger than
            # their logical shapes; clone/copy must use the logical shape.
            cloned=io.clone(views[0]);assert io.finite_f32_bytes(cloned)==expected[0]
            dst=io.zeros([b,h]);io.copy_into(views[1],dst)
            assert io.finite_f32_bytes(dst)==expected[1]
            check(views,expected)
            # Sources and destinations may be disjoint views of one allocation.
            io.copy_into(views[0],views[2]);expected[2]=expected[0];check(views,expected)
            replacement=[float(2000+row*h+j) for row in range(b) for j in range(h)]
            io.update(dst,replacement)
            io.copy_into(dst,views[1]);expected[1]=blob(replacement);check(views,expected)
            io.copy_into(views[0],views[0]);check(views,expected)
            assert [io.finite_f32_bytes(x) for x in source]==[blob(x) for x in rows]
            del source,views
            assert io.finite_f32_bytes(cloned)==expected[0]
            checks.append({'batch':b,'hidden':h,'logical_clone_copy_zero_and_disjoint_aliases_exact':True})
    # Original whole-allocation operations keep their behavior.
    original=io.from_host([1.,-0.,2.,3.],[2,2]);copy=io.clone(original)
    io.zero(original);assert io.finite_f32_bytes(original)==bytes(16)
    assert io.finite_f32_bytes(copy)==blob([1.,-0.,2.,3.])
    bad=False
    try:io.copy_into(copy,io.zeros([1,4]))
    except BaseException:bad=True
    assert bad,'Different logical shapes accepted'
    io.close();assert io.finite_f32_bytes(cloned)==expected[0]
    Path(sys.argv[1]).write_text(json.dumps({'all_passed':True,'cases':checks,
        'original_whole_allocations_preserved':True,'wrong_shape_rejected':True,
        'scope':'Logical spans for contiguous SDK views, neighboring storage preservation, source/owner lifetime and shared disjoint aliases. Not a model speed or independent quality proof.'},indent=2))
    print('CUDA_LOGICAL_VIEWS_PASS',len(checks),flush=True)
finally:io.close()
