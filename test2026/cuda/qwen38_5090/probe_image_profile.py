"""Native JPEG resize admission and normalization; not an image benchmark."""
import json,os,sys,math
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
fixture,output=sys.argv[1:3]
io=garnet.cuda_tensor_io()
try:
    legacy=io.prepare_image(fixture,1048576)
    current=io.prepare_image_profile(fixture,65536,1048576)
    assert [legacy[1],legacy[2]]==[14,14]
    assert [current[1],current[2]]==[16,16]
    values=garnet.tensor_to_cpu(current[0]).tolist()
    assert len(values)==256*1536 and all(math.isfinite(v) for v in values)
    # Constant red JPEG decodes as RGB[254,0,0]. Channel/temporal/patch order
    # must agree with the pinned mean/std .5 processor, including temporal repeat.
    maximum=0
    for i,value in enumerate(values):
        channel=(i%1536)//512
        expected=254.0/127.5-1 if channel==0 else -1
        maximum=max(maximum,abs(value-expected))
    assert maximum<.0001,maximum
    rejected=0
    for low,high in [[0,1048576],[65536,1024],[65536,1048577]]:
        try:io.prepare_image_profile(fixture,low,high)
        except:rejected+=1
    assert rejected==3
    result={'legacy_grid':[14,14],'checkpoint_grid':[16,16],
        'image_tokens':64,'max_normalization_error':maximum,'invalid_bounds_rejected':rejected,
        'scope':'224px constant JPEG frontend only; no vision model, PNG support or performance claim'}
    with open(output,'w') as f:json.dump(result,f,indent=2)
    print('NATIVE_IMAGE_PROFILE_PASS',result)
finally:io.close()
