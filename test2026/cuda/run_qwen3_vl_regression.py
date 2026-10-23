"""Current-DLL legacy serving smoke; isolated from Qwen3.5 model sessions."""
import json,os,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
checkpoint,xmodel,image,cache,output=sys.argv[1:6]
name='Qwen3-VL-2B-Instruct'
profile={'maxInputTokens':512,'patchCount':240,'kvPages':48,'minPixels':65536,'maxPixels':65536,'maxOutputTokens':32,'deviceId':0}
prompt='Describe the main objects in this image in one sentence.'
prepared=garnet.qwen_vl_prepare_request(checkpoint,image,prompt,65536,65536)
assert prepared is not None,'Native preprocessing failed'
preprocess={key:prepared[key] for key in ['pixel_values_shape','image_grid_thw','visual_token_count','source_height','source_width','height','width']}
profile['patchCount']=int(prepared['pixel_values_shape'][0])
print('QWEN3_VL_PREPROCESS_PROFILE',json.dumps(preprocess),flush=True)
prepared=None
start=time.monotonic();rows=[]
try:
    status=json.loads(garnet.serve_model(checkpoint,xmodel,cache,json.dumps(profile),name))
    print('QWEN3_VL_STARTUP_STATUS',json.dumps(status),flush=True)
    with open(output+'.startup.json','w') as stream:json.dump(status,stream,indent=2)
    assert status.get('ready') is True,status
    print('QWEN3_VL_REGRESSION_READY',time.monotonic()-start,flush=True)
    for trial in range(2):
        result=json.loads(garnet.infer_json(prompt,image,16,name))
        assert result['status']=='ok',result
        rows.append(result)
        with open(output,'w') as stream:json.dump({'profile':profile,'preprocess':preprocess,'status':status,'rows':rows,
            'limitation':'Legacy serving smoke, not broad quality or matched-engine performance'},stream,indent=2)
        print('QWEN3_VL_REGRESSION_CASE_COMPLETE',trial,flush=True)
    # Legacy public API exposes decoded text/counts, not raw token IDs.
    assert rows[0]['text']==rows[1]['text'],'Reset/replay text changed'
    assert rows[0]['output_tokens']==rows[1]['output_tokens']==16
    assert rows[0]['visual_tokens']==rows[1]['visual_tokens']==preprocess['visual_token_count']
    print('QWEN3_VL_REGRESSION_COMPLETE',flush=True)
finally:garnet.stop_serving(name)
