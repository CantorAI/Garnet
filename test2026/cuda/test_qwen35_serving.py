"""Public CUDA serve/status/infer/list/stop, profile conflicts and batch replay.

Real checkpoint smoke; run one model/profile at a time. Not a speed benchmark.
"""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
root,xmodel,image,output=sys.argv[1:5]
batch=int(sys.argv[5]) if len(sys.argv)>5 else 1
name=sys.argv[6] if len(sys.argv)>6 else 'Qwen3.5-2B'
profile={'batch':batch,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536}
def parsed(value):return json.loads(value)
def infer(prompts,images,limit):return parsed(garnet.infer_json(prompts,images,limit,name))
results=[]
try:
    invalid=dict(profile);invalid['batch']=3
    assert parsed(garnet.serve_model(root,xmodel,'',json.dumps(invalid),name))['status']=='error'
    status=parsed(garnet.serve_model(root,xmodel,'',json.dumps(profile),name))
    assert status['ready'] and status['batch']==batch,status
    assert parsed(garnet.serve_status_json(name))['ready']
    loaded=parsed(garnet.list_loaded_models_json())
    assert len(loaded['models'])==1 and loaded['models'][0]['batch']==batch,loaded
    assert parsed(garnet.serve_model(root,xmodel,'',json.dumps(profile),name))['ready']
    conflict=dict(profile);conflict['maxInputTokens']=257
    assert parsed(garnet.serve_model(root,xmodel,'',json.dumps(conflict),name))['error_code']=='model_instance_conflict'
    prompts=['Describe the main objects in this image in one sentence.' if b%2==0 else 'Describe this image briefly.' for b in range(batch)]
    images=[image]*batch
    assert infer(prompts,images,129)['status']=='error'
    assert infer(prompts,images[:-1],16)['status']=='error'
    for trial in range(2):
        result=infer(prompts,images,16);assert result['status']=='ok',result
        assert result['batch']==batch and result['generated_tokens']==[16]*batch,result
        assert result['visual_tokens_per_row']==[60]*batch,result
        results.append(result)
        with open(output,'w') as f:f.write(json.dumps({'status':status,'loaded':loaded,'results':results},indent=2))
        print('PUBLIC_SERVING_TRIAL_COMPLETE',trial,'batch',batch,flush=True)
    assert results[0]['generated_token_ids']==results[1]['generated_token_ids']
    assert results[1]['vision_compile_ms']==0
    print('QWEN35_PUBLIC_SERVING_COMPLETE',name,'batch',batch,flush=True)
finally:
    garnet.stop_serving(name)
    assert not parsed(garnet.list_loaded_models_json())['models']
    assert parsed(garnet.serve_status_json(name))['status']=='error'
