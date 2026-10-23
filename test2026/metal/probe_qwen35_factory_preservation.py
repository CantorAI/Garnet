"""One isolated old/current Metal factory process; compare outputs externally."""
import hashlib,importlib,importlib.util,json,os,sys,time
try:
    import resource
except ImportError:
    resource=None
runtime_root,repo,checkpoint,image,output,mode,overlay=sys.argv[1:8]
sys.path.insert(0,runtime_root);sys.path.insert(0,repo)
import garnet
assert 'Metal/MPS' in garnet.backend_info()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
if mode=='current':
    package=importlib.import_module('xModel.qwen35.vl_4b')
    package.model=load('xModel.qwen35.vl_4b.model',os.path.join(overlay,'factory_4b.py'))
    serving=load('preservation_metal_serving',os.path.join(overlay,'metal_serving.py'))
else:
    assert mode=='baseline'
    serving=importlib.import_module('runtime.metal_serving')
profile={'prefillChunk':16,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536}
start=time.monotonic()
model=serving.Model(checkpoint,os.path.join(repo,'xModel/qwen35/vl_4b'),
    os.path.join(overlay,'cache-'+mode),profile,'Qwen3.5-4B')
print('METAL_FACTORY_MODEL_READY',mode,time.monotonic()-start,flush=True)
rows=[]
try:
    for trial in range(2):
        result=model.infer('Describe the main objects in this image in one sentence.',image,8)
        assert result['status']=='ok'
        rows.append(result)
        with open(output,'w') as stream:json.dump({'mode':mode,'backend':garnet.backend_info(),
            'image_sha256':hashlib.sha256(open(image,'rb').read()).hexdigest(),'profile':profile,'rows':rows,
            'process_id':os.getpid(),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
            'limitation':'Short fixed-image factory preservation probe, not broad quality or comparative speed'},stream,indent=2)
        print('METAL_FACTORY_CASE_COMPLETE',mode,trial,result['generated_token_ids'],flush=True)
    assert rows[0]['generated_token_ids']==rows[1]['generated_token_ids'],'Reset changed output'
    print('QWEN35_METAL_FACTORY_PROBE_COMPLETE',mode,flush=True)
finally:
    if model.vision is not None:model.vision.close()
    model.decoder.close()
