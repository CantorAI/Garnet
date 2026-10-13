"""Common CPU frontend with observed image-to-first/final prediction clocks.

Arguments: worker executable, native build, source, model, case JSON, backend,
protocol directory. Both runtimes receive freshly prepared identical RGB pixels;
the native worker additionally receives the canonical FP32 patch representation.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from PIL import Image
from frame_preprocessing import prepare_frame

def main():
    executable,build,source,model,casefile,backend,folder=sys.argv[1:]
    source=Path(source);model=Path(model);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    workload=json.loads(Path(casefile).read_text());metadata=workload['image_metadata'];batch=workload['batch']
    configuration=json.loads((model/'config.json').read_text());processor=json.loads((model/'preprocessor_config.json').read_text())
    control=folder/'command.json'
    def send(message):
        temporary=folder/'command.tmp';temporary.write_text(json.dumps(message));os.replace(temporary,control)
    send(dict(request=-1))
    scripts=source/'test2026/metal'
    if backend=='garnet':command=[executable,str(scripts/'frame_worker_garnet.py'),build,str(source),str(model),casefile,str(control)]
    else:command=[executable,str(scripts/'frame_worker_vllm.py'),str(model),casefile,str(control)]
    worker_log=(folder/'worker.log').open('w')
    started=time.monotonic();worker=subprocess.Popen(command,stdout=worker_log,stderr=subprocess.STDOUT)
    def receive(marker,request=None):
        label='ready' if marker=='frame_ready' else str(request)+'-'+marker
        event=Path(str(control)+'.event-'+label+'.json');deadline=time.monotonic()+180
        while True:
            if event.exists():
                try:payload=json.loads(event.read_text())
                except json.JSONDecodeError:time.sleep(.001);continue
                return payload,time.monotonic()
            if worker.poll() is not None:raise RuntimeError('Image worker ended before '+marker)
            if time.monotonic()>deadline:raise TimeoutError('Image worker event timeout: '+marker)
            time.sleep(.001)
    try:
        ready,ready_time=receive('frame_ready');initialization=ready_time-started
        (folder/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
        print('frame_initialization',json.dumps(dict(seconds=initialization,command=command,backend=backend)),flush=True)
        for request in range(workload['requests']):
            begin=time.monotonic()
            with Image.open(metadata['image']) as image:patches,rgb,resized=prepare_frame(image,configuration['vision_config'],processor,batch)
            assert list(resized)==metadata['resized_hw'] and list(patches.shape)==metadata['shape']
            patchfile=folder/('frame-'+str(request)+'.f32');rgbfile=folder/('frame-'+str(request)+'.rgb')
            patches.tofile(patchfile);rgbfile.write_bytes(rgb)
            patch_sha=hashlib.sha256(patchfile.read_bytes()).hexdigest();rgb_sha=hashlib.sha256(rgb).hexdigest()
            assert patch_sha==metadata['patch_sha256']
            preprocessing=time.monotonic()-begin
            send(dict(request=request,patch_file=str(patchfile),rgb_file=str(rgbfile)))
            first,observed_first=receive('frame_first',request)
            done,end=receive('frame_done',request);result,_=receive('frame_result',request)
            assert first['request']==request and done['request']==request and result['request']==request
            result.update(runtime=backend,case=workload,initialization_seconds=initialization,
                clock='time.monotonic',ttft_seconds=observed_first-begin,total_seconds=end-begin,
                decode_window_seconds=end-observed_first,
                decode_wall_steps_per_second=result['decode_steps_after_first']/(end-observed_first),
                common_preprocessing_seconds=preprocessing,patch_sha256=patch_sha,rgb_sha256=rgb_sha,
                timing_scope='common CPU source-image preparation plus persistent worker protocol and model execution')
            print('matched_frame_result',json.dumps(result),flush=True)
        send(dict(request=workload['requests'],stop=True));worker.wait(timeout=45)
        worker_log.flush();remaining=(folder/'worker.log').read_text();print(remaining,end='',flush=True)
        assert worker.returncode==0 and 'Segfault encountered' not in remaining
    finally:
        if worker.poll() is None:
            send(dict(request=1000000,stop=True))
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=15)
        worker_log.close()

if __name__=='__main__':main()
