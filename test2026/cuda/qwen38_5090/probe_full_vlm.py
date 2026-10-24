"""Full native64-layer image/text smoke against matched reference token oracles."""
import hashlib,json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_decoder,create_cuda_vision,prepare_cuda_image
checkpoint,fixture,output=sys.argv[1:4]
chunk=int(os.environ.get('GARNET_STUDY_CHUNK','1'));assert chunk in [1,4,16]
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)
with open(os.path.join(fixture,'oracle.json')) as f:oracle=json.load(f)
tokenizer=garnet.tokenizer();tokenizer.open(checkpoint)
suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
image_id=config['image_token_id'];stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
for case in oracle['cases']:
    assert case['reference_text'].lower().rstrip('.')==case['expected_color']
    with open(os.path.join(fixture,case['image_file']),'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==case['image_sha256']
    actual=tokenizer.encode('<|im_start|>user\n<|vision_start|>')+[image_id]*64+tokenizer.encode('<|vision_end|>'+case['question']+suffix)
    assert actual==case['prompt_ids'],{'case':case['name'],'native_ids':actual,'reference_ids':case['prompt_ids']}
    assert len(actual)+32<=4096
if len(sys.argv)>4:
    from test2026.cuda.qwen38_5090.native_text_pilot import run_native_text_pilot
    with open(sys.argv[4]) as f:pilot_inputs=json.load(f)
    for item in pilot_inputs['requests']:
        prompt='<|im_start|>user\n'+item['body']['messages'][0]['content']+suffix
        assert len(tokenizer.encode(prompt))==41
decoder=None;vision=None;reports=[]
try:
    vision=create_cuda_vision(checkpoint,config,16,16)
    decoder=create_cuda_decoder(checkpoint,config,4096,chunk_size=chunk,fused_delta=chunk>1,fused_attention=chunk>1)
    decoder.enable_cuda_graph(True);vision.graph.enable_cuda_graph(True)
    if len(sys.argv)>4:
        run_native_text_pilot(decoder,tokenizer,sys.argv[4],output+'.text-pilot.json')
    for case in oracle['cases']:
        decoder.reset();io=decoder.io
        prepared=prepare_cuda_image(io,checkpoint,os.path.join(fixture,case['image_file']))
        assert [prepared[1],prepared[2]]==[16,16]
        visual=vision.encode_patches(prepared[0]);ids=case['prompt_ids'];image_start=ids.index(image_id)
        assert ids.count(image_id)==64 and ids[image_start:image_start+64]==[image_id]*64
        current=0;positions=[]
        for i in range(len(ids)):
            if image_start<=i<image_start+64:
                j=i-image_start;positions.append([current,current+j//8,current+j%8])
                if j==63:current+=8
            else:positions.append([current]*3);current+=1
        prediction=None
        if chunk>1:
            embeddings=io.embedding(ids)
            hidden=[io.visual_token(visual,i-image_start) if image_start<=i<image_start+64 else io.row(embeddings,i) for i in range(len(ids))]
            prediction=decoder.prefill_sequence(hidden,[positions])
        else:
            for i,token in enumerate(ids):
                if image_start<=i<image_start+64:
                    prediction=decoder.step_hidden(io.visual_token(visual,i-image_start),positions=[positions[i]],predict=i+1==len(ids))
                else:prediction=decoder.step([token],positions=[positions[i]],predict=i+1==len(ids))
        generated=[]
        for step in range(32):
            token=prediction[0];generated.append(token)
            if token in stops:break
            prediction=decoder.step([token],positions=[[current]*3]);current+=1
        text=tokenizer.decode([token for token in generated if token not in stops]).strip()
        row={'name':case['name'],'generated_ids':generated,'reference_generated_ids':case['reference_generated_ids'],
            'text':text,'expected_color':case['expected_color'],'exact_token_parity':generated==case['reference_generated_ids'],
            'semantic_smoke_pass':text.lower().rstrip('.')==case['expected_color'],
            'image_sha256':case['image_sha256'],'prompt_tokens':len(ids),'image_tokens':64}
        reports.append(row)
        result={'cases':reports,'layers':64,'capacity':4096,'batch':1,'chunk':chunk,'vision_projection_precision':'BF16 default',
            'io_memory':json.loads(io.memory_json()),'vision_stats':json.loads(vision.graph.stats_json()),
            'scope':'Four matched JPEG color/spatial smoke cases only; numerical vision failures and broader quality remain separate; no timing claim'}
        with open(output,'w') as f:json.dump(result,f,indent=2)
        print('NATIVE_FULL_VLM_SMOKE_CASE',row,flush=True)
    assert all(r['semantic_smoke_pass'] and r['exact_token_parity'] for r in reports),reports
    print('NATIVE_FULL_VLM_MATCHED_TOKEN_SMOKE_PASS',flush=True)
finally:
    if decoder is not None:decoder.close()
    if vision is not None:vision.close()
