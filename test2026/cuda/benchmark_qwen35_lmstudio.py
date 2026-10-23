"""One preloaded LM Studio/llama.cpp batch1 cohort with fresh JPEG variants.

Load with one parallel slot and no Draft MTP. Fresh per-round corner pixels
force image re-evaluation; runtime logs must audit remaining text prefix reuse.
"""
import json,time,sys
from pathlib import Path
import lmstudio as lms
dataset,output=sys.argv[1:3]
rounds=int(sys.argv[3]) if len(sys.argv)>3 else 7
template={'type':'manual','stopStrings':['<|im_end|>','<|endoftext|>'],
    'manualPromptTemplate':{'beforeSystem':'','afterSystem':'',
        'beforeUser':'<|im_start|>user\n','afterUser':'<|im_end|>\n',
        'beforeAssistant':'<|im_start|>assistant\n<think>\n\n</think>\n\n','afterAssistant':'<|im_end|>\n'}}
config={'temperature':0,'maxTokens':32,'topKSampling':1,'repeatPenalty':1.0,
    'contextOverflowPolicy':'stopAtLimit','promptTemplate':template}
images=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg'];rows=[]
with lms.Client() as client:
    loaded=client.llm.list_loaded()
    assert len(loaded)==1,'Benchmark requires exactly one loaded LM Studio model'
    model=client.llm.model('qwen35-2b-b1')
    Path(output).with_suffix('.config.json').write_text(json.dumps({'load_config':model.get_load_config().to_dict(),
        'prediction_config':config,'batch':1,'parallel_slots':1,
        'cache_policy':'Fresh JPEG image per round; remaining text-only prefix reuse must be audited in runtime logs.',
        'precision_limitation':'Original BF16 GGUF weights; internal residual/cache arithmetic is engine-specific.'},indent=2))
    try:
        for trial in range(rounds):
            for name in images:
                started=time.perf_counter()
                image=client.files.prepare_image(str(Path(dataset)/str(trial)/name))
                chat=lms.Chat();chat.add_user_message('Describe the main objects in this image in one sentence.',images=[image])
                result=model.respond(chat,config=config)
                elapsed=time.perf_counter()-started
                stats=result.stats.to_dict()
                row={'engine':'lmstudio-llama','model':'Qwen3.5-2B','batch':1,'trial':trial,'warmup':trial<2,
                    'images':[name],'dataset':dataset,'variant_round':trial,'wall_seconds':elapsed,
                    'text':result.content,'stats':stats,'request_output_limit':32,
                    'timing_scope':'SDK file/image preparation plus respond including decode/detokenization; model load excluded'}
                rows.append(row);Path(output).write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)
                assert stats['promptTokensCount']==85,'LM Studio prompt/image work differs from the85-token Garnet/vLLM request'
        print('QWEN35_LMSTUDIO_BENCHMARK_COMPLETE','requests',len(rows),flush=True)
    finally:model.unload()
