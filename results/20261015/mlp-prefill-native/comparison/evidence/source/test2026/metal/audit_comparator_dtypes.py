"""Read actual loaded MLX parameter dtypes through vLLM's worker RPC API."""
import json,os,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['VLLM_MLX_DEVICE']='gpu'
# Inspect the local worker directly; no callable pickle/RPC serialization.
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'

def inspect_worker(worker):
    from collections import Counter
    from mlx.utils import tree_flatten
    import mlx.core as mx
    runner=worker.model_runner
    model=runner.model;parameters=tree_flatten(model.parameters());registered=len(parameters)
    # Paged-attention wrappers keep their original projections in a private
    # _inner module, outside model.parameters(). Inspect those owners too.
    wrapped=[]
    for name,module in model.named_modules():
        inner=getattr(module,'_inner',None)
        if inner is not None and hasattr(inner,'parameters'):
            values=tree_flatten(inner.parameters())
            parameters.extend((name+'.wrapped.'+key,value) for key,value in values)
            wrapped.append(dict(name=name,type=type(module).__name__,parameter_tensors=len(values)))
    counts=Counter();elements=Counter();examples={}
    seen=set()
    for name,value in parameters:
        if id(value) in seen:continue
        seen.add(id(value))
        dtype=str(value.dtype);counts[dtype]+=1;elements[dtype]+=int(value.size)
        examples.setdefault(dtype,[])
        if len(examples[dtype])<5:examples[dtype].append(dict(name=name,shape=list(value.shape)))
    probes={}
    text=getattr(model,'language_model',model);backbone=getattr(text,'model',None)
    if backbone is not None and hasattr(backbone,'embed_tokens') and hasattr(backbone,'layers'):
        embedding=backbone.embed_tokens(mx.array([[0]],dtype=mx.int32))
        layer=backbone.layers[0];norm=layer.input_layernorm(embedding)
        mx.eval(embedding,norm)
        probes.update(embedding_output_dtype=str(embedding.dtype),first_norm_output_dtype=str(norm.dtype))
        attention=getattr(layer,'self_attn',getattr(layer,'linear_attn',None))
        inner=getattr(attention,'_inner',attention)
        projection=getattr(inner,'q_proj',getattr(inner,'in_proj_qkv',None))
        if projection is not None:
            result=projection(norm);mx.eval(result);probes['first_attention_projection_dtype']=str(result.dtype)
    return dict(configured_model_dtype=str(worker.vllm_config.model_config.dtype),
        actual_parameter_tensors=dict(counts),actual_parameter_elements=dict(elements),examples=examples,
        registered_parameter_tensors=registered,wrapped_attention_owners=wrapped,activation_dtype_probes=probes,
        runner_kv_dtype=str(getattr(runner,'kv_cache_dtype',None)),mlx_default_device=str(mx.default_device()))

def main():
    from vllm import LLM
    from importlib.metadata import version
    model=Path(sys.argv[1]);dtype=sys.argv[2]
    assert dtype in ('bfloat16','float32')
    memory_fraction=float(sys.argv[3]) if len(sys.argv)>3 else .30
    assert .1<=memory_fraction<=.6
    config=json.loads((model/'config.json').read_text());vl='vision_config' in config
    arguments=dict(model=str(model),dtype=dtype,max_model_len=256,max_num_seqs=1,max_num_batched_tokens=256,
        enable_prefix_caching=False,gpu_memory_utilization=memory_fraction,disable_log_stats=True,seed=0,
        distributed_executor_backend='uni')
    if vl:arguments.update(limit_mm_per_prompt={'image':1},mm_processor_cache_gb=0)
    llm=LLM(**arguments)
    try:
        result=llm.collective_rpc(inspect_worker,timeout=120)
        assert len(result)==1 and result[0]['actual_parameter_tensors']
        print('actual_dtype_audit',json.dumps(dict(model=str(model),requested_dtype=dtype,workers=result,
            engine_mode='in-process worker; serialization security defaults retained',
            gpu_memory_utilization=memory_fraction,
            versions={name:version(name) for name in ('vllm','vllm-metal','mlx','mlx-lm','mlx-vlm')},
            scope='Read-only loaded-model dtype audit; no inference speed claim')),flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
