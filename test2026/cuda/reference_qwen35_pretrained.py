"""Independent Transformers BF16 greedy reference; run sequentially on GPU0.

Uses recorded Garnet token IDs and optionally exported native image patches.
This isolates model behavior from tokenization/JPEG preprocessing differences.
Transformers BF16 residual/state arithmetic differs from Garnet mixed FP32 state;
record divergence rather than silently weakening a parity requirement.
"""
import argparse,json,time,types
import torch,transformers
from transformers import AutoModelForImageTextToText,AutoTokenizer

parser=argparse.ArgumentParser()
parser.add_argument('checkpoint');parser.add_argument('garnet_result');parser.add_argument('output')
parser.add_argument('--patches');parser.add_argument('--tokens',type=int,default=16)
parser.add_argument('--mixed-reference',action='store_true')
parser.add_argument('--row',type=int,default=0)
parser.add_argument('--batch-reference',action='store_true')
parser.add_argument('--serial-prefill',action='store_true')
parser.add_argument('--prefill-chunk',type=int,choices=[1,4,8,16],default=1)
parser.add_argument('--case-index',type=int,default=0)
parser.add_argument('--respect-stop',action='store_true')
parser.add_argument('--teacher-force',action='store_true')
parser.add_argument('--compare-hidden',action='store_true')
parser.add_argument('--native-input',action='store_true')
parser.add_argument('--strict-reductions',action='store_true')
args=parser.parse_args()
with open(args.garnet_result) as f:actual=json.load(f)[args.case_index]
if args.respect_stop:assert args.serial_prefill and not args.batch_reference,'Stop diagnostic supports one serial-reference row'
if args.teacher_force:assert args.serial_prefill and not args.batch_reference and not args.respect_stop,'Teacher forcing requires one serial row and its recorded trajectory'
if args.compare_hidden:assert args.teacher_force and 'normalized_hidden_trace' in actual,'Hidden comparison needs native trace and teacher forcing'
if args.native_input:assert args.teacher_force and 'prefill_embedding_trace' in actual,'Native-input diagnostic requires a recorded embedding trace'
teacher_tokens=actual['generated_token_ids'][args.row][:]
if actual.get('stop_token_ids',[None])[args.row] is not None:teacher_tokens.append(actual['stop_token_ids'][args.row])
if args.teacher_force:assert args.tokens<=len(teacher_tokens),'Requested teacher-forced trajectory exceeds recorded samples'
quantitative=[];input_comparison={}
torch.manual_seed(0)
model=AutoModelForImageTextToText.from_pretrained(args.checkpoint,dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda:0').eval()
tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
stop_ids={tokenizer.convert_tokens_to_ids(t) for t in ['<|im_end|>','<|endoftext|>']}
eos=model.config.text_config.eos_token_id
stop_ids.update(eos if isinstance(eos,list) else [eos])
reference_stop=None;sampled_steps=0
if args.mixed_reference:
    # Independent Transformers architecture with explicit Garnet arithmetic
    # boundaries, used to diagnose BF16 residual rounding divergence.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.strict_reductions:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    text_embedding=model.get_input_embeddings()
    # Keep large matrices/tied table BF16 even in the reference. Only small
    # norm/conv/state parameters need FP32; a full model.float() would exceed
    # a 16GiB device for 4B and distort diagnostic memory unnecessarily.
    for module in model.modules():
        for name,param in module.named_parameters(recurse=False):
            keep_bf16=(isinstance(module,(torch.nn.Linear,torch.nn.Conv3d)) and name=='weight') or module is text_embedding
            if not keep_bf16:param.data=param.data.float()
    text_embedding.register_forward_hook(lambda module,args,value:value.float())
    def mixed_linear(self,x):
        original_tokens=x.shape[1] if x.ndim==3 else 0
        chunk=getattr(self,'qwen35_projection_chunk',1)
        padded=chunk>1 and x.ndim==3 and x.shape[0]==1 and original_tokens<chunk
        if padded:x=torch.nn.functional.pad(x,(0,0,0,chunk-original_tokens))
        y=torch.nn.functional.linear(x.to(torch.bfloat16),self.weight.to(torch.bfloat16)).float()
        if self.bias is not None:y=y+self.bias.float()
        return y[:,:original_tokens] if padded else y
    def mixed_patch(self,x):
        y=torch.nn.functional.linear(x.flatten(1).to(torch.bfloat16),self.weight.flatten(1).to(torch.bfloat16)).float()
        if self.bias is not None:y=y+self.bias.float()
        return y[:,:,None,None,None]
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):module.forward=types.MethodType(mixed_linear,module)
        elif isinstance(module,torch.nn.Conv3d):module.forward=types.MethodType(mixed_patch,module)
if args.prefill_chunk>1:
    assert args.serial_prefill and args.mixed_reference and not args.batch_reference,'Chunk diagnostic currently requires mixed single-row serial-prefill reference'
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    def serial_delta(query,key,value,**kwargs):
        kwargs.pop('cu_seqlens',None)
        return torch_recurrent_gated_delta_rule(query,key,value,**kwargs)
    def full_chunk(self,hidden_states,position_embeddings,attention_mask=None,position_ids=None,past_key_values=None,**kwargs):
        normalized=self.input_layernorm(hidden_states);outputs=[]
        past=past_key_values.get_seq_length(self.self_attn.layer_idx)
        for t in range(hidden_states.shape[1]):
            mask=attention_mask[:,:,t:t+1,:past+t+1] if attention_mask is not None else None
            rotary=tuple(v[:,t:t+1] for v in position_embeddings)
            result,_=self.self_attn(hidden_states=normalized[:,t:t+1],attention_mask=mask,
                past_key_values=past_key_values,position_embeddings=rotary,**kwargs)
            outputs.append(result)
        residual=hidden_states+torch.cat(outputs,dim=1)
        return residual+self.mlp(self.post_attention_layernorm(residual))
    for name,module in model.named_modules():
        if name.startswith('model.language_model.layers.') and isinstance(module,torch.nn.Linear) and ('.mlp.' in name or '.linear_attn.' in name):
            module.qwen35_projection_chunk=args.prefill_chunk
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':layer.linear_attn.chunk_gated_delta_rule=serial_delta
        else:layer.forward=types.MethodType(full_chunk,layer)
rows=actual['prompt_token_ids'] if args.batch_reference else [actual['prompt_token_ids'][args.row]]
length=max(map(len,rows));pad=tokenizer.pad_token_id
ids=torch.tensor([[pad]*(length-len(row))+row for row in rows],device='cuda:0',dtype=torch.long)
mask=torch.tensor([[0]*(length-len(row))+[1]*len(row) for row in rows],device='cuda:0',dtype=torch.long)
inputs={'input_ids':ids,'attention_mask':mask}
if args.patches:
    with open(args.patches) as f:patches=json.load(f)
    inputs['pixel_values']=torch.tensor(patches['patches'],device='cuda:0',dtype=torch.bfloat16).reshape(-1,1536).repeat(len(rows),1)
    inputs['image_grid_thw']=torch.tensor([patches['grid_thw']]*len(rows),device='cuda:0',dtype=torch.long)
torch.cuda.synchronize();started=time.perf_counter()
with torch.inference_mode():
    if not args.serial_prefill:
        output=model.generate(**inputs,max_new_tokens=args.tokens,do_sample=False,use_cache=True)
    else:
        embeds=model.get_input_embeddings()(ids)
        if args.patches:
            # Garnet encodes each batch row's image separately with a shared
            # batch-one tower. Match that arithmetic shape in this diagnostic.
            features=[]
            pixels=inputs['pixel_values'];grid=inputs['image_grid_thw']
            image_size=int(grid[0].prod())
            for row in range(len(rows)):
                features.append(model.get_image_features(pixels[row*image_size:(row+1)*image_size],grid[row:row+1]).pooler_output[0])
            embeds[ids==model.config.image_token_id]=torch.cat(features)
        types_ids=(ids==model.config.image_token_id).long()
        positions,_=model.model.get_rope_index(ids,types_ids,image_grid_thw=inputs.get('image_grid_thw'),attention_mask=mask)
        if 'prefill_embedding_trace' in actual:
            native_embeds=torch.tensor(actual['prefill_embedding_trace'],device=embeds.device,dtype=torch.float32).reshape(embeds.shape)
            delta=native_embeds-embeds.float()
            native_positions=torch.tensor(actual['native_position_trace'],device=positions.device).permute(2,0,1)
            input_comparison={'embedding_max_abs':float(delta.abs().max()),'embedding_relative_l2':float(delta.norm()/embeds.float().norm().clamp_min(1e-12)),
                'positions_equal':bool(torch.equal(native_positions,positions)),'position_max_abs':int((native_positions-positions).abs().max())}
            if args.native_input:
                assert input_comparison['positions_equal'],'Native and reference RoPE positions differ'
                embeds=native_embeds
        cache=None
        for t in range(0,length,args.prefill_chunk):
            end=min(length,t+args.prefill_chunk)
            state=model.model.language_model(inputs_embeds=embeds[:,t:end],position_ids=positions[:,:,t:end],
                attention_mask=mask[:,:end],past_key_values=cache,use_cache=True)
            cache=state.past_key_values
        next_positions=positions.amax(dim=(0,2))+1
        generated_steps=[]
        for step in range(args.tokens):
            reference_hidden=state.last_hidden_state[:,-1]
            reference_logits=model.lm_head(reference_hidden)
            token=reference_logits.argmax(-1)
            if args.compare_hidden:
                native_hidden=torch.tensor(actual['normalized_hidden_trace'][step],device=reference_hidden.device,dtype=torch.float32).reshape(reference_hidden.shape)
                native_logits=model.lm_head(native_hidden)
                difference=native_hidden-reference_hidden.float();logit_difference=native_logits-reference_logits
                selected=teacher_tokens[step];top2=reference_logits.topk(2).values[0]
                quantitative.append({'step':step,'native_selected_token':selected,'reference_top1':int(token[0]),
                    'native_hidden_reconstructed_top1':int(native_logits.argmax(-1)[0]),
                    'hidden_max_abs':float(difference.abs().max()),'hidden_rms':float(difference.square().mean().sqrt()),
                    'hidden_relative_l2':float(difference.norm()/reference_hidden.float().norm().clamp_min(1e-12)),
                    'hidden_cosine':float(torch.nn.functional.cosine_similarity(native_hidden,reference_hidden.float()).mean()),
                    'logit_max_abs':float(logit_difference.abs().max()),'logit_rms':float(logit_difference.square().mean().sqrt()),
                    'reference_top1_gap':float(top2[0]-top2[1]),
                    'native_selected_reference_rank':int((reference_logits[0]>reference_logits[0,selected]).sum())+1})
            sampled_steps+=1
            if args.respect_stop and int(token[0]) in stop_ids:
                reference_stop=int(token[0]);break
            generated_steps.append(token)
            if step+1==args.tokens:break
            if args.teacher_force:token=torch.tensor([teacher_tokens[step]],device=token.device,dtype=torch.long)
            mask=torch.cat([mask,torch.ones((len(rows),1),device=mask.device,dtype=mask.dtype)],dim=1)
            state=model.model.language_model(input_ids=token[:,None],position_ids=next_positions[None,:,None].expand(3,-1,-1),
                attention_mask=mask,past_key_values=cache,use_cache=True)
            cache=state.past_key_values;next_positions+=1
        generated_tensor=torch.stack(generated_steps,dim=1) if generated_steps else ids[:,:0]
        output=torch.cat([ids,generated_tensor],dim=1)
torch.cuda.synchronize()
generated=output[:,ids.shape[1]:].tolist()
expected=actual['generated_token_ids'] if args.batch_reference else [actual['generated_token_ids'][args.row]]
common=[]
for a,b in zip(expected,generated):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    common.append(n)
result={'reference':'Transformers eager mixed FP32 state/BF16 projections' if args.mixed_reference else 'Transformers eager BF16','torch':torch.__version__,'transformers':transformers.__version__,
    'invocation':vars(args),
    'generated_token_ids':generated,'texts':tokenizer.batch_decode(generated,skip_special_tokens=False),
    'garnet_token_ids':expected,'common_prefix_tokens':common,'exact_token_match':generated==expected,
    'seconds':time.perf_counter()-started,'same_recorded_prompt_ids':True,
    'same_native_image_patches':bool(args.patches),'serial_prefill':args.serial_prefill,'prefill_chunk':args.prefill_chunk,'garnet_batch_row':args.row,'memory_allocated':torch.cuda.memory_allocated(),
    'peak_memory_allocated':torch.cuda.max_memory_allocated(),
    'case_index':args.case_index,'respect_stop':args.respect_stop,
    'sampled_tokens_including_stop':sampled_steps if args.serial_prefill else None,'stop_token_id':reference_stop,
    'teacher_forced':args.teacher_force,'quantitative_hidden_logit_comparison':quantitative,
    'native_input_embeddings':args.native_input,'input_comparison':input_comparison,
    'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    'limitation':'Reference correctness diagnostic, not a matched performance benchmark; arithmetic differs.'}
with open(args.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
print('QWEN35_TRANSFORMERS_REFERENCE_COMPLETE',flush=True)
