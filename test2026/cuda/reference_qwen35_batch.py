"""Independent HF mixed-precision, teacher-forced batch decoder diagnostic.

Native-input mode isolates decoder behavior from vision arithmetic. Ragged
rows use canonical left padding; only still-active recorded decisions are
compared. This is neither free-running quality acceptance nor a speed test.
"""
import argparse,json,time,types
import torch,transformers
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
parser=argparse.ArgumentParser()
parser.add_argument('checkpoint');parser.add_argument('native');parser.add_argument('output')
parser.add_argument('--case-index',type=int,default=0);parser.add_argument('--tokens',type=int,default=32)
parser.add_argument('--full-projection-chunk',action='store_true',help='Match the vectorized full-attention projection candidate, preserving old serial default')
args=parser.parse_args();actual=json.load(open(args.native))[args.case_index]
assert actual['batch'] in (1,2,4,8) and 'normalized_hidden_trace' in actual
batch=actual['batch'];chunk=actual['profile']['prefillChunk'];width=len(actual['prefill_embedding_trace'][0][0])
teacher=[list(row) for row in actual['generated_token_ids']]
for row,stop in zip(teacher,actual['stop_token_ids']):
    if stop is not None:row.append(stop)
assert args.tokens<=max(map(len,teacher)) and len(actual['normalized_hidden_trace'])>=args.tokens
torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
model=AutoModelForImageTextToText.from_pretrained(args.checkpoint,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
embedding=model.get_input_embeddings()
for module in model.modules():
    for name,param in module.named_parameters(recurse=False):
        keep=(isinstance(module,(torch.nn.Linear,torch.nn.Conv3d)) and name=='weight') or module is embedding
        if not keep:param.data=param.data.float()
embedding.register_forward_hook(lambda module,inputs,value:value.float())
def linear(self,x):
    count=x.shape[1] if x.ndim==3 else 0;size=getattr(self,'reference_chunk',1)
    padded=x.ndim==3 and count<size
    if padded:x=torch.nn.functional.pad(x,(0,0,0,size-count))
    value=torch.nn.functional.linear(x.to(torch.bfloat16),self.weight.to(torch.bfloat16)).float()
    if self.bias is not None:value=value+self.bias.float()
    return value[:,:count] if padded else value
for name,module in model.named_modules():
    if isinstance(module,torch.nn.Linear):
        module.forward=types.MethodType(linear,module)
        if name.startswith('model.language_model.layers.') and ('.mlp.' in name or '.linear_attn.' in name or (args.full_projection_chunk and '.self_attn.' in name)):module.reference_chunk=chunk
def delta(query,key,value,**kwargs):
    kwargs.pop('cu_seqlens',None);return torch_recurrent_gated_delta_rule(query,key,value,**kwargs)
def full_chunk(self,hidden_states,position_embeddings,attention_mask=None,position_ids=None,past_key_values=None,**kwargs):
    normalized=self.input_layernorm(hidden_states);outputs=[];past=past_key_values.get_seq_length(self.self_attn.layer_idx)
    if args.full_projection_chunk:
        value,_=self.self_attn(hidden_states=normalized,attention_mask=attention_mask,
            past_key_values=past_key_values,position_embeddings=position_embeddings,**kwargs)
        residual=hidden_states+value
        return residual+self.mlp(self.post_attention_layernorm(residual))
    for t in range(hidden_states.shape[1]):
        mask=attention_mask[:,:,t:t+1,:past+t+1] if attention_mask is not None else None
        value,_=self.self_attn(hidden_states=normalized[:,t:t+1],attention_mask=mask,
            past_key_values=past_key_values,position_embeddings=tuple(v[:,t:t+1] for v in position_embeddings),**kwargs)
        outputs.append(value)
    residual=hidden_states+torch.cat(outputs,dim=1)
    return residual+self.mlp(self.post_attention_layernorm(residual))
linear_modules=[]
def masked_linear(self,hidden_states,attention_mask=None,**kwargs):
    # HF normally skips recurrent masks after cache creation. Chunk-prefill may
    # still contain left padding then, so supply the current local input mask.
    return self.reference_original(hidden_states,attention_mask=self.reference_local_mask,**kwargs)
for layer in model.model.language_model.layers:
    if layer.block_type=='linear_attention':
        module=layer.linear_attn;module.chunk_gated_delta_rule=delta
        module.reference_original=module.forward;module.reference_local_mask=None
        module.forward=types.MethodType(masked_linear,module);linear_modules.append(module)
    else:layer.forward=types.MethodType(full_chunk,layer)
length=max(actual['prompt_tokens']);pad=model.config.text_config.pad_token_id or 0
ids=torch.tensor([[pad]*(length-len(row))+row for row in actual['prompt_token_ids']],device='cuda:0')
mask=torch.tensor([[0]*(length-n)+[1]*n for n in actual['prompt_tokens']],device='cuda:0')
embeds=torch.zeros((batch,length,width),device='cuda:0',dtype=torch.float32)
positions=torch.zeros((3,batch,length),device='cuda:0',dtype=torch.long)
for b,n in enumerate(actual['prompt_tokens']):
    embeds[b,-n:]=torch.tensor(actual['prefill_embedding_trace'][b],device='cuda:0').reshape(n,width)
    positions[:,b,-n:]=torch.tensor(actual['native_position_trace'][b],device='cuda:0').T
grid=actual['profile']['imageGrid']
reference_positions,_=model.model.get_rope_index(ids,(ids==model.config.image_token_id).long(),
    image_grid_thw=torch.tensor([[1,*grid]]*batch,device='cuda:0'),attention_mask=mask)
assert torch.equal(positions[:,mask.bool()],reference_positions[:,mask.bool()]),'Valid native/reference RoPE positions differ'
quantitative=[];cache=None;started=time.perf_counter()
with torch.inference_mode():
    for start in range(0,length,chunk):
        end=min(start+chunk,length)
        for module in linear_modules:module.reference_local_mask=mask[:,start:end]
        state=model.model.language_model(inputs_embeds=embeds[:,start:end],position_ids=positions[:,:,start:end],
            attention_mask=mask[:,:end],past_key_values=cache,use_cache=True)
        cache=state.past_key_values
    next_positions=positions.amax(dim=(0,2))+1
    for step in range(args.tokens):
        hidden=state.last_hidden_state[:,-1];scores=model.lm_head(hidden)
        native_hidden=torch.tensor(actual['normalized_hidden_trace'][step],device='cuda:0').reshape(batch,width)
        native_scores=model.lm_head(native_hidden)
        for b in range(batch):
            if step>=len(teacher[b]):continue
            selected=teacher[b][step];diff=native_hidden[b]-hidden[b];ldiff=native_scores[b]-scores[b];top2=scores[b].topk(2).values
            quantitative.append({'step':step,'row':b,'native_selected_token':selected,'reference_top1':int(scores[b].argmax()),
                'native_hidden_reconstructed_top1':int(native_scores[b].argmax()),'hidden_max_abs':float(diff.abs().max()),
                'hidden_rms':float(diff.square().mean().sqrt()),'hidden_relative_l2':float(diff.norm()/hidden[b].norm().clamp_min(1e-12)),
                'hidden_cosine':float(torch.nn.functional.cosine_similarity(native_hidden[b:b+1],hidden[b:b+1]).mean()),
                'logit_max_abs':float(ldiff.abs().max()),'logit_rms':float(ldiff.square().mean().sqrt()),
                'reference_top1_gap':float(top2[0]-top2[1]),'native_selected_reference_rank':int((scores[b]>scores[b,selected]).sum())+1})
        if step+1==args.tokens:break
        active=torch.tensor([int(step+1<len(row)) for row in teacher],device='cuda:0')
        token=torch.tensor([row[step] if step<len(row) else pad for row in teacher],device='cuda:0')
        mask=torch.cat([mask,active[:,None]],dim=1)
        for module in linear_modules:module.reference_local_mask=active[:,None]
        state=model.model.language_model(input_ids=token[:,None],position_ids=next_positions[None,:,None].expand(3,-1,-1),
            attention_mask=mask,past_key_values=cache,use_cache=True)
        cache=state.past_key_values;next_positions+=active
torch.cuda.synchronize()
result={'reference':'Independent Transformers mixed FP32 state/BF16 projections; fixed native-input batch trajectory',
    'torch':torch.__version__,'transformers':transformers.__version__,
    'teacher_forced':True,'batch':batch,'style':actual['style'],'case_index':args.case_index,'invocation':vars(args),
    'full_attention_projection_layout':'chunk_batched' if args.full_projection_chunk else 'token_serial',
    'native_input_embeddings':True,'positions_equal_on_valid_tokens':True,'allow_bf16_reduced_precision_reduction':False,
    'quantitative_hidden_logit_comparison':quantitative,'seconds':time.perf_counter()-started,
    'limitation':'Decoder-only diagnostic, not independent vision or free-running quality acceptance. Left padding is masked in every cached prefill chunk. Inactive post-stop row outputs are excluded; their unused reference state may continue updating.'}
with open(args.output,'w') as stream:json.dump(result,stream,indent=2)
print('QWEN35_BATCH_REFERENCE_COMPLETE','batch',batch,'style',actual['style'],'decisions',len(quantitative),flush=True)
