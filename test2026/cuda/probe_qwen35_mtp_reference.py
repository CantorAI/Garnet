"""Offline one-step MTP proposal study, not speculative inference or speed proof.

Independent PyTorch primitives follow checkpoint MTP norms/concat/fc, one full
decoder layer, final norm and shared head. Target hidden/input rows are traced
from Garnet. Future target hidden rows are never inputs to an earlier proposal.
Run only after every other model worker exits.
"""
import argparse,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('trace');p.add_argument('output');a=p.parse_args()
root=Path(a.checkpoint);trace=json.loads(Path(a.trace).read_text());c=json.loads((root/'config.json').read_text())['text_config']
assert len(trace['generated_token_ids'])==1
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
needed={'model.language_model.norm.weight','model.language_model.embed_tokens.weight'};weights={}
for file in sorted(root.glob('*.safetensors')):
    with safe_open(str(file),framework='pt',device='cpu') as f:
        for key in f.keys():
            if key.startswith('mtp.') or key in needed:weights[key]=f.get_tensor(key).to('cuda:0')
assert needed.issubset(weights) and 'mtp.fc.weight' in weights
rows=trace['rows'];hidden=torch.tensor([r['target_hidden_pre_final_norm'] for r in rows],device='cuda:0')
embedded=torch.tensor([r['input_embedding'] for r in rows],device='cuda:0')
positions=torch.tensor([r['position'] for r in rows],device='cuda:0',dtype=torch.float32)
def norm(x,key):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c['rms_norm_eps'])*(1+weights[key].float())
def project(x,key):return F.linear(x.to(torch.bfloat16),weights[key]).float()
def at(key):return 'mtp.layers.0.'+key
with torch.inference_mode():
    target=norm(hidden,'model.language_model.norm.weight')
    x=project(torch.cat([norm(embedded[1:],'mtp.pre_fc_norm_embedding.weight'),
        norm(target[:-1],'mtp.pre_fc_norm_hidden.weight')],-1),'mtp.fc.weight')
    length=len(x);heads=c['num_attention_heads'];kv=c['num_key_value_heads'];dim=c['head_dim']
    normalized=norm(x,at('input_layernorm.weight'))
    qg=project(normalized,at('self_attn.q_proj.weight')).reshape(length,heads,2*dim)
    q,gate=qg.chunk(2,-1);q=norm(q,at('self_attn.q_norm.weight'))
    key=norm(project(normalized,at('self_attn.k_proj.weight')).reshape(length,kv,dim),at('self_attn.k_norm.weight'))
    value=project(normalized,at('self_attn.v_proj.weight')).reshape(length,kv,dim)
    rope=c['rope_parameters'];rotary=int(dim*rope['partial_rotary_factor']);sections=rope['mrope_section']
    axes=[1 if i%3==1 and i<sections[1]*3 else 2 if i%3==2 and i<sections[2]*3 else 0 for i in range(rotary//2)]
    freq=torch.tensor([rope['rope_theta']**(-2*i/rotary) for i in range(rotary//2)],device='cuda:0')
    angle=positions[1:,axes]*freq;angle=torch.cat([angle,angle],-1)[:,None,:]
    def rotate(t):
        part=t[...,:rotary];left,right=part.chunk(2,-1)
        return torch.cat([part*angle.cos()+torch.cat([-right,left],-1)*angle.sin(),t[...,rotary:]],-1)
    q,key=rotate(q),rotate(key);key=key.repeat_interleave(heads//kv,1);value=value.repeat_interleave(heads//kv,1)
    scores=torch.einsum('thd,shd->hts',q,key)/math.sqrt(dim)
    scores=scores.masked_fill(torch.ones((length,length),device='cuda:0',dtype=torch.bool).triu(1),-1e30)
    core=torch.einsum('hts,shd->thd',scores.softmax(-1),value)*gate.sigmoid()
    x=x+project(core.reshape(length,heads*dim),at('self_attn.o_proj.weight'))
    normalized=norm(x,at('post_attention_layernorm.weight'))
    x=x+project(F.silu(project(normalized,at('mlp.gate_proj.weight')))*project(normalized,at('mlp.up_proj.weight')),at('mlp.down_proj.weight'))
    result=norm(x,'mtp.norm.weight');prompt=len(trace['prompt_token_ids'][0]);generated=trace['generated_token_ids'][0]
    assert len(rows)==prompt+len(generated)-1,'Study expects full non-stop trajectory'
    # H[prompt-1] + E[first generated] proposes the SECOND generated token.
    proposal_hidden=result[prompt-1:];assert len(proposal_hidden)==len(generated)-1
    head=weights['model.language_model.embed_tokens.weight']
    logits=F.linear(proposal_hidden.to(torch.bfloat16),head).float()
    target_logits=F.linear(target[prompt-1:].to(torch.bfloat16),head).float()
    expected=torch.tensor(generated[1:],device='cuda:0');top1=logits.argmax(-1)
    decisions=[{'generation_index':i+1,'proposed_token':int(top1[i]),'teacher_token':int(expected[i]),
        'teacher_rank_in_mtp':int((logits[i]>logits[i,expected[i]]).sum())+1} for i in range(len(expected))]
    shortlist=[]
    for size in (32,128,512,2048,8192,32768,65536):
        indices=target_logits[:-1].topk(size,-1).indices
        restricted=indices.gather(-1,logits.gather(-1,indices).argmax(-1,keepdim=True)).squeeze(-1)
        shortlist.append({'size':size,'matches':int((restricted==expected).sum()),'decisions':len(expected),
            'scope':'Restricted full-GEMM logits, not actual selected-weight projection timing or bitwise proof'})
    target_expected=torch.tensor(generated,device='cuda:0')
    result={'model':trace['model'],'trace':a.trace,'one_step_teacher_forced':True,'full_shifted_prompt_cache':True,
        'decisions':decisions,'proposal_matches':int((top1==expected).sum()),'proposal_count':len(expected),
        'target_head_reconstruction_matches':int((target_logits.argmax(-1)==target_expected).sum()),
        'shortlist_study':shortlist,'mtp_weight_bytes':sum(t.numel()*t.element_size() for k,t in weights.items() if k.startswith('mtp.')),
        'torch':torch.__version__,'speculative_serving_implemented':False,'goal_met':False,
        'limitation':'One teacher-forced native trajectory with independently recomputed final norm and MTP. Full-sequence GEMM geometry differs from iterative drafting. No rollout acceptance, rollback/commit correctness, target-equivalence, latency, memory-peak, quality or comparator-MTP claim.'}
Path(a.output).write_text(json.dumps(result,indent=2));print('QWEN35_MTP_REFERENCE_PROPOSAL_COMPLETE',result['proposal_matches'],result['proposal_count'])
