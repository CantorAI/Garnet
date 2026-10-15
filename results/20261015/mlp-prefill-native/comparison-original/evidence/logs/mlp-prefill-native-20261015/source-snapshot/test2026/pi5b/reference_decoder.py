import numpy as np
from reference_linear import step_linear
from reference_attention import step_attention
def rms_norm(x,weight,eps=1e-6):
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*(1+weight)
def finish_block(residual,w):
    normalized=rms_norm(residual,w['post_norm'])
    gate=normalized@w['mlp_gate'].T;up=normalized@w['mlp_up'].T
    return residual+(gate/(1+np.exp(-gate))*up)@w['mlp_down'].T
def linear_block(x,conv,state,active,w,kh,vh,kd,vd):
    x=x*active[:,None];normalized=rms_norm(x,w['input_norm'])
    output,next_conv,next_state=step_linear(normalized,conv,state,active,w,kh,vh,kd,vd)
    return finish_block(x+output,w),next_conv,next_state
def attention_block(x,key,value,positions,slots,active,w,heads,kv,dim,rotary,sections,theta):
    x=x*active[:,None];normalized=rms_norm(x,w['input_norm'])
    output,next_key,next_value=step_attention(normalized,key,value,positions,slots,active,w,heads,kv,dim,rotary,sections,theta)
    return finish_block(x+output,w),next_key,next_value
