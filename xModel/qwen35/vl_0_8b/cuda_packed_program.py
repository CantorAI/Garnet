# SPDX-License-Identifier: Apache-2.0
"""Unwired whole-decoder program with one canonical packed state input.

By default large matrices are readonly inputs from the caller's BF16 pool.
The opt-in embedded mode uses checkpoint constants and native mixed lowering.
Layer boundaries explicitly materialize FP32 hidden values. This component
does not yet replace the serving decoder or its prefix-commit helper.
"""
import json
import garnet
import tensor
from .decoder_layer import T
from .cuda_chunk import Qwen35CUDAChunk

def packed_view(value,start,width,shape):
    return value*T.unary_op('qwen35_packed_view',start=start,width=width,shape=shape)
def concat(values):
    result=values[0]
    for value in values[1:]:result=result*T.binary_op('qwen35_concat_packed')*value
    return result

def flat_concat(values):
    return values[0]*T.unary_op('qwen35_concat_parts',parts=values[1:]) if len(values)>1 else values[0]

def decoder_state_layout(config,batch,capacity):
    result=[];offset=0
    for index,kind in enumerate(config['layer_types']):
        if kind=='linear_attention':
            channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
            first=[batch,channels,config['linear_conv_kernel_dim']]
            second=[batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']]
        elif kind=='full_attention':first=second=[batch,config['num_key_value_heads'],capacity,config['head_dim']]
        else:raise ValueError('Unsupported packed decoder layer kind')
        widths=[]
        for shape in (first,second):
            width=1
            for d in shape[1:]:width*=d
            widths.append(width)
        result.append({'index':index,'kind':kind,'offset':offset,'state1_shape':first,'state2_shape':second,
            'state1_width':widths[0],'state2_width':widths[1]})
        offset+=sum(widths)
    return result,offset

class CUDAPackedDecoderProgram:
    def __init__(self,checkpoint,config,io,batch=1,capacity=512,chunk_size=4,retain_linear_inputs=False,projection_workspace_bytes=0,register_delta_128=False,flat_concat=False,join_input_projections=False,embedded_projection_weights=False):
        if batch not in (1,2,4,8) or not 32<=capacity<=8192 or chunk_size not in (4,8,16,32,64,128):raise ValueError('Unsupported packed decoder profile')
        if projection_workspace_bytes not in (0,16384):raise ValueError('Invalid projection workspace')
        self.config=config;self.io=io;self.batch=batch;self.capacity=capacity;self.size=chunk_size;self.retain=retain_linear_inputs
        self.register_delta_128=register_delta_128;self.flat_concat=flat_concat
        self.embedded_projection_weights=bool(embedded_projection_weights)
        self.join_input_projections=bool(join_input_projections)
        if self.embedded_projection_weights and self.join_input_projections:raise ValueError('Embedded projections do not support BF16 joined input pools')
        if self.join_input_projections and chunk_size!=4:raise ValueError('Joined input projections require decode width4')
        self.workspace=projection_workspace_bytes;self.layout,self.state_width=decoder_state_layout(config,batch,capacity)
        self.hidden_width=chunk_size*config['hidden_size'];self.payload_width=0;self.weight_names=[];self.weight_layers=[];self.weights=[];self.graph=None
        symbols=[tensor.input('packed_hidden',shape=[batch,chunk_size,config['hidden_size']],dtype=tensor.float32),
            tensor.input('packed_state',shape=[batch,self.state_width],dtype=tensor.float32),
            tensor.input('packed_active',shape=[batch,chunk_size],dtype=tensor.float32),
            tensor.input('packed_positions',shape=[batch,chunk_size,3],dtype=tensor.float32),
            tensor.input('packed_slots',shape=[batch,chunk_size],dtype=tensor.float32)]
        for layer in self.layout:
            index=layer['index'];kind=layer['kind'];p='model.language_model.layers.'+str(index)+'.'
            suffixes=['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
            joined=None;joined_name=p+'linear_attn.in_proj_joined.weight'
            if kind=='linear_attention' and self.join_input_projections:
                joined=io.group_weights([p+'linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')])
                suffixes+=['linear_attn.in_proj_joined.weight','linear_attn.out_proj.weight']
            else:suffixes+=['linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')] if kind=='linear_attention' else ['self_attn.'+name+'.weight' for name in ('q_proj','k_proj','v_proj','o_proj')]
            indices=[]
            # Empty external bindings deliberately select the existing native
            # checkpoint lookup/lowering, including mixed compressed constants.
            for suffix in ([] if self.embedded_projection_weights else suffixes):
                name=p+suffix
                if joined is not None and name==joined_name:
                    rows=2*config['linear_num_key_heads']*config['linear_key_head_dim']+2*config['linear_num_value_heads']*config['linear_value_head_dim']+2*config['linear_num_value_heads']
                    info={'dtype':'BF16','shape':[rows,config['hidden_size']]};value=joined
                else:info=json.loads(io.weight_info(name));value=io.weight(name)
                if info['dtype']!='BF16' or len(info['shape'])!=2:raise ValueError('Packed decoder requires BF16 matrix weights')
                indices.append(len(self.weight_names));self.weight_names.append(name);self.weights.append(value)
                symbols.append(tensor.input('packed_weight_'+str(len(self.weight_names)),shape=info['shape'],dtype=tensor.bfloat16))
            self.weight_layers.append(indices)
            layer['payload_width']=0
            if retain_linear_inputs and kind=='linear_attention':
                channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                layer['payload_width']=chunk_size*(channels+config['linear_num_value_heads']*config['linear_value_head_dim']+2*config['linear_num_value_heads'])
                layer['payload_offset']=self.hidden_width+self.state_width+self.payload_width;self.payload_width+=layer['payload_width']
        self.output_width=self.hidden_width+self.state_width+self.payload_width
        self.graph=garnet.compiled_graph();self.graph.set_builder_workspace_bytes(256<<20)
        try:self.graph.build(self._forward()(*symbols),symbols,checkpoint);self.graph.enable_cuda_graph(True)
        except:self.close();raise

    def _forward(self):
        config=self.config;b=self.batch;t=self.size;h=config['hidden_size'];layout=self.layout;join=flat_concat if self.flat_concat else concat
        @T.fusion(name='qwen35_packed_decoder_program',boundary='required')
        def forward(hidden,state,active,positions,slots,*weights):
            state_outputs=[];payload_outputs=[]
            for layer,indices in zip(layout,self.weight_layers):
                w1=layer['state1_width'];w2=layer['state2_width'];offset=layer['offset']
                first=packed_view(state,offset,w1,layer['state1_shape']);second=packed_view(state,offset+w1,w2,layer['state2_shape'])
                external={self.weight_names[i]:weights[i] for i in indices}
                packed=Qwen35CUDAChunk(hidden,first,second,active,positions,slots,config,layer['index'],layer['kind'],True,True,
                    self.workspace,self.retain,external,self.register_delta_128,self.join_input_projections and layer['kind']=='linear_attention')
                hidden=packed_view(packed,0,t*h,[b,t,h])
                state_outputs.append(packed_view(packed,t*h,w1+w2,[b,w1+w2]))
                if layer['payload_width']:
                    payload_outputs.append(packed_view(packed,t*h+w1+w2,layer['payload_width'],[b,layer['payload_width']]))
                if layer['index']+1<len(layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
            flat=hidden*T.unary_op('reshape',shape=[b,t*h])
            return join([flat]+state_outputs+payload_outputs)
        return forward

    def run(self,inputs):
        if self.graph is None:raise ValueError('Packed decoder program is closed')
        if len(inputs)!=5:raise ValueError('Packed decoder requires hidden/state/active/positions/slots')
        return self.graph.run_device(list(inputs)+self.weights)

    def memory_info(self):
        if self.graph is None:raise ValueError('Packed decoder program is closed')
        return {'graph':json.loads(self.graph.stats_json()),'embedded_projection_weights':self.embedded_projection_weights,'register_delta_128':self.register_delta_128,'flat_concat':self.flat_concat,'joined_input_projections':self.join_input_projections,'state_width':self.state_width,'hidden_width':self.hidden_width,
            'payload_width':self.payload_width,'external_weight_count':len(self.weights),'fp32_layer_barriers':max(0,len(self.layout)-1),
            'scope':'Program graph only; embedded weights belong to its engine, external projection pools belong to caller; canonical state and request buffers are separate. Not process peak.'}

    def close(self):
        if self.graph is not None:self.graph.close()
        self.graph=None;self.weights=[];self.io=None
