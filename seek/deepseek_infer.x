from garnet import garnet
import CpuTensor as T
import json
from xlang_os import fs

import numpy as np

# class Linear:
#     def Linear(in_features, out_features, bias=True):
#         this.in_features = in_features
#         this.out_features = out_features
#         this.bias = bias

#         # 初始化权重和梯度
#         this.weight = np.random.randn(out_features, in_features) * np.sqrt(2.0 / (in_features + out_features))
#         this.weight_grad = np.zeros_like(this.weight)
        
#         if this.bias:
#             this.bias = np.zeros(out_features)
#             this.bias_grad = np.zeros_like(this.bias)
#         else:
#             this.bias = None

#     def forward(x):
#         this.input = x  # 保存输入用于反向传播
#         output = np.dot(x, this.weight.T)
#         if this.bias is not None:
#             output += this.bias
#         return output


# class PreTrainedModel:#(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
#     config_class = None
#     base_model_prefix = ""
#     main_input_name = "input_ids"
#     model_tags = None

# class DeepseekPreTrainedModel(PreTrainedModel):
#     config_class = DeepseekConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["DeepseekDecoderLayer"]
#     _skip_keys_device_placement = "past_key_values"
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_cache_class = True

#     def _init_weights(module):
#         std = this.config.initializer_range
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()

# class DeepseekModel(DeepseekPreTrainedModel):
#     def DeepseekModel(config: DeepseekConfig):
#         super(config）
#         this.padding_idx = config.pad_token_id
#         this.vocab_size = config.vocab_size

#         this.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, this.padding_idx)
#         this.layers = nn.ModuleList(
#             [DeepseekDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         this._use_sdpa = config._attn_implementation == "sdpa"
#         this._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
#         this.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#         this.gradient_checkpointing = False
#         # Initialize weights and apply final processing
#         this.post_init()

#     def get_input_embeddings():
#         return this.embed_tokens

#     def set_input_embeddings(value):
#         this.embed_tokens = value

#     def forward(
#         input_ids: tensor = None,
#         attention_mask: tensor = None,
#         position_ids: tensor = None,
#         past_key_values: [List[tensor]] = None,
#         inputs_embeds: [tensor] = None,
#         use_cache: [bool] = None,
#         output_attentions: [bool] = None,
#         output_hidden_states: [bool] = None,
#         return_dict: [bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else this.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else this.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else this.config.use_cache

#         return_dict = return_dict if return_dict is not None else this.config.use_return_dict

#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape[:2]
#         elif inputs_embeds is not None:
#             batch_size, seq_length = inputs_embeds.shape[:2]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         if this.gradient_checkpointing and this.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
#                 )
#                 use_cache = False

#         past_key_values_length = 0
#         if use_cache:
#             use_legacy_cache = not isinstance(past_key_values, Cache)
#             if use_legacy_cache:
#                 past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#             past_key_values_length = past_key_values.get_usable_length(seq_length)

#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0)

#         if inputs_embeds is None:
#             inputs_embeds = this.embed_tokens(input_ids)

#         if this._use_flash_attention_2:
#             # 2d mask is passed through the layers
#             attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
#         elif this._use_sdpa and not output_attentions:
#             # output_attentions=True can not be supported when using SDPA, and we fall back on
#             # the manual implementation that requires a 4D causal mask in all cases.
#             attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
#                 attention_mask,
#                 (batch_size, seq_length),
#                 inputs_embeds,
#                 past_key_values_length,
#             )
#         else:
#             # 4d mask is passed through the layers
#             attention_mask = _prepare_4d_causal_attention_mask(
#                 attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
#             )

#         # embed positions
#         hidden_states = inputs_embeds

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in this.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if this.gradient_checkpointing and this.training:
#                 layer_outputs = this._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     attention_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = this.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = None
#         if use_cache:
#             next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )


# class DeepseekForCausalLM(DeepseekPreTrainedModel):
#     _tied_weights_keys = ["lm_head.weight"]

#     def DeepseekForCausalLM(config):
#         super().__init__(config)
#         this.model = DeepseekModel(config)
#         this.vocab_size = config.vocab_size
#         this.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         this.post_init()

#     def get_input_embeddings():
#         return this.model.embed_tokens

#     def set_input_embeddings(value):
#         this.model.embed_tokens = value

#     def get_output_embeddings():
#         return this.lm_head

#     def set_output_embeddings(new_embeddings):
#         this.lm_head = new_embeddings

#     def set_decoder(decoder):
#         this.model = decoder

#     def get_decoder():
#         return this.model

#     def forward(
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, DeepseekForCausalLM

#         >>> model = DeepseekForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
#         >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else this.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else this.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else this.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = this.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = outputs[0]
#         if this.config.pretraining_tp > 1:
#             lm_head_slices = this.lm_head.weight.split(this.vocab_size // this.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(this.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = this.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, this.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def prepare_inputs_for_generation(
#         input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
#     ):
#         if past_key_values is not None:
#             if isinstance(past_key_values, Cache):
#                 cache_length = past_key_values.get_seq_length()
#                 past_length = past_key_values.seen_tokens
#                 #max_cache_length = past_key_values.get_max_length()
#                 max_cache_length = past_key_values.get_max_cache_shape()

#             else:
#                 cache_length = past_length = past_key_values[0][0].shape[2]
#                 max_cache_length = None

#             # Keep only the unprocessed tokens:
#             # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
#             # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
#             # input)
#             if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
#                 input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
#             # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
#             # input_ids based on the past_length.
#             elif past_length < input_ids.shape[1]:
#                 input_ids = input_ids[:, past_length:]
#             # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

#             # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
#             if (
#                 max_cache_length is not None
#                 and attention_mask is not None
#                 and cache_length + input_ids.shape[1] > max_cache_length
#             ):
#                 attention_mask = attention_mask[:, -max_cache_length:]

#         position_ids = kwargs.get("position_ids", None)
#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             if past_key_values:
#                 position_ids = position_ids[:, -input_ids.shape[1] :]

#         # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#         if inputs_embeds is not None and past_key_values is None:
#             model_inputs = {"inputs_embeds": inputs_embeds}
#         else:
#             model_inputs = {"input_ids": input_ids}

#         model_inputs.update(
#             {
#                 "position_ids": position_ids,
#                 "past_key_values": past_key_values,
#                 "use_cache": kwargs.get("use_cache"),
#                 "attention_mask": attention_mask,
#             }
#         )
#         return model_inputs

#     def _reorder_cache(past_key_values, beam_idx):
#         reordered_past = ()
#         for layer_past in past_key_values:
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
#             )
#         return reordered_past


#

def scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False):
    # 获取键向量的维度
    d_k = key_states.shape[-1]

    # # 计算注意力分数
    attn_scores = np.matmul(query_states, key_states.transpose(0, 2, 1)) / np.sqrt(d_k)

    # # 应用因果掩码（如果需要）
    # if is_causal:
    #     seq_len_q, seq_len_k = query_states.shape[1], key_states.shape[1]
    #     causal_mask = np.triu(np.ones((seq_len_q, seq_len_k)), k=1) * -1e9  # 上三角掩码
    #     attn_scores = attn_scores + causal_mask[None, :, :]  # 广播到 batch 维度

    # # 应用注意力掩码（如果提供）
    # if attn_mask is not None:
    #     attn_scores = attn_scores + attn_mask

    # # 计算注意力权重
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))  # 数值稳定性
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # # 加权求和
    output = np.matmul(attn_weights, value_states)
    output = None

    return output

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = np.expand_dims(hidden_states, axis=2)  # 添加第3维 (None)
    hidden_states = np.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# class DeepseekRMSNorm():
#     def DeepseekRMSNorm(hidden_size, eps=1e-6):
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)
def deepseekRMSNorm(norm_weight, hidden_states):
    variance_epsilon = 0.000001#1e-6
    input_dtype = np.float16  # 假设原始数据类型是 float16
    hidden_states_f32 = hidden_states.astype(np.float32)
    variance = np.mean(hidden_states_f32 ** 2, axis=-1, keepdims=True)
    hidden_states_norm = hidden_states_f32 * (1 / np.sqrt(variance + variance_epsilon))
    hidden_states_final = norm_weight * hidden_states_norm.astype(input_dtype)
    return hidden_states_final

class DeepseekMLP:
    gate_proj
    up_proj
    down_proj
    act_fn
    def DeepseekMLP(cfg, weight_file,layer_idx):
        this.gate_proj = weight_file[f'model.layers.{layer_idx}.mlp.gate_proj.weight']
        this.up_proj = weight_file[f'model.layers.{layer_idx}.mlp.up_proj.weight']
        this.down_proj = weight_file[f'model.layers.{layer_idx}.mlp.down_proj.weight']
        this.act_fn = ACT2FN[config.hidden_act]
    def deepseekMLP():
        down_proj = this.down_proj(this.act_fn(this.gate_proj(x)) * this.up_proj(x))
        return down_proj


class DeepseekMoE:
    num_experts_per_tok
    experts
    gate
    shared_experts
    def DeepseekMoE(cfg,weight_file, layer_idx):
        this.num_experts_per_tok = config.num_experts_per_tok
        moe_intermediate_size = config['moe_intermediate_size']
        n_shared_experts = config['n_shared_experts']
        this.experts = nn.ModuleList([DeepseekMLP(cfg, intermediate_size = moe_intermediate_size, layer_idx) for i in range(config.n_routed_experts)])
        this.gate = MoEGate(cfg)
        intermediate_size = moe_intermediate_size * n_shared_experts
        this.shared_experts = DeepseekMLP(cfg=cfg, intermediate_size = intermediate_size,layer_idx)

    def moe_infer_numpy(x, flat_expert_indices, flat_expert_weights):
        expert_cache = np.zeros_like(x)
        
        idxs = flat_expert_indices.argsort()
        expert_token_counts = np.bincount(flat_expert_indices, minlength=self.num_experts)
        tokens_per_expert = expert_token_counts.cumsum()
        token_idxs = idxs // this.num_experts_per_tok
        for i in range(this.num_experts):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            end_idx = tokens_per_expert[i]
            if start_idx >= end_idx:
                continue  # 当前专家无 token 需要处理
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = this.experts[i](expert_tokens)
            expert_out *= flat_expert_weights[idxs[start_idx:end_idx], None]  # 保持维度
            np.add.at(expert_cache, exp_token_idx, expert_out)
        return expert_cache

    def deepseekMoE(hidden_states):
        
        # forward
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y




# text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
# inputs = tokenizer(text, return_tensors="pt")
# inputs_data = inputs.copy().data
# inputs_data['input_ids'] = inputs_data['input_ids'].tolist()
# inputs_data['attention_mask'] = inputs_data['attention_mask'].tolist()
# with open('inputs_data.json', 'w') as f:
#     json.dump(inputs_data, f, indent=4)

### 1. loading input tokens
def load_json(filename):
    fileObj = fs.File(filename,"r")
    content = fileObj.read(fileObj.size)
    fileObj.close()
    #print(content)
    inputs = json.loads(content,normalize=True)
    return inputs

inputs_filename='./data/inputs_data.json'
inputs = load_json(inputs_filename)
print("inputs:",inputs)


### 2. loading model weights
m001 = garnet.loadModel("C:/df/moe-16b/model-00001-of-00007.bin")

### 3. forward
config = load_json('C:/df/moe-16b/config.json')
num_hidden_layers = config['num_hidden_layers']
vocab_size = config['vocab_size']
hidden_size = config['hidden_size']
attention_dropout = config['attention_dropout']
num_heads = config['num_attention_heads']
head_dim = hidden_size // self.num_heads
num_key_value_heads = config['num_key_value_heads']
num_key_value_groups = self.num_heads // self.num_key_value_heads
max_position_embeddings = config['max_position_embeddings']
rope_theta = config['rope_theta']
first_k_dense_replace=config['first_k_dense_replace']
moe_layer_freq = config['moe_layer_freq']


input_ids = tensor(inputs['input_ids'])
position_ids =tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39]])
attention_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

model_embed_tokens_weight = m001['model.embed_tokens.weight']
#embed_tokens = nn.Embedding(vocab_size, hidden_size, 0)
inputs_embeds = model_embed_tokens_weight[input_ids]

hidden_states = inputs_embeds
for layer_idx in range(num_hidden_layers):
    input_layernorm =  m001[f'model.layers.{layer_idx}.input_layernorm.weight']
    residual = hidden_states
    ###########hidden_states = input_layernorm(hidden_states)
    input_layernorm_weight = m001[f'model.layers.{layer_idx}.input_layernorm.weight']
    hidden_states = deepseekRMSNorm(input_layernorm_weight, hidden_states)
    # Self Attention
    #self_attn =  m001[f'model.layers.{i}.input_layernorm.weight']
    bsz, q_len, _ = hidden_states.size()
    q_proj = m001[f'model.layers.{layer_idx}.self_attn.q_proj.weight']
    k_proj = m001[f'model.layers.{layer_idx}.self_attn.k_proj.weight']
    v_proj = m001[f'model.layers.{layer_idx}.self_attn.v_proj.weight']
    o_proj = m001[f'model.layers.{layer_idx}.self_attn.o_proj.weight']
    query_states = q_proj(hidden_states)
    key_states = k_proj(hidden_states)
    value_states = v_proj(hidden_states)

    # query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    query_states = query_states.reshape(bsz, q_len, num_heads, head_dim).swapaxes(1, 2)
    key_states = key_states.reshape(bsz, q_len, num_key_value_heads, head_dim).swapaxes(1, 2)
    value_states = value_states.reshape(bsz, q_len, num_key_value_heads, head_dim).swapaxes(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, layer_idx)
    
    # cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = 0, 0
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_output = scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal= attention_mask is None and q_len > 1
    )

    # attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    attn_output = attn_output.swapaxes(1, 2)  # 交换维度 1 和 2
    attn_output = np.ascontiguousarray(attn_output)  # 确保内存连续性（可选）
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    attn_output = o_proj(attn_output)
    hidden_states = attn_output

    hidden_states = residual + hidden_states
    # Fully Connected
    residual = hidden_states
    #hidden_states = post_attention_layernorm(hidden_states)
    post_attention_layernorm_weight = m001[f'model.layers.{layer_idx}.post_attention_layernorm.weight']
    hidden_states = deepseekRMSNorm(post_attention_layernorm_weight, hidden_states)
    #hidden_states = mlp(hidden_states)
    # self.mlp = DeepseekMoE(config) if (config.n_routed_experts is not None and  \
    #                                 layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0) \
    #                             else DeepseekMLP(config)
    if (n_routed_experts is not None and  layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0):
        deepseekMoE = DeepseekMoE(config, m001, layer_idx)
        hidden_states = deepseekMoE(hidden_states)
    else:
        deepseekMoE = DeepseekMLP(config, m001, layer_idx)
        gate_proj = m001[f'model.layers.{layer_idx}.mlp.gate_proj.weight']
        up_proj = m001[f'model.layers.{layer_idx}.mlp.up_proj.weight']
        down_proj = m001[f'model.layers.{layer_idx}.mlp.down_proj.weight']
        act_fn = ACT2FN[config.hidden_act]
        hidden_states = deepseekMLP(hidden_states)
    hidden_states = residual + hidden_states

#outputs = this_norm(hidden_states)

y_graph = T.graph(outputs)
y_graph.run()

print('hello1', m001)
print('hello2')