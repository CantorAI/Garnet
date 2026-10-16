"""Private text-only graph/session integration of confirmed exact heterogeneous QKV."""
import make_prepared_union_qkv as component
def replace(source,old,new,count=1):
    assert source.count(old)==count,(old,source.count(old),count);return source.replace(old,new)
GRAPH_METHODS='''
void MetalCompiledGraph::FindPreparedQkvGroups(){
    preparedQkvStarts_.clear();
#ifdef GARNET_METAL_PREPARED_QKV_TEXT
    const std::string suffix=".q_proj.weight|BF16";std::unordered_map<size_t,std::string> names;
    for(const auto& entry:weights_)names.emplace(entry.second,entry.first);
    auto projection=[&](const Op& op){return (op.kind==Kind::BF16MatVec||(op.kind==Kind::MatMul&&op.transpose))&&slots_[op.b].bf16.Valid();};
    for(size_t i=0;i+2<ops_.size();++i){const auto& q=ops_[i];const auto& k=ops_[i+1];const auto& v=ops_[i+2];
        if(!projection(q)||!projection(k)||!projection(v)||q.a!=k.a||q.a!=v.a||q.out==k.out||q.out==v.out||k.out==v.out)continue;
        auto found=names.find(q.b);if(found==names.end())continue;const auto& name=found->second;
        if(name.rfind("model.layers.",0)!=0||name.size()<suffix.size()||name.compare(name.size()-suffix.size(),suffix.size(),suffix)!=0)continue;
        auto stem=name.substr(0,name.size()-suffix.size());auto kw=weights_.find(stem+".k_proj.weight|BF16"),vw=weights_.find(stem+".v_proj.weight|BF16");
        if(kw==weights_.end()||vw==weights_.end()||kw->second!=k.b||vw->second!=v.b)continue;
        // Confirmed Qwen3-0.6B geometry only. Every original output remains materialized.
        if(slots_[q.a].columns!=1024||slots_[q.b].rows!=2048||slots_[k.b].rows!=1024||slots_[v.b].rows!=1024||
            slots_[q.b].columns!=1024||slots_[k.b].columns!=1024||slots_[v.b].columns!=1024)continue;
        preparedQkvStarts_.insert(i);
    }
#endif
}
void MetalCompiledGraph::PrepareQkvGroups(MetalContext& gpu,size_t physicalRows){
#ifdef GARNET_METAL_PREPARED_QKV_TEXT
    for(auto i:preparedQkvStarts_){const auto& q=ops_[i];const auto& k=ops_[i+1];const auto& v=ops_[i+2];size_t rows=physicalRows?physicalRows:slots_[q.a].rows;
        if(rows==24||rows==44)gpu.PrepareUnionQkv(slots_[q.b].bf16,slots_[k.b].bf16,slots_[v.b].bf16,rows);
    }
#else
    (void)gpu;(void)physicalRows;
#endif
}
'''
def graph(source):
    source=replace(source,'void MetalCompiledGraph::Clear() {',GRAPH_METHODS+'\nvoid MetalCompiledGraph::Clear() {')
    source=replace(source,'void MetalCompiledGraph::Clear() {','void MetalCompiledGraph::Clear() {\n    preparedQkvStarts_.clear();')
    source=replace(source,'sharedWeights_.clear();ready_=true;','FindPreparedQkvGroups();\n        sharedWeights_.clear();ready_=true;')
    source=replace(source,'visionVectorDispatches_=0;mlpGraphDispatches_=0;','visionVectorDispatches_=0;mlpGraphDispatches_=0;preparedQkvDispatches_=0;',2)
    source=replace(source,'    PrepareInputs(inputs);\n    gpu_->Begin();','    PrepareInputs(inputs);\n    PrepareQkvGroups(*gpu_);\n    gpu_->Begin();')
    source=replace(source,'mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();','mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();\n        preparedQkvDispatches_=gpu_->LastPreparedUnionQkvDispatches();')
    needle='#ifdef GARNET_METAL_MLP_PREFILL_GRAPH'
    code='''#ifdef GARNET_METAL_PREPARED_QKV_TEXT
        if(!profileOps&&(slots_[op.a].rows==24||slots_[op.a].rows==44)&&preparedQkvStarts_.count(i)&&i+2<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]))){
            const auto& k=ops_[i+1];const auto& v=ops_[i+2];
            if(gpu.EncodePreparedUnionQkv(a,slots_[op.b].bf16,slots_[k.b].bf16,slots_[v.b].bf16,
                out,slots_[k.out].matrix,slots_[v.out].matrix)){i+=2;continue;}
        }
#endif
'''
    return replace(source,needle,code+needle)
def graph_header(source):
    source=replace(source,'size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0;','size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0,preparedQkvDispatches_=0;\n    std::unordered_set<size_t> preparedQkvStarts_;\n    void FindPreparedQkvGroups();\n    void PrepareQkvGroups(MetalContext&,size_t physicalRows=0);')
    needle='size_t LastMlpGraphDispatches() const {return mlpGraphDispatches_;}'
    source=replace(source,needle,needle+'\n    size_t LastPreparedQkvDispatches() const {return preparedQkvDispatches_;}')
    needle='APISET().AddProp("mlp_prefill_graph_dispatches",&MetalCompiledGraph::LastMlpGraphDispatches);'
    return replace(source,needle,needle+'\n        APISET().AddProp("prepared_qkv_dispatches",&MetalCompiledGraph::LastPreparedQkvDispatches);')
def session(source):
    source=replace(source,'joinedGlobalMlpBlocks_=0;','joinedGlobalMlpBlocks_=0;preparedQkvDispatches_=0;',4)
    source=replace(source,'joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();','joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();preparedQkvDispatches_=gpu_->LastPreparedUnionQkvDispatches();')
    needle='states_.push_back(std::move(state));owners_.push_back(std::move(owner));'
    return replace(source,needle,'''#ifdef GARNET_METAL_PREPARED_QKV_TEXT
            // Shared BF16 owners are also used by later prefill graphs. Prepare
            // on this resident context during setup, never inside an encoded frame.
            if(batch==1||batch==2)graph->PrepareQkvGroups(*gpu_,batch==1?24:44);
#endif
            '''+needle)
def session_header(source):
    source=replace(source,'size_t joinedGlobalMlpBlocks_=0;','size_t joinedGlobalMlpBlocks_=0,preparedQkvDispatches_=0;')
    needle='size_t LastJoinedGlobalMlpBlocks() const {return joinedGlobalMlpBlocks_;}'
    source=replace(source,needle,needle+'\n    size_t LastPreparedQkvDispatches() const {return preparedQkvDispatches_;}')
    needle='APISET().AddProp("joined_global_mlp_blocks",&MetalDecoderSession::LastJoinedGlobalMlpBlocks);'
    return replace(source,needle,needle+'\n        APISET().AddProp("prepared_qkv_dispatches",&MetalDecoderSession::LastPreparedQkvDispatches);')
def cmake(source):
    needle='option(GARNET_METAL_JOINED_GLOBAL_MLP '
    text='''option(GARNET_METAL_PREPARED_QKV_TEXT "Test prepared exact heterogeneous QKV for confirmed text profiles" OFF)
if(GARNET_METAL_PREPARED_QKV_TEXT)
    if(NOT GARNET_METAL_GLOBAL_HALF_ALPHA)
        message(FATAL_ERROR "Prepared QKV requires retained exact common-alpha prefill")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PREPARED_QKV_TEXT=1)
endif()
'''
    return replace(source,needle,text+needle)
def entry(source):
    needle='#ifdef GARNET_METAL_JOINED_GLOBAL_MLP'
    return replace(source,needle,'#ifdef GARNET_METAL_PREPARED_QKV_TEXT\n        info+="; prepared_qkv_text";\n#endif\n'+needle)
def decoder(source):
    needle="if self.global_half_alpha_enabled:self.prefill_profile['global_half_alpha_dispatches']=0"
    return replace(source,needle,needle+"\n        if 'prepared_qkv_text' in garnet.backend_info():self.prefill_profile['prepared_qkv_dispatches']=0")
def worker(source):
    needle="mlp_graph_enabled='mlp_prefill_graph' in garnet.backend_info()"
    source=replace(source,needle,needle+";qkv_enabled='prepared_qkv_text' in garnet.backend_info()")
    needle="mlp_graph_counts=[decoder.prefill_profile.get('mlp_prefill_graph_dispatches',0) if mlp_graph_enabled else 0]"
    source=replace(source,needle,needle+";qkv_counts=[decoder.prefill_profile.get('prepared_qkv_dispatches',0) if qkv_enabled else 0]")
    needle='mlp_graph_counts.append(decoder.session.mlp_prefill_graph_dispatches if mlp_graph_enabled else 0)'
    source=replace(source,needle,needle+';qkv_counts.append(decoder.session.prepared_qkv_dispatches if qkv_enabled else 0)')
    needle='mlp_prefill_graph_enabled=mlp_graph_enabled,mlp_prefill_graph_counts=mlp_graph_counts,'
    return replace(source,needle,needle+'\n            prepared_qkv_text_enabled=qkv_enabled,prepared_qkv_counts=qkv_counts,')
def dense_reference(source):
    needle='  h=session.hidden().tolist();state=session.snapshot(0);a=state[0].tolist();b=state[1].tolist();resident=[]'
    code='''  if 'prepared_qkv_text' in garnet.backend_info():
   expected_qkv=int(len(sys.argv)>4 and sys.argv[4]=='bf16' and c['hidden_size']==1024 and c['num_attention_heads']*c['head_dim']==2048 and c['num_key_value_heads']*c['head_dim']==1024 and batch*chunk in (24,44))
   assert session.prepared_qkv_dispatches==expected_qkv,'Resident QKV route mismatch'
   assert graph.prepared_qkv_dispatches==expected_qkv,'Explicit graph QKV route mismatch'
   print('prepared_qkv_trained_route '+json.dumps(dict(batch=batch,chunk=chunk,expected=expected_qkv,session=session.prepared_qkv_dispatches,graph=graph.prepared_qkv_dispatches)),flush=True)
'''
    return replace(source,needle,code+needle)
OPERATIONS={'src/backends/metal/metal_context.mm':component.context,'src/backends/metal/metal_context.h':component.header,
 'src/backends/metal/compiled_graph.cpp':graph,'src/backends/metal/compiled_graph.h':graph_header,
 'src/backends/metal/decoder_session.cpp':session,'src/backends/metal/decoder_session.h':session_header,
 'cmake/GarnetMetal.cmake':cmake,'src/entry/metal_main.cpp':entry,
 'xModel/qwen35/vl_0_8b/metal_decoder.py':decoder,'test2026/metal/frame_worker_garnet.py':worker,
 'test2026/metal/run_matched_garnet.py':worker,'test2026/metal/run_dense_reference.py':dense_reference}
