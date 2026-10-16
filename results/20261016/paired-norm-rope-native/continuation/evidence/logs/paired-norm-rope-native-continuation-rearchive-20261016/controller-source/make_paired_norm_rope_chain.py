"""Build a distinct 32-stage carried-output component workload from the frozen one-pair probe."""
def probe(source):
 def replace(old,new,count=1):
  nonlocal source
  assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)
 replace('std::array<std::array<std::vector<float>,2>,4> inputs,expected;', 'std::array<std::array<std::vector<float>,2>,4> inputs;std::array<std::vector<std::array<std::vector<float>,2>>,4> expected;')
 replace('auto query=gpu.Upload(inputs[0][0],rows,qHeads*dim),key=gpu.Upload(inputs[0][1],rows,kHeads*dim),qo=gpu.Allocate(rows,qHeads*dim),ko=gpu.Allocate(rows,kHeads*dim);', '''auto query=gpu.Upload(inputs[0][0],rows,qHeads*dim),key=gpu.Upload(inputs[0][1],rows,kHeads*dim);
    constexpr size_t chainStages=32;std::vector<std::array<Garnet::MetalMatrix,2>> chain;for(size_t stage=0;stage<chainStages;++stage)chain.push_back({gpu.Allocate(rows,qHeads*dim),gpu.Allocate(rows,kHeads*dim)});''')
 replace('gpu.Begin();if(candidate)need(gpu.EncodePairedNormRope(query,key,', '''gpu.Begin();for(size_t stage=0;stage<chainStages;++stage){const auto& qInput=stage?chain[stage-1][0]:query;const auto& kInput=stage?chain[stage-1][1]:key;const auto& qo=chain[stage][0];const auto& ko=chain[stage][1];if(candidate)need(gpu.EncodePairedNormRope(qInput,kInput,''')
 replace('gpu.EncodeNormRope(query,weights[0]', 'gpu.EncodeNormRope(qInput,weights[0]')
 replace('gpu.EncodeNormRope(key,weights[1]', 'gpu.EncodeNormRope(kInput,weights[1]')
 replace('epsilon,offset,normalize);}auto encoded=', 'epsilon,offset,normalize);}}auto encoded=')
 replace('==size_t(candidate),"Actual paired route differs"', '==(candidate?chainStages:0),"Actual paired chain route differs"')
 start=source.index('    auto validate=[&](int variant)');end=source.index('    for(int variant=0;',start)
 source=source[:start]+'''    auto validate=[&](int variant){std::array<std::vector<float>,2> final;
     need(expected[variant].size()==chainStages,"Original carried reference missing");
     for(size_t stage=0;stage<chainStages;++stage){std::array<std::vector<float>,2> actual{gpu.Download(chain[stage][0]),gpu.Download(chain[stage][1])};for(size_t side=0;side<2;++side){const auto& control=expected[variant][stage][side];need(actual[side].size()==control.size()&&!std::memcmp(actual[side].data(),control.data(),actual[side].size()*sizeof(float)),"Original carried norm-RoPE stage output bits differ");for(float v:actual[side])need(std::isfinite(v),"Nonfinite carried output");}if(stage+1==chainStages)final=std::move(actual);}return final;};
'''+source[end:]
 replace('expected[variant]={gpu.Download(qo),gpu.Download(ko)};', 'expected[variant].clear();for(size_t stage=0;stage<chainStages;++stage)expected[variant].push_back({gpu.Download(chain[stage][0]),gpu.Download(chain[stage][1])});')
 replace('bitChecks+=actual[0].size()+actual[1].size();[outputHashes addObject:@[hash(actual[0]),hash(actual[1])]];', 'bitChecks+=chainStages*(actual[0].size()+actual[1].size());auto hashes=[NSMutableArray new];for(const auto& stage:expected[variant])[hashes addObject:@[hash(stage[0]),hash(stage[1])]];[outputHashes addObject:hashes];')
 replace('@"output_sha256":outputHashes', '@"stage_output_sha256":outputHashes,@"chain_stages":@(chainStages),@"candidate_dispatches_per_execution":@(chainStages)')
 replace('@"complete_output_bits_checked_after_window":@YES', '@"all_stage_output_bits_checked_after_window":@YES,@"chain_stages":@(chainStages),@"paired_dispatches_per_execution":@(candidate?chainStages:0)')
 replace('@"original_bits_equal":@YES});\n   }}', '@"original_bits_equal":@YES,@"chain_stages":@(chainStages)});\n   }}')
 replace('Learned kernel component only; host fixtures are reused for kernel-level ABBA, no native/model speed proof', '32-stage carried learned component only; every stage output retained and checked, no native/model speed proof')
 return source
