"""Interleave four output columns for one vector load; preserve index values/order."""
from make_screened_seed_abba import generate_harness as base_harness
from make_screened_seed_probe import generate_shader as seed_shader
def generate_shader(source):
    control=seed_shader(source,1024).replace('prepare_head_bounds','control_prepare_head_bounds').replace('screened_head(','control_screened_head(')
    candidate=seed_shader(source,1024)
    start=candidate.index('        char4 b0=');end=candidate.index('\n    }',start)
    candidate=candidate[:start]+'''        uint4 packed=*(device const uint4*)(approximate+column*p.inner+k*4);
        for(uint j=0;j<4;++j){float4 w=float4(as_type<char4>(packed[j]));sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}
'''+candidate[end:]
    return candidate+'\n'+control
def generate_harness(source):
    source=base_harness(source)
    def change(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    change('batch*8192*4','batch*1024*4')
    change('Params controlSeedParams{batch,inner,8192,inner,8192};','Params controlSeedParams{batch,inner,1024,inner,1024};')
    needle='        auto metadata=static_cast<const float*>(remaining.contents);'
    change(needle,'''        uint64_t paddedColumns=(columns+3)/4*4;
        auto interleaved=[device newBufferWithLength:paddedColumns*inner options:MTLResourceStorageModeShared];
        if(!interleaved)throw std::runtime_error("Interleaved index allocation failed");
        auto originalIndex=static_cast<const int8_t*>(approximate.contents);auto packedIndex=static_cast<int8_t*>(interleaved.contents);
        for(uint64_t group=0;group<paddedColumns/4;++group)for(uint64_t k=0;k<inner;++k)for(uint64_t c=0;c<4;++c){uint64_t row=group*4+c;packedIndex[(group*inner+k)*4+c]=row<columns?originalIndex[row*inner+k]:0;}
        for(uint64_t row=0;row<columns;++row)for(uint64_t k=0;k<inner;++k)if(packedIndex[((row/4)*inner+k)*4+row%4]!=originalIndex[row*inner+k])throw std::runtime_error("Interleaved index changed a value");
        printf("{\\"kind\\":\\"index_roundtrip\\",\\"elements\\":%llu,\\"all_values_exact\\":true,\\"storage\\":\\"interleaved-int8\\",\\"padded_columns\\":%llu}\\n",columns*inner,paddedColumns);fflush(stdout);
'''+needle)
    change('[encoder setBuffer:approximate offset:0 atIndex:8];','[encoder setBuffer:pruned?interleaved:approximate offset:0 atIndex:8];')
    change('double((columns-seedRows)*inner+2*reads)/double(2*columns*inner)','double((paddedColumns-seedRows)*inner+2*reads)/double(2*columns*inner)')
    change('            timingControl=true;run(false);','''            std::vector<float> candidateMasked(batch*columns);std::memcpy(candidateMasked.data(),output.contents,batch*columns*4);
            std::vector<uint32_t> candidateFlags(columns*2);std::memcpy(candidateFlags.data(),visits.contents,columns*2*4);
            timingControl=true;run(false);
            if(std::memcmp(candidateMasked.data(),output.contents,batch*columns*4)||std::memcmp(candidateFlags.data(),visits.contents,columns*2*4))throw std::runtime_error("Interleaved masked output or flags changed bits");
            printf("{\\"kind\\":\\"layout_check\\",\\"query\\":\\"%s\\",\\"batch\\":%llu,\\"masked_values_and_flags_bit_exact\\":true}\\n",name.UTF8String,batch);fflush(stdout);''')
    source=source.replace('pruned?"seed1024":"seed8192"','pruned?"interleaved-index":"row-index"')
    return source
