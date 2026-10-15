"""Same-process exact 8192 versus 1024 seed; preserve original full-logit gates."""
from make_screened_seed_probe import generate_harness as seed_harness,generate_shader as seed_shader
def generate_shader(source):
    candidate=seed_shader(source,1024)
    control=seed_shader(source,8192).replace('prepare_head_bounds','control_prepare_head_bounds').replace('screened_head(','control_screened_head(')
    return candidate+'\n'+control
def generate_harness(source):
    source=seed_harness(source,1024)
    def change(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    change('NSString* folder=[NSString stringWithUTF8String:argv[2]];',
        'auto controlPrepare=pipeline(device,strictLibrary,@"control_prepare_head_bounds"),controlScreen=pipeline(device,strictLibrary,@"control_screened_head");\n        NSString* folder=[NSString stringWithUTF8String:argv[2]];')
    change('auto seed=[device newBufferWithLength:batch*seedRows*4 options:MTLResourceStorageModeShared];',
        'auto seed=[device newBufferWithLength:batch*seedRows*4 options:MTLResourceStorageModeShared];\n            auto controlSeed=[device newBufferWithLength:batch*8192*4 options:MTLResourceStorageModeShared];bool timingControl=false;')
    change('ArgmaxParams ap{batch,columns,columns,2};','Params controlSeedParams{batch,inner,8192,inner,8192};\n            ArgmaxParams ap{batch,columns,columns,2};')
    change('if(pruned) {','if(pruned||timingControl) {\n                    auto activeSeed=pruned?seed:controlSeed;const auto& activeSeedParams=pruned?seedParams:controlSeedParams;')
    change('encodeRaw(cb,seedParams,seed);','encodeRaw(cb,activeSeedParams,activeSeed);')
    change('[prep setComputePipelineState:preparePipeline];','[prep setComputePipelineState:pruned?preparePipeline:controlPrepare];')
    change('[encoder setComputePipelineState:prunePipeline];','[encoder setComputePipelineState:pruned?prunePipeline:controlScreen];')
    change('[prep setBuffer:seed offset:0 atIndex:1];','[prep setBuffer:activeSeed offset:0 atIndex:1];')
    change('[encoder setBuffer:seed offset:0 atIndex:7];','[encoder setBuffer:activeSeed offset:0 atIndex:7];')
    change('            auto commonWarm=std::chrono::steady_clock::now();','''            timingControl=true;run(false);
            {
                auto values=static_cast<const float*>(output.contents),full=static_cast<const float*>(reference.contents);
                auto flags=static_cast<const uint32_t*>(visits.contents);auto threshold=static_cast<const float*>(stats.contents);
                auto chosen=static_cast<const float*>(prediction.contents);
                for(uint64_t row=0;row<batch;++row){
                    for(uint64_t col=0;col<columns;++col){auto index=row*columns+col;
                        if(flags[columns+col]&(1u<<row)){if(std::memcmp(values+index,full+index,4))throw std::runtime_error("Control retained bits changed");}
                        else if(!(full[index]<threshold[row*(blocks+2)]))throw std::runtime_error("Control rejection can beat seed");
                    }
                    if(chosen[row*2]!=[query[@"predictions"][row] floatValue]||chosen[row*2+1]!=0)throw std::runtime_error("Control greedy mismatch");
                }
                printf("{\\"kind\\":\\"control_check\\",\\"query\\":\\"%s\\",\\"batch\\":%llu,\\"all_values_checked\\":true,\\"greedy_exact\\":true}\\n",name.UTF8String,batch);fflush(stdout);
            }
            auto commonWarm=std::chrono::steady_clock::now();''')
    source=source.replace('pruned?"pruned":"raw"','pruned?"seed1024":"seed8192"')
    return source
