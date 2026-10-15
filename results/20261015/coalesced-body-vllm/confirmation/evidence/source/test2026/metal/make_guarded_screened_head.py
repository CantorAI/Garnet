"""Separate strict-math filter kernels and original default-math control/argmax."""
from pathlib import Path
import runpy
def generate_shader(source):
    source=source.replace('threadgroup float partial[4];float best=-INFINITY;','threadgroup float partial[4];threadgroup uint invalid[4];float best=-INFINITY;')
    old='    float square=0;for(ulong j=tid;j<p.inner;j+=128){float x=input[row*p.aStride+j];square+=x*x;}'
    assert source.count(old)==1
    source=source.replace(old,'''    float square=0;uint bad=0;
    for(ulong j=tid;j<p.inner;j+=128){float x=input[row*p.aStride+j];bad|=uint(!isfinite(x)||abs(x)>1e6f);square=fma(x,x,square);}
    bad=simd_max(bad);if(lane==0)invalid[simd]=bad;''')
    old='    if(tid==0)stats[row*stride+1]=sqrt(partial[0]+partial[1]+partial[2]+partial[3])*1.01f+1e-20f;';assert source.count(old)==1
    source=source.replace(old,'''    if(tid==0){
        float norm=sqrt(partial[0]+partial[1]+partial[2]+partial[3])*1.01f+1e-20f;
        stats[row*stride+1]=norm;
        stats[row*stride+2]=float(!(invalid[0]|invalid[1]|invalid[2]|invalid[3])&&isfinite(norm)&&norm>=1e-10f&&norm<=1e10f);
    }''')
    source=source.replace('sum0+=x[j]*w;','sum0=fma(x[j],w,sum0);').replace('sum1+=y[j]*w;','sum1=fma(y[j],w,sum1);')
    source=source.replace('keep0[j]=!(sum0[j]*m.w+stats[1]*error+1e-6f<stats[0]);','keep0[j]=stats[2]==0||!(sum0[j]*m.w+stats[1]*error+1e-6f<stats[0]);')
    source=source.replace('keep1[j]=!(sum1[j]*m.w+stats[stride+1]*error+1e-6f<stats[stride]);','keep1[j]=stats[stride+2]==0||!(sum1[j]*m.w+stats[stride+1]*error+1e-6f<stats[stride]);')
    return source
def generate_harness(source):
    source=runpy.run_path(str(Path(__file__).with_name('make_screened_head_harness.py')))['generate'](source)
    old='        auto rawPipeline=pipeline(device,library,@"bf16_decode"),preparePipeline=pipeline(device,library,@"prepare_head_bounds");'
    assert source.count(old)==1
    source=source.replace(old,'''        auto strictOptions=[MTLCompileOptions new];strictOptions.fastMathEnabled=NO;
        auto strictLibrary=[device newLibraryWithSource:source options:strictOptions error:&error];
        if(!strictLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto rawPipeline=pipeline(device,library,@"bf16_decode"),preparePipeline=pipeline(device,strictLibrary,@"prepare_head_bounds");''')
    source=source.replace('pipeline(device,library,@"screened_head")','pipeline(device,strictLibrary,@"screened_head")')
    needle='        for(NSDictionary* query in manifest[@"queries"]) {'
    source=source.replace(needle,'''        auto metadata=static_cast<const float*>(remaining.contents);
        for(uint64_t row=0;row<columns;++row){
            for(int j=0;j<4;++j)if(!std::isfinite(metadata[row*4+j])||metadata[row*4+j]<0||metadata[row*4+j]>1e6f)throw std::runtime_error("Screen metadata outside admitted finite range");
            if(!(metadata[row*4+3]>0))throw std::runtime_error("Screen scale must be positive");
        }
'''+needle)
    needle='            auto commonWarm=std::chrono::steady_clock::now();';assert source.count(needle)==1
    source=source.replace(needle,'''            if([name isEqualToString:@"b1-step0"]||[name isEqualToString:@"b2-step0"]){
                std::vector<float> saved(batch*inner);std::memcpy(saved.data(),a.contents,saved.size()*4);
                for(int pattern=0;pattern<(batch==2?6:5);++pattern){
                    auto input=static_cast<float*>(a.contents);
                    for(uint64_t r=0;r<batch;++r)for(uint64_t j=0;j<inner;++j){
                        float v=pattern==0?0.f:pattern==1?1e-20f:pattern==2?1e20f:pattern==3?NAN:INFINITY;
                        input[r*inner+j]=(pattern==5&&r==1)?saved[r*inner+j]:v;
                    }
                    run(false);std::vector<float> full(batch*columns),rawPrediction(batch*2);
                    std::memcpy(full.data(),output.contents,full.size()*4);std::memcpy(rawPrediction.data(),prediction.contents,rawPrediction.size()*4);
                    run(true);auto actual=static_cast<const float*>(output.contents);auto flags=static_cast<const uint32_t*>(visits.contents);
                    auto bound=static_cast<const float*>(stats.contents);auto selected=static_cast<const float*>(prediction.contents);
                    for(uint64_t r=0;r<batch;++r){bool fallback=pattern!=5||r==0;
                        if(fallback&&bound[r*(blocks+2)+2]!=0)throw std::runtime_error("Unsafe input did not disable screening");
                        if(!fallback&&bound[r*(blocks+2)+2]!=1)throw std::runtime_error("Safe mixed row unexpectedly disabled screening");
                        for(uint64_t c=0;c<columns;++c){uint64_t i=r*columns+c;bool kept=flags[columns+c]&(1u<<r);
                            if(fallback&&!kept)throw std::runtime_error("Fallback dropped a vocabulary row");
                            if(kept){if(std::isfinite(full[i])){if(std::memcmp(actual+i,full.data()+i,4))throw std::runtime_error("Finite fallback/refinement changed bits");}
                                else if(std::isnan(full[i])?!std::isnan(actual[i]):actual[i]!=full[i])throw std::runtime_error("Nonfinite fallback changed classification");
                            }else if(!(full[i]<bound[r*(blocks+2)]))throw std::runtime_error("Mixed-row rejection can beat seed");
                        }
                        if(selected[r*2+1]!=rawPrediction[r*2+1]||(selected[r*2+1]==0&&selected[r*2]!=rawPrediction[r*2]))throw std::runtime_error("Fallback greedy/error flag mismatch");
                    }
                    printf("{\\"kind\\":\\"fallback_check\\",\\"batch\\":%llu,\\"pattern\\":%d,\\"all_values_checked\\":true,\\"greedy_and_error_flags_match\\":true}\\n",batch,pattern);fflush(stdout);
                }
                std::memcpy(a.contents,saved.data(),saved.size()*4);run(true);
            }
'''+needle)
    return source
