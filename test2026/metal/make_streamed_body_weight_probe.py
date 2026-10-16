"""Prepare separate numerical and fixed64 own-null/comparison component modes."""
from make_streamed_body_weights import harness as base_harness,replace

def harness(source):
 source=base_harness(source)
 source=replace(source,'need(argc==3,"Expected assets and model");','need(argc==4,"Expected assets, model and numeric/timing mode");')
 source=replace(source,'controls();Garnet::MetalContext gpu;','bool timing=std::strcmp(argv[3],"timing")==0;need(timing||std::strcmp(argv[3],"numeric")==0,"Unknown mode");if(!timing)controls();Garnet::MetalContext gpu;')
 source=replace(source,'for(int pattern=0;pattern<4;++pattern)','if(!timing)for(int pattern=0;pattern<4;++pattern)')
 source=replace(source,'@"passed":@(passed)});}','@"passed":@(passed)});need(passed,"Original bits/numerical gate failed");}')
 start=source.index('        inputs(0,false);auto warm=');end=source.index('\n    }return 0;',start)
 timing='''        if(!timing)continue;
        inputs(0,false);auto warm=std::chrono::steady_clock::now();
        do{run(0);run(1);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.);
        for(bool null:{true,false})for(int round=0;round<10;++round)for(int position=0;position<4;++position){
            bool candidate=position==1||position==2;int actual=candidate&&!null?1:0;
            for(int warm=0;warm<3;++warm)run(actual);
            for(int sample=0;sample<5;++sample){auto t=run(actual);need(std::isfinite(t.first)&&std::isfinite(t.second)&&t.first>0&&t.second>0,"Invalid interval");
                emit(@{@"kind":@"timing",@"batch":@(batch),@"rows":@(rows),@"null":@(null),@"round":@(round),@"position":@(position),@"sample":@(sample),@"candidate":@(candidate),@"actual_policy":@(actual),@"gpu_ms":@(t.first),@"wall_ms":@(t.second)});
            }
        }'''
 source=source[:start]+timing+source[end:]
 return replace(source,'    }return 0;','    }emit(@{@"kind":@"complete",@"mode":timing?@"timing":@"numeric",@"numeric_checks":@(timing?0:8),@"timing_samples":@(timing?800:0),@"full_goal_met":@NO});return 0;')
