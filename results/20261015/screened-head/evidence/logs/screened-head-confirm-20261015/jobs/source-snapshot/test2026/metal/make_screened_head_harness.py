"""Reuse the complete captured-logit/greedy checks and original raw head control."""
from pathlib import Path
def generate(source):
    source=source.replace('wide?@"pruned_head_wide":@"pruned_head"','@"screened_head"')
    source=source.replace('remaining.length!=columns*(blocks+1)*4','remaining.length!=columns*4*4')
    needle='        for(NSDictionary* query in manifest[@"queries"]) {';assert source.count(needle)==1
    source=source.replace(needle,'        auto approximate=load(device,folder,@"approximate.bin");if(approximate.length!=columns*inner)throw std::runtime_error("Index byte mismatch");\n'+needle)
    needle='[encoder setBuffer:seed offset:0 atIndex:7];';assert source.count(needle)==1
    source=source.replace(needle,needle+'[encoder setBuffer:approximate offset:0 atIndex:8];')
    source=source.replace('uint64_t groupColumns=wide?16:4;','uint64_t groupColumns=16;')
    source=source.replace('\\"physical_weight_fraction\\"','\\"counted_weight_byte_fraction\\"').replace('double(reads)/double(columns*inner)','double((columns-seedRows)*inner+2*reads)/double(2*columns*inner)')
    needle='            for(int round=0;round<2;++round)'
    source=source.replace(needle,'''            auto commonWarm=std::chrono::steady_clock::now();size_t warmEpoch=0;
            do{run((warmEpoch++%2)!=0);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-commonWarm).count()<.25);
'''+needle)
    source=source.replace('std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());','''auto rawGpu=gpu,rawWall=wall;
                printf("{\\"kind\\":\\"samples\\",\\"query\\":\\"%s\\",\\"batch\\":%llu,\\"round\\":%d,\\"candidate\\":%s,\\"gpu_ms\\":[",name.UTF8String,batch,round,pruned?"true":"false");
                for(size_t i=0;i<rawGpu.size();++i)printf("%s%.12g",i?",":"",rawGpu[i]);printf("],\\"wall_ms\\":[");
                for(size_t i=0;i<rawWall.size();++i)printf("%s%.12g",i?",":"",rawWall[i]);puts("]}");
                std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());''')
    return source
if __name__=='__main__':
    import sys
    Path(sys.argv[1]).write_text(generate(Path(__file__).with_name('pruned_head_gpu.mm').read_text()))
