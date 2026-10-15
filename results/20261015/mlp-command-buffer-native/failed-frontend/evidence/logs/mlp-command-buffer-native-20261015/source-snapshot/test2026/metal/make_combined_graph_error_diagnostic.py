"""Derive per-projection error reporting without changing the tested graph math."""
from pathlib import Path
import sys

source=Path(__file__).with_name('combined_graph_prefill_gpu.mm').read_text()
source=source.replace('size_t n,k;Garnet::MetalBF16Weight', 'std::string name;size_t n,k;Garnet::MetalBF16Weight')
needle='bool key=name.rfind';assert source.count(needle)==1
source=source.replace(needle,'p.name=name;bool key=name.rfind')
start=source.index('                for(auto& p:projections){auto actual=')
end=source.index('                bool passed=finite',start)
source=source[:start]+'''                size_t matrix=0;
                for(auto& p:projections){
                    auto actual=static_cast<const float*>(p.graphOutput[policy-1].contents);
                    double pw=0,pc=0,cw=0;size_t at=0;bool pf=true;
                    for(size_t i=0;i<rows*p.n;++i){
                        cw=std::max(cw,std::abs(double(p.control[i])-p.exact[i])/p.bound[i]);
                        if(!std::isfinite(actual[i])){pf=false;continue;}
                        pw=std::max(pw,std::abs(double(actual[i])-p.exact[i])/p.bound[i]);
                        double ratio=std::abs(double(actual[i])-p.control[i])/(5e-6+5e-5*std::abs(double(p.control[i])));
                        if(ratio>pc){pc=ratio;at=i;}
                    }
                    finite&=pf;worst=std::max(worst,pw);comparison=std::max(comparison,pc);
                    emit(@{@"kind":@"projection_check",@"batch":@(batch),@"rows":@(rows),@"policy":@(policy),@"pattern":@(pattern),
                        @"matrix":@(matrix++),@"tensor":[NSString stringWithUTF8String:p.name.c_str()],@"columns":@(p.n),@"inner":@(p.k),@"values":@(rows*p.n),
                        @"control_fp64_worst":@(cw),@"fp64_worst":@(pw),@"comparison_worst":@(pc),@"finite":@(pf),@"passed":@(pf&&cw<=1&&pw<=1&&pc<=1),
                        @"worst_index":@(at),@"control_at_worst":@(p.control[at]),@"candidate_at_worst":@(actual[at]),@"fp64_at_worst":@(p.exact[at]),@"fp64_bound_at_worst":@(p.bound[at]),
                        @"comparison_bound_at_worst":@(5e-6+5e-5*std::abs(double(p.control[at])))});
                }
'''+source[end:]
start=source.index('        inputs(0,false);auto warm=')
end=source.index('    }}return 0;',start)
source=source[:start]+source[end:]
Path(sys.argv[1]).write_bytes(source.encode())
