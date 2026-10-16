"""Change only exact seed width; retain strict bounds and full-row checks."""
from make_guarded_screened_head import generate_harness as strict_harness,generate_shader as strict_shader
SEEDS=(1024,4096,8192,32768)
def generate_harness(source,seed):
    assert seed in SEEDS
    source=strict_harness(source)
    needle='std::min<uint64_t>(8192,columns)';assert source.count(needle)==1
    return source.replace(needle,'std::min<uint64_t>('+str(seed)+',columns)')
def generate_shader(source,seed):
    assert seed in SEEDS
    source=strict_shader(source)
    needle='min(ulong(8192),p.columns)';assert source.count(needle)==2
    return source.replace(needle,'min(ulong('+str(seed)+'),p.columns)')
