"""Reconstruct each seed variant and audit every captured rejection and timing."""
from pathlib import Path
source=Path(__file__).with_name('verify_screened_head.py').read_text()
needle="for trial,status in proof['jobs'].items():\n    out=root/'evidence/logs'/trial/'jobs'";assert source.count(needle)==1
source=source.replace(needle,"trials=[(trial+'/jobs/seed-'+str(seed),status) for trial,status in proof['jobs'].items() for seed in (8192,1024,4096,32768)]\nfor trial,status in trials:\n    out=root/'evidence/logs'/trial")
source=source.replace("'logs/'+trial+'/jobs/","'logs/'+trial+'/")
needle="    expected='#include <metal_stdlib>";assert source.count(needle)==1
source=source.replace(needle,"    seed=meta['seed_rows'];assert seed in (8192,1024,4096,32768) and strict and bits==8\n    screen=screen.replace('min(ulong(8192),p.columns)','min(ulong('+str(seed)+'),p.columns)')\n"+needle)
needle="    if strict:assert 'strictOptions.fastMathEnabled=NO'";assert source.count(needle)==1
source=source.replace(needle,"    harness=harness.replace('std::min<uint64_t>(8192,columns)','std::min<uint64_t>('+str(seed)+',columns)')\n"+needle)
needle='result=dict(verified_files=';assert source.count(needle)==1
source=source.replace(needle,"for trial in proof['jobs']:\n    out=root/'evidence/logs'/trial/'jobs';outer=json.loads((out/'statuses.json').read_text())\n    assert len(outer)==4 and {p['seed_rows'] for p in outer}=={8192,1024,4096,32768} and all(p['exit_code']==0 for p in outer)\n    combined=json.loads((out/'summary.json').read_text())['profiles']\n    assert combined==[dict(seed_rows=seed,**p) for seed in (8192,1024,4096,32768) for p in json.loads((out/('seed-'+str(seed))/'summary.json').read_text())['profiles']]\n"+needle)
exec(compile(source,__file__,'exec'))
