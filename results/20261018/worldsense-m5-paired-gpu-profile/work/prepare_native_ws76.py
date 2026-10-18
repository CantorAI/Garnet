"""Prepare unchanged-reference legacy guards and program dumps before GPU tests."""
from pathlib import Path
import hashlib, json, re, shlex, shutil, subprocess
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i')
w=b/'work/worldsense-paired-gpu-profile';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
assert load(w/'build-terminal.json')['all_six_builds_pass']
assert not (w/'native-preparation.json').exists()
prior=b/'work/worldsense-validated-text-worker/legacy_projection_test.cpp'
shutil.copy2(prior,w/'legacy_projection_test.cpp')
for name in ['restore_normal.py','wait_cantor.py','cases.json']:
    shutil.copy2(b/'work/worldsense-minimal-selective-defaults'/name,w/name)
assert sha(n/'libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
shutil.copy2(n/'libgarnet.dylib',w/'before-libgarnet.dylib')
dump='''#include "metal_context.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("output prefix required");
 Dl_info image{};
 if(!dladdr(reinterpret_cast<const void*>(&Garnet::MetalContext::AvailableDeviceName),&image)||!image.dli_fname||std::filesystem::weakly_canonical(image.dli_fname)!=std::filesystem::weakly_canonical(LEGACY_LIBRARY_PATH))throw std::runtime_error("program probe bound another library");
 Garnet::MetalContext gpu;auto programs=gpu.ShaderProgramSources();
 if(programs.size()!=2||programs[0].empty())throw std::runtime_error("program source contract failed");
 for(size_t i=0;i<2;++i){std::ofstream out(std::string(argv[1])+(i?".prefill.metal":".main.metal"),std::ios::binary);out.write(programs[i].data(),programs[i].size());out.flush();if(!out)throw std::runtime_error("cannot save full program source");}
 std::cout<<"Actual constructor MAIN/prefill programs saved; loaded library path verified\\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\\n';return 1;}}
'''
(w/'program_source_probe.cpp').write_text(dump)
record=dict(commands={},preprocessing={},legacy_reference_source=str(prior),legacy_reference_sha256=sha(prior),original_legacy_reference_source_exact=True)
for family in ['legacy','current']:
    s=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source'
    seed=load(w/(family+'-seed-source-manifest.json'));manifest=load(w/(family+'-source-manifest.json'))
    assert all(sha(s/k)==v for k,v in manifest.items())
    operations=load(w/(family+'-source-transformations.json'))
    for rel,ops in operations.items():
        text=(s/rel).read_text()
        for op in reversed(ops):
            if op.get('append'):
                assert text.endswith(op['after']);text=text[:-len(op['after'])]
            else:
                assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
        assert hashlib.sha256(text.encode()).hexdigest()==seed[rel]
    for mode in ['quiet','trace','fault']:
        role=family+'-'+mode;build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role);lib=build/'libgarnet.dylib'
        record['commands'][role]={}
        for kind in (['program','projection'] if family=='legacy' else ['program']):
            source=w/('program_source_probe.cpp' if kind=='program' else 'legacy_projection_test.cpp');exe=w/(role+'-'+kind+'-probe')
            cmd=['/usr/bin/c++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-I',str(s/'src/backends/metal'),str(source),'-DLEGACY_LIBRARY_PATH="'+str(lib)+'"',str(lib),str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(build),'-Wl,-rpath,'+str(n),'-o',str(exe)]
            with (w/('build-'+role+'-'+kind+'-probe.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            record['commands'][role][kind]=dict(command=cmd,source_sha256=sha(source),library_sha256=sha(lib),executable_sha256=sha(exe),no_GPU_or_model_dispatch=True)
    # Actual host preprocessing with the quiet role's generated flags.
    build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+family+'-quiet')
    flags=build/'CMakeFiles/garnet_metal.dir/flags.make';text=flags.read_text()
    values={key:shlex.split(re.search(r'^'+key+r' = (.*)$',text,re.M).group(1)) for key in ['OBJCXX_DEFINES','OBJCXX_INCLUDES','OBJCXX_FLAGS']}
    cmd=['/usr/bin/c++','-x','objective-c++','-E','-P']+sum(values.values(),[])+[str(s/'src/backends/metal/metal_context.mm')]
    completed=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True);expanded=completed.stdout.decode();a=expanded.index('void MetalContext::EncodeBF16MatVec(');z=expanded.index('void MetalContext::Update(',a);part=expanded[a:z]
    assert 'PassTraceEncoder' not in part and 'std::string("bf16:")' not in part and '[(*impl_).command computeCommandEncoder]' in part
    record['preprocessing'][family]=dict(command=cmd,flags_sha256=sha(flags),preprocessed_source_sha256=hashlib.sha256(completed.stdout).hexdigest(),projection_sha256=hashlib.sha256(part.encode()).hexdigest(),diagnostic_label_arguments_and_wrapper_absent=True,original_encoder_call_present=True,scope='Preprocessing only; no GPU ISA or timing equivalence claim')
assert sha(w/'legacy_projection_test.cpp')==sha(prior)
(w/'native-preparation.json').write_text(json.dumps(record,indent=2)+'\n')
print('Nine external probes built; original legacy186record guard exact; both quiet OFF-label preprocessing proofs pass. No GPU/model run.')
