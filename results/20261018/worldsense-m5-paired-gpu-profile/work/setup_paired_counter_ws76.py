"""Apply the same bounded WS62 counter observer to two immutable source trees."""
from pathlib import Path
import ast, hashlib, json, re, shutil

base=Path('/Users/shawn/Documents/Codex/2026-10-15/i')
w=base/'work/worldsense-paired-gpu-profile'
r=Path('/Users/shawn/CantorAI')
prior=base/'work/worldsense-gpu-counter-profile'
protocol=json.loads((w/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(w/'native-design-before-implementation.txt')==protocol['design_sha256']
prior_ops=json.loads((prior/'source-transformations.json').read_text())
fault_before='GARNET_METAL_PRIMARY_TEST_FAULTS'
fault_after='GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS'
helper=(prior/'gpu_pass_counter_trace.inc').read_text().replace(fault_before,fault_after)
test=(prior/'gpu_pass_counter_test.cpp').read_text().replace(fault_before,fault_after)
for family in ['legacy','current']:
    old=Path(protocol[family+'_source'])
    seed=json.loads((w/(family+'-seed-source-manifest.json')).read_text())
    s=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source'
    s.mkdir(parents=True,exist_ok=False)
    assert all(sha(old/k)==v for k,v in seed.items())
    for rel in seed:
        p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
    changes={}
    def replace(name,before,after,count=1):
        p=s/name;text=p.read_text();assert text.count(before)==count,(family,name,before[:100],text.count(before),count)
        p.write_text(text.replace(before,after))
        changes.setdefault(name,[]).append(dict(before=before,after=after,count=count))
    h='src/backends/metal/metal_context.h'
    mm='src/backends/metal/metal_context.mm'
    entry='src/entry/metal_main.cpp'
    runtime='runtime/metal_serving.py'
    replace(h,prior_ops[h][0]['before'],prior_ops[h][0]['after'])
    api=prior_ops[h][1]['after'].split('\n',1)[1].replace(fault_before,fault_after)
    replace(h,'    static std::string AvailableDeviceName();','    static std::string AvailableDeviceName();\n'+api)
    for op in prior_ops[mm]:
        # The three large prior function replacements only specialize labels.
        # Apply equivalent small substitutions inside this seed's own functions.
        if op['before'].startswith(('void MetalContext::EncodeBF16MatVec(', 'void MetalContext::EncodePointwise(', 'void MetalContext::EncodeStateKernel(')):
            continue
        replace(mm,op['before'],op['after'],op['count'])
    labels=[
      ('void MetalContext::EncodeBF16MatVec(','void MetalContext::Update(',
       'std::string("bf16:")+std::to_string(a.rows)+":"+std::to_string(a.columns)+":"+std::to_string(o.columns)+":"+'+('(primary?"primary":"raw")' if family=='current' else '"raw"')),
      ('void MetalContext::EncodePointwise(','void MetalContext::EncodeRMSNorm(',
       'std::string("pointwise:")+std::to_string(operation)+":"+std::to_string(a.rows)+":"+std::to_string(a.columns)'),
      ('void MetalContext::EncodeStateKernel(','void MetalContext::EncodeLayout(',
       'std::string("state:")+std::to_string(operation)+":"+std::to_string(c.rows)+":"+std::to_string(c.columns)')]
    for start,end,label in labels:
        text=(s/mm).read_text();at=text.index(start);stop=text.index(end,at);part=text[at:stop]
        assert 'PassTraceEncoder(*impl_,__func__)' in part
        replace(mm,part,part.replace('PassTraceEncoder(*impl_,__func__)','PassTraceEncoder(*impl_,'+label+')'))
    if family=='legacy':
        replace(h,'    std::string DeviceName() const;','    std::string DeviceName() const;\n    std::vector<std::string> ShaderProgramSources() const;')
        replace(mm,'struct MetalPipelines {\n    id<MTLLibrary> library;','struct MetalPipelines {\n    std::string mainProgram;\n    id<MTLLibrary> library;')
        replace(mm,'        impl_->kernels->library=library;','        impl_->kernels->library=library;\n        impl_->kernels->mainProgram=source.UTF8String;')
        getter='''
namespace Garnet {
std::vector<std::string> MetalContext::ShaderProgramSources() const {
 if(!impl_->kernels)throw std::logic_error("Shader program sources require ready kernels");
 return {impl_->kernels->mainProgram,{}};
}
}
'''
        p=s/mm;p.write_text(p.read_text()+getter)
        changes[mm].append(dict(before='',after=getter,count=1,append=True))
    for op in prior_ops[entry]:
        replace(entry,op['before'],op['after'].replace(fault_before,fault_after),op['count'])
    for op in prior_ops[runtime][:2]:
        replace(runtime,op['before'],op['after'])
    text=(s/runtime).read_text()
    match=re.search(r'(?m)^( +)prediction = self\.decoder\.step\(prediction, active=\[1\], positions=\[\[current, current, current\]\]\)$',text)
    assert match
    indent=match.group(1);line=match.group()
    replace(runtime,line,indent+'if _gpu_pass_trace and counter_request:\n'+indent+'    garnet.counter_trace_step(index + 1)\n'+line)
    prefix='        return dict(input_prompt_sha256=' if family=='legacy' else '            result = dict(input_prompt_sha256='
    indent=' '* (8 if family=='legacy' else 12)
    replace(runtime,prefix,indent+'if _gpu_pass_trace:\n'+indent+'    stages["counter_trace_request_id"] = counter_request\n'+prefix)
    cmake='cmake/GarnetMetal.cmake'
    extra=prior_ops[cmake][0]['after']+'''
option(GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS "Correctness-only counter fault controls" OFF)
if(GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS)
 if(NOT GARNET_METAL_GPU_PASS_COUNTER_PROFILE)
  message(FATAL_ERROR "Counter fault controls require counter profile")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS=1)
endif()
'''
    p=s/cmake;p.write_text(p.read_text()+extra)
    changes[cmake]=[dict(before='',after=extra,count=1,append=True)]
    for rel,text in [('src/backends/metal/gpu_pass_counter_trace.inc',helper),('test2026/metal/gpu_pass_counter_test.cpp',test)]:
        (s/rel).write_text(text)
    for rel,ops in changes.items():
        text=(s/rel).read_text()
        for op in reversed(ops):
            if op.get('append'):
                assert text.endswith(op['after']);text=text[:-len(op['after'])]
            else:
                assert text.count(op['after'])==op['count'],(family,rel,'inverse',op['after'][:80])
                text=text.replace(op['after'],op['before'])
        assert text==(old/rel).read_text(),(family,rel)
    literals=lambda text:re.findall(r'R"([A-Za-z_]+)\((.*?)\)\1"',text,re.S)
    assert literals((s/mm).read_text())==literals((old/mm).read_text())
    ast.parse((s/runtime).read_text())
    manifest={str(p.relative_to(s)):sha(p) for p in sorted(s.rglob('*')) if p.is_file()}
    assert len(manifest)==len(seed)+2
    expected=['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/gpu_pass_counter_trace.inc','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/gpu_pass_counter_test.cpp']
    assert sorted(k for k,v in manifest.items() if seed.get(k)!=v)==expected
    assert all(manifest[k]==v for k,v in seed.items() if k not in changes)
    (w/(family+'-source-transformations.json')).write_text(json.dumps(changes,indent=2)+'\n')
    (w/(family+'-source-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    (w/(family+'-source-inverse-proof.json')).write_text(json.dumps(dict(seed_files=len(seed),source_files=len(manifest),changed_or_added=expected,all_existing_changes_invert_to_immutable_seed=True,all_original_shader_literals_graph_session_xModel_image_CUDA_engine_tests_unchanged=True,legacy_actual_program_getter_added=family=='legacy',common_helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),common_test_sha256=hashlib.sha256(test.encode()).hexdigest(),fault_macro_only_rename_from_WS62=True,no_build_or_model=True),indent=2)+'\n')
    print(family,len(manifest),'source files; seven diagnostic paths; all existing changes inverse-exact',flush=True)
shutil.copytree(base/'work/worldsense-minimal-selective-defaults/build-profile',w/'build-profile')
protocol.update(phase='source-implemented-before-build',source_implemented=True,no_successor_source_build_model_or_timing_yet=False,no_build_or_model_yet=True)
(w/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
