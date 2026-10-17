from pathlib import Path
import json,hashlib,shutil,ast
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=root/'work/worldsense-packed-only-defaults';s=r/'out/research/worldsense-packed-only-defaults/source';seed=r/'out/research/worldsense-isolated-primary/source';m=json.loads((root/'work/worldsense-isolated-primary/correctness-source-manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert not s.exists()and len(m)==1718 and all(sha(seed/k)==h for k,h in m.items());assert json.loads((w/'protocol.json').read_text())['experiment']=='WS-M5-57';s.mkdir(parents=True)
for rel in m:(s/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/rel,s/rel)
(w/'ws54-seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p=s/'src/backends/metal/compiled_graph.cpp';t=p.read_text();anchor='''        if(!ready_||outputParts_.size()!=3||inputs_.size()<3||residentOps_.size()!=ops_.size())return;
''';assert t.count(anchor)==1
addition='''#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS
        // Raw graphs retain their original default buffers. Eligibility is local
        // to this graph's immutable weights, never the device's global owners.
        if(std::none_of(slots_.begin(),slots_.end(),[](const auto& slot){return slot.bf16.PrimaryReady();})) {
            lazyDefaultSlots_.clear();lazyDefaultPlanReady_=true;return;
        }
#endif
''';p.write_text(t.replace(anchor,anchor+addition,1))
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();t+='''
option(GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS "Retire resident defaults only for actual packed-weight graphs" OFF)
if(GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_WIDE_PRIMARY_STORAGE OR NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS)
  message(FATAL_ERROR "Packed-only default retirement requires primary weights, wide storage and lazy defaults")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS=1)
endif()
''';p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();anchor='        return info;';assert t.count(anchor)==1;marker='#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS\n        info+="; primary_only_lazy_defaults";\n#endif\n';p.write_text(t.replace(anchor,marker+anchor,1))
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};changed={k for k,h in manifest.items()if m.get(k)!=h};assert changed=={'cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/entry/metal_main.cpp'}
assert(s/'src/backends/metal/compiled_graph.cpp').read_text().replace(addition,'',1)==(seed/'src/backends/metal/compiled_graph.cpp').read_text();assert(s/'src/entry/metal_main.cpp').read_text().replace(marker,'',1)==(seed/'src/entry/metal_main.cpp').read_text();assert(s/'cmake/GarnetMetal.cmake').read_text().startswith((seed/'cmake/GarnetMetal.cmake').read_text())
(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'starting-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'implementation-audit.json').write_text(json.dumps(dict(source_files=len(manifest),seed_changed=sorted(changed),narrow_graph_and_entry_inverse_exact=True,all_shaders_math_codec_loader_session_serving_xModel_CUDA_original_tests_exact=True,scope='Source audit only; no build/native/model/performance evidence yet.'),indent=2)+'\n')
print('WS57 isolated source1718 files, exact3file policy delta; no build/test/performance yet')
