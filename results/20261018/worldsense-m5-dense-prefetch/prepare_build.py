from pathlib import Path
import json,hashlib,shutil,ast
w=Path(__file__).resolve().parent
s=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-prefetch/source')
p=s/'src/backends/metal/metal_context.mm';t=p.read_text()
block='#ifdef GARNET_METAL_DENSE_PREFETCH\nstd::atomic<uint64_t> densePrefetchEncoded{0};\n#endif'
assert t.count(block)==1;t=t.replace(block,'')
needle='namespace Garnet {\n#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS'
assert t.count(needle)==1
t=t.replace(needle,'namespace Garnet {\n#ifdef GARNET_METAL_DENSE_PREFETCH\nnamespace {std::atomic<uint64_t> densePrefetchEncoded{0};}\n#endif\n#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS')
t=t.replace('#include <cstring>','#include <cstring>\n#include <atomic>');p.write_text(t)
(w/'prebuild-contract-review.json').write_text(json.dumps(dict(prefetch_counter_independent_of_PRIMARY_compile_flag=True,reason='New policy admits raw-only builds; atomic definition must not be inside PRIMARY conditional',initial_preparation_assertion_failed_before_write=True,attempted_missing_build_driver_exit2_no_build_started=True,no_model_or_GPU_run=True),indent=2)+'\n')
manifest={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()};assert len(manifest)==1720
(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copytree(w.parent/'worldsense-isolated-primary/build-profile',w/'build-profile')
t=(w.parent/'worldsense-isolated-primary/build.py').read_text().replace('worldsense-isolated-primary','worldsense-dense-prefetch')
needle="'-DGARNET_METAL_PRIMARY_BF16_BLOCKS='+primary,";assert t.count(needle)==1
t=t.replace(needle,needle+"'-DGARNET_METAL_DENSE_PREFETCH='+primary,");ast.parse(t);(w/'build.py').write_text(t)
print('Prebuild RAW-only counter contract corrected; four build roles prepared')
