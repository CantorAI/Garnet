"""Prepare original-bound numerical-only projection guard against exact WS10 DLL."""
from pathlib import Path
import re,subprocess,json,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-prefix-cache/source';n=r/'out/build/macos-arm64-release/bin';lib=r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'
t=(r/'out/research/worldsense-native-text-worker/source/test2026/metal/primary_projection_test.cpp').read_text()
original=t
t=t.replace('#include <algorithm>','#include <algorithm>\n#include <dlfcn.h>\n#include <filesystem>')
needle='static uint64_t Encoded(){auto s=MetalContext::PrimaryStats();return s.decodeEncoded+s.headEncoded+s.prefillEncoded;}';assert t.count(needle)==1;t=t.replace(needle,'')
t=t.replace('cases=0,passes=0,widePasses=0','cases=0,passes=0')
a=t.index('  bool eligible=false;');b=t.index('  std::vector<float> saved;',a);t=t[:a]+t[b:]
t=t.replace('auto before=Encoded();','').replace('auto invalidBefore=Encoded();','')
needle='require(Encoded()-before==size_t(eligible),"Incorrect encoded wide route");widePasses+=eligible;';assert t.count(needle)==1;t=t.replace(needle,'')
t=t.replace('require(rejected&&Encoded()==invalidBefore,"Invalid encoding reached wide dispatch");','require(rejected,"Invalid projection unexpectedly accepted");')
needle='require(Encoded()-before==2*size_t(eligible),"Queued route count");';assert t.count(needle)==1;t=t.replace(needle,'')
a=t.index('#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS');b=t.index(' if(binary.is_open())',a);t=t[:a]+t[b:]
t=t.replace('<<" wide="<<widePasses','')
t=t.replace('Primary numerical/reference/poison/input/tail272/fallback/small/vocab/queued owner/Abort recovery PASS','Pinned WS10 RAW DLL numerical/reference/poison/input/tail272/fallback/small/vocab/queued owner/Abort recovery PASS')
needle=' MetalContext gpu;';assert t.count(needle)==1
t=t.replace(needle,''' Dl_info image{};require(dladdr(reinterpret_cast<const void*>(&MetalContext::AvailableDeviceName),&image)&&image.dli_fname,"Legacy numerical function address unavailable");require(std::filesystem::weakly_canonical(image.dli_fname)==std::filesystem::weakly_canonical(LEGACY_LIBRARY_PATH),"Projection guard resolved another native library");
 MetalContext gpu;''')
assert 'PrimaryStats'not in t and 'PrimaryReady'not in t and 'Encoded()'not in t and 'eligible'not in t
bound='double bound=rows<=2?1e-4+2e-6*l1:5e-6+5e-5*std::abs(ref);';assert bound in t and bound in original
assert 'require(cases==93&&passes==186' in t
p=w/'legacy_projection_test.cpp';p.write_text(t)
cmd=['/usr/bin/c++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-I',str(s/'src/backends/metal'),str(p),'-DLEGACY_LIBRARY_PATH="'+str(lib)+'"',str(lib),str(n/'libxlang3_runtime.dylib'),'-Wl,-rpath,'+str(lib.parent),'-Wl,-rpath,'+str(n),'-o',str(w/'legacy-projection-test')]
with(w/'build-legacy-projection.log').open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(w/'legacy-projection-build-identity.json').write_text(json.dumps(dict(command=cmd,source_sha256=sha(p),library_sha256=sha(lib),executable_sha256=sha(w/'legacy-projection-test'),original_93shape_186record_FP64_bounds_and_poison_input_tail_fallback_queued_Abort_checks_retained=True,removed_only_newer_PRIMARY_format_counter_assertions_unavailable_in_old_header=True,no_fake_counter_or_speed_claim=True),indent=2)+'\n')
print('Exact WS10-DLL numerical projection guard built; no GPU dispatch or timing result',flush=True)
