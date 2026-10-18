"""Inspect actual HOST ARM64 code; never a GPU ISA or runtime-cost proof."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-owned-storage-local';r=Path('/Users/shawn/CantorAI')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
tool=Path('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/llvm-objdump')
symbols={'prepare':'__ZN6Garnet12MetalContext20PrepareBF16BodyShapeERKNS_11MetalMatrixERKNS_15MetalBF16WeightES3_','encode':'__ZN6Garnet12MetalContext16EncodeBF16MatVecERKNS_11MetalMatrixERKNS_15MetalBF16WeightES3_'}
roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']
assert (w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+x for x in roles]
assert not (w/'host-codegen-review.json').exists()
details={};identities={str(tool):sha(tool)}
for role in ['off','raw-borrow','control','on']:
 build=r/('out/build/garnet-metal-worldsense-owned-storage-local-'+role)
 obj=build/'CMakeFiles/garnet_metal.dir/src/backends/metal/metal_context.mm.o'
 for path in [obj,build/'CMakeFiles/garnet_metal.dir/flags.make',build/'CMakeCache.txt',build/'libgarnet.dylib']:
  identities[str(path)]=sha(path)
 flags=(build/'CMakeFiles/garnet_metal.dir/flags.make').read_text();assert '-fobjc-arc' in flags and '-O3' in flags
 details[role]={}
 for kind,symbol in symbols.items():
  command=[str(tool),'--macho','--disassemble','--dis-symname',symbol,str(obj)]
  p=subprocess.run(command,capture_output=True,text=True,check=True)
  path=w/(role+'-'+kind+'.arm64.txt');path.write_text(p.stdout);(w/(role+'-'+kind+'.objdump.stderr')).write_text(p.stderr)
  lines=p.stdout.splitlines();owner=next(i for i,line in enumerate(lines)if re.search(r'fmov\s+x\d+, d0',line));last=next(i for i in range(owner+1,len(lines))if '_objc_msgSend' in lines[i]);window=lines[owner:last]
  retained=sum('_objc_retain' in x for x in window);released=sum('_objc_release' in x for x in window)
  enabled=role in ['raw-borrow','on'];assert (retained,released)==((0,0)if enabled else((1,0)if role=='off'else(2,1)))
  before=lines[max(0,owner-12):owner];assert any('0xf8290108' in x for x in before)
  # q0 is copied from the weight handle, preserving both Impl and control-block.
  assert any('ldr\tq0' in x for x in before) and any('str\tq0' in x for x in before)
  # Normal cleanup decrements the copied weight shared_ptr AFTER preparation
  # accesses; in Encode it occurs after the unchanged dispatch/endEncoding.
  release_weak=next(i for i in range(last+1,len(lines))if '__shared_weak_count14__release_weakEv' in lines[i])
  assert any('0xf8e90108' in x for x in lines[release_weak-10:release_weak])
  if kind=='encode':assert any('_objc_msgSend$endEncoding' in x for x in lines[last:release_weak])
  if role=='on':assert any('csel' in x for x in window) and any('ldr' in x for x in window)
  details[role][kind]=dict(command=command,disassembly_sha256=sha(path),selection_window=window,selection_static_retain_calls=retained,selection_static_release_calls=released,owner_copy_and_atomic_increment_window=before,normal_weight_owner_release_window=lines[release_weak-10:release_weak+1],normal_owner_release_after_binding_and_endEncoding_if_encode=True,whole_function_static_retain_call_sites=p.stdout.count('bl\t_objc_retain'),whole_function_static_release_call_sites=p.stdout.count('bl\t_objc_release'))
for a,c in [('off','raw-borrow'),('control','on')]:
 def profile(role):
  return dict(re.findall(r'^(GARNET[^:]*):(?:BOOL|STRING|PATH|FILEPATH)=(.*)$',(r/('out/build/garnet-metal-worldsense-owned-storage-local-'+role)/'CMakeCache.txt').read_text(),re.M))
 aa,cc=profile(a),profile(c);assert aa.pop('GARNET_METAL_OWNED_STORAGE_LOCAL')=='OFF' and cc.pop('GARNET_METAL_OWNED_STORAGE_LOCAL')=='ON';assert aa==cc
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build_observed_session=93228,build_observed_terminal_exit_code=0,all_eight_builds_pass=True,selected_local_ownership_removed_both_raw_and_packed=True,matched_pairs_differ_only_new_GARNET_flag=True,ordinary_weight_owner_copy_and_normal_cleanup_remain=True,details=details,inputs=identities,limits='Actual HOST ARM64 static code windows and normal ownership cleanup, not dynamic call counts, cost, every error-path runtime safety, complete GPU operations/ISA/register/occupancy or speed proof. Original and additional native/learned error/lifetime checks still required. .long opcodes retained exactly as emitted by tool; shared_ptr identity determined from C++ handle/q0 control-block tracking and original compiler lowering, not ARC retainCount.')
(w/'host-codegen-review.json').write_text(json.dumps(result,indent=2)+'\n')
print('Actual HOST codegen: selected-local retains/releases OFF raw1/0 and packed2/1 -> ON0/0; copied weight owner and normal later cleanup remain. Original gates still pending.')
