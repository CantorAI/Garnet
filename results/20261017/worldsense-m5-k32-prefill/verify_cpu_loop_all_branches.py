from pathlib import Path
import subprocess,re,json,hashlib
w=Path(__file__).resolve().parent;obj=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-k32-prefill-on/CMakeFiles/garnet_metal.dir/src/backends/metal/metal_context.mm.o');cmd=['xcrun','llvm-objdump','--disassemble','--demangle','--no-show-raw-insn',str(obj)];out=subprocess.check_output(cmd,text=True);lines=out.splitlines();at=next(i for i,x in enumerate(lines)if ' <Garnet::MetalContext::UploadBF16AsFloat('in x);end=next(i for i in range(at+1,len(lines))if re.match(r'^[0-9a-f]+ <',lines[i]));fragment=lines[at:end];start=int(re.match(r'^([0-9a-f]+)',lines[at]).group(1),16);stop=int(re.match(r'^([0-9a-f]+)',lines[end]).group(1),16)
relcmd=['xcrun','llvm-objdump','--reloc',str(obj)];relocations=subprocess.check_output(relcmd,text=True);row=[]
for line in relocations.splitlines():
 m=re.match(r'^([0-9a-f]+)\s+ARM64_RELOC_BRANCH26\s+_objc_msgSend\$rowBytes$',line)
 if m and start<=int(m.group(1),16)<stop:row.append((int(m.group(1),16),line))
assert len(row)==1,row
backedges=[]
for line in fragment:
 m=re.match(r'^\s*([0-9a-f]+):\s*(?:b(?:\.[a-z]+)?|cbz|cbnz|tbz|tbnz)\s+.*?0x([0-9a-f]+)',line)
 if m:
  pc,target=(int(x,16)for x in m.groups())
  if target<pc and start<=target:backedges.append(dict(start=target,end=pc))
assert backedges and all(not x['start']<=row[0][0]<=x['end']for x in backedges),'Getter remains in a backward branch interval'
simd=[x for x in fragment if re.search(r'\b(?:ushll|ushll2|shll|shll2)\b|\bshl\s+v\d+\.',x)]
(w/'direct-upload-cpu-object.masm.txt').write_text('\n'.join(fragment)+'\n');(w/'direct-upload-cpu-relocation.txt').write_text(row[0][1]+'\n');(w/'cpu-loop-proof.json').write_text(json.dumps({'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'disassembly_command':cmd,'relocation_command':relcmd,'rowBytes_calls_in_function':1,'rowBytes_call_outside_all_backward_branch_intervals':True,'rowBytes_call_address':row[0][0],'backward_branch_intervals':backedges,'branch_families_checked':['b','b.cond','cbz','cbnz','tbz','tbnz'],'vector_shift_instructions':simd,'scope':'Existing built CPU object, not a timing/performance claim. One getter remains before conversion loops. Native bit tests and WorldSense metrics required.'},indent=2)+'\n');print('CPU loop getter outside allbackedges; vector shifts',len(simd))
