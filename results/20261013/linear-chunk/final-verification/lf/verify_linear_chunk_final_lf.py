
import pathlib,subprocess,json,hashlib
r=pathlib.Path("/Users/zsdx/garnet-metal-investigation");out=r/"logs/linear-chunk-final-lf-20261013"
commands=[["/opt/homebrew/bin/cmake","--build",str(r/"build-linear-chunk"),"--parallel","4"],["/opt/homebrew/bin/ctest","--test-dir",str(r/"build-linear-chunk"),"--output-on-failure"]]
for command in commands:
 print("Executing",json.dumps(command),flush=True);subprocess.run(command,check=True)
actual=hashlib.file_digest((r/"build-linear-chunk/libgarnet.dylib").open("rb"),"sha256").hexdigest()
proof=dict(candidate_sha256=actual,confirmed_sha256="26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671",formatting_only=True,identical_binary=actual=="26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671",commands=commands)
(out/"binary-proof.json").write_text(json.dumps(proof,indent=2)+"\n")
assert proof["identical_binary"],"Formatting rebuild differs; preserve confirmed binary and audit before use"
print("Final formatted source produces byte-identical tested binary",flush=True)
