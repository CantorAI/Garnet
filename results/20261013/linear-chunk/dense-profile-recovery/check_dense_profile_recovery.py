
import pathlib,subprocess,json,hashlib
r=pathlib.Path("/Users/zsdx/garnet-metal-investigation");out=r/"logs/dense-profile-recovery-20261013/jobs"
out.mkdir(parents=True,exist_ok=False);statuses=[];records=[]
base=r/"logs/vllm-matched-parallel-delta-text-20261013/jobs"
for batch in (1,2):
 for name in ("Qwen3-0.6B","Qwen2.5-Coder-0.5B-Instruct"):
  label=name+"-b"+str(batch);case=base/(label+"-case.json")
  guard=r/("guard-dense-recovery-"+label+".py")
  guard.write_text((r/"guard_extended_prefill.py").read_text().replace("extended-prefill-preflight-20261013","dense-recovery-preflight-"+label+"-20261013"))
  subprocess.run(["/opt/homebrew/bin/python3",str(guard)],check=True)
  command=["/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3",str(r/"source/test2026/metal/run_dense_profile_guard.py"),str(r/"build-linear-chunk"),str(r/"source"),str(r/"models"/name),str(case)]
  log=out/(label+".log")
  with log.open("w") as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  statuses.append(dict(name=label,command=command,exit_code=result.returncode,case_sha256=hashlib.file_digest(case.open("rb"),"sha256").hexdigest()))
  (out/"statuses.json").write_text(json.dumps(statuses,indent=2)+"\n")
  if result.returncode:raise SystemExit(result.returncode)
  lines=log.read_text().splitlines();assert "dense_profile_guard PASS" in lines
  values=[json.loads(v.split(" ",1)[1]) for v in lines if v.startswith("matched_result ")]
  frozen=base/(label+"-garnet-0/output.log")
  expected=[json.loads(v.split(" ",1)[1]) for v in frozen.read_text().splitlines() if v.startswith("matched_result ")]
  assert len(values)==len(expected)==3
  for v,e in zip(values,expected):
   for key in ("token_ids","finished","prompt_token_ids"):assert v[key]==e[key],(label,key)
  records.append(dict(name=label,exact_generated_outputs=True,requests=3,unsupported_profiles_rejected=[32,64]))
  (out/"summary.json").write_text(json.dumps(dict(scope="Correctness after rejected extended dense profiles; no speed claim",groups=records),indent=2)+"\n")
  print(label,"guard and complete outputs PASS",flush=True)
