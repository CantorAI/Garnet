
import pathlib,subprocess,json
r=pathlib.Path("/Users/zsdx/garnet-metal-investigation");py="/opt/homebrew/bin/python3";s=r/"source/test2026/metal"
text=r/"logs/native-packed-decode-text-ab-20261013/jobs";vl=r/"logs/native-packed-decode-vl-ab-20261013/jobs"
commands=[[py,str(s/"summarize_native_builds.py"),str(text)],[py,str(s/"summarize_native_frames.py"),str(vl)],[py,str(s/"summarize_native_startup.py"),str(text),str(vl),str(r/"logs/packed-decode-summary-20261013/startup.json")]]
for command in commands:
 print("Executing",json.dumps(command),flush=True);subprocess.run(command,check=True)
rss={}
for jobs in (text,vl):
 for log in jobs.glob("*/output.log"):
  processes=[json.loads(x.split(" ",1)[1]) for x in log.read_text().splitlines() if x.startswith("matched_process ")]
  if processes:rss[log.parent.name]=processes[-1]
(r/"logs/packed-decode-summary-20261013/memory.json").write_text(json.dumps(dict(scope="High-water individual native child RSS; not aggregate/unique GPU memory",processes=rss),indent=2)+"\n")
