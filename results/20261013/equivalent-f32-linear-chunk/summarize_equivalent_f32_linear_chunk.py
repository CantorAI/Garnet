
import pathlib,json,subprocess
r=pathlib.Path("/Users/zsdx/garnet-metal-investigation");p="/opt/homebrew/bin/python3"
for category,script in [("text","equivalent_f32_summarize_vllm_comparison.py"),("vl","equivalent_f32_summarize_frame_comparison.py")]:
 command=[p,str(r/script),str(r/("logs/equivalent-f32-linear-chunk-"+category+"-20261013/jobs"))]
 print("Executing",json.dumps(command),flush=True);subprocess.run(command,check=True)
