import sys,pathlib,json,time
build,source,label=sys.argv[1:4];sys.path.insert(0,build)
import garnet
sys.path.insert(0,source)
from runtime.metal_serving import Model
m=Model('/Users/shawn/CantorAI/out/models/Qwen3-1.7B','','/Users/shawn/Documents/Codex/2026-10-15/i/work/4b-handover/cache-llm-'+label,{'maxInputTokens':1536,'maxOutputTokens':512},'Qwen3-1.7B')
for prompt in ['Return only JSON with current_state as [{"key":"rabbit_visible","values":[true]}].','Return only JSON with current_state as [{"key":"rabbit_visible","values":[null]}].','Extract search terms as JSON from: a passenger car with headlights.']:
 print(json.dumps({'prompt':prompt,'call':m.infer(prompt,None,100)}),flush=True)
