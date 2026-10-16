from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-dual-chunk64/source');shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-chunk64/source',s)
p=s/'runtime/metal_serving.py';t=p.read_text();old="                changed += 1\n        return json.dumps(dict(supported=changed > 0, enabled=bool(enabled)))";new="""                changed += 1
            elif not model.visual:
                if model.text_config['hidden_size'] != 2048 or model.text_config['num_hidden_layers'] != 28:
                    raise ValueError('Dual prefill admits only the validated Qwen3-1.7B reasoner')
                model.decoder.chunk = 64 if enabled else 16
                model.prefill_chunk = model.decoder.chunk
                changed += 1
        return json.dumps(dict(supported=changed > 0, enabled=bool(enabled)))""";assert old in t;t=t.replace(old,new);p.write_text(t)
for name in ['controller.py','cases.json','analyze.py','state_guard.py']:
 t=(w.parent/'worldsense-chunk64'/name).read_text().replace('worldsense-chunk64','worldsense-dual-chunk64')
 if name=='analyze.py':t=t.replace("64 if b['model_id']=='Qwen3.5-4B' and policy=='candidate' else 16","64 if policy=='candidate' else 16")
 (w/name).write_text(t)
shutil.copy2(w.parent/'worldsense-chunk64/before-libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-08','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Both Qwen3.5-4B VLM and Qwen3-1.7B reasoner use physical prefill64 versus16; same original BF16/FP32 kernels and shared weights. Separate from rejected VLM-only WS07.','hypothesis':'Larger chunks may reduce repeated weight reads in both model stages. WS07 found useful VLM gain but text-stage regression; no causal claim for that regression.','tradeoffs':'Both chunk-keyed graph sets kept warm for ablation, larger activation/cache footprint and padding; measure memory provenance, do not call RSS peakGPU.','stage':'Native/reference and all-layer state/token equivalence for BOTH models before WorldSense single-car ABBA screen, four samples/policy/model; no promotion without fresh full three-frame/search mirrored workers.','gates':{'exact_tokens_and_all_state_bits':True,'actual_chunk_plan':True,'4b_ttft_min_gain':.10,'4b_total_min_gain':.03,'decode_max_regression':.02,'text_total_ttft_max_regression':.02}},indent=2))
print('WS-M5-08 prepared, no timing')
