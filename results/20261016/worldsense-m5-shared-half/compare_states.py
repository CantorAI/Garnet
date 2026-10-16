import pathlib,json,array,math
w=pathlib.Path(__file__).resolve().parent;a=json.loads((w/'state-control.json').read_text());b=json.loads((w/'state-candidate.json').read_text());parts=[]
for layer in range(32):
 for part in range(2):
  names=['state-control.json','state-candidate.json'];vs=[]
  for name in names:
   v=array.array('f');v.frombytes((w/(name+'.layer'+str(layer)+'.part'+str(part)+'.f32')).read_bytes());vs.append(v)
  x,y=vs;assert len(x)==len(y);worst=0;maximum=0;maxref=0;sumdiff=0;sumref=0;finite=True
  for c,d in zip(x,y):
   difference=abs(c-d);maximum=max(maximum,difference);maxref=max(maxref,abs(c));sumdiff+=difference*difference;sumref+=c*c;finite=finite and math.isfinite(c)and math.isfinite(d);worst=max(worst,difference/(5e-6+5e-5*abs(c)))
  parts.append({'layer':layer,'part':part,'values':len(x),'bitwise_equal':a['states'][layer][part]['sha256']==b['states'][layer][part]['sha256'],'finite':finite,'max_abs_error':maximum,'max_reference_magnitude':maxref,'l2_error':math.sqrt(sumdiff),'l2_reference':math.sqrt(sumref),'worst_pointwise_dense_coefficient_ratio':worst})
r={'tokens_equal':a['generated_token_ids']==b['generated_token_ids'],'all_state_parts_bitwise_equal':all(p['bitwise_equal']for p in parts),'all_states_finite':all(p['finite']for p in parts),'parts':parts,'scope':'One fixed image/prompt, 16 output tokens at 512 capacity. State differences diagnose changed MPS reduction order; not a full original multi-profile numerical suite.'};(w/'state-comparison.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items()if k!='parts'});print('max abs state error',max(p['max_abs_error']for p in parts),'worst coefficient ratio',max(p['worst_pointwise_dense_coefficient_ratio']for p in parts))
