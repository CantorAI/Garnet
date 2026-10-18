import sys,json,pathlib
sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-main-macos-verify-20261018')
import garnet
r=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/native-image/fixtures');options={'patch_size':16,'merge_size':2,'temporal_patch_size':2,'pixels':4096,'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]};checks=[]
for name in ['bad.jpg','truncated.jpg']:
 try:garnet.prepare_image(str(r/name),json.dumps(options))
 except Exception:checks.append({'invalid':name,'rejected':True})
 else:raise AssertionError('Corrupt image admitted')
bad=dict(options);bad['image_std']=[0,.5,.5]
try:garnet.prepare_image(str(r/'rgb.png'),json.dumps(bad))
except Exception:checks.append({'invalid':'zero normalization std','rejected':True})
else:raise AssertionError('Invalid normalization admitted')
pathlib.Path(__file__).with_name('image-rejection-results.json').write_text(json.dumps(checks,indent=2));print('Native invalid image/config rejection PASS',len(checks))
