"""After terminal build95115, require the intended packed-owner rejection case."""
from pathlib import Path
import json,hashlib,shutil,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-raw-wide-vocab';s=Path('/Users/shawn/CantorAI/out/research/worldsense-raw-wide-vocab/source');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text());roles=['off','primary','on','fault','compat56','compat56-fault']
assert(w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+x for x in roles]and not(w/'native-results.json').exists();old=load(w/'correctness-source-manifest.json');assert len(old)==1723 and all(sha(s/k)==v for k,v in old.items())
archive=w/'pre-native-fixture-review';archive.mkdir(exist_ok=False)
for name in ['correctness-source-manifest.json','source-inverse-proof.json','source-transformations.json','build-plan.json','build.py','raw_wide_vocab_test.cpp','build-controller.log','pre-native-fixture-review.txt']+['build-'+v+'.log'for v in roles]:shutil.copy2(w/name,archive/name)
rel='test2026/metal/raw_wide_vocab_test.cpp';t=(s/rel).read_text();before='{std::vector<uint16_t> constant(4*128,0x3f80);auto packed=gpu.UploadBF16(constant.data(),constant.size()*2,4,128);if(packed.PrimaryReady())rejected([&]{gpu.RawBF16ReadGPUWords(packed,0,4);});}'
after='''{std::vector<uint16_t> constant(260*128,0x3f80);auto packed=gpu.UploadBF16(constant.data(),constant.size()*2,260,128);
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(packed.PrimaryReady(),"Packed-owner negative fixture did not pack");rejected([&]{gpu.RawBF16ReadGPUWords(packed,0,4);});
#else
 need(!packed.PrimaryReady(),"Disabled primary unexpectedly packed fixture");
#endif
 }'''
assert t.count(before)==1;t=t.replace(before,after);(s/rel).write_text(t);(w/'raw_wide_vocab_test.cpp').write_text(t)
current={str(p.relative_to(s)):sha(p)for p in s.rglob('*')if p.is_file()};assert current.keys()==old.keys()and [k for k in current if current[k]!=old[k]]==[rel]
(w/'correctness-source-manifest.json').write_text(json.dumps(current,indent=2)+'\n')
(w/'pre-native-fixture-repair.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_initial_build_session=95115,observed_initial_build_exit=0,all_six_initial_builds_pass=True,no_native_model_or_timing_attempt_yet=True,path=rel,before=before,after=after,before_sha256=old[rel],after_sha256=current[rel],original_shape_admission_preserved=True,only_new_fixture_coverage_changed=True,readback_policy_math_original_tests_source_unchanged=True,observation_scope='Initial log already contained sixBUILD PASS at review; prior note stillconsuming wording was a cautious unverified assumption. Authoritative poll95115returned terminal0 before any edit. No consumed edits during live build.'),indent=2)+'\n')
d=load(b/'work/worldsense-next.json');d.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='NEW_FIXTURE_COVERAGE_REPAIRED_REBUILD_NEXT',initial_build_session=95115,initial_build_exit=0,live_job=None,pre_native_fixture_repair=True);(b/'work/worldsense-next.json').write_text(json.dumps(d,indent=2)+'\n');print('Initialsixbuilds95115terminal0 preserved; ONLY newfixture4->260rows with requiredpacked admission repaired, no originaltest/API/math change. Sixrebuild next; no native/model yet.')
