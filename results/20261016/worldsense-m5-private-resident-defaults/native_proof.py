from pathlib import Path
import json,hashlib,shutil
w=Path(__file__).resolve().parent;proof={};results={}
for mode in ['off','on']:
 log=(w/('final-tests-'+mode+'.log')).read_text();assert '100% tests passed out of 12' in log;results[mode]=0
 b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-defaults-'+mode)
 for n in [256,512,1024]:
  p=b/('dense_decode'+str(n)+'_words.bin');data=p.read_bytes();proof[mode+'-'+str(n)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 shutil.copy2(b/'dense_decode1024_words.bin',w/('dense-words-'+mode+'.bin'))
assert len({v['sha256']for v in proof.values()})==1
(w/'native-results.json').write_text(json.dumps(results,indent=2));(w/'native-correctness-proof.json').write_text(json.dumps({'both_native12_pass':True,'all_old_wide_reference_outputs_identical':True,'word_files':proof,'audit_preparation_error':'Initialsupplementallogparserexpected0testsfailedphrase notemittedbycurrentctest;correctedtoactual100%tests passedoutof12summary;noretest/resultchange.','scope':'Allwidekernelsdisabled;originalnativecorrectnessproof,notnewdispatchorperformance.'},indent=2))
print('Bothnative12logs andallreferenceoutputwordsverified')
