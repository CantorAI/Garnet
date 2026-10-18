"""Build exact corpus guards, create fixtures, freeze proof inputs; no inference."""
from pathlib import Path
import subprocess, sys, hashlib, json, datetime

w = Path(__file__).resolve().parent.parent / 'worldsense-validated-text-worker'
r = Path('/Users/shawn/CantorAI')
assert (w / 'text-independent-audit.json').is_file()
for name in ['build_corpus.py', 'build_direct_corpus.py', 'make_resident_fixtures.py']:
    with (w / ('prepare-' + name + '.log')).open('w') as log:
        subprocess.run([sys.executable, str(w / name)], stdout=log, stderr=subprocess.STDOUT, check=True)
    print('PASS ' + name, flush=True)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

frozen = json.loads((w / 'first-model-identities-before.json').read_text())
for key, root in [('parent_source', r / 'out/research/worldsense-validated-text-worker/source'),
                  ('legacy_source', r / 'out/research/worldsense-prefix-cache/source')]:
    assert all(sha(root / rel) == value for rel, value in frozen[key].items())
assert all(sha(Path(p)) == value for p, value in frozen['external'].items())
paths = [p for p in w.iterdir() if p.suffix in ['.py', '.cpp']]
paths += [p for d in ['fixture-1-8', 'fixture-2-64'] for p in (w / d).rglob('*') if p.is_file()]
paths += [w / 'direct-upload-selection.json',
          w.parent / 'worldsense-primary-blocks/matrix-selection.json',
          w.parent / 'worldsense-prefix-cache/state-guard.json',
          w.parent / 'worldsense-request-residency/vision-grid-off.json']
paths += [w.parent / 'skill-expansion' / name for name in ['car.jpg', 'traffic.jpg', 'bunny.jpg']]
b = r / 'out/build/garnet-metal-worldsense-validated-text-worker-on'
paths += [b / 'primary_corpus_test', b / 'direct_float_corpus_test', b / 'libgarnet_metal.a']
paths += [Path(__file__).with_name('ws61_' + name + '.py')
          for name in ['owner_death', 'owner_parent', 'post_owner_recovery']]
frozen['external'].update({str(p): sha(p) for p in paths})
frozen['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
frozen['scope'] = 'Full learned proof source/assets/oracles/drivers/corpus executables; no complete timing build-host closure claim.'
(w / 'learned-input-identities-before.json').write_text(json.dumps(frozen, indent=2) + '\n')
print('Full learned proof inputs frozen', flush=True)
