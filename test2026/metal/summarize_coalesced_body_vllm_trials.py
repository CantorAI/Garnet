"""Compute all original matched targets from two independently audited cohorts."""
from pathlib import Path
import json,sys

root=Path(sys.argv[1]).resolve()
bases=[root/'evidence',root/'confirmation/evidence']
assert (bases[0]/'selection.json').read_bytes()==(bases[1]/'selection.json').read_bytes()
jobs=[base/'logs'/name/'jobs' for base,name in zip(bases,('coalesced-body-vllm-20261015','coalesced-body-vllm-confirm-20261015'))]
assert json.loads((jobs[0]/'versions.json').read_text())==json.loads((jobs[1]/'versions.json').read_text())
summaries=[json.loads((job/'summary.json').read_text()) for job in jobs]
for initial,repeat in zip(*(s['profiles'] for s in summaries)):
    assert (initial['model'],initial['batch'])==(repeat['model'],repeat['batch'])
    folders=[job/Path(p['evidence']).name for job,p in zip(jobs,(initial,repeat))]
    metas=[json.loads((p/'metadata.json').read_text()) for p in folders]
    for key in ('native_library_sha256','source_checkpoint_sha256','comparator_checkpoint_sha256','equivalence_proof_sha256','source_sha256','native_build_policy','selection_sha256','common_frontend','threaded_decoder_sha256'):
        assert metas[0][key]==metas[1][key],key
    cases=[json.loads((p/'case.json').read_text()) for p in folders]
    assert cases[0]==cases[1]

source=Path(__file__).with_name('summarize_wide_batch_vllm_trials.py').read_text()
source=source.replace('wide-batch-vllm','coalesced-body-vllm')
source=source.replace('full_goal_met=False)',"full_goal_met=all(all(v['decode_ratio']>=2 and v['ttft_fraction']<=.5 for v in p) for p in by_profile.values()))")
source=source.replace("(root/'remaining-budgets.json').write_bytes", "result['identical_selection_versions_sources_assets_and_cases']=True\n(root/'remaining-budgets.json').write_bytes")
exec(compile(source,__file__,'exec'))
