"""Preserve audit-path failure; confirm admitted unchanged native candidate without rerunning initial."""
from pathlib import Path
p=Path(__file__).with_name('continue_vector_vision_native.py');wrapper=p.read_text();ns={'__file__':str(p)}
marker="exec(compile(source,__file__,'exec'))";assert wrapper.count(marker)==1
exec(compile(wrapper[:wrapper.index(marker)],str(p),'exec'),ns);source=ns['source']
source=source.replace('continue_vector_vision_native.py','continue_vector_vision_native_repaired.py')
source=source.replace('verify_vector_vision_native_profiles.py','verify_vector_vision_native_profiles_repaired.py')
old="for label,trial in [('initial','vector-vision-native-20261016'),('repeat','vector-vision-native-confirm-20261016')]:";assert source.count(old)==1
source=source.replace(old,"for label,trial in [('repeat','vector-vision-native-confirm-20261016')]:")
old='        labels=[]';assert source.count(old)==1
source=source.replace(old,"""        prior=parent/'logs/vector-vision-native-continuation-20261016'
        assert json.loads((prior/'exit.json').read_text())['exit_code']==1
        failure=parent/'vector-native-auditor-path-failure-20261016.tar.gz'
        assert digest(failure)=='252aeda0ce11fbe3849f9127111504715f16cc8f4ceb17362bab6d3c28168c94'
        initial=parent/'vector-vision-native-20261016.tar.gz'
        assert digest(initial)=='4391422a668b2e4b551c981635e252e00e5274f3ffa8eabd7259a325759bded5'
        evidence=root/'validation-audits/vector-vision-native-results/initial'
        repaired=json.loads((evidence/'verification.json').read_text())
        assert repaired['audit_path_repair_only'] and repaired['regenerated_native_comparators']==6 and repaired['distinct_expanded_script_files']==3
        decision=json.loads((evidence.parent/'decision-initial.json').read_text());assert decision['repeat_admitted']
        (out/'initial-archive.json').write_bytes((prior/'initial-archive.json').read_bytes())
        (out/'resume-provenance.json').write_text(json.dumps(dict(reused_complete_initial_archive_sha256=digest(initial),
            preserved_controller_failure_archive_sha256=digest(failure),initial_repaired_audit=repaired,
            scope='Audit filename only repaired. All runtime/benchmark sources, bounds, cases and predeclared selection unchanged; initial model timings never rerun.'),indent=2))
        labels=['initial']""")
exec(compile(source,__file__,'exec'))
