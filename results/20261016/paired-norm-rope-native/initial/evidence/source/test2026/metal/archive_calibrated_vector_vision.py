"""Freeze terminal calibration/candidate cohorts under the new prospective design."""
from pathlib import Path
p=Path(__file__).with_name('archive_vector_vision_component.py');source=p.read_text()
source=source.replace("['vector-vision-context-20261016']","['vector-vision-calibration-repaired-20261016']")
source=source.replace("'vector-vision-context-20261016','vector-vision-context-confirm-20261016'",
    "'vector-vision-calibration-20261016','vector-vision-calibration-confirm-20261016','vector-vision-calibration-repaired-20261016','vector-vision-calibration-repaired-confirm-20261016','vector-vision-calibrated-20261016','vector-vision-calibrated-confirm-20261016'")
source=source.replace("'archive_vector_vision_component.py','launch_vector_vision_context.py','verify_vector_vision_context.py','analyze_vector_vision_context.py'",
    "'archive_calibrated_vector_vision.py','archive_vector_vision_component.py','launch_calibrated_vector_vision.py','verify_calibrated_vector_vision.py','analyze_calibrated_vector_null.py','analyze_vector_vision_context.py'")
exec(compile(source,__file__,'exec'))
