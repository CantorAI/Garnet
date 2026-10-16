"""Complete sequential native cohorts only behind exact terminal/archived/audited full32 correctness."""
from pathlib import Path
p=Path(__file__).with_name('continue_paired_norm_rope_native.py');source=p.read_text()
source=source.replace('paired-norm-rope-native','delta-value-tile64-native').replace('full34','full32')
source=source.replace("gate['validation_stages']==34","gate['validation_stages']==32")
for old,new in [('launch_paired_norm_rope_native_profiles.py','launch_delta_value_tiles_native_profiles.py'),('archive_paired_norm_rope_native_profiles.py','archive_delta_value_tiles_native_profiles.py'),('verify_paired_norm_rope_native_profiles.py','verify_delta_value_tiles_native_profiles.py'),('analyze_paired_norm_rope_native_results.py','analyze_delta_value_tiles_native_results.py')]:
 assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
