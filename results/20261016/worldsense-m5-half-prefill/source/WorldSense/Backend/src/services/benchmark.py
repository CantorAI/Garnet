"""Fixed archived-frame experiments through the production Galaxy Skill filter.

Explicit current-frame Skills only: no scheduling, detector, temporal memory,
WorldState commits, event delivery or DB writes are exercised by this path.
Ordinary live-video tests remain necessary for those integration behaviors.
"""
import hashlib
import json
import time
import threading
from pathlib import Path

from cantorproxy import GetGalaxy
from perception.native_values import to_native, to_python
from services.skill_assignment_runtime import INFERENCE_PIPELINE, _skill_specs, _vision_source_filter, sync_skill_assignments
from services.skill_lab import read_observations, public_values

_benchmark_lock = threading.Lock()


def recorded_suite(root, cases, repeats, queries):
    if not _benchmark_lock.acquire(blocking=False):
        raise ValueError('Another recorded-frame benchmark is running')
    results = []
    failure = None
    restored = False
    plans = []
    try:
        source = _vision_source_filter()
        source.SetSkillAssignments(to_native([]))
        # Every case warms once in the declared order. First requests retain
        # setup and queued-live-work cost and are never pooled with warm calls.
        for repetition in range(repeats + 1):
            for case in cases:
                sample = recorded_frame(root, case['observation_id'], case['start_ms'],
                                        case['end_ms'], case['skill_ref'])
                sample['repetition'] = repetition
                sample['warmup'] = repetition == 0
                results.append(sample)
            instances = GetGalaxy().QueryPipelineInstancesByFileName(str(INFERENCE_PIPELINE))
            vlm = instances[0]['object'].QueryFilter('Skill-VLM')
            for query in queries:
                started = time.monotonic()
                plan = to_python(vlm.PlanSearch(query, to_native({'kind': 'llm', 'benchmark': True})))
                plans.append(dict(query=query, repetition=repetition, warmup=repetition == 0,
                                  native_call_seconds=time.monotonic()-started, result=public_values(plan)))
    except Exception as error:
        failure = str(error)
    finally:
        try:
            sync_skill_assignments()
            restored = True
        except Exception as error:
            failure = (failure or '') + '; assignment restore failed: ' + str(error)
        _benchmark_lock.release()
    return dict(valid=failure is None and restored and all(
                    sample['result'].get('status') == 'ok' for sample in results),
                error=failure, assignments_restored=restored, items=results, search_plans=plans,
                scope='Fixed recorded frames; production Galaxy current-Skill filter; live Skill inference suspended during batch')


def current_candidate(skill, ref, row):
    event, search, vlm = skill.get('event', {}), skill.get('search', {}), skill.get('vlm', {})
    if event.get('enabled') and event.get('evaluation') != 'current':
        raise ValueError('Benchmark admits explicit current-frame or search-only Skills')
    if not skill.get('enabled', True) or not (event.get('enabled') or search.get('enabled')):
        raise ValueError('Benchmark requires an enabled Skill with event or search output')
    filtering = skill.get('filter', {})
    if filtering.get('geometry', {}).get('enabled'):
        raise ValueError('Geometry/tracking Skills require the live detector pipeline')
    prompt = vlm.get('prompt', '')
    if filtering.get('semantic_query'):
        prompt += '\nRequired visual condition: ' + filtering['semantic_query']
    if filtering.get('objects'):
        prompt += '\nObject constraints (' + filtering.get('combine', '') + '):'
        prompt += ''.join(' ' + item for item in filtering['objects'])
    return dict(kind='skill_candidate', image_id=row['image_id'], skill_ref=ref,
                skill_id=skill['id'], skill_name=skill.get('name', skill['id']),
                jpeg=row['image_path'], detections=[], event_evaluation='current',
                event_enabled=bool(event.get('enabled')), direct_event=False, verify_event=True,
                search_enabled=bool(search.get('enabled')), vlm_prompt=prompt,
                output_language=vlm.get('output_language', 'en'), search_focus=search.get('focus', ''))


def recorded_frame(root, observation_id, start_ms, end_ms, skill_ref):
    rows, truncated = read_observations(root, start_ms, end_ms, 1000)
    matches = [row for row in rows if row['observation_id'] == observation_id]
    if truncated or len(matches) != 1:
        raise ValueError('Select a unique recorded observation in a narrower time window')
    row = matches[0]
    path = Path(row['image_path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Recorded frame is unavailable or outside the TimeDB archive')
    skill = _skill_specs().get(skill_ref)
    if skill is None:
        raise ValueError('Unknown Skill reference')
    candidate = current_candidate(skill, skill_ref, row)
    instances = GetGalaxy().QueryPipelineInstancesByFileName(str(INFERENCE_PIPELINE))
    if not instances or not len(instances):
        raise RuntimeError('WorldSense Skill pipeline is not running')
    vlm = instances[0]['object'].QueryFilter('Skill-VLM')
    if not vlm:
        raise RuntimeError('WorldSense Skill-VLM filter is unavailable')
    # Archive identity before entering the timed Galaxy call. Local snapshot
    # path is resolved by native Garnet just as for other recorded-file inputs.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = json.dumps(skill)
    started = time.monotonic()
    result = to_python(vlm.BenchmarkCurrentFrame(to_native(candidate)))
    wall = time.monotonic() - started
    result.pop('jpeg', None)
    return dict(scope='WorldSense recorded-frame current-Skill → Galaxy → Garnet; no state/event commits',
                observation_id=observation_id, image_sha256=digest, skill_ref=skill_ref,
                skill_sha256=hashlib.sha256(spec.encode()).hexdigest(), skill=skill,
                native_call_seconds=wall, result=public_values(result))
