"""Generic contiguous-layer placement and sequential GPU pipeline execution.

Memory estimates are supplied by an xModel adapter. This initial scheduler
serializes stage execution; it does not overlap independent requests.
"""
try:
    from _sha2 import sha256
except ImportError:
    from _sha256 import sha256
import json


def plan_layers(devices, layer_bytes, first_bytes=0, last_bytes=0,
                reserve_bytes=1 << 30, memory_fraction=.9):
    if not devices or len(devices) > len(layer_bytes):
        raise ValueError('select between one and num_layers GPUs')
    if len(set(d['id'] for d in devices)) != len(devices):
        raise ValueError('GPU IDs must be unique')
    if not 0 < memory_fraction <= 1 or reserve_bytes < 0:
        raise ValueError('invalid memory budget')
    if any(n < 0 for n in layer_bytes) or first_bytes < 0 or last_bytes < 0:
        raise ValueError('memory estimates must be nonnegative')
    budgets = [int(min(d['free_bytes'], d['total_bytes'] * memory_fraction)) - reserve_bytes
               for d in devices]
    prefix = [0]
    for size in layer_bytes:
        prefix.append(prefix[-1] + size)
    # Dynamic programming finds contiguous cuts that fit every individual GPU,
    # minimizing the largest fraction of any GPU's usable budget.
    states = {0: (0., [])}
    for rank, budget in enumerate(budgets):
        next_states = {}
        if budget <= 0:
            raise ValueError('GPU has no usable memory after reserve')
        for start, (score, cuts) in states.items():
            for end in range(start + 1, len(layer_bytes) - (len(devices) - rank - 1) + 1):
                size = prefix[end] - prefix[start]
                if rank == 0:
                    size += first_bytes
                if rank == len(devices) - 1:
                    size += last_bytes
                if size > budget:
                    break
                candidate = max(score, size / budget)
                if end not in next_states or candidate < next_states[end][0]:
                    next_states[end] = (candidate, cuts + [(start, end, size)])
        states = next_states
    if len(layer_bytes) not in states:
        raise ValueError('model does not fit selected GPUs; reduce context/batch, add VRAM, or implement offloading')
    cuts = states[len(layer_bytes)][1]
    hardware = [{k: d[k] for k in ('id', 'name', 'total_bytes', 'compute_major',
                'compute_minor', 'pci_bus_id', 'peer_access')} for d in devices]
    identity = {'schema': 1, 'hardware': hardware, 'cuts': cuts,
                'reserve_bytes': reserve_bytes, 'memory_fraction': memory_fraction}
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return {'schema': 1, 'cache_key': key, 'hardware': hardware,
            'stages': [{'device_id': d['id'], 'start': cut[0], 'end': cut[1],
                        'estimated_bytes': cut[2], 'budget_bytes': budget}
                       for d, cut, budget in zip(devices, cuts, budgets)]}


class Pipeline:
    """Models and persistent inputs are owned by their assigned GPU stage."""
    def __init__(self, stages):
        self.stages = stages

    def forward(self, activation, controls, sample=False):
        import garnet as G
        previous = G.cuda_set_device(self.stages[0]['device_id'])
        try:
            result = None
            for rank, stage in enumerate(self.stages):
                G.cuda_set_device(stage['device_id'])
                activation = G.tensor_to_device(activation, stage['device_id'])
                local = [G.tensor_to_device(t, stage['device_id']) for t in controls]
                # controls: position, page table, context length, slot, active mask
                request = {'inputs': [activation, local[0], stage['keys'], stage['values']] + local[1:]}
                if sample and rank == len(self.stages) - 1:
                    request['sample'] = 'greedy'
                result = stage['model'].forward(request)
                if result['status'] != 'ok':
                    raise RuntimeError(str(result))
                G.cuda_synchronize()
                if rank < len(self.stages) - 1 or not sample:
                    activation = result['output']
            return result
        finally:
            G.cuda_set_device(previous)

    def release(self):
        import garnet as G
        previous = G.cuda_set_device(self.stages[0]['device_id'])
        try:
            for stage in self.stages:
                G.cuda_set_device(stage['device_id'])
                stage['model'].release_runtime()
        finally:
            G.cuda_set_device(previous)


class ResidentTensorParallel:
    """Own two serial TP models with shared rank-local KV and fenced handoff.

    TensorParallel waits for rank stream completion before returning. This
    owner serializes whole forward calls across the two phase executors.
    """
    def __init__(self, prefill, decode):
        from threading import Lock
        self._lock = Lock()
        self._prefill, self._decode = prefill, decode
        self._operator_phase_binder = None
        self._operator_group_releaser = None
        if len(prefill.stages) != len(decode.stages):
            raise ValueError('Resident phase rank counts differ')
        for first, second in zip(prefill.stages, decode.stages):
            _extra_stage_inputs(first)
            _extra_stage_inputs(second)
            if (first['device_id'] != second['device_id'] or
                    first['keys'] is not second['keys'] or first['values'] is not second['values']):
                raise ValueError('Resident phases must share exact rank-local KV objects')
            extras = first.get('shared_resources', {})
            other = second.get('shared_resources', {})
            if (first.get('shared_resource_layout') != second.get('shared_resource_layout') or
                    first.get('extra_input_names', ()) != second.get('extra_input_names', ()) or
                    extras.keys() != other.keys() or
                    any(value is not other[name] for name, value in extras.items())):
                raise ValueError('Resident phases must share every auxiliary object and layout')
        self.prefill_stages, self.decode_stages = prefill.stages, decode.stages

    @classmethod
    def build(cls, build_prefill, build_decode):
        prefill = decode = None
        try:
            prefill = build_prefill()
            kv = cls.shared_rank_resources(prefill)
            decode = build_decode(kv)
            return cls(prefill, decode)
        except Exception:
            try:
                if decode is not None:
                    decode.release()
            finally:
                if prefill is not None:
                    prefill.release()
            raise

    @staticmethod
    def shared_rank_resources(model):
        """Old stages retain tuple K/V; explicit auxiliary layouts carry all tensors."""
        result = []
        for stage in model.stages:
            _extra_stage_inputs(stage)
            if stage.get('shared_resources'):
                result.append(dict(keys=stage['keys'], values=stage['values'],
                    shared_resources=dict(stage['shared_resources']),
                    shared_resource_layout=json.loads(json.dumps(stage['shared_resource_layout'])),
                    extra_input_names=stage['extra_input_names']))
            else:
                result.append((stage['keys'], stage['values']))
        return result

    def attach_operator_execution_group(self, group, bind_phase, release_group=None):
        with self._lock:
            if self._prefill is None or self._operator_phase_binder is not None:
                raise RuntimeError('Released or already bound resident execution group')
            if not callable(bind_phase):
                raise ValueError('Explicit phase binder required')
            if release_group is not None and not callable(release_group):
                raise ValueError('Explicit group releaser must be callable')
            self._prefill.operator_execution_group = group
            self._prefill.operator_execution_phase = 0
            self._decode.operator_execution_group = group
            self._decode.operator_execution_phase = 1
            self._operator_phase_binder = lambda phase: bind_phase(group, phase)
            if release_group is not None:
                self._operator_group_releaser = lambda: release_group(group)

    def forward_prefill(self, rank_inputs, **kwargs):
        with self._lock:
            if self._prefill is None:
                raise RuntimeError('Resident models were released')
            if self._operator_phase_binder is not None:
                self._operator_phase_binder(0)
            return self._prefill.forward_rank_local(rank_inputs, **kwargs)

    def forward_decode(self, rank_inputs, **kwargs):
        with self._lock:
            if self._decode is None:
                raise RuntimeError('Resident models were released')
            if self._operator_phase_binder is not None:
                self._operator_phase_binder(1)
            return self._decode.forward_rank_local(rank_inputs, **kwargs)

    def release(self):
        with self._lock:
            try:
                if self._decode is not None:
                    self._decode.release()
            finally:
                self._decode = None
                try:
                    if self._prefill is not None:
                        self._prefill.release()
                finally:
                    self._prefill = None
                    self._operator_phase_binder = None
                    release_group = self._operator_group_releaser
                    self._operator_group_releaser = None
                    if release_group is not None:
                        release_group()


def _extra_stage_inputs(stage):
    resources = stage.get('shared_resources', {})
    names = stage.get('extra_input_names', ())
    if any(key in stage for key in ('shared_resources', 'extra_input_names', 'shared_resource_layout')):
        if (not isinstance(resources, dict) or not isinstance(names, tuple) or
                not resources or any(not isinstance(name, str) or not name for name in names) or
                len(set(names)) != len(names) or set(names) != set(resources) or
                not stage.get('shared_resource_layout')):
            raise ValueError('Auxiliary tensor inputs need an exact ordered resource/layout contract')
    return [resources[name] for name in names]


class TensorParallel:
    """Lockstep two-rank TensorRT execution with NCCL collectives."""
    def __init__(self, stages, greedy_candidate_pairs=False):
        from concurrent.futures import ThreadPoolExecutor
        if len(stages) != 2:
            raise ValueError('TensorParallel requires exactly two rank stages')
        self.stages = stages
        self.greedy_candidate_pairs = greedy_candidate_pairs
        self.operator_execution_group = None
        self.operator_execution_phase = 0
        self.executor = ThreadPoolExecutor(max_workers=2)

    def forward(self, activation, controls, sample=False, sample_batch=False,
                reuse_output=False, native_candidate_merge=False, native_greedy_state=None):
        import garnet as G
        import os

        prepared = []
        for stage in self.stages:
            previous = G.cuda_set_device(stage['device_id'])
            try:
                trace = os.environ.get('GARNET_TP_TRACE')
                if trace:
                    print('tp-rank', stage['rank'], 'copying inputs', flush=True)
                local_activation = G.tensor_to_device(activation, stage['device_id'])
                local = [G.tensor_to_device(t, stage['device_id']) for t in controls]
                request = {'inputs': [local_activation, local[0], stage['keys'],
                    stage['values'], local[1], local[2], local[3], local[4]] + _extra_stage_inputs(stage)}
                if reuse_output:
                    request['reuse_output'] = True
                if sample and stage['rank'] == 0 and not self.greedy_candidate_pairs:
                    request['sample'] = 'greedy_batch' if sample_batch else 'greedy'
                prepared.append((stage, request))
            finally:
                G.cuda_set_device(previous)

        return self._run_prepared(prepared, compact_sample=sample and self.greedy_candidate_pairs,
                                  native_candidate_merge=native_candidate_merge,
                                  native_greedy_state=native_greedy_state)

    def forward_rank_local(self, rank_inputs, sample=False, scalar_values=None,
                           sample_batch=False, vector_values=None, reuse_output=False,
                           native_candidate_merge=False, pattern_values=None,
                           native_greedy_state=None):
        """Run tensors already resident on their corresponding rank device."""
        if len(rank_inputs) != len(self.stages):
            raise ValueError('one input group per tensor-parallel rank is required')
        if scalar_values is not None and len(scalar_values) != 4:
            raise ValueError('rank-local scalar update requires token, position, length and slot')
        if vector_values is not None and (len(vector_values) != 4 or scalar_values is not None):
            raise ValueError('rank-local update requires either four scalars or four vectors')
        if pattern_values is not None and (len(pattern_values) != 4 or
                scalar_values is not None or vector_values is not None):
            raise ValueError('rank-local pattern update requires four exclusive integer patterns')
        prepared = []
        updates = []
        for stage, (activation, controls) in zip(self.stages, rank_inputs):
            if len(controls) != 5:
                raise ValueError('rank-local controls must contain position, page table, length, slot and active')
            request = {'inputs': [activation, controls[0], stage['keys'], stage['values'],
                controls[1], controls[2], controls[3], controls[4]] + _extra_stage_inputs(stage)}
            if reuse_output:
                request['reuse_output'] = True
            if sample and stage['rank'] == 0 and not self.greedy_candidate_pairs:
                request['sample'] = 'greedy_batch' if sample_batch else 'greedy'
            prepared.append((stage, request))
            if scalar_values is not None:
                updates.append(([activation, controls[0], controls[2], controls[3]], scalar_values))
            elif vector_values is not None:
                updates.append(([activation, controls[0], controls[2], controls[3]], vector_values))
            elif pattern_values is not None:
                updates.append(([activation, controls[0], controls[2], controls[3]], pattern_values))
        return self._run_prepared(prepared, updates if updates else None,
                                  vector_updates=vector_values is not None,
                                  pattern_updates=pattern_values is not None,
                                  compact_sample=sample and self.greedy_candidate_pairs,
                                  native_candidate_merge=native_candidate_merge,
                                  native_greedy_state=native_greedy_state)

    def _run_prepared(self, prepared, updates=None, vector_updates=False, compact_sample=False,
                      native_candidate_merge=False, pattern_updates=False, native_greedy_state=None):
        import garnet as G
        import os

        def run(stage, request, update):
            previous = G.cuda_set_device(stage['device_id'])
            try:
                trace = os.environ.get('GARNET_TP_TRACE')
                if update is not None:
                    updater = (G.tensor_update_int_patterns_async if pattern_updates else
                               G.tensor_update_int_vectors_async if vector_updates else
                               G.tensor_update_int_scalars_async)
                    if not updater(*update):
                        raise RuntimeError('rank-local integer update failed')
                if trace:
                    print('tp-rank', stage['rank'], 'enter model forward', flush=True)
                if self.operator_execution_group is not None:
                    request.update(operator_execution_group=self.operator_execution_group,
                        operator_execution_rank=stage['rank'], operator_execution_phase=self.operator_execution_phase)
                result = stage['model'].forward(request)
                if trace:
                    print('tp-rank', stage['rank'], 'returned model forward', flush=True)
                if result['status'] != 'ok':
                    raise RuntimeError(str(result))
                G.cuda_synchronize()
                if trace:
                    print('tp-rank', stage['rank'], 'synchronized', flush=True)
                return result
            finally:
                G.cuda_set_device(previous)

        futures = [self.executor.submit(run, stage, request,
            updates[index] if updates is not None else None)
            for index, (stage, request) in enumerate(prepared)]
        results = [future.result() for future in futures]
        if compact_sample:
            if native_greedy_state is not None:
                previous = G.cuda_set_device(self.stages[0]['device_id'])
                try:
                    return {'status': 'ok', 'token_ids': G.greedy_batch_state_step(
                        native_greedy_state, results[0]['output'])}
                finally:
                    G.cuda_set_device(previous)
            previous = G.cuda_set_device(self.stages[0]['device_id'])
            try:
                if native_candidate_merge:
                    return G.merge_greedy_candidate_pairs(results[0]['output'])
                pairs = G.tensor_to_cpu(results[0]['output']).tolist()
            finally:
                G.cuda_set_device(previous)
            if not pairs or len(pairs) % 4:
                raise ValueError('Greedy candidate output must contain two score/ID pairs per row')
            ids, values = [], []
            for i in range(0, len(pairs), 4):
                a, ai, b, bi = pairs[i:i + 4]
                if ai != int(ai) or bi != int(bi) or not (0 <= ai < (1 << 24) and 0 <= bi < (1 << 24)):
                    raise ValueError('Greedy candidate IDs must be exact nonnegative FP32 integers')
                take_b = b > a or (b == a and bi < ai)
                ids.append(int(bi if take_b else ai))
                values.append(b if take_b else a)
            return {'status': 'ok', 'token_ids': ids, 'token_values': values,
                    'token_id': ids[0]}
        return results[0]

    def release(self):
        import garnet as G
        previous = G.cuda_set_device(self.stages[0]['device_id'])
        try:
            self.executor.shutdown(wait=True)
            for stage in self.stages:
                G.cuda_set_device(stage['device_id'])
                stage['model'].release_runtime()
            self.operator_execution_group = None
        finally:
            G.cuda_set_device(previous)
