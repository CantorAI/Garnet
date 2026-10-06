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


class TensorParallel:
    """Lockstep two-rank TensorRT execution with NCCL collectives."""
    def __init__(self, stages):
        from concurrent.futures import ThreadPoolExecutor
        if len(stages) != 2:
            raise ValueError('TensorParallel requires exactly two rank stages')
        self.stages = stages
        self.executor = ThreadPoolExecutor(max_workers=2)

    def forward(self, activation, controls, sample=False):
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
                    stage['values'], local[1], local[2], local[3], local[4]]}
                if sample and stage['rank'] == 0:
                    request['sample'] = 'greedy'
                prepared.append((stage, request))
            finally:
                G.cuda_set_device(previous)

        return self._run_prepared(prepared)

    def forward_rank_local(self, rank_inputs, sample=False, scalar_values=None):
        """Run tensors already resident on their corresponding rank device."""
        if len(rank_inputs) != len(self.stages):
            raise ValueError('one input group per tensor-parallel rank is required')
        if scalar_values is not None and len(scalar_values) != 4:
            raise ValueError('rank-local scalar update requires token, position, length and slot')
        prepared = []
        updates = []
        for stage, (activation, controls) in zip(self.stages, rank_inputs):
            if len(controls) != 5:
                raise ValueError('rank-local controls must contain position, page table, length, slot and active')
            request = {'inputs': [activation, controls[0], stage['keys'], stage['values'],
                controls[1], controls[2], controls[3], controls[4]]}
            if sample and stage['rank'] == 0:
                request['sample'] = 'greedy'
            prepared.append((stage, request))
            if scalar_values is not None:
                updates.append(([activation, controls[0], controls[2], controls[3]], scalar_values))
        return self._run_prepared(prepared, updates if scalar_values is not None else None)

    def _run_prepared(self, prepared, updates=None):
        import garnet as G
        import os

        def run(stage, request, update):
            previous = G.cuda_set_device(stage['device_id'])
            try:
                trace = os.environ.get('GARNET_TP_TRACE')
                if update is not None and not G.tensor_update_int_scalars_async(*update):
                    raise RuntimeError('rank-local scalar update failed')
                if trace:
                    print('tp-rank', stage['rank'], 'enter model forward', flush=True)
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
        return results[0]

    def release(self):
        import garnet as G
        previous = G.cuda_set_device(self.stages[0]['device_id'])
        try:
            self.executor.shutdown(wait=True)
            for stage in self.stages:
                G.cuda_set_device(stage['device_id'])
                stage['model'].release_runtime()
        finally:
            G.cuda_set_device(previous)
