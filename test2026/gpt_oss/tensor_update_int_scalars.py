"""XLang3: validate batched decode-control copies on every available GPU."""
import json
import garnet as G


devices = json.loads(G.cuda_devices_json())
if not devices:
    raise RuntimeError('CUDA GPU required')

for device in devices[:2]:
    previous = G.cuda_set_device(device['id'])
    try:
        tensors = [
            G.tensor_from_host([0], dtype='int64', shape=[1], device='cuda'),
            G.tensor_from_host([0], dtype='int64', shape=[1], device='cuda'),
            G.tensor_from_host([0], dtype='int32', shape=[1], device='cuda'),
            G.tensor_from_host([0], dtype='int32', shape=[1], device='cuda'),
        ]
        for values in ([200005, 4096, 4097, 4096], [0, 2**33, 1, -1]):
            assert G.tensor_update_int_scalars(tensors, values)
            assert [G.tensor_to_cpu(t).tolist()[0] for t in tensors] == values
        assert not G.tensor_update_int_scalars(tensors, [1, 2])
        assert not G.tensor_update_int_scalars(tensors, [1, 2, 2**33, 4])
        for values in ([17, 2**33, 3, -2], [200005, 4096, 4097, 4096]):
            assert G.tensor_update_int_scalars_async(tensors, values)
            G.cuda_synchronize()
            assert [G.tensor_to_cpu(t).tolist()[0] for t in tensors] == values
        assert not G.tensor_update_int_scalars_async(tensors, [1, 2])
        assert not G.tensor_update_int_scalars_async(tensors, [1, 2, 2**33, 4])
        print('batched scalar update passed on GPU', device['id'], flush=True)
    finally:
        G.cuda_set_device(previous)
