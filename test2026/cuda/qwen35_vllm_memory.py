"""Named local worker extension for diagnostic-only scalar memory RPCs."""
class MemoryExtension:
    def qwen35_memory_snapshot(self,reset=False):
        import torch,os
        torch.cuda.synchronize()
        if reset:torch.cuda.reset_peak_memory_stats()
        free,total=torch.cuda.mem_get_info()
        return {'worker_process_id':os.getpid(),'allocated_bytes':torch.cuda.memory_allocated(),
            'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'whole_device_used_bytes':total-free,
            'scope':'Worker torch allocator; driver/library allocations may be absent. mem_get_info is whole-device.'}
