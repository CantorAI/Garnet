from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-memory/source')
shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source',s,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('class MetalContext {', '''struct MetalMemoryStats {
    uint64_t allocated=0,recommended=0,registry=0,physical=0,resident=0,compressed=0,internal=0,virtualBytes=0;
    int taskStatus=-1;
};
class MetalContext {''',1).replace('    static std::string AvailableDeviceName();','    static std::string AvailableDeviceName();\n    static MetalMemoryStats MemoryStats();');p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text().replace('#include <cstring>','#include <cstring>\n#include <mach/mach.h>');idx=t.index('bool MetalContext::PrepareBF16BodyShape');t=t[:idx]+'''MetalMemoryStats MetalContext::MemoryStats() {
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device)throw std::runtime_error("No Metal device for memory snapshot");
        MetalMemoryStats stats;stats.allocated=device.currentAllocatedSize;
        stats.recommended=device.recommendedMaxWorkingSetSize;stats.registry=device.registryID;
        task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
        stats.taskStatus=task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count);
        if(stats.taskStatus==KERN_SUCCESS){stats.physical=info.phys_footprint;stats.resident=info.resident_size;
            stats.compressed=info.compressed;stats.internal=info.internal;stats.virtualBytes=info.virtual_size;}
        return stats;
    }
}
'''+t[idx:];p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text().replace('    X::Value PrepareImage(','''    std::string MemoryStatsJson() {
        const auto m=MetalContext::MemoryStats();
        return nlohmann::json{{"metal_allocated_bytes",m.allocated},{"metal_recommended_working_set_bytes",m.recommended},
            {"metal_registry_id",m.registry},{"task_info_status",m.taskStatus},{"task_physical_footprint_bytes",m.physical},
            {"task_resident_bytes",m.resident},{"task_compressed_bytes",m.compressed},{"task_internal_bytes",m.internal},
            {"task_virtual_bytes",m.virtualBytes},{"scope","process Metal device resources and current task; snapshots, not peak or system free RAM"}}.dump();
    }
    X::Value PrepareImage(''',1);t=t.replace('        APISET().AddFunc<0>("backend_info"','        APISET().AddFunc<0>("memory_stats_json",&GarnetMetalAPI::MemoryStatsJson);\n        APISET().AddFunc<0>("backend_info"');p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text().replace('        self.root, self.name = root, name','        self.initialization_memory = dict(before=json.loads(garnet.memory_stats_json()), other_model_ids_at_start=list(_models))\n        self.root, self.name = root, name',1)
t=t.replace('    def _prefill_with_prefix(', "        self.initialization_memory['after_decoder_ready'] = json.loads(garnet.memory_stats_json())\n\n    def _prefill_with_prefix(",1)
t=t.replace('        stages = {}',"        stages = dict(initialization_memory=self.initialization_memory, memory_before_request=json.loads(garnet.memory_stats_json()))",1)
t=t.replace('        prefill_start = time.monotonic()',"        stages['memory_after_input'] = json.loads(garnet.memory_stats_json())\n        prefill_start = time.monotonic()",1)
t=t.replace('        first = time.monotonic()',"        stages['memory_after_prefill'] = json.loads(garnet.memory_stats_json())\n        first = time.monotonic()",1)
t=t.replace("        stages['decode_seconds'] = time.monotonic() - first", "        stages['decode_seconds'] = time.monotonic() - first\n        stages['memory_after_decode'] = json.loads(garnet.memory_stats_json())",1);p.write_text(t)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
p=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-prefix-cache/controller.py').read_text().replace('worldsense-prefix-cache','worldsense-memory');(w/'controller.py').write_text(p)
cases=json.loads(Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-mixed-chunk/confirmation-cases.json').read_text());cases['paired_prefill']=False;cases['repeats']=1;(w/'cases.json').write_text(json.dumps(cases,indent=2))
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-12 diagnostic','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Recent large phase variability may involve Metal allocation/working-set or task compression pressure; measure before attributing cause or choosing memory change.','method':'Unchanged WS10 inference math/policy; passive native device currentAllocatedSize/recommendedMaxWorkingSetSize plus mach task_vm_info snapshots at model init start/ready and each real WorldSense request input/prefill/decode boundary. All snapshots process scoped, not per-model allocations/peak/free system RAM.','workload':'Three fixed archived frames/Skills and one query per window, two windows total. Fresh image encodes; VLM64 and text16 native prefix retained.12 expected model calls including first window, not a performance comparison.','validation':'Native reference tests and snapshot lifecycle allocation/release tests, complete full model output/prompt/config/input identity parity against WS10 confirmation. No memory/speed improvement claim or promotion from diagnostic instrumentation.','restoration':'Controller finally restores exact validated WS10 binary/live assignments; restart normal without benchmark access.'},indent=2))
print('Prepared passive WorldSense memory diagnostic')
