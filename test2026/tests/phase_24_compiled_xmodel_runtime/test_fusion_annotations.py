# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def discover_garnet_dll():
    explicit = os.environ.get("GARNET_DLL_PATH", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.extend([
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise AssertionError("garnet.dll not found")


def add_windows_dll_dirs(garnet_dll):
    handles = []
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return handles
    for path in [
        garnet_dll.parent,
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
    ]:
        if path.exists():
            handles.append(os.add_dll_directory(str(path)))
    return handles


garnet_dll = discover_garnet_dll()
_dll_handles = add_windows_dll_dirs(garnet_dll)

import xlang3

garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
model_path = SCRIPT_DIR / "compiled_fusion_regions_model.x"
cache_dir = SCRIPT_DIR / "cache" / "fusion_annotations"
shutil.rmtree(cache_dir, ignore_errors=True)


def load_fixture():
    return garnet.load_model(
        str(model_path),
        runtime_mode="compiled_xmodel",
        entry_function="Model",
        input_shapes=[[1, 4]],
        cache_dir=str(cache_dir),
    )


model = load_fixture()
status = model.runtime_status()
assert status["state"] == "compiled_engine_ready", status
assert status["engines_prepared"], status
assert float(status["engine_prepare_ms"]) >= 0.0, status
assert status["scheduler"] == "cpu_control_gpu_execution", status
assert status["engine_partition_count"] == 2, status
plan = json.loads(status["execution_plan_json"])
assert plan["control_plane"] == "cpu", plan
assert plan["tensor_plane"] == "gpu", plan
assert plan["intermediate_host_copies"] is False, plan
assert plan["device_wide_synchronization"] is False, plan
assert plan["execution_mode"] == "partitioned_engines", plan
assert plan["partition_state"] == "physical_engines_compiled", plan
assert len(plan["engine_partitions"]) == 2, plan
assert all(Path(partition["engine_path"]).exists() for partition in plan["engine_partitions"]), plan
edge_output = plan["engine_partitions"][0]["outputs"][0]
edge_input = plan["engine_partitions"][1]["inputs"][0]
assert edge_output["name"] == edge_input["name"], plan
assert edge_output["tensor_id"] == edge_input["tensor_id"], plan

regions = plan["regions"]
assert [region["name"] for region in regions] == [
    "fixture_root",
    "input_projection",
    "AtomicBlock",
    "AtomicBlock",
    "AtomicBlock",
    "AtomicBlock",
    "AtomicBlock",
    "final_projection",
], regions
assert [region["invocation"] for region in regions if region["name"] == "AtomicBlock"] == list(range(5))
assert all(region["atomic"] for region in regions if region["name"] == "AtomicBlock")
assert all(len(region["input_tensor_ids"]) == 1 for region in regions), regions
assert all(len(region["output_tensor_ids"]) == 1 for region in regions), regions
assert regions[0]["operation_count"] == 0, regions
assert all(region["operation_count"] == 1 for region in regions[1:]), regions
assert next(region for region in regions if region["name"] == "input_projection")["boundary"] == "preferred"
assert next(region for region in regions if region["name"] == "final_projection")["candidate_partition"] == 1
assert regions[0]["cuda_graph"] is True
assert len(plan["operations"]) == 7, plan
assert [operation["candidate_partition"] for operation in plan["operations"]] == [
    0, 0, 0, 0, 0, 0, 1
], plan

input_tensor = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
result = model.forward({"inputs": [input_tensor]})
assert result["status"] == "ok", result
actual = np.asarray(garnet.tensor_to_cpu(result["output"]).tolist(), dtype=np.float32)
np.testing.assert_allclose(actual, np.asarray([[128.0, 256.0, 384.0, 512.0]], dtype=np.float32))

cached_model = load_fixture()
cached_status = cached_model.runtime_status()
cached_plan = json.loads(cached_status["execution_plan_json"])
assert cached_plan == plan
assert cached_status["engine_partition_count"] == 2, cached_status
assert cached_status["forbidden_path_counters"]["graph_cache_hits"] == 1
assert cached_status["forbidden_path_counters"]["root_x_executions"] == 0

preferred_cache = SCRIPT_DIR / "cache" / "fusion_preferred_planner"
shutil.rmtree(preferred_cache, ignore_errors=True)
preferred_model = garnet.load_model(
    str(model_path),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    compile={"partition": {"preferred_min_operations": 1}},
    cache_dir=str(preferred_cache),
)
preferred_status = preferred_model.runtime_status()
preferred_plan = json.loads(preferred_status["execution_plan_json"])
assert preferred_status["engine_partition_count"] == 3, preferred_status
assert preferred_plan["planner_options"]["preferred_min_operations"] == 1
assert preferred_plan["operations"][0]["partition_reason"].startswith("preferred:"), preferred_plan
preferred_result = preferred_model.forward({"inputs": [input_tensor]})
assert preferred_result["status"] == "ok", preferred_result
preferred_actual = np.asarray(
    garnet.tensor_to_cpu(preferred_result["output"]).tolist(), dtype=np.float32
)
np.testing.assert_allclose(preferred_actual, actual)

atomic_cache = SCRIPT_DIR / "cache" / "fusion_atomic_planner"
shutil.rmtree(atomic_cache, ignore_errors=True)
atomic_model = garnet.load_model(
    str(model_path),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    compile={"partition": {
        "enable_preferred_boundaries": False,
        "max_atomic_regions_per_partition": 2,
    }},
    cache_dir=str(atomic_cache),
)
atomic_status = atomic_model.runtime_status()
atomic_plan = json.loads(atomic_status["execution_plan_json"])
assert atomic_status["engine_partition_count"] == 5, atomic_status
atomic_layer_operations = [
    operation for operation in atomic_plan["operations"]
    if operation["region_id"] in {
        region["id"] for region in atomic_plan["regions"]
        if region["name"] == "AtomicBlock"
    }
]
assert [operation["candidate_partition"] for operation in atomic_layer_operations] == [
    1, 1, 2, 2, 3
], atomic_plan
atomic_result = atomic_model.forward({"inputs": [input_tensor]})
assert atomic_result["status"] == "ok", atomic_result
atomic_actual = np.asarray(
    garnet.tensor_to_cpu(atomic_result["output"]).tolist(), dtype=np.float32
)
np.testing.assert_allclose(atomic_actual, actual)

keyword_cache = SCRIPT_DIR / "cache" / "fusion_keyword_boundary"
shutil.rmtree(keyword_cache, ignore_errors=True)
keyword_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_keyword_boundary_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 2], [1, 2, 4], [2, 4]],
    input_dtypes=["int64", "float32", "float32"],
    compile={"partition": {"preferred_min_operations": 1}},
    cache_dir=str(keyword_cache),
)
keyword_status = keyword_model.runtime_status()
keyword_plan = json.loads(keyword_status["execution_plan_json"])
assert keyword_status["engine_partition_count"] == 2, keyword_status
merge_operation = keyword_plan["operations"][-1]
assert len(merge_operation["input_tensor_ids"]) == 3, merge_operation
merge_partition = keyword_plan["engine_partitions"][-1]
assert sorted(
    binding["request_input_index"]
    for binding in merge_partition["inputs"]
    if binding["request_input_index"] >= 0
) == [0, 1], merge_partition
keyword_result = keyword_model.forward({"inputs": [
    np.asarray([[1, 2]], dtype=np.int64),
    np.zeros((1, 2, 4), dtype=np.float32),
    np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
]})
assert keyword_result["status"] == "ok", keyword_result
keyword_actual = np.asarray(
    garnet.tensor_to_cpu(keyword_result["output"]).tolist(), dtype=np.float32
)
np.testing.assert_allclose(
    keyword_actual,
    np.asarray([[[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]]], dtype=np.float32),
)

invalid_model_path = cache_dir / "invalid_fusion_parameter.x"
invalid_model_path.write_text(
    "from garnet import garnet\n"
    "T = garnet.tensor()\n"
    "@T.fusion(stage=\"decode\")\n"
    "def Model(x):\n"
    "    return x + x\n",
    encoding="utf-8",
)
invalid_model = garnet.load_model(
    str(invalid_model_path),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "invalid_parameter_cache"),
)
invalid_status = invalid_model.runtime_status()
assert invalid_status["state"] == "failed", invalid_status
assert invalid_status["error_code"] == "invalid_fusion_annotation", invalid_status
assert "unsupported parameter 'stage'" in invalid_status["error_message"], invalid_status

conflict_model_path = cache_dir / "conflicting_fusion_regions.x"
conflict_model_path.write_text(
    "from garnet import garnet\n"
    "T = garnet.tensor()\n"
    "@T.fusion(name=\"inner\", boundary=\"required\")\n"
    "def Inner(x):\n"
    "    return x + x\n"
    "@T.fusion(name=\"outer\", atomic=True)\n"
    "def Model(x):\n"
    "    return Inner(x)\n",
    encoding="utf-8",
)
conflict_model = garnet.load_model(
    str(conflict_model_path),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "conflict_cache"),
)
conflict_status = conflict_model.runtime_status()
assert conflict_status["state"] == "failed", conflict_status
assert conflict_status["error_code"] == "invalid_fusion_annotation", conflict_status
assert "inside atomic region 'outer'" in conflict_status["error_message"], conflict_status

print("fusion annotations: PASS")
print(f"regions={len(regions)} candidate_partitions={plan['candidate_partition_count']}")
