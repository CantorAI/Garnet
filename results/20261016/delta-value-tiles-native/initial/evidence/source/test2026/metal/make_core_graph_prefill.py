"""Test explicit core-only compilation for all original FP32 graph projections."""
from pathlib import Path
import sys

source=Path(__file__).with_name('combined_graph_prefill_gpu.mm').read_text()
needle='MPSGraph* graphs[2]={[MPSGraph new],[MPSGraph new]};';assert source.count(needle)==1
source=source.replace(needle,needle+'''auto compilation=[MPSGraphCompilationDescriptor new];compilation.optimizationLevel=MPSGraphOptimizationLevel0;compilation.waitForCompilationCompletion=YES;auto execution=[MPSGraphExecutionDescriptor new];execution.compilationDescriptor=compilation;execution.waitUntilCompleted=YES;''')
needle='[graphs[index] runWithMTLCommandQueue:queue feeds:feeds[index] targetOperations:nil resultsDictionary:results[index]];';assert source.count(needle)==1
source=source.replace(needle,'[graphs[index] runAsyncWithMTLCommandQueue:queue feeds:feeds[index] targetOperations:nil resultsDictionary:results[index] executionDescriptor:execution];')
needle='@"fp32_activation_weight_product_output":@YES,';assert source.count(needle)==1
source=source.replace(needle,needle+'@"graph_optimization_level":@(compilation.optimizationLevel),@"graph_wait_until_completed":@(execution.waitUntilCompleted),')
Path(sys.argv[1]).write_bytes(source.encode())
