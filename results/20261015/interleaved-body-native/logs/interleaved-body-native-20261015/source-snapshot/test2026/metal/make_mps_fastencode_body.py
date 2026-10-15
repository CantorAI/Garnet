"""Keep direct mixed-storage MPS arithmetic; vary only its CPU API validation."""
from make_direct_mps_body import context as direct_context,harness
def context(source):
    source=direct_context(source)
    needle='if(!product)throw std::runtime_error("Cannot create global half-alpha MPS product");';assert source.count(needle)==1
    return source.replace(needle,needle+'\n                    product.options=MPSKernelOptionsSkipAPIValidation;')
