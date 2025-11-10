// Force-instantiate FasterTransformer's ParallelGptContextDecoder templates
// into the th_transformer shared library, ensuring the required templated
// constructors and methods are available at runtime to resolve references
// from Torch ops without depending on transformer-shared.

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"

// Explicit template instantiations for float, half, and bfloat16
namespace fastertransformer {

template class ParallelGptContextDecoder<float>;
template class ParallelGptContextDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptContextDecoder<__nv_bfloat16>;
#endif

}

