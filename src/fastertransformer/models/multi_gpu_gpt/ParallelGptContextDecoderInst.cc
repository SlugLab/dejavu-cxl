// Explicit template instantiations for ParallelGptContextDecoder
// This file ensures all required template specializations are compiled into the library

#include "ParallelGptContextDecoder.cc"

namespace fastertransformer {

// Explicitly instantiate all required template specializations
template class ParallelGptContextDecoder<float>;
template class ParallelGptContextDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptContextDecoder<__nv_bfloat16>;
#endif

}
