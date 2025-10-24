// This TU force-instantiates FasterTransformer's ParallelGptContextDecoder
// into the th_transformer shared library, ensuring its templated symbols
// (constructors, forward, dtor) are available to resolve references from
// ParallelGpt and Torch ops without depending on transformer-shared.
//
// Note: We intentionally include the .cc implementation unit here to
// guarantee the explicit template instantiations are emitted in this
// shared object. This avoids cross-library device-link issues.

// Provide a minimal stub for DejaVuClient so the implementation can
// compile without pulling in protobuf-dependent headers.
class DejaVuClient {
public:
    void MarkComplete(int) {}
};

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.cc"
