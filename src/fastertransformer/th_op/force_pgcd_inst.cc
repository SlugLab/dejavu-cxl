// Force-instantiate FasterTransformer's ParallelGptContextDecoder templates
// into the th_transformer shared library, ensuring the required templated
// constructors and methods are available at runtime to resolve references
// from Torch ops without depending on transformer-shared.

// Provide a minimal stub for DejaVuClient so the implementation can
// compile without pulling in protobuf-dependent headers.
class DejaVuClient {
public:
    void MarkComplete(int) {}
};

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.cc"

