// Stub implementation for WindowAttentionINT8 constructors and destructors
// These are not used in the GPT codebase but are referenced by the linker

#include "src/fastertransformer/layers/attention_layers_int8/WindowAttentionINT8.h"

namespace fastertransformer {

// Stub constructor for float
template<>
WindowAttentionINT8<float>::WindowAttentionINT8(
    int              max_batch,
    int              window_size,
    cudaStream_t     stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator*      allocator,
    bool             is_free_buffer_after_forward,
    bool             sparse,
    float            q_scaling,
    int              version):
    BaseAttentionLayer<float>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse)
{
    // Stub - not implemented
    throw std::runtime_error("WindowAttentionINT8<float> is not implemented in this build");
}

// Stub destructor for float
template<>
WindowAttentionINT8<float>::~WindowAttentionINT8()
{
    // Stub - nothing to clean up
}

// Stub constructor for half
template<>
WindowAttentionINT8<half>::WindowAttentionINT8(
    int              max_batch,
    int              window_size,
    cudaStream_t     stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator*      allocator,
    bool             is_free_buffer_after_forward,
    bool             sparse,
    float            q_scaling,
    int              version):
    BaseAttentionLayer<half>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse)
{
    // Stub - not implemented
    throw std::runtime_error("WindowAttentionINT8<half> is not implemented in this build");
}

// Stub destructor for half
template<>
WindowAttentionINT8<half>::~WindowAttentionINT8()
{
    // Stub - nothing to clean up
}

// Stub allocateBuffer for float
template<>
void WindowAttentionINT8<float>::allocateBuffer()
{
    // Stub - not implemented
    throw std::runtime_error("WindowAttentionINT8<float>::allocateBuffer is not implemented in this build");
}

// Stub allocateBuffer for half
template<>
void WindowAttentionINT8<half>::allocateBuffer()
{
    // Stub - not implemented
    throw std::runtime_error("WindowAttentionINT8<half>::allocateBuffer is not implemented in this build");
}

// Stub freeBuffer for float
template<>
void WindowAttentionINT8<float>::freeBuffer()
{
    // Stub - not implemented
}

// Stub freeBuffer for half
template<>
void WindowAttentionINT8<half>::freeBuffer()
{
    // Stub - not implemented
}

// Stub forward for float
template<>
void WindowAttentionINT8<float>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<float, float>* attention_weights)
{
    // Stub - not implemented
    throw std::runtime_error("WindowAttentionINT8<float>::forward is not implemented in this build");
}

// Stub forward for half
template<>
void WindowAttentionINT8<half>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<half, half>* attention_weights)
{
    // Stub - not implemented
    throw std::runtime_error("WindowAttentionINT8<half>::forward is not implemented in this build");
}

}  // namespace fastertransformer
