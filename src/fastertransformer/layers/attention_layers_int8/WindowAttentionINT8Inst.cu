// Explicit template instantiations for WindowAttentionINT8
#include "WindowAttentionINT8.cu"

namespace fastertransformer {

template class WindowAttentionINT8<float>;
template class WindowAttentionINT8<half>;

}
