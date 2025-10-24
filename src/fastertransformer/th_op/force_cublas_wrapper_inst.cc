// Force-emit FasterTransformer cublasMMWrapper definitions into th_transformer
// to satisfy runtime references without depending on transformer-shared.

#include "src/fastertransformer/utils/cublasMMWrapper.cc"

