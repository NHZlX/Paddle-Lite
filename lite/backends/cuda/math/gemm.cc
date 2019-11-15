// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/cuda/math/gemm.h"
#include <iostream>
#include "lite/core/device_info.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <>
bool Gemm<float, float>::init(const bool trans_a,
                              bool trans_b,
                              const int m,
                              const int n,
                              const int k,
                              Context<TARGET(kCUDA)> *ctx) {
  if (cu_handle_ == nullptr) {
    this->exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasCreate(&cu_handle_));
    CUBLAS_CALL(cublasSetStream(cu_handle_, this->exe_stream_));
  }
  lda_ = (!trans_a) ? k : m;
  ldb_ = (!trans_b) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return true;
}

template <>
bool Gemm<float, float>::run(const float alpha,
                             const float beta,
                             const float *a,
                             const float *b,
                             float *c,
                             Context<TARGET(kCUDA)> *ctx) {
  CUBLAS_CALL(cublasSgemm(cu_handle_,
                          cu_trans_b_,
                          cu_trans_a_,
                          n_,
                          m_,
                          k_,
                          &alpha,
                          b,
                          ldb_,
                          a,
                          lda_,
                          &beta,
                          c,
                          ldc_));
  return true;
}

template <>
bool Gemm<int8_t, float>::init(const bool trans_a,
                               bool trans_b,
                               const int m,
                               const int n,
                               const int k,
                               Context<TARGET(kCUDA)> *ctx) {
  if (cu_handle_ == nullptr) {
    cublasCreate_v2(&cu_handle_);
    exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasSetStream(cu_handle_, exe_stream_));
  }
  lda_ = (!trans_a) ? k : m;
  ldb_ = (!trans_b) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return true;
}

template <>
bool Gemm<int8_t, float>::run(const float alpha,
                              const float beta,
                              const int8_t *a,
                              const int8_t *b,
                              float *c,
                              Context<TARGET(kCUDA)> *ctx) {
  int generate_arch =
      Env<TARGET(kCUDA)>::Global()[ctx->device_id()].generate_arch();

  bool arch_check = generate_arch == 61;
  if (arch_check) {
#if __CUDACC_VER_MAJOR__ >= 9
    CUBLAS_CALL(cublasGemmEx(cu_handle_,
                             cu_trans_b_,
                             cu_trans_a_,
                             n_,
                             m_,
                             k_,
                             &alpha,
                             b,
                             CUDA_R_8I,
                             ldb_,
                             a,
                             CUDA_R_8I,
                             lda_,
                             &beta,
                             c,
                             CUDA_R_32F,
                             ldc_,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT));
#else
    CUBLAS_CALL(cublasSgemmEx(cu_handle_,
                              cu_trans_b_,
                              cu_trans_a_,
                              n_,
                              m_,
                              k_,
                              &alpha,
                              b,
                              CUDA_R_8I,
                              ldb_,
                              a,
                              CUDA_R_8I,
                              lda_,
                              &beta,
                              c,
                              CUDA_R_32F,
                              ldc_));
#endif
  } else {
    CUBLAS_CALL(cublasSgemmEx(cu_handle_,
                              cu_trans_b_,
                              cu_trans_a_,
                              n_,
                              m_,
                              k_,
                              &alpha,
                              b,
                              CUDA_R_8I,
                              ldb_,
                              a,
                              CUDA_R_8I,
                              lda_,
                              &beta,
                              c,
                              CUDA_R_32F,
                              ldc_));
  }
  return true;
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
