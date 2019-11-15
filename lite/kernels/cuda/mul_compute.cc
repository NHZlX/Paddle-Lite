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

#include "lite/kernels/cuda/mul_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void MulComputeInt8::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  int x_h = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  int y_w =
      static_cast<int>(param.y->dims()
                           .Slice(param.y_num_col_dims, param.y->dims().size())
                           .production());
  m_ = x_h;
  n_ = y_w;
  k_ = y_h;
  gemm_impl_.reset(new lite::cuda::math::Gemm<int8_t, float>);
  gemm_impl_->init(false, false, x_h, y_w, y_h, &context);
}

void MulComputeInt8::Run() {
  CHECK(ctx_) << "running context should be set first";
  auto& context = this->ctx_->template As<CUDAContext>();
  auto& param = this->Param<param_t>();
  const auto* x_data = param.x->data<int8_t>();
  const auto* y_data = param.y->data<int8_t>();
  auto* out_data = param.output->mutable_data<float>(TARGET(kCUDA));
  float in_scale = param.input_scale;
  std::vector<float> weight_scale = param.weight_scale;
  CHECK(weight_scale.size() == 1)
      << "In int8 mode, The mul's scale size should be 1";
  float alpha = in_scale * weight_scale[0];

  int x_h = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int x_w =
      static_cast<int>(param.x->dims()
                           .Slice(param.x_num_col_dims, param.x->dims().size())
                           .production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  int y_w =
      static_cast<int>(param.y->dims()
                           .Slice(param.y_num_col_dims, param.y->dims().size())
                           .production());
  if (!(x_h == m_ && k_ == y_h && n_ == y_w)) {
    m_ = x_h;
    k_ = y_h;
    n_ = y_w;
    gemm_impl_->init(false, false, x_h, y_w, y_h, &context);
  }
  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  gemm_impl_->run(alpha, 0., x_data, y_data, out_data, &context);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mul, kCUDA, kFloat, kNCHW, paddle::lite::kernels::cuda::MulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    mul, kCUDA, kInt8, kNCHW, paddle::lite::kernels::cuda::MulComputeInt8, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt8))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt8))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();
