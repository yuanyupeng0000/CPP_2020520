/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // bias is a learned parameter; initialize it
    const BiasParameter& param = this->layer_param_.bias_param();
    const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
          << "bias blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> bias_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(param.filler()));
    filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BiasParameter& param = this->layer_param_.bias_param();
  Blob<Dtype>* bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis == 0 in special case where bias is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis == 0 and (therefore) outer_dim_ == 1.
  const int axis = (bias->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
      << "bias blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis;
  for (int i = 0; i < bias->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis + i
        << ") and bias->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis);
  bias_dim_ = bias->count();
  inner_dim_ = bottom[0]->count(axis + bias->num_axes());
  dim_ = bias_dim_ * inner_dim_;
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
  bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
  if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
    caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  for (int n = 0; n < outer_dim_; ++n) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
        inner_dim_, 1, Dtype(1), bias_data,
        bias_multiplier_.cpu_data(), Dtype(1), top_data);
    top_data += dim_;
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_cpu_diff();
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_cpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, bias_multiplier_.cpu_data(), Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(BiasLayer);
#endif

INSTANTIATE_CLASS(BiasLayer);
REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe