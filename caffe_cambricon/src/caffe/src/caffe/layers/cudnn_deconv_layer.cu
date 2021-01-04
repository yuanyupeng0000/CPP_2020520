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

#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_deconv_layer.hpp"

namespace caffe {

__global__ void sync_deconv_groups() {}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          handle_[g],
          cudnn::dataType<Dtype>::one,
          filter_desc_,
          weight + this->weight_offset_ * g,
          bottom_descs_[i],
          bottom_data + bottom_offset_ * g,
          conv_descs_[i],
          bwd_data_algo_[i],
          workspace[g],
          workspace_bwd_data_sizes_[i],
          cudnn::dataType<Dtype>::zero,
          top_descs_[i],
          top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
                                   cudnn::dataType<Dtype>::one,
                                   bias_desc_,
                                   bias_data + bias_offset_ * g,
                                   cudnn::dataType<Dtype>::one,
                                   top_descs_[i],
                                   top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_deconv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0 * this->group_ + g],
                                                 cudnn::dataType<Dtype>::one,
                                                 top_descs_[i],
                                                 top_diff + top_offset_ * g,
                                                 cudnn::dataType<Dtype>::one,
                                                 bias_desc_,
                                                 bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            handle_[1 * this->group_ + g],
            cudnn::dataType<Dtype>::one,
            top_descs_[i],
            top_diff + top_offset_ * g,
            bottom_descs_[i],
            bottom_data + bottom_offset_ * g,
            conv_descs_[i],
            bwd_filter_algo_[i],
            workspace[1 * this->group_ + g],
            workspace_bwd_filter_sizes_[i],
            cudnn::dataType<Dtype>::one,
            filter_desc_,
            weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(
            cudnnConvolutionForward(handle_[2 * this->group_ + g],
                                    cudnn::dataType<Dtype>::one,
                                    top_descs_[i],
                                    top_diff + top_offset_ * g,
                                    filter_desc_,
                                    weight + this->weight_offset_ * g,
                                    conv_descs_[i],
                                    fwd_algo_[i],
                                    workspace[2 * this->group_ + g],
                                    workspace_fwd_sizes_[i],
                                    cudnn::dataType<Dtype>::zero,
                                    bottom_descs_[i],
                                    bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_deconv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNDeconvolutionLayer);

}  // namespace caffe
#endif
