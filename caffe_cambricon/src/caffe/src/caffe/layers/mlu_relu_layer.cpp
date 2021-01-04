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

#ifdef USE_MLU
#include <vector>
#include "caffe/layers/mlu_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  ReLULayer<Dtype>::LayerSetUp(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  this->upper_limit_ = this->layer_param_.relu_param().upper_limit();
  if (this->upper_limit_ > 0 && negative_slope)
     LOG(FATAL) << "Parameter setting is not supported.";
  if (negative_slope != 0) {
    vector<int> slope_shape(4, 1);
    slope_data_.Reshape(slope_shape, cpu_dtype, mlu_dtype, CNML_CONST);
    caffe_set(slope_data_.count(),
              Dtype(negative_slope),
              slope_data_.mutable_cpu_data());
  }
}

template <typename Dtype>
void MLUReLULayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == 1 && top.size() == 1);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  if (top[0] != bottom[0]) {
    top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  } else {
    LOG(INFO) << this->type() << " has in-place bottom/top, skip blob reshape";
  }
}

template <typename Dtype>
void MLUReLULayer<Dtype>::MLUDestroyOp() {
  if (relu_active_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&relu_active_op_ptr_));
    relu_active_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUReLULayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.relu_param().negative_slope() != 0) {
    MLU_CHECK(cnmlCreatePreluOp(&relu_active_op_ptr_,
                               bottom[0]->mlu_tensor(),
                               top[0]->mlu_tensor(),
                               slope_data_.mlu_tensor()));
    MLU_CHECK(cnmlBindConstData_V2(slope_data_.mlu_tensor(),
                                   slope_data_.sync_data(),
                                   false));
  } else {
    cnmlActiveFunction_t function = cnmlActiveFunction_t::CNML_ACTIVE_RELU;
    if (this->upper_limit_ == 1)
      function = cnmlActiveFunction_t::CNML_ACTIVE_RELU1;
    else if (this->upper_limit_ == 6)
      function = cnmlActiveFunction_t::CNML_ACTIVE_RELU6;
    else if (this->upper_limit_ != 0)
      LOG(FATAL) << "The specified upper_limit is not supported.";
    
    MLU_CHECK(cnmlCreateActiveOp(&relu_active_op_ptr_,
                                function,
                                bottom[0]->mlu_tensor(),
                                top[0]->mlu_tensor()));
  }
}

template<typename Dtype>
void MLUReLULayer<Dtype>::MLUCompileOp() {
    MLU_CHECK(cnmlCompileBaseOp(relu_active_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
}

template <typename Dtype>
MLUReLULayer<Dtype>::~MLUReLULayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUReLULayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeActiveOpForward_V4(relu_active_op_ptr_,
                                      bottom[0]->mlu_tensor_rt(),
                                      bottom[0]->mutable_mlu_data(),
                                      top[0]->mlu_tensor_rt(),
                                      top[0]->mutable_mlu_data(),
                                      Caffe::queue(), nullptr));
}
template<typename Dtype>
void MLUReLULayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(relu_active_op_ptr_);
}

INSTANTIATE_CLASS(MLUReLULayer);

}  // namespace caffe
#endif  // USE_MLU
