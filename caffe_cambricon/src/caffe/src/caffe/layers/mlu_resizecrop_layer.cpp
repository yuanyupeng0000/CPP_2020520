/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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
#include "caffe/layers/mlu_resizecrop_layer.hpp"
#include <vector>

namespace caffe {
typedef uint16_t half;

template <typename Dtype>
void MLUResizecropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ResizecropParameter resize_param = this->layer_param().resize_crop_param();
  resize_h_ = resize_param.resize_h();
  resize_w_ = resize_param.resize_w();
  keepAspectRatio_ = resize_param.equalproportion_mode();
  bottom[0]->set_preprocess(false);
  top[0]->set_preprocess(false);
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = DT_UINT8;
  BaseDataType mlu_dtype = DT_UINT8;
  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 4;
  top_shape[2] = resize_h_;
  top_shape[3] = resize_w_;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  bottom[0]->set_cpu_type(cpu_dtype);
  bottom[0]->set_mlu_type(mlu_dtype);
  bottom[0]->set_preprocess(false);
  top[0]->set_preprocess(false);
  for (int i = 1; i < 4; i++) {
    bottom[i]->set_cpu_type(DT_FLOAT32);
    bottom[i]->set_mlu_type(DT_INT32);
    bottom[i]->set_preprocess(false);
  }
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int s_col = bottom[0]->width();
  int s_row = bottom[0]->height();
  int d_col = top[0]->width();
  int d_row = top[0]->height();
  int batchNum = bottom[0]->num();
  ioParams mode;
  mode.color = RGBA_TO_RGBA;
  mode.datatype = UINT8_TO_UINT8;
  cnmlCreatePluginCropAndResizeOpParam_V2(&param_, s_row, s_col, d_row, d_col,
                                          mode, batchNum, keepAspectRatio_,
                                          Caffe::rt_core());
  // prepare cnml tensor
  auto dst_cnml = top[0]->mlu_tensor();
  cnmlTensor_t src_cnmls[4];
  src_cnmls[0] = bottom[0]->mlu_tensor();
  src_cnmls[1] = bottom[1]->mlu_tensor();
  src_cnmls[2] = bottom[2]->mlu_tensor();
  src_cnmls[3] = bottom[3]->mlu_tensor();
  cnmlCreatePluginCropAndResizeOp_V2(&crop_resize_op_, param_, &dst_cnml, src_cnmls);
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(crop_resize_op_, Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  auto dst_cnml = top[0]->mlu_tensor();
  cnmlTensor_t src_cnmls[4];
  src_cnmls[0] = bottom[0]->mlu_tensor();
  src_cnmls[1] = bottom[1]->mlu_tensor();
  src_cnmls[2] = bottom[2]->mlu_tensor();
  src_cnmls[3] = bottom[3]->mlu_tensor();
  void* src_addrs_[4];
  void* dst_addrs_[1];
  src_addrs_[0] = bottom[0]->mutable_mlu_data();
  src_addrs_[1] = bottom[1]->mutable_mlu_data();
  src_addrs_[2] = bottom[2]->mutable_mlu_data();
  src_addrs_[3] = bottom[3]->mutable_mlu_data();
  dst_addrs_[0] = top[0]->mutable_mlu_data();

  MLU_CHECK(cnmlComputePluginCropAndResizeOpForward_V2(
      crop_resize_op_, param_, src_cnmls, src_addrs_,
      &dst_cnml, dst_addrs_, Caffe::queue()));
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(crop_resize_op_);
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::MLUDestroyOp() {
  if (param_ != nullptr) {
    cnmlDestroyPluginCropAndResizeOpParam(&param_);
    param_ = nullptr;
  }
  if (crop_resize_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&crop_resize_op_));
    crop_resize_op_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUResizecropLayer);

}  // namespace caffe
#endif  // USE_MLU
